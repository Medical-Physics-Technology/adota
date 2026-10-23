"""The difficulty score refit per engine, out of fold by scan.

Same form as the deployed score s(c) (:mod:`src.acquisition.scorer`,
``scripts/analysis/acquisition_frozen_test.py``): every metric goes through its
101-point percentile transform, with the grids taken from the training rows,
and a ridge regression (alpha 1) combines them. Two changes make it a
**within-cell** score, so that it can only explain what differs between beamlets
of one scan at one energy:

- the target is the within-cell rank of the engine's error, centred per cell,
  instead of ``log(1 + RDE)``;
- the percentile features are centred per cell and then standardised with the
  training rows' spread, so the coefficients are per standard deviation.

Centring a feature per cell only shifts a score by a constant inside each cell,
so the within-cell ordering of ``b + sum_k w_k p_k`` is the fitted score's, and
:meth:`ScoreFit.scorer` returns it as a :class:`DifficultyScorer`. Folds are
whole scans (leave one scan out); a cell never spans two scans, so centring
never mixes folds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.acquisition.scorer import FULL, DifficultyScorer, percentile_grids, percentile_transform
from src.active_learning.ensemble.agreement import scan_resamples
from src.active_learning.hardness.cells import (
    MetricClusters,
    bootstrap_interval,
    cell_ranks,
    cell_spearman,
    centre_within_cells,
    nanmedian_columns,
)

RIDGE_ALPHA = 1.0
"""The full score's ridge penalty in the study."""


@dataclass
class Design:
    """Standardised, cell-centred percentile features of some rows."""

    z: np.ndarray
    grids: Dict[str, np.ndarray]
    mean: np.ndarray
    scale: np.ndarray


def design(frame: pd.DataFrame, metrics: Sequence[str], cells: pd.Series, train: np.ndarray) -> Design:
    """Features of every row of ``frame`` with grids, mean and spread from the
    rows flagged in ``train``."""
    grids = percentile_grids(frame.loc[train], metrics)
    p = pd.DataFrame({m: percentile_transform(grids[m], frame[m].to_numpy()) for m in metrics}, index=frame.index)
    pc = centre_within_cells(p, cells).to_numpy()
    mean = pc[train].mean(axis=0)
    scale = pc[train].std(axis=0)
    scale[scale <= 1e-12] = 1.0    # no within-cell variation: the column stays zero
    return Design((pc - mean) / scale, grids, mean, scale)


def rank_target(values: pd.Series, cells: pd.Series) -> np.ndarray:
    """Within-cell rank of ``values``, centred per cell: what the score is fitted to."""
    ranks = cell_ranks(values.to_frame(), cells)
    return centre_within_cells(ranks, cells).iloc[:, 0].to_numpy()


@dataclass
class ScoreFit:
    """One ridge fit on standardised features."""

    metrics: List[str]
    coef: np.ndarray            # per standard deviation of the cell-centred percentile
    design: Design

    def scorer(self, label: str = "") -> DifficultyScorer:
        """The fit as a frozen linear score on raw percentiles (rank-equivalent within cells)."""
        weights = self.coef / self.design.scale
        intercept = float(-(weights * self.design.mean).sum())
        return DifficultyScorer(variant=FULL, intercept=intercept,
                                weights={m: float(w) for m, w in zip(self.metrics, weights)},
                                grids=self.design.grids, source=label)


def fit_score(frame: pd.DataFrame, metrics: Sequence[str], target: pd.Series, cells: pd.Series,
              train: np.ndarray, alpha: float = RIDGE_ALPHA) -> ScoreFit:
    d = design(frame, metrics, cells, train)
    y = rank_target(target, cells)
    model = Ridge(alpha=alpha).fit(d.z[train], y[train])
    return ScoreFit(list(metrics), np.asarray(model.coef_, dtype=float), d)


@dataclass
class OutOfFold:
    """Leave-one-scan-out scores: ``score`` holds each row's prediction by the
    fit that did not see its scan."""

    score: np.ndarray
    fold_of_row: np.ndarray
    fits: Dict[str, ScoreFit]


def out_of_fold(frame: pd.DataFrame, metrics: Sequence[str], target: pd.Series, cells: pd.Series,
                scans: pd.Series, alpha: float = RIDGE_ALPHA) -> OutOfFold:
    scan_values = scans.to_numpy()
    score = np.full(len(frame), np.nan)
    fits = {}
    for scan in sorted(pd.unique(scan_values)):
        test = scan_values == scan
        if test.all():
            raise ValueError("leave-one-scan-out needs at least two scans")
        fit = fit_score(frame, metrics, target, cells, ~test, alpha)
        score[test] = fit.design.z[test] @ fit.coef
        fits[str(scan)] = fit
    return OutOfFold(score, scan_values.astype(str), fits)


def evaluate_scores(scores: pd.DataFrame, targets: pd.DataFrame, cells: pd.Series, scans: pd.Series,
                    min_cell_size: int, n_boot: int, seed: int, ci: float) -> pd.DataFrame:
    """Median within-cell Spearman of every score column with every target
    column, with scan-bootstrap intervals: one row per ``(score, target)``."""
    rows = []
    for s in scores.columns:
        corr = cell_spearman(targets, scores[s], cells, scans, min_cell_size)
        point, lo, hi = bootstrap_interval(corr.values, corr.cells["scan"], nanmedian_columns, n_boot, seed, ci)
        for j, t in enumerate(targets.columns):
            rows.append({"score": s, "target": t, "median_cell_spearman": point[j], "ci_lo": lo[j], "ci_hi": hi[j],
                         "n_cells": int(np.isfinite(corr.values[:, j]).sum()),
                         "n_scans": int(corr.cells["scan"].nunique()), "n_records": int(corr.cells["n"].sum())})
    return pd.DataFrame(rows)


def coefficients(frame: pd.DataFrame, metrics: Sequence[str], target: pd.Series, cells: pd.Series,
                 scans: pd.Series, clusters: MetricClusters, n_boot: int, seed: int, ci: float,
                 alpha: float = RIDGE_ALPHA) -> pd.DataFrame:
    """Standardised coefficients of the fit on every row, with intervals from
    refits on scan-bootstrap resamples."""
    every = np.ones(len(frame), dtype=bool)
    full = fit_score(frame, metrics, target, cells, every, alpha)
    unique_scans = np.asarray(sorted(pd.unique(scans.to_numpy())))
    rows_of = {s: np.flatnonzero(scans.to_numpy() == s) for s in unique_scans}
    draws = []
    for picked in scan_resamples(unique_scans, n_boot, seed):
        rows = np.concatenate([rows_of[s] for s in unique_scans[picked]])
        sub = frame.iloc[rows].reset_index(drop=True)
        sub_cells = pd.Series(cells.to_numpy()[rows])
        if pd.unique(scans.to_numpy()[rows]).size < 2:
            continue
        fit = fit_score(sub, metrics, target.iloc[rows].reset_index(drop=True), sub_cells,
                        np.ones(len(sub), dtype=bool), alpha)
        draws.append(fit.coef)
    draws = np.asarray(draws)
    lo, hi = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0
    return pd.DataFrame({"metric": list(metrics), "cluster": [clusters.assignment[m] for m in metrics],
                         "cluster_name": [clusters.names[clusters.assignment[m]] for m in metrics],
                         "coef_std": full.coef,
                         "ci_lo": np.percentile(draws, lo, axis=0) if len(draws) else np.nan,
                         "ci_hi": np.percentile(draws, hi, axis=0) if len(draws) else np.nan,
                         "n_bootstrap": len(draws)})


def permutation_importance(oof: OutOfFold, target: pd.Series, cells: pd.Series, scans: pd.Series,
                           clusters: MetricClusters, min_cell_size: int, n_repeats: int, n_boot: int,
                           seed: int, ci: float) -> pd.DataFrame:
    """Drop in out-of-fold median within-cell Spearman when one cluster's metrics
    are permuted jointly inside every cell of the held-out scan.

    Permuting inside cells keeps each metric's cell-level distribution, so the
    drop measures only what the cluster contributes within cells. Per cell the
    permuted Spearman is averaged over ``n_repeats``; the interval is the scan
    bootstrap of ``median(base) - median(permuted)``.
    """
    rng = np.random.default_rng(seed)
    cell_values = cells.to_numpy()
    metrics = next(iter(oof.fits.values())).metrics
    groups = {f: [np.flatnonzero((oof.fold_of_row == f) & (cell_values == c))
                  for c in pd.unique(cell_values[oof.fold_of_row == f])] for f in oof.fits}
    base = cell_spearman(pd.DataFrame({"score": oof.score}), target, cells, scans, min_cell_size)
    columns = [base.values[:, 0]]
    for c in clusters.ids:
        cols = [metrics.index(m) for m in clusters.members(c)]
        acc = np.zeros(len(base.values))
        for _ in range(n_repeats):
            score = oof.score.copy()
            for f, fit in oof.fits.items():
                z = fit.design.z
                for idx in groups[f]:
                    zp = z[idx].copy()
                    zp[:, cols] = z[rng.permutation(idx)][:, cols]
                    score[idx] = zp @ fit.coef
            acc += cell_spearman(pd.DataFrame({"s": score}), target, cells, scans, min_cell_size).values[:, 0]
        columns.append(acc / n_repeats)
    stacked = np.column_stack(columns)

    def drop(values: np.ndarray) -> np.ndarray:
        med = nanmedian_columns(values)
        return med[0] - med[1:]

    point, lo, hi = bootstrap_interval(stacked, base.cells["scan"], drop, n_boot, seed, ci)
    return pd.DataFrame({"cluster": clusters.ids, "cluster_name": [clusters.names[c] for c in clusters.ids],
                         "n_metrics": [len(clusters.members(c)) for c in clusters.ids],
                         "importance": point, "ci_lo": lo, "ci_hi": hi,
                         "base_median_cell_spearman": float(nanmedian_columns(stacked[:, :1])[0]),
                         "n_repeats": n_repeats})


def transfer_summary(matrix: pd.DataFrame, cell_values: Mapping[str, np.ndarray], cell_scans: Sequence[str],
                     engines: Sequence[str], n_boot: int, seed: int, ci: float) -> pd.DataFrame:
    """Diagonal against off-diagonal of the cross-transfer matrix.

    ``cell_values[score]`` is the ``(n_cells, n_engines)`` within-cell Spearman of
    one engine-fitted score with every engine's error, columns in ``engines``
    order. Returns the mean diagonal, the mean off-diagonal and their
    difference, each with a scan-bootstrap interval.
    """
    k = len(engines)
    stacked = np.concatenate([cell_values[e] for e in engines], axis=1)    # (n_cells, k * k), score-major

    def stat(values: np.ndarray) -> np.ndarray:
        med = nanmedian_columns(values).reshape(k, k)
        diag = np.nanmean(np.diag(med))
        off = np.nanmean(med[~np.eye(k, dtype=bool)]) if k > 1 else np.nan
        return np.array([diag, off, diag - off])

    point, lo, hi = bootstrap_interval(stacked, cell_scans, stat, n_boot, seed, ci)
    return pd.DataFrame({"quantity": ["diagonal_mean", "off_diagonal_mean", "diagonal_minus_off"],
                         "value": point, "ci_lo": lo, "ci_hi": hi, "n_engines": k})


def scorer_json(fits: Mapping[str, ScoreFit], note: str, extra: Optional[Dict] = None) -> Dict:
    """Full-data fits in the arms-keyed layout :meth:`DifficultyScorer.load` reads
    (``arm`` = target name)."""
    arms = {}
    for name, fit in fits.items():
        s = fit.scorer(name)
        arms[name] = {"weights": {FULL: {"intercept": s.intercept, "coefficients": dict(s.weights)}},
                      "percentile_grids_dev": {m: g.tolist() for m, g in s.grids.items()}}
    return {"note": note, "features": next(iter(fits.values())).metrics if fits else [], "arms": arms,
            **(extra or {})}
