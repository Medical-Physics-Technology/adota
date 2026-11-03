"""Is the difficulty score model-independent? The EXP-0014 test, without training.

Every member of the ensemble has an out-of-sample error on the same beamlets.
Each member's ``log(1 + RDE)`` is standardised by its own mean and spread on the
fit set, and the shared target is the mean of the standardised errors over
members. The score is the deployed form (:mod:`src.acquisition.scorer`): every
metric through its 101-point percentile transform, a ridge on the 30 metrics
(alpha 1) or a Lasso for the sparse variant.

The primary endpoint is the leave-one-member-out gap. For each member ``k`` two
scores are fitted on the fit set: one to ``k``'s own standardised error, one to
the mean of the others. On a test set,

    gap_k = Pearson(own score, z_k) - Pearson(others' score, z_k),

and the endpoint is the median of ``gap_k`` over members, with an interval from
a bootstrap over test scans (patients). A small gap says a score fitted without
a model ranks that model's errors as well as one fitted to it.

Pearson over a resample of whole scans is a function of six per-scan sums, so the
bootstrap draws scan counts and never touches the rows again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold

from src.acquisition.scorer import FULL, SPARSE, DifficultyScorer, percentile_grids, percentile_transform

RIDGE_ALPHA = 1.0
SPARSE_ALPHA = 0.0008
"""The study's Lasso penalty, for its target ``log(1 + RDE)``."""
REFERENCE_TARGET_SD = 0.0704
"""Spread of the study's ``log(1 + RDE)`` (``results.csv`` of run 20260707_124010):
the Lasso penalty is rescaled by the target's spread over this, so the sparse
score keeps the study's sparsity on a standardised target."""
ENERGY_BIN_MEV = 10.0


def log_errors(paths: Mapping[str, str], metric: str = "rde_pct") -> pd.DataFrame:
    """``log(1 + metric)`` of every member, one column per label, indexed by sample id."""
    cols = {label: pd.read_csv(path, usecols=["sample_id", metric]).set_index("sample_id")[metric]
            for label, path in paths.items()}
    return np.log1p(pd.DataFrame(cols))


@dataclass(frozen=True)
class Standardiser:
    """Each member's mean and spread on the fit set, frozen for every test set."""

    mean: pd.Series
    sd: pd.Series

    @classmethod
    def fit(cls, errors: pd.DataFrame) -> "Standardiser":
        return cls(errors.mean(), errors.std(ddof=0))

    def __call__(self, errors: pd.DataFrame) -> pd.DataFrame:
        return (errors - self.mean[errors.columns]) / self.sd[errors.columns]


def fit_scorer(features: pd.DataFrame, metrics: Sequence[str], y: np.ndarray, variant: str = FULL,
               label: str = "") -> DifficultyScorer:
    """The deployed score form fitted to ``y``: grids from these rows, ridge or Lasso."""
    grids = percentile_grids(features, metrics)
    x = np.column_stack([percentile_transform(grids[m], features[m].to_numpy()) for m in metrics])
    if variant == FULL:
        model = Ridge(alpha=RIDGE_ALPHA)
    elif variant == SPARSE:
        model = Lasso(alpha=SPARSE_ALPHA * float(np.std(y)) / REFERENCE_TARGET_SD, max_iter=20000)
    else:
        raise ValueError(f"variant must be {FULL!r} or {SPARSE!r}, got {variant!r}")
    model.fit(x, y)
    weights = {m: float(w) for m, w in zip(metrics, model.coef_) if w != 0.0}
    return DifficultyScorer(variant=variant, intercept=float(model.intercept_), weights=weights,
                            grids=grids, source=label)


def energy_baseline(fit_energy: np.ndarray, y: np.ndarray, energy: np.ndarray,
                    bin_mev: float = ENERGY_BIN_MEV) -> np.ndarray:
    """The energy-only score: the fit set's mean target per energy bin, the
    nearest populated bin for an energy the fit set never had."""
    fit_bins = np.floor(np.asarray(fit_energy) / bin_mev).astype(int)
    means = pd.Series(y).groupby(fit_bins).mean()
    bins = np.floor(np.asarray(energy) / bin_mev).astype(int)
    nearest = means.index.to_numpy()[np.abs(bins[:, None] - means.index.to_numpy()[None, :]).argmin(axis=1)]
    return means.loc[nearest].to_numpy()


def scan_sums(x: np.ndarray, y: np.ndarray, scans: np.ndarray, order: Sequence[str]) -> np.ndarray:
    """Per-scan ``(n, sum x, sum y, sum xx, sum yy, sum xy)``, rows in ``order``."""
    frame = pd.DataFrame({"s": scans, "n": 1.0, "x": x, "y": y, "xx": x * x, "yy": y * y, "xy": x * y})
    return frame.groupby("s")[["n", "x", "y", "xx", "yy", "xy"]].sum().loc[list(order)].to_numpy()


def pearson_from_sums(sums: np.ndarray) -> np.ndarray:
    """Pearson from summed statistics; ``sums`` is ``(..., 6)``."""
    n, sx, sy, sxx, syy, sxy = np.moveaxis(sums, -1, 0)
    cov = sxy - sx * sy / n
    return cov / np.sqrt((sxx - sx * sx / n) * (syy - sy * sy / n))


def scan_resample_counts(n_scans: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot, n_scans)`` multiplicities of a bootstrap over scans."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_scans, size=(n_boot, n_scans))
    return np.stack([np.bincount(d, minlength=n_scans) for d in draws]).astype(float)


@dataclass
class EvalSet:
    """One evaluation population: features, standardised errors and scan keys."""

    name: str
    features: pd.DataFrame
    z: pd.DataFrame
    scans: np.ndarray

    @property
    def scan_order(self) -> List[str]:
        return sorted(pd.unique(self.scans).astype(str))


def pearson_with_interval(score: np.ndarray, target: np.ndarray, test: EvalSet, counts: np.ndarray,
                          ci: float = 0.95) -> Tuple[float, float, float, np.ndarray]:
    sums = scan_sums(score, target, test.scans.astype(str), test.scan_order)
    boot = pearson_from_sums(counts @ sums)
    lo, hi = np.nanquantile(boot, [(1 - ci) / 2, (1 + ci) / 2])
    return float(pearson_from_sums(sums.sum(axis=0))), float(lo), float(hi), boot


def leave_one_member_out(fit_features: pd.DataFrame, fit_z: pd.DataFrame, metrics: Sequence[str],
                         tests: Sequence[EvalSet], members: Sequence[str], variant: str = FULL,
                         n_boot: int = 2000, seed: int = 0, ci: float = 0.95
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per member and test set, the own-fit and others'-fit Pearson and their
    gap; plus the median gap over members with its scan-bootstrap interval."""
    own = {k: fit_scorer(fit_features, metrics, fit_z[k].to_numpy(), variant, f"own:{k}") for k in members}
    others = {k: fit_scorer(fit_features, metrics, fit_z[[m for m in members if m != k]].mean(axis=1).to_numpy(),
                            variant, f"others:{k}") for k in members}
    rows, summary = [], []
    for test in tests:
        counts = scan_resample_counts(len(test.scan_order), n_boot, seed)
        gaps_boot = []
        for k in members:
            target = test.z[k].to_numpy()
            r_own, _, _, b_own = pearson_with_interval(own[k].score(test.features), target, test, counts, ci)
            r_oth, _, _, b_oth = pearson_with_interval(others[k].score(test.features), target, test, counts, ci)
            g = b_own - b_oth
            gaps_boot.append(g)
            lo, hi = np.nanquantile(g, [(1 - ci) / 2, (1 + ci) / 2])
            rows.append({"test_set": test.name, "variant": variant, "member": k, "pearson_own": r_own,
                         "pearson_others": r_oth, "gap": r_own - r_oth, "gap_ci_lo": lo, "gap_ci_hi": hi,
                         "n_records": len(target), "n_scans": len(test.scan_order)})
        point = float(np.median([r["gap"] for r in rows[-len(members):]]))
        med_boot = np.median(np.stack(gaps_boot), axis=0)
        lo, hi = np.nanquantile(med_boot, [(1 - ci) / 2, (1 + ci) / 2])
        summary.append({"test_set": test.name, "variant": variant, "median_gap": point, "ci_lo": float(lo),
                        "ci_hi": float(hi), "n_members": len(members), "n_records": len(test.z),
                        "n_scans": len(test.scan_order)})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def verdict(median_gap: float, ci_hi: float, tight: float = 0.03, tight_ci: float = 0.05,
            loose: float = 0.10) -> str:
    """The EXP-0014 decision rule, fixed before any number existed."""
    if median_gap <= tight and ci_hi <= tight_ci:
        return "model-independent"
    if median_gap <= loose:
        return "approximately model-independent"
    return "model-specific"


def within_test_cv_gap(test: EvalSet, metrics: Sequence[str], engines: Sequence[str], ladder: Sequence[str],
                       n_splits: int = 5, variant: str = FULL) -> pd.DataFrame:
    """The gap with both scores fitted inside the test set, out of fold by scan:
    the own score against one fitted to the ladder mean without the engine.
    Engines outside the ladder (no fit-set errors) get their gap only here, and
    neither score has a domain advantage over the other."""
    folds = list(GroupKFold(n_splits=n_splits).split(test.features, groups=test.scans))
    rows = []
    for k in engines:
        others = [m for m in ladder if m != k]
        y_own, y_oth = test.z[k].to_numpy(), test.z[others].mean(axis=1).to_numpy()
        p_own, p_oth = np.full(len(y_own), np.nan), np.full(len(y_own), np.nan)
        for train, held in folds:
            tf = test.features.iloc[train]
            p_own[held] = fit_scorer(tf, metrics, y_own[train], variant).score(test.features.iloc[held])
            p_oth[held] = fit_scorer(tf, metrics, y_oth[train], variant).score(test.features.iloc[held])
        r_own, r_oth = np.corrcoef(p_own, y_own)[0, 1], np.corrcoef(p_oth, y_own)[0, 1]
        rows.append({"test_set": test.name, "variant": variant, "engine": k, "in_ladder": k in ladder,
                     "pearson_own_cv": r_own, "pearson_ladder_cv": r_oth, "gap": r_own - r_oth})
    return pd.DataFrame(rows)


def transfer_table(scores: Mapping[str, np.ndarray], test: EvalSet, targets: Mapping[str, np.ndarray],
                   n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> pd.DataFrame:
    """Pearson of every score with every target on one test set, with intervals."""
    counts = scan_resample_counts(len(test.scan_order), n_boot, seed)
    rows = []
    for s, values in scores.items():
        for t, target in targets.items():
            r, lo, hi, _ = pearson_with_interval(np.asarray(values), np.asarray(target), test, counts, ci)
            rows.append({"test_set": test.name, "score": s, "target": t, "pearson": r, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def scorer_json(scorers: Mapping[str, DifficultyScorer], description: str,
                extra: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """The layout :meth:`DifficultyScorer.load` reads: weights per variant and
    one set of grids (all variants of one fit share them)."""
    first = next(iter(scorers.values()))
    return {"description": description, **dict(extra or {}),
            "weights": {v: {"intercept": s.intercept, "coefficients": s.weights} for v, s in scorers.items()},
            "percentile_grids_dev": {m: np.asarray(g).tolist() for m, g in first.grids.items()}}
