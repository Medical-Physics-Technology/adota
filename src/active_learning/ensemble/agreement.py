"""Pairwise agreement of ensemble members on which records are hard.

Members reach different accuracy levels, so their errors are compared as ranks,
not values, and within cells of one scan and one 20 MeV energy bin, so that a
pair does not agree merely because both find high energies or one anatomy hard
(EXP-0014, stage 3 item 2). Per pair of members:

- the Spearman correlation of the errors inside every cell with at least
  ``min_cell_size`` records, summarised as the median over cells;
- the pooled Spearman over all records, as a secondary number that keeps the
  energy and scan structure;
- the Jaccard overlap of the top ``top_fraction`` of records, within cells and
  pooled.

A pair's **level** is read from the training ids, never from the run names:
two members trained on the same ids differ only in their seed (``seed_only``,
the floor of the agreement), any other pair differs in its training data
(``data``). Intervals come from a bootstrap over scans, the unit the
evaluation set was held out by.
"""
from __future__ import annotations

import itertools
import logging
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.active_learning.ensemble.members import Member
from src.active_learning.retrospective.sampling import energy_bin

logger = logging.getLogger(__name__)

LEVEL_SEED = "seed_only"
LEVEL_DATA = "data"
LEVEL_UNKNOWN = "unknown"

ERROR_METRICS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "rde_pct": lambda f: f["rde_pct"],
    "gamma_fail": lambda f: 1.0 - f["gpr"],
    "mape_pct": lambda f: f["mape_pct"],
    "abs_dr80_mm": lambda f: f["dr80_mm"].abs(),
}
"""Per-record error, oriented so that larger is worse. Rank statistics are
invariant to monotone transforms, so ``rde_pct`` stands for log(1 + RDE) too."""


def error_matrix(members: Sequence[Member], metric: str) -> pd.DataFrame:
    """Records by members, complete cases only (dR80 or gamma can be undefined)."""
    if metric not in ERROR_METRICS:
        raise KeyError(f"unknown error metric {metric!r}; known {sorted(ERROR_METRICS)}")
    columns = {m.label: ERROR_METRICS[metric](m.errors.set_index("sample_id")).astype(float)
               for m in members}
    frame = pd.DataFrame(columns)
    complete = frame.dropna()
    if len(complete) < len(frame):
        logger.info("%s: %d of %d records have an undefined error in some member and are dropped",
                    metric, len(frame) - len(complete), len(frame))
    return complete


def assign_cells(metadata: pd.DataFrame, ids: Sequence[str], group_column: str = "patient_key",
                 energy_column: str = "energy_mev") -> pd.DataFrame:
    """``scan``, ``energy_bin`` and their ``cell`` for every id."""
    meta = metadata.assign(sample_id=metadata["sample_id"].astype(str))
    meta = meta.drop_duplicates("sample_id").set_index("sample_id")
    missing = [i for i in ids if i not in meta.index]
    if missing:
        raise KeyError(f"{len(missing)} evaluation records have no metadata, e.g. {missing[0]!r}")
    part = meta.loc[list(ids)]
    out = pd.DataFrame({"scan": part[group_column].astype(str).to_numpy(),
                        "energy_bin": energy_bin(part[energy_column].astype(float)).to_numpy()},
                       index=pd.Index(list(ids), name="sample_id"))
    out["cell"] = out["scan"] + "|" + out["energy_bin"]
    return out


def pair_levels(members: Sequence[Member]) -> pd.DataFrame:
    """One row per pair ``(a, b)`` in member order: the Jaccard overlap of their
    training ids and the level it implies."""
    rows = []
    for a, b in itertools.combinations(members, 2):
        row = {"a": a.label, "b": b.label, "training_jaccard": np.nan, "level": LEVEL_UNKNOWN}
        if a.training_ids is not None and b.training_ids is not None:
            sa, sb = set(a.training_ids), set(b.training_ids)
            row["training_jaccard"] = len(sa & sb) / max(len(sa | sb), 1)
            row["level"] = LEVEL_SEED if sa == sb else LEVEL_DATA
            seeds = (a.provenance.get("seed"), b.provenance.get("seed"))
            if sa == sb and seeds[0] is not None and seeds[0] == seeds[1]:
                logger.warning("%s and %s share training ids and seed: a duplicate model?",
                               a.label, b.label)
        rows.append(row)
    return pd.DataFrame(rows)


def cell_correlations(matrix: pd.DataFrame, cells: pd.DataFrame,
                      min_cell_size: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """Within-cell Spearman of every pair: an array ``(n_cells, n_pairs)`` with
    pairs in :func:`pair_levels` order, and the cells it covers."""
    labels = list(matrix.columns)
    iu = np.triu_indices(len(labels), k=1)
    grouped = cells.loc[matrix.index].groupby("cell", sort=True)
    rows, kept = [], []
    for cell, part in grouped:
        if len(part) < min_cell_size:
            continue
        ranks = rankdata(matrix.loc[part.index].to_numpy(), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(ranks, rowvar=False)
        rows.append(np.atleast_2d(corr)[iu])
        kept.append({"cell": cell, "scan": part["scan"].iloc[0],
                     "energy_bin": part["energy_bin"].iloc[0], "n": len(part)})
    if not rows:
        raise ValueError(f"no cell has {min_cell_size} or more complete records")
    return np.vstack(rows), pd.DataFrame(kept)


def _top_flags(values: pd.DataFrame, fraction: float, groups: Optional[pd.Series]) -> pd.DataFrame:
    ranked = values.groupby(groups).rank(pct=True) if groups is not None else values.rank(pct=True)
    return ranked > 1.0 - fraction


def _jaccard(x: np.ndarray, y: np.ndarray) -> float:
    union = np.logical_or(x, y).sum()
    return float(np.logical_and(x, y).sum() / union) if union else float("nan")


def pair_table(matrix: pd.DataFrame, cells: pd.DataFrame, pairs: pd.DataFrame,
               corr: np.ndarray, top_fraction: float) -> pd.DataFrame:
    """Point estimates per pair; ``pairs`` comes from :func:`pair_levels` on the
    same members, in the same order as ``matrix``'s columns."""
    groups = cells.loc[matrix.index, "cell"]
    top_cell = _top_flags(matrix, top_fraction, groups)
    top_pool = _top_flags(matrix, top_fraction, None)
    out = pairs.copy()
    with np.errstate(all="ignore"):
        out["median_cell_spearman"] = np.nanmedian(corr, axis=0)
    out["n_cells"] = np.isfinite(corr).sum(axis=0)
    out["pooled_spearman"] = [float(spearmanr(matrix[a], matrix[b])[0])
                              for a, b in zip(pairs["a"], pairs["b"])]
    out["top_jaccard_within_cell"] = [_jaccard(top_cell[a].to_numpy(), top_cell[b].to_numpy())
                                      for a, b in zip(pairs["a"], pairs["b"])]
    out["top_jaccard_pooled"] = [_jaccard(top_pool[a].to_numpy(), top_pool[b].to_numpy())
                                 for a, b in zip(pairs["a"], pairs["b"])]
    out["n_records"] = len(matrix)
    return out


def _level_means(values: np.ndarray, levels: np.ndarray, names: Sequence[str]) -> Dict[str, float]:
    with np.errstate(all="ignore"):
        return {name: float(np.nanmean(values[levels == name])) for name in names}


def scan_resamples(cell_scans: Sequence[str], n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """``n_boot`` bootstrap draws over scans, as row indices into the cells.

    Scans are drawn with replacement and every drawn scan brings all of its
    cells, as often as it was drawn. The one resampling scheme behind every
    interval of the agreement and hardness analyses, so one seed gives one
    sequence of draws everywhere.
    """
    scans, scan_of_cell = np.unique(np.asarray(cell_scans), return_inverse=True)
    rng = np.random.default_rng(seed)
    cells = np.arange(len(scan_of_cell))
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(scans), len(scans)), minlength=len(scans))
        yield np.repeat(cells, counts[scan_of_cell])


def scan_bootstrap(corr: np.ndarray, cell_scans: Sequence[str], levels: Sequence[str],
                   n_boot: int, seed: int, ci: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resample scans with replacement; every drawn scan brings all its cells.

    Returns the per-pair interval of the median within-cell Spearman, and per
    level the mean over its pairs with an interval, plus the contrast
    ``data - seed_only`` when both levels are present.
    """
    scans = np.unique(np.asarray(cell_scans))
    levels = np.asarray(levels)
    names = [n for n in (LEVEL_SEED, LEVEL_DATA, LEVEL_UNKNOWN) if (levels == n).any()]
    pair_draws = np.empty((n_boot, corr.shape[1]))
    level_draws: Dict[str, List[float]] = {n: [] for n in names}
    contrast: List[float] = []
    for i, rows in enumerate(scan_resamples(cell_scans, n_boot, seed)):
        with np.errstate(all="ignore"):
            pair_draws[i] = np.nanmedian(corr[rows], axis=0) if rows.size else np.nan
        means = _level_means(pair_draws[i], levels, names)
        for name in names:
            level_draws[name].append(means[name])
        if LEVEL_SEED in means and LEVEL_DATA in means:
            contrast.append(means[LEVEL_DATA] - means[LEVEL_SEED])
    lo, hi = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0
    with np.errstate(all="ignore"):
        pair_ci = pd.DataFrame({"median_cell_spearman_lo": np.nanpercentile(pair_draws, lo, axis=0),
                                "median_cell_spearman_hi": np.nanpercentile(pair_draws, hi, axis=0)})
        point = np.nanmedian(corr, axis=0)
    point_means = _level_means(point, levels, names)
    rows_out = []
    for name in names:
        values = point[levels == name]
        rows_out.append({"level": name, "n_pairs": int((levels == name).sum()),
                         "mean_median_cell_spearman": point_means[name],
                         "min_pair": float(np.nanmin(values)), "max_pair": float(np.nanmax(values)),
                         "ci_lo": float(np.nanpercentile(level_draws[name], lo)),
                         "ci_hi": float(np.nanpercentile(level_draws[name], hi))})
    if contrast:
        rows_out.append({"level": f"{LEVEL_DATA}_minus_{LEVEL_SEED}", "n_pairs": np.nan,
                         "mean_median_cell_spearman": point_means[LEVEL_DATA] - point_means[LEVEL_SEED],
                         "min_pair": np.nan, "max_pair": np.nan,
                         "ci_lo": float(np.nanpercentile(contrast, lo)),
                         "ci_hi": float(np.nanpercentile(contrast, hi))})
    levels_frame = pd.DataFrame(rows_out)
    levels_frame["n_scans"] = len(scans)
    levels_frame["n_bootstrap"] = n_boot
    return pair_ci, levels_frame


def agreement_matrix(pairs: pd.DataFrame, labels: Sequence[str], column: str) -> pd.DataFrame:
    """The symmetric members-by-members matrix of one pair statistic, NaN on the
    diagonal."""
    out = pd.DataFrame(np.nan, index=list(labels), columns=list(labels))
    for a, b, value in zip(pairs["a"], pairs["b"], pairs[column]):
        out.loc[a, b] = out.loc[b, a] = value
    return out
