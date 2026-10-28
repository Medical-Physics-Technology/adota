"""Within-cell statistics: ranks, Spearman correlations, intervals and metric clusters.

A **cell** is one scan times one 20 MeV energy bin, the cell of
:func:`src.active_learning.ensemble.agreement.assign_cells`; only cells with at
least ``min_cell_size`` records count. Every correlation here is computed
inside cells and summarised as the median over cells, so that a metric does
not "explain" an error merely because both differ between scans or energies.
Intervals come from :func:`src.active_learning.ensemble.agreement.scan_resamples`,
the bootstrap over scans that the agreement analysis uses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from src.active_learning.ensemble.agreement import scan_resamples


def cell_ranks(values: pd.DataFrame, cells: pd.Series) -> pd.DataFrame:
    """Percentile rank of every column within its cell, in ``(0, 1]``, ties averaged."""
    return values.groupby(cells.to_numpy(), sort=False).rank(pct=True, method="average")


def centre_within_cells(values: pd.DataFrame, cells: pd.Series) -> pd.DataFrame:
    """Every column minus its cell mean (the within-cell part)."""
    return values - values.groupby(cells.to_numpy(), sort=False).transform("mean")


def eligible_cells(cells: pd.Series, min_cell_size: int) -> pd.Series:
    """Boolean mask of the records whose cell has at least ``min_cell_size`` records."""
    sizes = cells.map(cells.value_counts())
    return sizes >= int(min_cell_size)


@dataclass
class CellCorrelation:
    """Within-cell Spearman of ``k`` columns against one target: ``values`` is
    ``(n_cells, k)``, ``cells`` names each row's cell, scan and size."""

    values: np.ndarray
    columns: List[str]
    cells: pd.DataFrame

    def median(self) -> pd.Series:
        with np.errstate(all="ignore"):
            return pd.Series(np.nanmedian(self.values, axis=0), index=self.columns)


def cell_spearman(x: pd.DataFrame, y: pd.Series, cells: pd.Series, scans: pd.Series,
                  min_cell_size: int) -> CellCorrelation:
    """Spearman of every column of ``x`` with ``y`` inside each eligible cell.

    A column that is constant inside a cell (energy in a single-energy cell, a
    saturated count) has no correlation there and gives NaN for that cell.
    """
    rows, kept = [], []
    xv, yv = x.to_numpy(dtype=float), np.asarray(y, dtype=float)
    for cell, index in pd.Series(np.arange(len(x))).groupby(cells.to_numpy(), sort=True):
        idx = index.to_numpy()
        if len(idx) < min_cell_size:
            continue
        rx = rankdata(xv[idx], axis=0)
        ry = rankdata(yv[idx])
        rx = rx - rx.mean(axis=0)
        ry = ry - ry.mean()
        with np.errstate(all="ignore"):
            denom = np.sqrt((rx ** 2).sum(axis=0) * (ry ** 2).sum())
            corr = (rx * ry[:, None]).sum(axis=0) / denom
        corr[~np.isfinite(corr)] = np.nan
        rows.append(corr)
        kept.append({"cell": cell, "scan": scans.iloc[idx[0]], "n": len(idx)})
    if not rows:
        raise ValueError(f"no cell has {min_cell_size} or more records")
    return CellCorrelation(np.vstack(rows), list(x.columns), pd.DataFrame(kept))


def bootstrap_interval(values: np.ndarray, cell_scans: Sequence[str],
                       statistic: Callable[[np.ndarray], np.ndarray], n_boot: int, seed: int,
                       ci: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimate and scan-bootstrap interval of ``statistic(values[rows])``.

    ``values`` is any per-cell array (cells along axis 0); ``statistic`` maps it
    to a vector (for example the median over cells per column, or a difference
    of two such medians). Returns ``(point, lo, hi)``.
    """
    with np.errstate(all="ignore"):
        point = np.atleast_1d(statistic(values))
        draws = np.full((n_boot, point.size), np.nan)
        for i, rows in enumerate(scan_resamples(cell_scans, n_boot, seed)):
            if rows.size:
                draws[i] = np.atleast_1d(statistic(values[rows]))
        lo, hi = 100.0 * (1.0 - ci) / 2.0, 100.0 * (1.0 + ci) / 2.0
        return point, np.nanpercentile(draws, lo, axis=0), np.nanpercentile(draws, hi, axis=0)


def nanmedian_columns(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmedian(values, axis=0)


def summarise_correlation(corr: CellCorrelation, n_boot: int, seed: int, ci: float = 0.95,
                          label: str = "column") -> pd.DataFrame:
    """Median within-cell Spearman per column with its scan-bootstrap interval."""
    point, lo, hi = bootstrap_interval(corr.values, corr.cells["scan"], nanmedian_columns, n_boot, seed, ci)
    return pd.DataFrame({label: corr.columns, "median_cell_spearman": point, "ci_lo": lo, "ci_hi": hi,
                         "n_cells": np.isfinite(corr.values).sum(axis=0),
                         "n_scans": corr.cells["scan"].nunique(), "n_records": int(corr.cells["n"].sum())})


# --------------------------------------------------------------------------- clusters

@dataclass
class MetricClusters:
    """Groups of near-duplicate metrics: hierarchical clustering on
    ``1 - |rho|`` of their within-cell ranks."""

    assignment: Dict[str, int]                 # metric -> cluster id, 1..k
    names: Dict[int, str]                      # cluster id -> name (its medoid)
    medoid: Dict[int, str]
    signs: Dict[str, float]                    # +1 / -1 relative to the medoid
    correlation: pd.DataFrame                  # metrics x metrics within-cell Spearman
    threshold: float

    def members(self, cluster: int) -> List[str]:
        return [m for m, c in self.assignment.items() if c == cluster]

    @property
    def ids(self) -> List[int]:
        return sorted(self.names)

    def table(self) -> pd.DataFrame:
        rows = []
        for c in self.ids:
            members = self.members(c)
            sub = self.correlation.loc[members, members].abs().to_numpy()
            off = sub[~np.eye(len(members), dtype=bool)]
            rows.append({"cluster": c, "name": self.names[c], "medoid": self.medoid[c], "n_metrics": len(members),
                         "members": " ".join(members),
                         "signs": " ".join("+" if self.signs[m] > 0 else "-" for m in members),
                         "min_abs_rho": float(off.min()) if off.size else np.nan,
                         "mean_abs_rho": float(off.mean()) if off.size else np.nan})
        return pd.DataFrame(rows)


def within_cell_correlation(features: pd.DataFrame, cells: pd.Series) -> pd.DataFrame:
    """Spearman between metrics after ranking each inside its cell (the pooled
    correlation of within-cell ranks)."""
    ranks = centre_within_cells(cell_ranks(features, cells), cells)
    return ranks.corr()


def cluster_metrics(features: pd.DataFrame, cells: pd.Series, threshold: float,
                    method: str = "average") -> MetricClusters:
    """Cluster the columns of ``features`` so that metrics closer than
    ``threshold`` in ``1 - |rho|`` share a cluster.

    A metric with no within-cell variation (NaN correlations) is its own cluster.
    Each cluster is named after its medoid, the member with the largest mean
    ``|rho|`` to the others.
    """
    corr = within_cell_correlation(features, cells)
    metrics = list(corr.columns)
    dist = 1.0 - corr.abs().fillna(0.0).to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 1.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method=method), t=threshold, criterion="distance")
    # Renumber by size, largest first, so cluster ids are stable to read.
    order = pd.Series(labels).value_counts().sort_index().sort_values(ascending=False, kind="stable").index
    renumber = {old: new for new, old in enumerate(order, start=1)}
    assignment = {m: renumber[int(c)] for m, c in zip(metrics, labels)}
    names, medoids, signs = {}, {}, {}
    for c in sorted(set(assignment.values())):
        members = [m for m in metrics if assignment[m] == c]
        sub = corr.loc[members, members].abs().fillna(0.0)
        medoid = str(sub.mean(axis=1).idxmax())
        medoids[c] = medoid
        names[c] = medoid if len(members) == 1 else f"{medoid} (+{len(members) - 1})"
        for m in members:
            rho = corr.loc[m, medoid]
            signs[m] = -1.0 if (np.isfinite(rho) and rho < 0) else 1.0
    return MetricClusters(assignment, names, medoids, signs, corr, float(threshold))


def apply_clusters(clusters: MetricClusters, features: pd.DataFrame, cells: pd.Series) -> MetricClusters:
    """The same assignment on another table, with that table's correlations
    (for the composites and for reporting how well the grouping carries over)."""
    corr = within_cell_correlation(features[list(clusters.assignment)], cells)
    return MetricClusters(dict(clusters.assignment), dict(clusters.names), dict(clusters.medoid),
                          dict(clusters.signs), corr, clusters.threshold)


def cluster_composites(clusters: MetricClusters, features: pd.DataFrame, cells: pd.Series,
                       ids: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Per record and cluster, the mean within-cell rank of its members, each
    oriented like the medoid: one number per cluster for univariate summaries
    and for picking examples."""
    ranks = cell_ranks(features[list(clusters.assignment)], cells)
    out = {}
    for c in ids or clusters.ids:
        members = clusters.members(c)
        oriented = [ranks[m] if clusters.signs[m] > 0 else 1.0 - ranks[m] for m in members]
        out[clusters.names[c]] = pd.concat(oriented, axis=1).mean(axis=1)
    return pd.DataFrame(out, index=features.index)


def co_assignment_agreement(a: MetricClusters, b: MetricClusters) -> float:
    """Adjusted Rand index of two clusterings of the same metrics."""
    from sklearn.metrics import adjusted_rand_score

    metrics = sorted(set(a.assignment) & set(b.assignment))
    return float(adjusted_rand_score([a.assignment[m] for m in metrics], [b.assignment[m] for m in metrics]))
