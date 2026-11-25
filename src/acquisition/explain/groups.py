"""Groups of correlated difficulty features, so credit split between them can be summed.

Correlated features share credit in a linear fit and often carry large weights
of opposite sign (``sobel_dw_edge_energy`` against ``sobel_dw_mean``,
``bp_range_max_mm`` against ``energy_mev``). A single feature's contribution is
then not interpretable on its own; the sum over its group is.

Groups are hierarchical clusters (average linkage) on ``1 - |Spearman|``
between the features' percentiles over the reference population, cut so that
features with ``|rho|`` at least ``threshold`` join. This reuses
:func:`src.active_learning.hardness.cells.cluster_metrics` with a single cell,
which makes its within-cell correlation the plain Spearman correlation: no
centring by energy or anatomy, so the energy-to-depth link is kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import pandas as pd

from src.active_learning.hardness.cells import MetricClusters, cluster_metrics

DEFAULT_THRESHOLD = 0.7
"""``|rho|`` at which features join a group (CHG-0021 used the same cut, 0.3 in ``1 - |rho|``)."""

FAMILY: Mapping[str, str] = {
    "energy_mev": "depth and energy", "bp_range_min_mm": "depth and energy",
    "bp_range_max_mm": "depth and energy", "max_grad_depth_mm": "depth and energy",
    "wepl_mean": "depth and energy",
    "n_density_regions": "tissue changes in the peak zone", "total_hu_change": "tissue changes in the peak zone",
    "max_hu_jump": "tissue changes in the peak zone", "sigma_hu_bp": "tissue changes in the peak zone",
    "max_hu_gradient": "tissue changes in the peak zone", "hetero_fraction": "tissue changes in the peak zone",
    "interface_bp_distance": "tissue changes in the peak zone",
    "ct_max_hu": "densest voxel in the crop", "lateral_hu_var_bp": "lateral HU spread at the peak",
    "mean_sobel_axial": "edges around the peak", "p95_sobel_bp": "edges around the peak",
    "sum_sobel_bp": "edges around the peak", "sobel_dw_mean": "edges around the peak",
    "sobel_dw_anisotropy": "edges around the peak", "sobel_dw_edge_energy": "edges around the peak",
    "sobel_th_mean": "edges around the peak", "sobel_th_anisotropy": "edges around the peak",
    "sobel_th_edge_energy": "edges around the peak", "lateral_edge_energy": "edges around the peak",
    "pflugfelder_hi": "WEPL spread across the beamlet", "wepl_std": "WEPL spread across the beamlet",
    "isi_sum": "interface severity", "isi_max": "interface severity", "isi_mean": "interface severity",
    "isi_axial_sum": "interface severity",
    "air_gap_mm": "air in front of the patient", "lowdensity_path_mm": "low-density path to the peak",
}
"""The physical family of each of the 30 features (and of the two path features of
:mod:`src.engine_drivers.geometry`), used only to name groups."""


@dataclass(frozen=True)
class FeatureGroups:
    """Feature groups and their names."""

    clusters: MetricClusters
    names: Dict[str, str]            # feature -> group name

    @property
    def threshold(self) -> float:
        return 1.0 - self.clusters.threshold

    def table(self) -> pd.DataFrame:
        t = self.clusters.table()
        t["group"] = [self.names[m.split()[0]] for m in t["members"]]
        t["abs_rho_threshold"] = self.threshold
        return t[["group", "cluster", "medoid", "n_metrics", "members", "signs", "min_abs_rho", "mean_abs_rho",
                  "abs_rho_threshold"]]


def _group_name(members, medoid: str) -> str:
    counts = pd.Series([FAMILY.get(m, "other") for m in members]).value_counts()
    family = FAMILY.get(medoid, "other") if counts.iloc[0] == counts.get(FAMILY.get(medoid), 0) else counts.index[0]
    return f"{family} [{medoid}]"


def feature_groups(reference_percentiles: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> FeatureGroups:
    """Group the columns of ``reference_percentiles`` (one row per reference beamlet).

    Args:
        reference_percentiles: Features through the scorer's percentile transform.
        threshold: ``|rho|`` at or above which two features may share a group.

    Returns:
        The clusters and a ``{feature: group name}`` map. A group is named after
        the physical family most of its members belong to, with its medoid in
        brackets, for example ``"edges around the peak [sobel_dw_edge_energy]"``.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    one_cell = pd.Series(0, index=reference_percentiles.index)
    clusters = cluster_metrics(reference_percentiles, one_cell, threshold=1.0 - threshold)
    names: Dict[str, str] = {}
    for c in clusters.ids:
        members = clusters.members(c)
        name = _group_name(members, clusters.medoid[c])
        for m in members:
            names[m] = name
    return FeatureGroups(clusters=clusters, names=names)


__all__ = ["DEFAULT_THRESHOLD", "FAMILY", "FeatureGroups", "feature_groups"]
