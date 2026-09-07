"""The input-derived difficulty metrics of a beamlet.

The thirty metrics behind the difficulty score of
``research/acquisition_function_final_summary.md``, computed from a beamlet's CT
crop, flux and energy plus a *dose* used only to locate the Bragg peak and to
weight the edge metrics. In the reference study that dose was the Monte Carlo
ground truth; at selection time no such dose exists and
:mod:`src.acquisition.surrogate` supplies an analytic one instead. Either way the
metric code is identical, which is the point of having the dose as an argument.

``analyse_density_regions`` and ``compute_advanced_metrics`` moved here verbatim
from ``scripts/training_set_analysis_advanced_metrics.py``; the remaining metrics
already lived in ``src.metrics.sobel``, ``src.processing.pflugfelder_hi`` and
``src.processing.interface_severity``. :func:`compute_features` is the one
orchestrator that calls them all in the reference run's order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.figures.ct_visualizations import segment_hu, smooth_ct
from src.metrics.sobel import (
    compute_sobel_metrics,
    compute_sobel_metrics_sphere,
    compute_structure_tensor_metrics_sphere,
)
from src.processing.interface_severity import interface_severity
from src.processing.pflugfelder_hi import pflugfelder_hi
from src.utils.dose_grid_utils import estimate_bp_range

# The metrics with a weight in ``frozen_final_scorer.json``, in the order the
# reference results.csv carries them. Everything :func:`compute_features` returns
# beyond these is diagnostic.
FEATURE_NAMES: Tuple[str, ...] = (
    "energy_mev", "ct_max_hu", "bp_range_min_mm", "bp_range_max_mm", "max_grad_depth_mm",
    "n_density_regions", "total_hu_change", "max_hu_jump", "sigma_hu_bp", "max_hu_gradient",
    "lateral_hu_var_bp", "hetero_fraction", "interface_bp_distance",
    "mean_sobel_axial", "p95_sobel_bp", "sum_sobel_bp",
    "sobel_dw_mean", "sobel_dw_anisotropy", "sobel_dw_edge_energy",
    "sobel_th_mean", "sobel_th_anisotropy", "sobel_th_edge_energy", "lateral_edge_energy",
    "pflugfelder_hi", "wepl_mean", "wepl_std",
    "isi_sum", "isi_max", "isi_mean", "isi_axial_sum",
)


@dataclass(frozen=True)
class FeatureConfig:
    """How the metrics are computed.

    Defaults are the reference run's settings
    (``/scratch/mstryja/adota_runs/20260707_124010/config_analysis_advanced_metrics.yaml``),
    which are what the frozen score's weights and percentile grids assume. They
    differ from :class:`src.schemas.configs.AdvancedAnalysisConfig`'s defaults
    (sphere instead of zone for the edge metrics, 5 mm instead of 10 mm, raw
    instead of smoothed CT), so a caller that changes them is computing a
    different feature vector.
    """

    resolution_mm: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    proximal_fraction: float = 0.50
    fall_fraction: float = 0.10
    flux_threshold_frac: float = 0.10
    sobel_percentile: float = 95.0
    sobel_use_raw: bool = True
    smoothing_method: str = "gaussian"
    smoothing_sigma: float = 1.0
    region_method: str = "sphere"        # "sphere" | "bp_range"
    sphere_radius_mm: float = 5.0
    isi_severity_mode: str = "rsp_sq"


def analyse_density_regions(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    z_min: float,
    z_max: float,
    flux_threshold_frac: float = 0.10,
) -> Tuple[int, float, list[dict]]:
    """Count distinct density regions along the beamlet path in the BP zone.

    For each depth slice between *z_min* and *z_max* the lateral flux
    (fast beamlet-shape projection) is used as a weight mask to compute
    a flux-weighted mean HU value.  The resulting 1-D HU profile is
    then segmented using :func:`segment_hu`'s tissue classes, and
    consecutive slices with the same tissue class are grouped into
    contiguous *density regions*.

    Args:
        ct_hu: 3-D CT volume ``(D, H, W)`` in HU.
        flux: 3-D flux / fast beamlet-shape volume ``(D, H, W)``.
        z_min: Proximal BP boundary (depth-slice index, float).
        z_max: Distal BP boundary (depth-slice index, float).
        flux_threshold_frac: Fraction of the per-slice flux maximum
            below which voxels are ignored (default 10 %).

    Returns:
        ``(n_regions, total_hu_change, region_details)``

        * **n_regions** -- number of distinct contiguous tissue-class
          regions along the beam path.
        * **total_hu_change** -- sum of absolute mean-HU differences
          between consecutive regions.
        * **region_details** -- list of dicts, one per region, each
          containing ``class_idx``, ``label``, ``mean_hu``,
          ``start_slice``, ``end_slice``.
    """
    from src.figures.ct_visualizations import HU_LUT

    k_start = int(np.ceil(z_min))
    k_end = int(np.floor(z_max))

    if k_end <= k_start:
        return 0, 0.0, []

    # -- Flux-weighted mean HU per depth slice --------------------------------
    mean_hu_per_slice = np.zeros(k_end - k_start + 1)
    for i, k in enumerate(range(k_start, k_end + 1)):
        flux_slice = np.abs(flux[k])  # (H, W)
        ct_slice = ct_hu[k]  # (H, W)

        # Threshold: only consider voxels where flux is significant
        f_max = flux_slice.max()
        if f_max < 1e-12:
            mean_hu_per_slice[i] = ct_slice.mean()
            continue

        mask = flux_slice >= flux_threshold_frac * f_max
        if mask.sum() == 0:
            mean_hu_per_slice[i] = ct_slice.mean()
            continue

        weights = flux_slice[mask]
        mean_hu_per_slice[i] = np.average(ct_slice[mask], weights=weights)

    # -- Segment the 1-D mean-HU profile into tissue classes ------------------
    class_per_slice = segment_hu(mean_hu_per_slice)  # (N,) int array

    # -- Group consecutive slices of the same class into regions --------------
    regions: list[dict] = []
    current_class = int(class_per_slice[0])
    region_start = k_start

    for i in range(1, len(class_per_slice)):
        if int(class_per_slice[i]) != current_class:
            # Close the current region
            region_end = k_start + i - 1
            region_mask = slice(region_start - k_start, region_end - k_start + 1)
            regions.append(
                {
                    "class_idx": current_class,
                    "label": HU_LUT[current_class][0],
                    "mean_hu": float(np.mean(mean_hu_per_slice[region_mask])),
                    "start_slice": region_start,
                    "end_slice": region_end,
                }
            )
            current_class = int(class_per_slice[i])
            region_start = k_start + i

    # Close the last region
    region_mask = slice(region_start - k_start, len(class_per_slice))
    regions.append(
        {
            "class_idx": current_class,
            "label": HU_LUT[current_class][0],
            "mean_hu": float(np.mean(mean_hu_per_slice[region_mask])),
            "start_slice": region_start,
            "end_slice": k_end,
        }
    )

    # -- Total absolute HU change between consecutive regions -----------------
    total_hu_change = 0.0
    for j in range(1, len(regions)):
        total_hu_change += abs(regions[j]["mean_hu"] - regions[j - 1]["mean_hu"])

    return len(regions), total_hu_change, regions


def compute_advanced_metrics(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    gt_dose: np.ndarray,
    z_min: float,
    z_max: float,
    region_details: list[dict],
    flux_threshold_frac: float = 0.10,
) -> dict:
    """Compute advanced heterogeneity metrics for a single beamlet.

    Returns a dict with keys:
        max_hu_jump, sigma_hu_bp, max_hu_gradient,
        lateral_hu_var_bp, hetero_fraction, interface_bp_distance.
    """

    k_start = int(np.ceil(z_min))
    k_end = int(np.floor(z_max))

    # -- Defaults for degenerate cases ------------------------------------
    defaults = dict(
        max_hu_jump=0.0,
        sigma_hu_bp=0.0,
        max_hu_gradient=0.0,
        lateral_hu_var_bp=0.0,
        hetero_fraction=0.0,
        interface_bp_distance=0.0,
    )
    if k_end <= k_start:
        return defaults

    # -- Flux-weighted mean HU profile (recomputed, cheap) ----------------
    n_slices = k_end - k_start + 1
    mean_hu = np.zeros(n_slices)
    for i, k in enumerate(range(k_start, k_end + 1)):
        flux_slice = np.abs(flux[k])
        ct_slice = ct_hu[k]
        f_max = flux_slice.max()
        if f_max < 1e-12:
            mean_hu[i] = ct_slice.mean()
            continue
        mask = flux_slice >= flux_threshold_frac * f_max
        if mask.sum() == 0:
            mean_hu[i] = ct_slice.mean()
            continue
        weights = flux_slice[mask]
        mean_hu[i] = np.average(ct_slice[mask], weights=weights)

    # (1) max_hu_jump: largest |mean_hu| difference between consecutive regions
    max_hu_jump = 0.0
    if len(region_details) >= 2:
        for j in range(1, len(region_details)):
            jump = abs(region_details[j]["mean_hu"] - region_details[j - 1]["mean_hu"])
            if jump > max_hu_jump:
                max_hu_jump = jump

    # (2) sigma_hu_bp: std of the flux-weighted mean HU profile
    sigma_hu_bp = float(np.std(mean_hu))

    # (3) max_hu_gradient: max |dH/dk| along beam path at slice resolution
    if n_slices >= 2:
        hu_grad = np.abs(np.diff(mean_hu))
        max_hu_gradient = float(np.max(hu_grad))
    else:
        max_hu_gradient = 0.0

    # (4) lateral_hu_var_bp: flux-weighted HU variance at BP slice
    idd = gt_dose.sum(axis=(1, 2))
    bp_idx = int(np.argmax(idd))
    flux_bp = np.abs(flux[bp_idx])
    ct_bp = ct_hu[bp_idx]
    f_max_bp = flux_bp.max()
    lateral_hu_var_bp = 0.0
    if f_max_bp > 1e-12:
        mask_bp = flux_bp >= flux_threshold_frac * f_max_bp
        if mask_bp.sum() > 1:
            w_bp = flux_bp[mask_bp]
            mu_bp = np.average(ct_bp[mask_bp], weights=w_bp)
            lateral_hu_var_bp = float(
                np.average((ct_bp[mask_bp] - mu_bp) ** 2, weights=w_bp)
            )

    # (5) hetero_fraction: fraction of slices NOT in the dominant class
    class_per_slice = segment_hu(mean_hu)
    unique, counts = np.unique(class_per_slice, return_counts=True)
    dominant_count = counts.max()
    hetero_fraction = 1.0 - dominant_count / len(class_per_slice)

    # (6) interface_bp_distance: distance (slices) from BP to nearest
    #     tissue-class transition
    bp_local = bp_idx - k_start  # BP index in local array
    bp_local = max(0, min(bp_local, len(class_per_slice) - 1))
    interface_bp_distance = float(len(class_per_slice))  # fallback: max
    for i in range(1, len(class_per_slice)):
        if class_per_slice[i] != class_per_slice[i - 1]:
            # transition between slice i-1 and i
            transition_pos = (i - 1 + i) / 2.0
            dist = abs(transition_pos - bp_local)
            if dist < interface_bp_distance:
                interface_bp_distance = dist

    return dict(
        max_hu_jump=max_hu_jump,
        sigma_hu_bp=sigma_hu_bp,
        max_hu_gradient=max_hu_gradient,
        lateral_hu_var_bp=lateral_hu_var_bp,
        hetero_fraction=hetero_fraction,
        interface_bp_distance=interface_bp_distance,
    )


def compute_features(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    energy_mev: float,
    dose: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
) -> Dict[str, float]:
    """The thirty difficulty metrics of one beamlet, plus a few diagnostics.

    Args:
        ct_hu: CT crop in Hounsfield units, ``(D, H, W)`` with depth first.
        flux: The analytical flux channel on the same grid.
        energy_mev: Beam energy.
        dose: The dose used to locate the Bragg peak and to weight the edge
            metrics: the Monte Carlo ground truth in the reference study, the
            analytic surrogate at selection time. Same grid as ``ct_hu``.
        config: Metric settings; the defaults reproduce the reference run.

    Returns:
        ``{name: value}`` for every entry of :data:`FEATURE_NAMES`, plus
        ``bp_slice`` (Bragg-peak depth index) and ``sobel_dw_beam_angle`` /
        ``sobel_th_beam_angle`` (degrees) as diagnostics.
    """
    dz_mm = float(config.resolution_mm[0])
    out: Dict[str, float] = {"energy_mev": float(energy_mev), "ct_max_hu": float(np.max(ct_hu))}

    z_min, z_max = estimate_bp_range(ct_hu, dose, proximal_fraction=config.proximal_fraction,
                                     fall_fraction=config.fall_fraction)
    out["bp_range_min_mm"] = z_min * dz_mm
    out["bp_range_max_mm"] = z_max * dz_mm
    idd = dose.sum(axis=(1, 2))
    out["bp_slice"] = int(np.argmax(idd))
    out["max_grad_depth_mm"] = float(int(np.argmax(np.gradient(idd)))) * dz_mm

    n_regions, hu_change, region_details = analyse_density_regions(
        ct_hu, flux, z_min, z_max, flux_threshold_frac=config.flux_threshold_frac)
    out["n_density_regions"] = n_regions
    out["total_hu_change"] = hu_change

    adv = compute_advanced_metrics(ct_hu, flux, dose, z_min, z_max, region_details,
                                   flux_threshold_frac=config.flux_threshold_frac)
    out.update(adv)

    ct_for_sobel = ct_hu if config.sobel_use_raw else smooth_ct(
        ct_hu, method=config.smoothing_method, sigma=config.smoothing_sigma)
    if config.region_method == "sphere":
        sobel = compute_sobel_metrics_sphere(
            ct_for_sobel, dose, radius_mm=config.sphere_radius_mm, resolution=config.resolution_mm,
            flux=flux, flux_threshold_frac=config.flux_threshold_frac,
            sobel_percentile=config.sobel_percentile)
    else:
        sobel = compute_sobel_metrics(ct_for_sobel, flux, z_min, z_max,
                                      flux_threshold_frac=config.flux_threshold_frac,
                                      sobel_percentile=config.sobel_percentile)
    out.update({k: sobel[k] for k in ("mean_sobel_axial", "p95_sobel_bp", "sum_sobel_bp")})

    st = compute_structure_tensor_metrics_sphere(
        ct_for_sobel, dose, radius_mm=config.sphere_radius_mm, resolution=config.resolution_mm)
    out.update(st)
    theta_rad = np.radians(st["sobel_dw_beam_angle"])
    out["lateral_edge_energy"] = float(st["sobel_dw_edge_energy"] * np.sin(theta_rad) ** 2)

    hi = pflugfelder_hi(ct_hu, flux, dose, resolution_mm=config.resolution_mm,
                        flux_threshold_frac=config.flux_threshold_frac)
    out["pflugfelder_hi"] = hi["hi"]
    out["wepl_mean"] = hi["wepl_mean"]
    out["wepl_std"] = hi["wepl_std"]

    isi = interface_severity(ct_hu, flux, dose, resolution_mm=config.resolution_mm,
                             severity_mode=config.isi_severity_mode,
                             sphere_radius_mm=config.sphere_radius_mm,
                             flux_threshold_frac=config.flux_threshold_frac)
    out.update({k: isi[k] for k in ("isi_sum", "isi_max", "isi_mean", "isi_axial_sum")})
    return out
