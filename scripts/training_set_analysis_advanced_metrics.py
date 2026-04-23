"""
Training Set Analysis – Advanced Metrics

Iterates over all samples in the training set, extracts per-beamlet
data (CT, flux / fast beamlet-shape projection, initial energy, and
ground-truth dose) and logs basic statistics.

No model inference or figure generation is performed — this script is
the skeleton for future advanced-metric experiments.
"""

import csv
import logging
import math
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
import yaml
from scipy.ndimage import sobel as ndimage_sobel
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import (
    DEFAULT_GAMMA_PARAMS,
    DEFAULT_SCALE,
    denormalize_energy,
    get_device,
    load_yaml_config,
    setup_logging,
    setup_run_directory,
)
from src.adota.models import DoTA3D_v3
from src.adota.utils import (
    count_parameters_per_block,
    count_total_parameters,
    load_model,
)
from src.figures.ct_visualizations import (
    plot_ct_with_segmentation,
    segment_hu,
    smooth_ct,
)
from src.figures.single_beam import publication_figure
from src.loaders.generator import H5PYGenerator
from src.loaders.utils import validate_inputs
from src.metrics.classic import (
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from src.metrics.gamma_pass_rate import gamma_index_torch
from src.processing.interface_severity import interface_severity
from src.processing.pflugfelder_hi import pflugfelder_hi
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)

app = typer.Typer(help="Training-set advanced-metrics analysis")


from src.schemas.configs import AdvancedAnalysisConfig as AnalysisConfig
from src.schemas.results import SampleRecord

# ── Helpers ─────────────────────────────────────────────────────────────────


# ── Per-sample extraction ─────────────────────────────────────────────────────


def extract_single_sample(
    sample_idx: int,
    record_id: str,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
) -> Optional[tuple[SampleRecord, np.ndarray, np.ndarray, float, np.ndarray]]:
    """Extract data for a single beamlet and compute summary statistics.

    Args:
        sample_idx: Index into the H5PYGenerator dataset.
        record_id: Unique identifier for the sample.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.

    Returns:
        Tuple of ``(SampleRecord, ct_hu, flux, energy_mev, gt_dose)``
        where:

        - *ct_hu* – 3-D CT volume in Hounsfield Units ``(D, H, W)``.
        - *flux* – fast beamlet-shape projection ``(D, H, W)``.
        - *energy_mev* – initial beam energy in MeV.
        - *gt_dose* – ground-truth dose grid ``(D, H, W)``.

        Returns ``None`` if the sample is skipped (zero flux or energy
        above threshold).
    """
    scale = config.scale
    start_time = perf_counter()

    # Load data from HDF5
    x, energy, y = dataset[sample_idx]

    energy_mev = denormalize_energy(energy.item(), scale)

    # ── Zero-flux guard ─────────────────────────────────────────────────
    flux_channel = x[1]  # (D, H, W)
    if torch.abs(flux_channel).max().item() < 1e-9:
        logger.warning(
            f"Skipping sample {record_id} (idx={sample_idx}): "
            f"flux channel is all zeros"
        )
        return None

    # ── Energy threshold guard ──────────────────────────────────────────
    if energy_mev > config.max_energy_mev:
        logger.debug(
            f"Skipping sample {record_id} (idx={sample_idx}): "
            f"energy {energy_mev:.1f} MeV > threshold {config.max_energy_mev:.1f} MeV"
        )
        return None

    # ── De-normalise to physical units ──────────────────────────────────
    ct_norm = x[0].cpu().numpy()  # (D, H, W) – CT channel
    ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])

    flux_np = x[1].cpu().numpy()  # (D, H, W) – flux / fast beamlet shape

    gt_dose_norm = y.cpu().numpy()  # (1, D, H, W) or (D, H, W)
    gt_dose = inverse_minmax(
        gt_dose_norm if gt_dose_norm.ndim == 4 else gt_dose_norm[np.newaxis],
        scale["min_ds"],
        scale["max_ds"],
    ).squeeze()

    extract_time = perf_counter() - start_time

    record = SampleRecord(
        sample_id=record_id,
        energy_mev=energy_mev,
        ct_min_hu=float(np.min(ct_hu)),
        ct_max_hu=float(np.max(ct_hu)),
        flux_max=float(np.max(np.abs(flux_np))),
        gt_dose_min=float(np.min(gt_dose)),
        gt_dose_max=float(np.max(gt_dose)),
        bp_range_min_mm=0.0,
        bp_range_max_mm=0.0,
        max_grad_depth_mm=0.0,
        n_density_regions=0,
        total_hu_change=0.0,
        max_hu_jump=0.0,
        sigma_hu_bp=0.0,
        max_hu_gradient=0.0,
        lateral_hu_var_bp=0.0,
        hetero_fraction=0.0,
        interface_bp_distance=0.0,
        mean_sobel_axial=0.0,
        p95_sobel_bp=0.0,
        sum_sobel_bp=0.0,
        gpr=0.0,
        rde=0.0,
        extract_time=extract_time,
    )
    return record, ct_hu, flux_np, energy_mev, gt_dose


# ── Bragg-peak range estimation ─────────────────────────────────────────────


def estimate_bp_range(
    ct_grid: np.ndarray,
    dose_grid: np.ndarray,
    proximal_fraction: float = 0.50,
    fall_fraction: float = 0.10,
) -> Tuple[float, float]:
    """Estimate the depth range of the Bragg peak from a dose grid.

    The algorithm works on the Integrated Depth Dose (IDD) -- the dose
    summed over the lateral dimensions at each depth slice::

        IDD[k] = dose_grid[k, :, :].sum()

    **Proximal boundary (z_min)** -- walk backward from the Bragg-peak
    maximum towards the entrance and find the first depth slice where
    the IDD drops below ``proximal_fraction * IDD_max``.  This mirrors
    the clinical definition of the proximal range (e.g. proximal R80
    or R50) and is inherently robust against plateau noise because it
    is anchored to the peak rather than the entrance.

    **Distal boundary (z_max)** -- starting from the Bragg-peak
    maximum, walk towards deeper depths until the IDD drops below
    ``fall_fraction * IDD_max``.  This marks the distal fall-off.

    Both boundaries are returned as **depth-slice indices** (float).
    Multiply by the voxel spacing to convert to physical units.

    Args:
        ct_grid: 3-D CT volume ``(D, H, W)`` in HU (reserved for
            future density-aware refinements; currently unused).
        dose_grid: 3-D ground-truth dose grid ``(D, H, W)``.
        proximal_fraction: Fraction of peak IDD used as the proximal
            threshold when walking backward from the BP (default 50 %).
        fall_fraction: Fraction of peak IDD used as the distal
            threshold (default 10 %).

    Returns:
        ``(z_min, z_max)`` -- proximal and distal depth-slice indices
        bounding the Bragg peak.  If the dose is effectively zero
        everywhere, ``(0.0, 0.0)`` is returned.
    """
    # -- Integrated Depth Dose ------------------------------------------------
    idd = dose_grid.sum(axis=(1, 2))  # (D,)

    idd_max = idd.max()
    if idd_max < 1e-9:
        return 0.0, 0.0

    bp_idx = int(np.argmax(idd))

    # -- Proximal boundary: walk backward from BP -----------------------------
    proximal_threshold = proximal_fraction * idd_max
    z_min = 0.0  # fallback: entrance
    for k in range(bp_idx, -1, -1):
        if idd[k] < proximal_threshold:
            z_min = float(k)
            break

    # -- Distal boundary: walk from BP towards exit ---------------------------
    n_slices = len(idd)
    fall_threshold = fall_fraction * idd_max
    z_max = float(n_slices - 1)
    for k in range(bp_idx, n_slices):
        if idd[k] < fall_threshold:
            z_max = float(k)
            break

    return z_min, z_max


# ── Density region analysis ─────────────────────────────────────────────────


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
    from src.figures.ct_visualizations import HU_LUT

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


def compute_sobel_metrics(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    z_min: float,
    z_max: float,
    flux_threshold_frac: float = 0.10,
    sobel_percentile: float = 95.0,
) -> dict:
    """Compute 3-D Sobel-based edge metrics in the Bragg-peak zone.

    The CT volume is filtered with 3-D Sobel operators along each axis.
    Gradient magnitudes are aggregated using the flux as a spatial
    weight mask so that only edges traversed by the beam contribute.

    Returns a dict with keys:
        mean_sobel_axial  -- flux-weighted mean of |G_z| in BP zone.
        p95_sobel_bp      -- 95th percentile of the total gradient
                             magnitude among flux-weighted voxels.
    """
    k_start = int(np.ceil(z_min))
    k_end = int(np.floor(z_max))

    defaults = dict(mean_sobel_axial=0.0, p95_sobel_bp=0.0, sum_sobel_bp=0.0)
    if k_end <= k_start:
        return defaults

    # -- 3-D Sobel on the full CT volume (cheap: ~ms for 160x30x30) -------
    g_z = ndimage_sobel(ct_hu, axis=0)  # axial
    g_y = ndimage_sobel(ct_hu, axis=1)
    g_x = ndimage_sobel(ct_hu, axis=2)
    grad_mag = np.sqrt(g_z**2 + g_y**2 + g_x**2)

    # -- Restrict to BP zone ----------------------------------------------
    g_z_bp = np.abs(g_z[k_start : k_end + 1])  # (N, H, W)
    grad_mag_bp = grad_mag[k_start : k_end + 1]  # (N, H, W)
    flux_bp = np.abs(flux[k_start : k_end + 1])  # (N, H, W)

    # -- Build flux weight mask (per-slice 10% threshold) -----------------
    weights = np.zeros_like(flux_bp)
    for i in range(flux_bp.shape[0]):
        f_max = flux_bp[i].max()
        if f_max < 1e-12:
            continue
        mask = flux_bp[i] >= flux_threshold_frac * f_max
        weights[i][mask] = flux_bp[i][mask]

    w_sum = weights.sum()
    if w_sum < 1e-12:
        return defaults

    # -- mean_sobel_axial: flux-weighted mean of |G_z| --------------------
    mean_sobel_axial = float(np.sum(weights * g_z_bp) / w_sum)

    # -- p95_sobel_bp: 95th percentile of grad magnitude (weighted) -------
    #    Collect gradient magnitudes at voxels with non-zero weight.
    active = weights > 0
    if active.sum() == 0:
        return defaults
    active_grads = grad_mag_bp[active]
    p95_sobel_bp = float(np.percentile(active_grads, sobel_percentile))
    sum_sobel_bp = float(np.sum(active_grads))

    return dict(
        mean_sobel_axial=mean_sobel_axial,
        p95_sobel_bp=p95_sobel_bp,
        sum_sobel_bp=sum_sobel_bp,
    )


def compute_sobel_metrics_sphere(
    ct_hu: np.ndarray,
    gt_dose: np.ndarray,
    radius_mm: float,
    resolution: tuple,
    flux: np.ndarray,
    flux_threshold_frac: float = 0.10,
    sobel_percentile: float = 95.0,
) -> dict:
    """Compute 3-D Sobel metrics inside a sphere around the Bragg peak.

    The Bragg peak is located from the ground-truth dose grid.  A
    spherical neighbourhood of the given radius (in mm) is extracted
    and the 3-D Sobel gradient magnitude is computed on the raw CT.

    Returns a dict with keys:
        mean_sobel_axial  -- mean of |G_z| inside the sphere.
        p95_sobel_bp      -- percentile of gradient magnitude inside sphere.
        sum_sobel_bp      -- sum of gradient magnitudes inside sphere.
    """
    D, H, W = ct_hu.shape
    dz, dy, dx = resolution

    # Locate Bragg peak from dose
    idd = gt_dose.sum(axis=(1, 2))
    bp_k = int(np.argmax(idd))
    # Lateral BP position from dose at BP depth
    dose_bp_slice = gt_dose[bp_k]
    if dose_bp_slice.max() > 1e-9:
        bp_y = int(np.argmax(dose_bp_slice.max(axis=1)))
        bp_x = int(np.argmax(dose_bp_slice.max(axis=0)))
    else:
        bp_y, bp_x = H // 2, W // 2

    # Build sphere mask
    rz = int(np.ceil(radius_mm / dz))
    ry = int(np.ceil(radius_mm / dy))
    rx = int(np.ceil(radius_mm / dx))

    z_lo, z_hi = max(0, bp_k - rz), min(D, bp_k + rz + 1)
    y_lo, y_hi = max(0, bp_y - ry), min(H, bp_y + ry + 1)
    x_lo, x_hi = max(0, bp_x - rx), min(W, bp_x + rx + 1)

    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    dist_sq = (
        ((zz - bp_k) * dz) ** 2 + ((yy - bp_y) * dy) ** 2 + ((xx - bp_x) * dx) ** 2
    )
    sphere_mask = dist_sq <= radius_mm**2

    defaults = dict(mean_sobel_axial=0.0, p95_sobel_bp=0.0, sum_sobel_bp=0.0)
    if sphere_mask.sum() == 0:
        return defaults

    # 3-D Sobel on raw CT
    g_z = ndimage_sobel(ct_hu, axis=0)
    g_y = ndimage_sobel(ct_hu, axis=1)
    g_x = ndimage_sobel(ct_hu, axis=2)
    grad_mag = np.sqrt(g_z**2 + g_y**2 + g_x**2)

    # Extract values inside sphere
    g_z_sphere = np.abs(g_z[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi])[sphere_mask]
    grad_sphere = grad_mag[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]

    mean_sobel_axial = float(np.mean(g_z_sphere))
    p95_sobel_bp = float(np.percentile(grad_sphere, sobel_percentile))
    sum_sobel_bp = float(np.sum(grad_sphere))

    return dict(
        mean_sobel_axial=mean_sobel_axial,
        p95_sobel_bp=p95_sobel_bp,
        sum_sobel_bp=sum_sobel_bp,
    )


def extract_all_samples(
    model: DoTA3D_v3,
    record_ids: list,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
    device: torch.device,
    figures_dir: Path,
    show_progress: bool = True,
) -> list[SampleRecord]:
    """Iterate over all beamlets, run inference, and extract metrics.

    For each processed sample a CT segmentation figure is saved to
    ``figures_dir``.

    Args:
        model: The loaded DoTA model.
        record_ids: List of record IDs.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.
        device: Target device for computation.
        figures_dir: Directory where per-sample segmentation figures
            are saved.
        show_progress: Whether to show a progress bar.

    Returns:
        List of SampleRecord objects.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    results: list[SampleRecord] = []
    n_skipped = 0
    n_samples = len(dataset)
    iterator = (
        tqdm(range(n_samples), desc="Extracting samples")
        if show_progress
        else range(n_samples)
    )

    for i in iterator:
        out = extract_single_sample(
            sample_idx=i,
            record_id=record_ids[i],
            dataset=dataset,
            config=config,
        )
        if out is None:
            n_skipped += 1
            continue

        record, ct_hu, flux, energy_mev, gt_dose = out

        # ── Bragg-peak range estimation ─────────────────────────────
        z_min, z_max = estimate_bp_range(
            ct_hu,
            gt_dose,
            proximal_fraction=config.proximal_fraction,
            fall_fraction=config.fall_fraction,
        )
        voxel_spacing_mm = config.resolution[0]  # isotropic: 2 mm
        bp_min_mm = z_min * voxel_spacing_mm
        bp_max_mm = z_max * voxel_spacing_mm
        record.bp_range_min_mm = bp_min_mm
        record.bp_range_max_mm = bp_max_mm

        # Depth of maximum IDD gradient
        idd = gt_dose.sum(axis=(1, 2))
        idd_grad = np.gradient(idd)
        max_grad_slice = int(np.argmax(idd_grad))
        record.max_grad_depth_mm = float(max_grad_slice) * voxel_spacing_mm

        # ── Density region analysis along the beamlet path ──────────
        n_regions, hu_change, region_details = analyse_density_regions(
            ct_hu,
            flux,
            z_min,
            z_max,
            flux_threshold_frac=config.flux_threshold_frac,
        )
        record.n_density_regions = n_regions
        record.total_hu_change = hu_change

        # ── Advanced heterogeneity metrics ───────────────────────
        adv = compute_advanced_metrics(
            ct_hu,
            flux,
            gt_dose,
            z_min,
            z_max,
            region_details,
            flux_threshold_frac=config.flux_threshold_frac,
        )
        record.max_hu_jump = adv["max_hu_jump"]
        record.sigma_hu_bp = adv["sigma_hu_bp"]
        record.max_hu_gradient = adv["max_hu_gradient"]
        record.lateral_hu_var_bp = adv["lateral_hu_var_bp"]
        record.hetero_fraction = adv["hetero_fraction"]
        record.interface_bp_distance = adv["interface_bp_distance"]

        # ── Sobel edge metrics ──────────────────────────────────────
        # Select CT volume for Sobel: raw or smoothed
        if config.sobel_use_raw:
            ct_for_sobel = ct_hu
        else:
            ct_for_sobel = smooth_ct(
                ct_hu, method=config.smoothing_method, sigma=config.smoothing_sigma
            )

        if config.region_method == "sphere":
            sobel = compute_sobel_metrics_sphere(
                ct_for_sobel,
                gt_dose,
                radius_mm=config.sphere_radius_mm,
                resolution=config.resolution,
                flux=flux,
                flux_threshold_frac=config.flux_threshold_frac,
                sobel_percentile=config.sobel_percentile,
            )
        else:
            sobel = compute_sobel_metrics(
                ct_for_sobel,
                flux,
                z_min,
                z_max,
                flux_threshold_frac=config.flux_threshold_frac,
                sobel_percentile=config.sobel_percentile,
            )
        record.mean_sobel_axial = sobel["mean_sobel_axial"]
        record.p95_sobel_bp = sobel["p95_sobel_bp"]
        record.sum_sobel_bp = sobel["sum_sobel_bp"]

        # ── Pflugfelder (2007) heterogeneity index ──────────────────
        hi_result = pflugfelder_hi(
            ct_hu,
            flux,
            gt_dose,
            resolution_mm=config.resolution,
            flux_threshold_frac=config.flux_threshold_frac,
        )
        record.pflugfelder_hi = hi_result["hi"]
        record.wepl_mean = hi_result["wepl_mean"]
        record.wepl_std = hi_result["wepl_std"]

        # ── Interface Severity Index (Schneider 24-class) ───────────
        isi_result = interface_severity(
            ct_hu,
            flux,
            gt_dose,
            resolution_mm=config.resolution,
            severity_mode=config.isi_severity_mode,
            sphere_radius_mm=config.sphere_radius_mm,
            flux_threshold_frac=config.flux_threshold_frac,
        )
        record.isi_sum = isi_result["isi_sum"]
        record.isi_max = isi_result["isi_max"]
        record.isi_mean = isi_result["isi_mean"]
        record.isi_axial_sum = isi_result["isi_axial_sum"]

        # ── Model inference ──────────────────────────────────────
        x_tensor, energy_tensor, y_tensor = dataset[i]
        x_tensor = x_tensor.to(device)
        energy_tensor = energy_tensor.to(device)
        y_tensor = y_tensor.to(device)

        with torch.no_grad():
            y_pred = model(x_tensor.unsqueeze(0), energy_tensor.unsqueeze(0))[0]

        # De-normalise prediction to physical units
        scale = config.scale
        y_np = inverse_minmax(
            y_tensor.unsqueeze(0).detach().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )
        y_pred_np = inverse_minmax(
            y_pred.detach().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )

        # ── Relative Dose Error ─────────────────────────────────────
        rde = calculate_relative_dose_error(to_gy(y_pred_np), to_gy(y_np))
        record.rde = rde

        # ── Gamma Pass Rate ─────────────────────────────────────────
        scale_gpr = {"y_min": scale["min_ds"], "y_max": scale["max_ds"]}
        gpr_result = gamma_index_torch(
            y_tensor.unsqueeze(0),
            y_pred,
            scale=scale_gpr,
            gamma_params=config.gamma_params,
            resolution=config.resolution,
        )
        gpr = gpr_result[1][0] * 100
        record.gpr = gpr

        logger.info(
            f"Sample {record.sample_id}: "
            f"BP range = [{bp_min_mm:.1f}, {bp_max_mm:.1f}] mm, "
            f"max grad @ {record.max_grad_depth_mm:.1f} mm, "
            f"density regions = {n_regions}, "
            f"total HU change = {hu_change:.1f}, "
            f"GPR = {gpr:.2f}%, RDE = {rde:.4f}% "
            f"(slices [{z_min:.0f}, {z_max:.0f}], "
            f"E = {energy_mev:.1f} MeV)"
        )

        results.append(record)

        if show_progress:
            iterator.set_postfix(
                E=f"{record.energy_mev:.0f}",
                ct_max=f"{record.ct_max_hu:.0f}",
            )

    if n_skipped > 0:
        logger.info(
            f"Skipped {n_skipped}/{n_samples} samples "
            f"(zero flux or energy > {config.max_energy_mev:.0f} MeV) "
            f"({100.0 * n_skipped / n_samples:.1f}%)"
        )

    return results


# ── Results CSV ─────────────────────────────────────────────────────────────


def save_results_csv(results: list[SampleRecord], output_path: Path) -> None:
    """Save per-beamlet results to a CSV file.

    Args:
        results: List of SampleRecord objects.
        output_path: Path to the output CSV file.
    """
    fieldnames = [
        "sample_id",
        "energy_mev",
        "ct_min_hu",
        "ct_max_hu",
        "flux_max",
        "gt_dose_min",
        "gt_dose_max",
        "bp_range_min_mm",
        "bp_range_max_mm",
        "max_grad_depth_mm",
        "n_density_regions",
        "total_hu_change",
        "max_hu_jump",
        "sigma_hu_bp",
        "max_hu_gradient",
        "lateral_hu_var_bp",
        "hetero_fraction",
        "interface_bp_distance",
        "mean_sobel_axial",
        "p95_sobel_bp",
        "sum_sobel_bp",
        "pflugfelder_hi",
        "wepl_mean_mm",
        "wepl_std_mm",
        "isi_sum",
        "isi_max",
        "isi_mean",
        "isi_axial_sum",
        "gpr_pct",
        "rde_pct",
        "extract_time_s",
    ]

    sorted_results = sorted(results, key=lambda r: r.energy_mev)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_results:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "energy_mev": f"{r.energy_mev:.2f}",
                    "ct_min_hu": f"{r.ct_min_hu:.2f}",
                    "ct_max_hu": f"{r.ct_max_hu:.2f}",
                    "flux_max": f"{r.flux_max:.6e}",
                    "gt_dose_min": f"{r.gt_dose_min:.6e}",
                    "gt_dose_max": f"{r.gt_dose_max:.6e}",
                    "bp_range_min_mm": f"{r.bp_range_min_mm:.1f}",
                    "bp_range_max_mm": f"{r.bp_range_max_mm:.1f}",
                    "max_grad_depth_mm": f"{r.max_grad_depth_mm:.1f}",
                    "n_density_regions": r.n_density_regions,
                    "total_hu_change": f"{r.total_hu_change:.1f}",
                    "max_hu_jump": f"{r.max_hu_jump:.1f}",
                    "sigma_hu_bp": f"{r.sigma_hu_bp:.2f}",
                    "max_hu_gradient": f"{r.max_hu_gradient:.1f}",
                    "lateral_hu_var_bp": f"{r.lateral_hu_var_bp:.2f}",
                    "hetero_fraction": f"{r.hetero_fraction:.4f}",
                    "interface_bp_distance": f"{r.interface_bp_distance:.2f}",
                    "mean_sobel_axial": f"{r.mean_sobel_axial:.2f}",
                    "p95_sobel_bp": f"{r.p95_sobel_bp:.2f}",
                    "sum_sobel_bp": f"{r.sum_sobel_bp:.2f}",
                    "pflugfelder_hi": f"{r.pflugfelder_hi:.6f}",
                    "wepl_mean_mm": f"{r.wepl_mean:.2f}",
                    "wepl_std_mm": f"{r.wepl_std:.2f}",
                    "isi_sum": f"{r.isi_sum:.4f}",
                    "isi_max": f"{r.isi_max:.4f}",
                    "isi_mean": f"{r.isi_mean:.4f}",
                    "isi_axial_sum": f"{r.isi_axial_sum:.4f}",
                    "gpr_pct": f"{r.gpr:.2f}",
                    "rde_pct": f"{r.rde:.4f}",
                    "extract_time_s": f"{r.extract_time:.4f}",
                }
            )

    logger.info(f"Results CSV saved to {output_path}")


# ── Figure generation for selected samples ──────────────────────────────────


def _generate_figures_for_records(
    model: DoTA3D_v3,
    records: list[SampleRecord],
    record_ids: list,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
    device: torch.device,
    output_dir: Path,
    category_label: str,
) -> None:
    """Generate CT segmentation and publication figures for a list of records.

    Both figure types are saved into *output_dir*.

    Args:
        model: The loaded DoTA model.
        records: SampleRecord objects to plot.
        record_ids: Ordered list of record IDs matching the dataset.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.
        device: Target device for computation.
        output_dir: Directory where figures are stored.
        category_label: Human-readable label for logging (e.g. "worst_gpr").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx: dict[str, int] = {rid: i for i, rid in enumerate(record_ids)}
    scale = config.scale

    for record in tqdm(records, desc=f"Figures ({category_label})"):
        sample_idx = id_to_idx.get(record.sample_id)
        if sample_idx is None:
            logger.warning(
                f"Sample {record.sample_id} not found in dataset -- skipping"
            )
            continue

        # Reload data
        x, energy, y = dataset[sample_idx]
        x = x.to(device)
        energy = energy.to(device)
        y = y.to(device)

        # De-normalise CT and dose
        ct_norm = x[0].cpu().numpy()
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])
        flux_np = x[1].cpu().numpy()

        gt_dose_norm = y.cpu().numpy()
        gt_dose = inverse_minmax(
            gt_dose_norm if gt_dose_norm.ndim == 4 else gt_dose_norm[np.newaxis],
            scale["min_ds"],
            scale["max_ds"],
        ).squeeze()

        # Re-estimate BP range for figure annotation
        z_min, z_max = estimate_bp_range(
            ct_hu,
            gt_dose,
            proximal_fraction=config.proximal_fraction,
            fall_fraction=config.fall_fraction,
        )
        voxel_spacing_mm = config.resolution[0]

        # ── CT segmentation figure ──────────────────────────────────
        ct_hu_smooth = smooth_ct(
            ct_hu, method=config.smoothing_method, sigma=config.smoothing_sigma
        )

        g_z_raw = ndimage_sobel(ct_hu, axis=0)
        g_y_raw = ndimage_sobel(ct_hu, axis=1)
        g_x_raw = ndimage_sobel(ct_hu, axis=2)
        sobel_mag_raw = np.sqrt(g_z_raw**2 + g_y_raw**2 + g_x_raw**2)

        g_z = ndimage_sobel(ct_hu_smooth, axis=0)
        g_y = ndimage_sobel(ct_hu_smooth, axis=1)
        g_x = ndimage_sobel(ct_hu_smooth, axis=2)
        sobel_mag = np.sqrt(g_z**2 + g_y**2 + g_x**2)

        fig_path = output_dir / f"{record.sample_id}_ct_seg.png"
        plot_ct_with_segmentation(
            ct_hu_smooth,
            record.sample_id,
            fig_path,
            ct_hu_unsmoothed=ct_hu,
            gt_dose=gt_dose,
            bp_range_slices=(z_min, z_max),
            voxel_spacing_mm=voxel_spacing_mm,
            sobel_magnitude=sobel_mag,
            sobel_magnitude_raw=sobel_mag_raw,
        )

        # ── Publication figure (GT vs prediction) ───────────────────
        with torch.no_grad():
            y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]

        y_np = inverse_minmax(
            y.unsqueeze(0).detach().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )
        y_pred_np = inverse_minmax(
            y_pred.detach().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )

        gt_sq = y_np.squeeze()
        pred_sq = y_pred_np.squeeze()
        rmse = calculate_rmse(to_gy(pred_sq), to_gy(gt_sq))
        mask = gt_sq > 0.1 * np.max(gt_sq)
        mape = calculate_pure_mape(gt_sq[mask], pred_sq[mask])

        x_input = x.squeeze().cpu().numpy()  # (2, D, H, W)
        pub_path = output_dir / f"{record.sample_id}_E{record.energy_mev:.2f}MeV.svg"

        publication_figure(
            x_input,
            record.energy_mev,
            gt_sq,
            pred_sq,
            str(pub_path),
            rmse,
            mape,
            record.gpr,
            gamma_params=config.gamma_params,
            beamlet_shape=True,
        )

        logger.info(
            f"[{category_label}] Figures saved for {record.sample_id}: "
            f"GPR={record.gpr:.2f}%, pflugfelder_hi={record.pflugfelder_hi:.4f}, "
            f"sum_sobel_bp={record.sum_sobel_bp:.2f}"
        )


def generate_figures_for_selection(
    model: DoTA3D_v3,
    results: list[SampleRecord],
    record_ids: list,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
    device: torch.device,
    figures_dir: Path,
    n_samples_requested: Optional[int],
) -> None:
    """Generate CT segmentation and publication figures for selected samples.

    Generates figures for four categories of interest:
    - 3 worst performing cases (lowest GPR)
    - 3 best performing cases (highest GPR)
    - 3 cases with the highest Pflugfelder heterogeneity index
    - 3 cases with the highest sum_sobel_bp

    Each category gets its own subdirectory under ``publications/``.

    Args:
        model: The loaded DoTA model.
        results: List of SampleRecord objects (with metrics populated).
        record_ids: Ordered list of record IDs matching the dataset.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.
        device: Target device for computation.
        figures_dir: Output directory for figures.
        n_samples_requested: The n_samples value from the config/CLI.
    """
    publications_dir = figures_dir / "publications"
    publications_dir.mkdir(parents=True, exist_ok=True)

    # ── 3 worst GPR (lowest) ────────────────────────────────────────────
    sorted_by_gpr_asc = sorted(results, key=lambda r: r.gpr)
    worst_gpr = sorted_by_gpr_asc[:3]
    logger.info("3 worst GPR cases:")
    for rank, r in enumerate(worst_gpr):
        logger.info(f"  #{rank+1}: {r.sample_id}  GPR={r.gpr:.2f}%")

    # ── 3 best GPR (highest) ────────────────────────────────────────────
    best_gpr = sorted_by_gpr_asc[-3:][::-1]
    logger.info("3 best GPR cases:")
    for rank, r in enumerate(best_gpr):
        logger.info(f"  #{rank+1}: {r.sample_id}  GPR={r.gpr:.2f}%")

    # ── 3 highest Pflugfelder HI ────────────────────────────────────────
    sorted_by_phi = sorted(results, key=lambda r: r.pflugfelder_hi, reverse=True)
    top_phi = sorted_by_phi[:3]
    logger.info("3 highest pflugfelder_hi cases:")
    for rank, r in enumerate(top_phi):
        logger.info(
            f"  #{rank+1}: {r.sample_id}  pflugfelder_hi={r.pflugfelder_hi:.4f}"
        )

    # ── 3 highest sum_sobel_bp ──────────────────────────────────────────
    sorted_by_sobel = sorted(results, key=lambda r: r.sum_sobel_bp, reverse=True)
    top_sobel = sorted_by_sobel[:3]
    logger.info("3 highest sum_sobel_bp cases:")
    for rank, r in enumerate(top_sobel):
        logger.info(f"  #{rank+1}: {r.sample_id}  sum_sobel_bp={r.sum_sobel_bp:.2f}")

    # ── Generate figures for each category ──────────────────────────────
    categories = [
        ("worst_gpr", worst_gpr),
        ("best_gpr", best_gpr),
        ("highest_pflugfelder_hi", top_phi),
        ("highest_sum_sobel", top_sobel),
    ]

    for category_name, selected in categories:
        cat_dir = publications_dir / category_name
        _generate_figures_for_records(
            model=model,
            records=selected,
            record_ids=record_ids,
            dataset=dataset,
            config=config,
            device=device,
            output_dir=cat_dir,
            category_label=category_name,
        )

    logger.info("Figure generation complete")


# ── Summary ─────────────────────────────────────────────────────────────────


def print_summary(results: list[SampleRecord], total_time: float) -> None:
    """Print extraction summary to the logger.

    Args:
        results: List of SampleRecord objects.
        total_time: Total extraction time in seconds.
    """
    energies = np.array([r.energy_mev for r in results])
    ct_mins = np.array([r.ct_min_hu for r in results])
    ct_maxs = np.array([r.ct_max_hu for r in results])
    dose_mins = np.array([r.gt_dose_min for r in results])
    dose_maxs = np.array([r.gt_dose_max for r in results])
    gprs = np.array([r.gpr for r in results])
    rdes = np.array([r.rde for r in results])
    extract_times = np.array([r.extract_time for r in results])

    logger.info(f"Total elapsed time: {total_time:.2f}s")
    logger.info(f"Samples processed: {len(results)}")
    logger.info(f"Average time per sample: {np.mean(extract_times):.4f}s")
    logger.info(
        f"Energy [MeV]     – "
        f"mean: {np.mean(energies):.2f}, "
        f"std: {np.std(energies):.2f}, "
        f"min: {np.min(energies):.2f}, "
        f"max: {np.max(energies):.2f}"
    )
    logger.info(
        f"CT HU min        – "
        f"mean: {np.mean(ct_mins):.2f}, "
        f"min: {np.min(ct_mins):.2f}, "
        f"max: {np.max(ct_mins):.2f}"
    )
    logger.info(
        f"CT HU max        – "
        f"mean: {np.mean(ct_maxs):.2f}, "
        f"min: {np.min(ct_maxs):.2f}, "
        f"max: {np.max(ct_maxs):.2f}"
    )
    logger.info(
        f"GT dose min      – "
        f"mean: {np.mean(dose_mins):.6e}, "
        f"min: {np.min(dose_mins):.6e}, "
        f"max: {np.max(dose_mins):.6e}"
    )
    logger.info(
        f"GT dose max      – "
        f"mean: {np.mean(dose_maxs):.6e}, "
        f"min: {np.min(dose_maxs):.6e}, "
        f"max: {np.max(dose_maxs):.6e}"
    )
    logger.info(
        f"GPR [%]          – "
        f"mean: {np.mean(gprs):.2f}, "
        f"std: {np.std(gprs):.2f}, "
        f"min: {np.min(gprs):.2f}, "
        f"max: {np.max(gprs):.2f}"
    )
    logger.info(
        f"RDE [%]          – "
        f"mean: {np.mean(rdes):.4f}, "
        f"std: {np.std(rdes):.4f}, "
        f"min: {np.min(rdes):.4f}, "
        f"max: {np.max(rdes):.4f}"
    )


# ── Scatter plots ───────────────────────────────────────────────────────────


def generate_scatter_plots(
    results: list[SampleRecord],
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Generate scatter plots for all heterogeneity metrics vs GPR.

    Produces individual scatter plots for each metric and a correlation
    summary table (both CSV and PNG).

    Args:
        results: List of SampleRecord objects.
        output_dir: Directory where figures will be saved.
        config: Analysis configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) < 3:
        logger.info("Skipping scatter plots -- fewer than 3 samples")
        return

    gpr_arr = np.array([r.gpr for r in results])

    gpr_label = (
        f"GPR ({config.gamma_params['dose_percent_threshold']}%/"
        f"{config.gamma_params['distance_mm_threshold']}mm, "
        f"{config.gamma_params['lower_percent_dose_cutoff']}% cutoff) [%]"
    )

    # ── Define all metrics to plot ───────────────────────────────────────
    METRICS = [
        (
            "n_density_regions",
            "Number of density regions",
            "n_density_regions",
            "#4C72B0",
        ),
        ("total_hu_change", "Total HU change [HU]", "total_hu_change", "#DD8452"),
        ("max_hu_jump", "Max single HU jump [HU]", "max_hu_jump", "#55A868"),
        ("sigma_hu_bp", r"$\sigma_{HU}$ in BP zone [HU]", "sigma_hu_bp", "#C44E52"),
        (
            "max_hu_gradient",
            "Max axial HU gradient [HU/slice]",
            "max_hu_gradient",
            "#8172B2",
        ),
        (
            "lateral_hu_var_bp",
            r"Lateral HU variance at BP [HU$^2$]",
            "lateral_hu_var_bp",
            "#937860",
        ),
        (
            "hetero_fraction",
            "Hetero fraction (non-dominant class)",
            "hetero_fraction",
            "#DA8BC3",
        ),
        (
            "mean_sobel_axial",
            "Mean Sobel axial (flux-weighted) [HU/vox]",
            "mean_sobel_axial",
            "#6ACC65",
        ),
        (
            "p95_sobel_bp",
            "P95 Sobel gradient magnitude [HU/vox]",
            "p95_sobel_bp",
            "#D65F5F",
        ),
        (
            "sum_sobel_bp",
            "Sum Sobel gradient magnitude [HU/vox]",
            "sum_sobel_bp",
            "#FF6347",
        ),
        (
            "interface_bp_distance",
            "Interface-to-BP distance [slices]",
            "interface_bp_distance",
            "#8C8C8C",
        ),
        (
            "pflugfelder_hi",
            "Pflugfelder HI (WEPL CV)",
            "pflugfelder_hi",
            "#2CA02C",
        ),
        (
            "wepl_mean",
            "WEPL mean [mm]",
            "wepl_mean",
            "#17BECF",
        ),
        (
            "wepl_std",
            r"WEPL $\sigma$ [mm]",
            "wepl_std",
            "#BCBD22",
        ),
        (
            "isi_sum",
            r"ISI sum $(\Delta\mathrm{RSP})^2$ — BP sphere",
            "isi_sum",
            "#E377C2",
        ),
        (
            "isi_max",
            r"ISI max $(\Delta\mathrm{RSP})^2$ — BP sphere",
            "isi_max",
            "#7F7F7F",
        ),
        (
            "isi_mean",
            r"ISI mean $(\Delta\mathrm{RSP})^2$ — BP sphere",
            "isi_mean",
            "#1F77B4",
        ),
        (
            "isi_axial_sum",
            r"ISI axial sum $(\Delta\mathrm{RSP})^2$ — BP sphere",
            "isi_axial_sum",
            "#FF7F0E",
        ),
    ]

    # ── Also include derived ratios ─────────────────────────────────────
    bp_range = np.array([r.bp_range_max_mm - r.bp_range_min_mm for r in results])
    n_regions = np.array([r.n_density_regions for r in results], dtype=float)
    hu_change = np.array([r.total_hu_change for r in results])

    DERIVED = [
        (
            "hu_change_per_region",
            "HU change per region [HU]",
            hu_change / np.clip(n_regions, 1, None),
            "#64B5CD",
        ),
        (
            "hu_change_per_mm",
            "HU change per mm [HU/mm]",
            hu_change / np.clip(bp_range, 1, None),
            "#B07AA1",
        ),
        ("bp_range_mm", "BP range [mm]", bp_range, "#E5A836"),
    ]

    # ── Collect correlation results ─────────────────────────────────────
    corr_rows: list[dict] = []

    def _plot_single(metric_key, xlabel, values, colour, fname):
        r_p, p_p = pearsonr(values, gpr_arr)
        r_s, p_s = spearmanr(values, gpr_arr)

        corr_rows.append(
            dict(
                metric=metric_key,
                pearson_r=r_p,
                pearson_p=p_p,
                spearman_r=r_s,
                spearman_p=p_s,
            )
        )

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        ax.scatter(
            values,
            gpr_arr,
            s=18,
            alpha=0.6,
            color=colour,
            edgecolors="k",
            linewidths=0.3,
        )

        # Linear fit
        z = np.polyfit(values.astype(float), gpr_arr, 1)
        x_fit = np.linspace(values.min(), values.max(), 200)
        ax.plot(
            x_fit,
            np.polyval(z, x_fit),
            "r--",
            linewidth=1.2,
            label=(
                f"Pearson r = {r_p:.3f} (p = {p_p:.2e})\n"
                f"Spearman r = {r_s:.3f} (p = {p_s:.2e})"
            ),
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(gpr_label)
        ax.set_title(f"{xlabel}  vs  GPR  (N = {len(results)})")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(
            f"{metric_key:30s}  "
            f"Pearson r={r_p:+.3f} (p={p_p:.2e})  "
            f"Spearman r={r_s:+.3f} (p={p_s:.2e})"
        )

    # Plot primary metrics
    for key, label, attr, colour in METRICS:
        values = np.array([getattr(r, attr) for r in results])
        _plot_single(key, label, values, colour, f"{key}_vs_gpr.png")

    # Plot derived metrics
    for key, label, values, colour in DERIVED:
        _plot_single(key, label, values, colour, f"{key}_vs_gpr.png")

    # ── Correlation summary table (sorted by |Spearman r|) ──────────────
    corr_rows.sort(key=lambda d: abs(d["spearman_r"]), reverse=True)

    # Save as CSV
    corr_csv = output_dir / "correlation_summary.csv"
    with open(corr_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
        )
        w.writeheader()
        for row in corr_rows:
            w.writerow(
                {
                    "metric": row["metric"],
                    "pearson_r": f"{row['pearson_r']:+.4f}",
                    "pearson_p": f"{row['pearson_p']:.4e}",
                    "spearman_r": f"{row['spearman_r']:+.4f}",
                    "spearman_p": f"{row['spearman_p']:.4e}",
                }
            )
    logger.info(f"Correlation summary CSV saved to {corr_csv}")

    # Log the ranked summary
    logger.info("")
    logger.info("=" * 72)
    logger.info("CORRELATION SUMMARY (sorted by |Spearman r|)")
    logger.info("=" * 72)
    logger.info(f"{'Metric':30s}  {'Pearson r':>10s}  {'Spearman r':>10s}")
    logger.info("-" * 72)
    for row in corr_rows:
        logger.info(
            f"{row['metric']:30s}  {row['pearson_r']:>+10.4f}  {row['spearman_r']:>+10.4f}"
        )
    logger.info("=" * 72)

    # ── Summary bar chart of |correlations| ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    names = [r["metric"] for r in corr_rows]
    pearson_abs = [abs(r["pearson_r"]) for r in corr_rows]
    spearman_abs = [abs(r["spearman_r"]) for r in corr_rows]
    x_pos = np.arange(len(names))
    bar_w = 0.35
    ax.barh(x_pos - bar_w / 2, pearson_abs, bar_w, label="|Pearson r|", color="#4C72B0")
    ax.barh(
        x_pos + bar_w / 2, spearman_abs, bar_w, label="|Spearman r|", color="#DD8452"
    )
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Absolute correlation with GPR")
    ax.set_title(f"Metric correlation ranking  (N = {len(results)})")
    ax.legend()
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"Correlation ranking chart saved to {output_dir / 'correlation_ranking.png'}"
    )


# ── Energy-stratified analysis ──────────────────────────────────────────────


def _partial_correlation(
    df: pd.DataFrame, x_col: str, y_col: str, control_col: str
) -> float:
    """Partial Spearman correlation controlling for *control_col*."""
    from scipy.stats import spearmanr

    rxy = spearmanr(df[x_col], df[y_col]).correlation
    rxz = spearmanr(df[x_col], df[control_col]).correlation
    ryz = spearmanr(df[y_col], df[control_col]).correlation
    denom = math.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom < 1e-12:
        return float("nan")
    return (rxy - rxz * ryz) / denom


def generate_energy_stratified_analysis(
    results: list,
    output_dir: Path,
    config: "AnalysisConfig",
) -> None:
    """Produce energy-stratified correlation analysis."""
    from scipy.stats import spearmanr, pearsonr

    strat_dir = output_dir / "energy_stratified"
    strat_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in results])
    if "gpr" not in df.columns or df["gpr"].isna().all():
        logger.warning("No GPR data – skipping energy-stratified analysis.")
        return

    # Derived metrics
    if "n_density_regions" in df.columns:
        mask = df["n_density_regions"] > 0
        df.loc[mask, "hu_change_per_region"] = (
            df.loc[mask, "total_hu_change"] / df.loc[mask, "n_density_regions"]
        )
    if "bp_range_start" in df.columns and "bp_range_end" in df.columns:
        df["bp_range_mm"] = (
            df["bp_range_end"] - df["bp_range_start"]
        ) * config.resolution[0]
        mask = df["bp_range_mm"] > 0
        df.loc[mask, "hu_change_per_mm"] = (
            df.loc[mask, "total_hu_change"] / df.loc[mask, "bp_range_mm"]
        )

    # Metric columns to analyse
    candidate_metrics = [
        "ct_mean",
        "ct_std",
        "n_density_regions",
        "total_hu_change",
        "max_hu_jump",
        "sigma_hu_bp",
        "max_hu_gradient",
        "lateral_hu_var_bp",
        "hetero_fraction",
        "interface_bp_distance",
        "mean_sobel_axial",
        "p95_sobel_bp",
        "sum_sobel_bp",
        "pflugfelder_hi",
        "wepl_mean",
        "wepl_std",
        "isi_sum",
        "isi_max",
        "isi_mean",
        "isi_axial_sum",
        "hu_change_per_region",
        "hu_change_per_mm",
        "bp_range_mm",
    ]
    metrics = [
        m for m in candidate_metrics if m in df.columns and df[m].notna().sum() > 5
    ]

    bins = sorted(config.energy_bins)
    bin_labels = ["{}-{}".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    df["energy_bin"] = pd.cut(
        df["energy_mev"], bins=bins, labels=bin_labels, include_lowest=True
    )

    # ── 1. Binned correlation heatmaps ──────────────────────────────────
    spearman_mat = pd.DataFrame(index=bin_labels, columns=metrics, dtype=float)
    pearson_mat = pd.DataFrame(index=bin_labels, columns=metrics, dtype=float)
    count_mat = pd.DataFrame(index=bin_labels, columns=metrics, dtype=int)

    for label in bin_labels:
        sub = df[df["energy_bin"] == label].dropna(subset=["gpr"])
        for m in metrics:
            valid = sub.dropna(subset=[m])
            n = len(valid)
            count_mat.loc[label, m] = n
            if n >= 5:
                spearman_mat.loc[label, m] = spearmanr(
                    valid[m], valid["gpr"]
                ).correlation
                pearson_mat.loc[label, m] = pearsonr(valid[m], valid["gpr"]).statistic
            else:
                spearman_mat.loc[label, m] = float("nan")
                pearson_mat.loc[label, m] = float("nan")

    for name, mat in [("spearman", spearman_mat), ("pearson", pearson_mat)]:
        fig, ax = plt.subplots(
            figsize=(max(10, len(metrics) * 0.8), max(4, len(bin_labels) * 0.7))
        )
        sns.heatmap(
            mat.astype(float),
            annot=True,
            fmt=".2f",
            center=0,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("{} correlation with GPR by energy bin".format(name.capitalize()))
        ax.set_ylabel("Energy bin (MeV)")
        ax.set_xlabel("Metric")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(
            strat_dir / "{}_heatmap_by_energy.png".format(name),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Save binned correlations CSV
    binned_csv = []
    for label in bin_labels:
        for m in metrics:
            binned_csv.append(
                {
                    "energy_bin": label,
                    "metric": m,
                    "spearman": spearman_mat.loc[label, m],
                    "pearson": pearson_mat.loc[label, m],
                    "n": count_mat.loc[label, m],
                }
            )
    pd.DataFrame(binned_csv).to_csv(strat_dir / "binned_correlations.csv", index=False)
    logger.info("Binned correlation heatmaps saved.")

    # ── 2. Energy-coloured scatter plots (top 6 metrics) ───────────────
    # Rank by absolute overall Spearman
    overall_corr = {}
    for m in metrics:
        valid = df.dropna(subset=[m, "gpr"])
        if len(valid) >= 5:
            overall_corr[m] = abs(spearmanr(valid[m], valid["gpr"]).correlation)
    top_metrics = sorted(overall_corr, key=overall_corr.get, reverse=True)[:6]

    for m in top_metrics:
        sub = df.dropna(subset=[m, "gpr", "energy_mev"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            sub[m],
            sub["gpr"],
            c=sub["energy_mev"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="none",
            s=20,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Energy (MeV)")
        ax.set_xlabel(m)
        ax.set_ylabel("GPR (%)")
        ax.set_title("{} vs GPR (coloured by energy)".format(m))
        fig.tight_layout()
        fig.savefig(
            strat_dir / "{}_vs_gpr_energy_coloured.png".format(m),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    logger.info(
        "Energy-coloured scatter plots saved for top %d metrics.", len(top_metrics)
    )

    # ── 3. Partial correlations controlling for energy ──────────────────
    partial_rows = []
    for m in metrics:
        valid = df.dropna(subset=[m, "gpr", "energy_mev"])
        if len(valid) >= 10:
            raw_spearman = spearmanr(valid[m], valid["gpr"]).correlation
            partial_spearman = _partial_correlation(valid, m, "gpr", "energy_mev")
            partial_rows.append(
                {
                    "metric": m,
                    "raw_spearman": raw_spearman,
                    "partial_spearman": partial_spearman,
                    "difference": partial_spearman - raw_spearman,
                    "n": len(valid),
                }
            )
    partial_df = pd.DataFrame(partial_rows)
    if len(partial_df) > 0:
        partial_df = partial_df.sort_values(
            "partial_spearman", key=abs, ascending=False
        )
    partial_df.to_csv(strat_dir / "partial_correlations.csv", index=False)

    # Bar chart: raw vs partial
    if len(partial_df) > 0:
        fig, ax = plt.subplots(figsize=(max(10, len(partial_df) * 0.7), 6))
        x = range(len(partial_df))
        w = 0.35
        ax.bar(
            [i - w / 2 for i in x],
            partial_df["raw_spearman"],
            w,
            label="Raw Spearman",
            color="steelblue",
        )
        ax.bar(
            [i + w / 2 for i in x],
            partial_df["partial_spearman"],
            w,
            label="Partial (ctrl energy)",
            color="coral",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(partial_df["metric"], rotation=45, ha="right")
        ax.set_ylabel("Spearman correlation with GPR")
        ax.set_title("Raw vs Partial Spearman (controlling for energy)")
        ax.legend()
        ax.axhline(0, color="grey", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(
            strat_dir / "partial_vs_raw_correlation.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    logger.info("Partial correlation analysis saved.")


# ── Main CLI ────────────────────────────────────────────────────────────────


@app.command()
def main(
    model_name: Annotated[
        Optional[str],
        typer.Argument(help="Name of the model directory under models/"),
    ] = None,
    h5_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the HDF5 dataset file"),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML configuration file"),
    ] = None,
    excluded_indexes_file: Annotated[
        Optional[Path],
        typer.Option(help="Path to file listing record IDs to exclude"),
    ] = None,
    model_fname: Annotated[Optional[str], typer.Option(help="Model filename")] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index (-1 for CPU)")
    ] = None,
    max_energy_mev: Annotated[
        Optional[float],
        typer.Option(
            help="Skip beamlets with initial energy above this threshold [MeV]"
        ),
    ] = None,
    n_samples: Annotated[
        Optional[int],
        typer.Option(help="Limit extraction to the first N samples (default: all)"),
    ] = None,
    no_progress: Annotated[
        Optional[bool], typer.Option(help="Disable progress bar")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose output")
    ] = None,
) -> None:
    """Extract per-beamlet data from the training set (advanced metrics).

    Iterates over every beamlet in the HDF5 training set, extracts the
    CT volume, fast beamlet-shape projection (flux), initial energy, and
    ground-truth dose, and logs summary statistics.

    Can be configured via CLI arguments, a YAML config file (--config),
    or both.  CLI arguments take precedence over YAML values.
    """
    # ── Load & merge config ─────────────────────────────────────────────
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    model_name = model_name or yaml_config.get("model_name")
    h5_path = h5_path or (
        Path(yaml_config["h5_path"]) if "h5_path" in yaml_config else None
    )
    excluded_indexes_file = excluded_indexes_file or (
        Path(yaml_config["excluded_indexes_file"])
        if "excluded_indexes_file" in yaml_config
        else None
    )
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )
    max_energy_mev = (
        max_energy_mev
        if max_energy_mev is not None
        else yaml_config.get("max_energy_mev", 250.0)
    )
    n_samples = n_samples if n_samples is not None else yaml_config.get("n_samples")
    no_progress = (
        no_progress
        if no_progress is not None
        else yaml_config.get("no_progress", False)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    # ── Validate required arguments ─────────────────────────────────────
    if model_name is None:
        raise typer.BadParameter(
            "MODEL_NAME is required (via CLI argument or YAML config)"
        )
    if h5_path is None:
        raise typer.BadParameter(
            "H5_PATH is required (via CLI argument or YAML config)"
        )

    # ── Setup run directory & logging ───────────────────────────────────
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir)
    log_file = setup_logging(run_dir, verbose=verbose)

    # Copy config for reproducibility
    if config_path is not None:
        shutil.copy2(config_path, run_dir / config_path.name)
        logger.info(f"Config file copied to {run_dir / config_path.name}")

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    # ── Resolve paths ───────────────────────────────────────────────────
    model_hub = PROJECT_ROOT / "models"
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"

    if not h5_path.is_absolute():
        h5_path = PROJECT_ROOT / h5_path

    validate_inputs(h5_path, model_path, hyperparams_path)

    # ── Log run configuration ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RUN CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model file: {model_fname}")
    logger.info(f"HDF5 dataset: {h5_path}")
    logger.info(f"Excluded indexes file: {excluded_indexes_file}")
    logger.info(f"Device index: {device_index}")
    logger.info(f"Max energy: {max_energy_mev} MeV")
    logger.info(f"N samples: {n_samples if n_samples is not None else 'all'}")
    logger.info(f"Region method: {yaml_config.get('region_method', 'bp_range')}")
    logger.info(f"Sphere radius: {yaml_config.get('sphere_radius_mm', 10.0)} mm")
    logger.info(f"Sobel on raw CT: {yaml_config.get('sobel_use_raw', False)}")
    logger.info(f"ISI severity mode: {yaml_config.get('isi_severity_mode', 'rsp_sq')}")
    logger.info("=" * 60)

    # ── Setup analysis configuration ────────────────────────────────────
    energy_bins = yaml_config.get("energy_bins", [70, 100, 130, 160, 190, 220, 250])
    smoothing_sigma = yaml_config.get("smoothing_sigma", 1.0)
    smoothing_method = yaml_config.get("smoothing_method", "gaussian")
    proximal_fraction = yaml_config.get("proximal_fraction", 0.50)
    fall_fraction = yaml_config.get("fall_fraction", 0.10)
    flux_threshold_frac = yaml_config.get("flux_threshold_frac", 0.10)
    sobel_percentile = yaml_config.get("sobel_percentile", 95.0)
    region_method = yaml_config.get("region_method", "bp_range")
    sphere_radius_mm = yaml_config.get("sphere_radius_mm", 10.0)
    sobel_use_raw = yaml_config.get("sobel_use_raw", False)
    isi_severity_mode = yaml_config.get("isi_severity_mode", "rsp_sq")
    resolution = tuple(yaml_config.get("resolution", [2.0, 2.0, 2.0]))

    # Override gamma params from YAML if provided
    gamma_params = DEFAULT_GAMMA_PARAMS.copy()
    yaml_gamma = yaml_config.get("gamma_params", {})
    gamma_params.update(yaml_gamma)

    analysis_config = AnalysisConfig(
        max_energy_mev=max_energy_mev,
        smoothing_sigma=smoothing_sigma,
        smoothing_method=smoothing_method,
        energy_bins=energy_bins,
        proximal_fraction=proximal_fraction,
        fall_fraction=fall_fraction,
        flux_threshold_frac=flux_threshold_frac,
        sobel_percentile=sobel_percentile,
        region_method=region_method,
        sphere_radius_mm=sphere_radius_mm,
        sobel_use_raw=sobel_use_raw,
        isi_severity_mode=isi_severity_mode,
        resolution=resolution,
        gamma_params=gamma_params,
    )

    # ── Load excluded indexes ───────────────────────────────────────────
    excluded_indexes: list[str] = []
    if excluded_indexes_file is not None:
        exc_path = (
            excluded_indexes_file
            if excluded_indexes_file.is_absolute()
            else PROJECT_ROOT / excluded_indexes_file
        )
        if exc_path.exists():
            with open(exc_path, "r") as f:
                excluded_indexes = [line.strip() for line in f if line.strip()]
            logger.info(
                f"Loaded {len(excluded_indexes)} excluded indexes from {exc_path}"
            )
        else:
            logger.warning(f"Excluded indexes file not found: {exc_path}")

    # ── Discover samples in HDF5 ────────────────────────────────────────
    with h5py.File(h5_path, "r") as ds:
        all_record_ids = list(ds.keys())
    logger.info(f"Total records in HDF5: {len(all_record_ids)}")

    record_ids = [rid for rid in all_record_ids if rid not in excluded_indexes]
    logger.info(
        f"Records after exclusion: {len(record_ids)} "
        f"(excluded {len(all_record_ids) - len(record_ids)})"
    )

    if n_samples is not None:
        record_ids = record_ids[:n_samples]
        logger.info(f"Limited to first {n_samples} samples → using {len(record_ids)}")

    if not record_ids:
        logger.error("No samples remaining after exclusion")
        raise typer.Exit(code=1)

    # ── Build dataset (no augmentation for evaluation) ──────────────────
    dataset = H5PYGenerator(
        file_path=str(h5_path),
        indexes=record_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )
    logger.info(f"H5PYGenerator created with {len(dataset)} samples")

    # ── Setup device & load model ───────────────────────────────────────
    device = get_device(device_index)
    logger.info(f"Using device: {device}")

    model = load_model(model_path, hyperparams_path, device)
    total_params = count_total_parameters(model)
    params_per_block = count_parameters_per_block(model)
    logger.info(f"Model loaded – {total_params:,} parameters")
    logger.info(f"Parameters per block: {params_per_block}")

    # ── Run extraction ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING EXTRACTION")
    logger.info("=" * 60)

    figures_dir = run_dir / "figures"

    start_time = perf_counter()
    results = extract_all_samples(
        model=model,
        record_ids=record_ids,
        dataset=dataset,
        config=analysis_config,
        device=device,
        figures_dir=figures_dir,
        show_progress=not no_progress,
    )
    total_time = perf_counter() - start_time

    # ── Save results CSV ────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    save_results_csv(results, run_dir / "results.csv")

    # ── Print summary ───────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    print_summary(results, total_time)

    # ── Generate figures for selected samples ───────────────────────
    generate_figures_for_selection(
        model=model,
        results=results,
        record_ids=record_ids,
        dataset=dataset,
        config=analysis_config,
        device=device,
        figures_dir=figures_dir,
        n_samples_requested=n_samples,
    )

    # ── Generate scatter plots ──────────────────────────────────────
    generate_scatter_plots(
        results=results,
        output_dir=figures_dir,
        config=analysis_config,
    )

    # ── Energy-stratified analysis ──────────────────────────────────
    generate_energy_stratified_analysis(
        results=results,
        output_dir=figures_dir,
        config=analysis_config,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Analysis complete! Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
