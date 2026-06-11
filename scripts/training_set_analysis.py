"""
Training Set Analysis: Tissue-Interface Beamlet Prevalence & Performance

Quantifies how many beamlets in the training set have their Bragg peak
at a density interface (tissue boundary) and shows that ADoTA performs
worse on those cases.

Steps implemented:
  1. Locate the Bragg peak per beamlet from the MC ground-truth dose.
  2. Extract CT density in a spherical neighbourhood around the BP.
  3. Compute a heterogeneity metric (σ_HU) for each beamlet.
  4. Classify beamlets as "interface" (σ_HU > threshold) vs "homogeneous".
  5. Run ADoTA inference and compute per-beamlet γ (2%/2mm) pass rate.
  6. Report prevalence + performance split; generate violin & scatter plots.
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import yaml
from scipy.stats import pearsonr
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import (
    DEFAULT_GAMMA_PARAMS,
    DEFAULT_SCALE,
    denormalize_energy,
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
from src.loaders.generator import H5PYGenerator
from src.loaders.utils import validate_inputs
from src.metrics.classic import (
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from src.figures.single_beam import publication_figure
from src.metrics.gamma_pass_rate import gamma_index_torch
from src.utils.dose_grid_utils import estimate_bragg_peak
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)

app = typer.Typer(help="Training-set tissue-interface analysis")


from src.schemas.analysis import BeamletResult
from src.schemas.configs import AnalysisConfig
from src.evaluation.cli import resolve_device
from src.evaluation.engine import InferenceContext, evaluate
from src.evaluation.outputs import CsvColumn, save_results_csv as save_csv
from src.evaluation.sources import H5Source

# ── Bragg-peak heterogeneity ────────────────────────────────────────────────


def _extract_sphere_voxels(
    ct_hu: np.ndarray,
    bp_idx: tuple,
    radius_mm: float,
    resolution: tuple,
) -> np.ndarray:
    """Extract HU voxels inside a spherical neighbourhood around a point.

    This is a shared helper used by :func:`compute_bp_sigma_hu`,
    :func:`compute_bp_tv`, and :func:`compute_bp_cv`.

    Args:
        ct_hu: 3-D CT volume in Hounsfield Units, shape (D, H, W).
        bp_idx: (depth, y, x) voxel indices of the centre.
        radius_mm: Radius of the sphere in millimetres.
        resolution: Voxel spacing (dz, dy, dx) in millimetres.

    Returns:
        1-D array of HU values inside the sphere.
    """
    D, H, W = ct_hu.shape
    dz, dy, dx = resolution

    # Compute voxel radii in each direction
    rz = int(np.ceil(radius_mm / dz))
    ry = int(np.ceil(radius_mm / dy))
    rx = int(np.ceil(radius_mm / dx))

    # Build index ranges (clamped to volume bounds)
    z0, y0, x0 = bp_idx
    z_lo, z_hi = max(0, z0 - rz), min(D, z0 + rz + 1)
    y_lo, y_hi = max(0, y0 - ry), min(H, y0 + ry + 1)
    x_lo, x_hi = max(0, x0 - rx), min(W, x0 + rx + 1)

    # Extract the bounding box
    patch = ct_hu[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]

    # Build a distance mask (ellipsoidal in voxel space → spherical in mm)
    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    dist_sq = ((zz - z0) * dz) ** 2 + ((yy - y0) * dy) ** 2 + ((xx - x0) * dx) ** 2
    sphere_mask = dist_sq <= radius_mm**2

    return patch[sphere_mask[: patch.shape[0], : patch.shape[1], : patch.shape[2]]]


def compute_bp_sigma_hu(
    ct_hu: np.ndarray,
    bp_idx: tuple,
    radius_mm: float,
    resolution: tuple,
) -> float:
    """Compute σ_HU in a spherical neighbourhood around the Bragg peak.

    Args:
        ct_hu: 3-D CT volume in Hounsfield Units, shape (D, H, W).
        bp_idx: (depth, y, x) voxel indices of the Bragg peak.
        radius_mm: Radius of the sphere in millimetres.
        resolution: Voxel spacing (dz, dy, dx) in millimetres.

    Returns:
        Standard deviation of HU values inside the sphere.
    """
    voxels = _extract_sphere_voxels(ct_hu, bp_idx, radius_mm, resolution)
    if len(voxels) < 2:
        return 0.0
    return float(np.std(voxels))


def compute_bp_tv(
    ct_hu: np.ndarray,
    bp_idx: tuple,
    radius_mm: float,
    resolution: tuple,
) -> float:
    """Compute Total Variation (TV) of HU values within a sphere at the BP.

    TV is the sum of absolute differences between consecutive voxel values
    (sorted by their flat index inside the sphere)::

        TV = Σ |v_{i+1} − v_i|

    A high TV indicates a rough / heterogeneous tissue composition within
    the Bragg-peak neighbourhood.

    Args:
        ct_hu: 3-D CT volume in Hounsfield Units, shape (D, H, W).
        bp_idx: (depth, y, x) voxel indices of the Bragg peak.
        radius_mm: Radius of the sphere in millimetres.
        resolution: Voxel spacing (dz, dy, dx) in millimetres.

    Returns:
        Total variation of HU values inside the sphere.
    """
    voxels = _extract_sphere_voxels(ct_hu, bp_idx, radius_mm, resolution)
    if len(voxels) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(voxels))))


def compute_bp_cv(
    ct_hu: np.ndarray,
    bp_idx: tuple,
    radius_mm: float,
    resolution: tuple,
) -> float:
    """Compute Coefficient of Variation (CV) of HU values within a sphere.

    CV is the ratio of the standard deviation to the absolute mean::

        CV = σ(v) / |μ(v)|

    A high CV indicates large relative spread in tissue density around
    the Bragg peak.

    Args:
        ct_hu: 3-D CT volume in Hounsfield Units, shape (D, H, W).
        bp_idx: (depth, y, x) voxel indices of the Bragg peak.
        radius_mm: Radius of the sphere in millimetres.
        resolution: Voxel spacing (dz, dy, dx) in millimetres.

    Returns:
        Coefficient of variation of HU values inside the sphere.
    """
    voxels = _extract_sphere_voxels(ct_hu, bp_idx, radius_mm, resolution)
    if len(voxels) < 2:
        return 0.0
    mu = np.mean(voxels)
    sigma = np.std(voxels)
    return float(sigma / np.abs(mu)) if np.abs(mu) > 1e-9 else 0.0


# ── Per-sample evaluation ───────────────────────────────────────────────────


def _make_per_sample_fn(config: AnalysisConfig, compute_gpr: bool):
    """Build the per-sample callback for the evaluation engine.

    Reproduces the original ``evaluate_single_sample``: zero-flux and
    energy-threshold skip guards (returning ``None``), Bragg-peak heterogeneity
    (sigma_HU / TV / CV) with the interface/homogeneous label, RMSE / MAPE / RDE,
    and an optional gamma pass rate.

    Args:
        config: Analysis configuration.
        compute_gpr: Whether to compute the gamma pass rate per beamlet.

    Returns:
        A callable mapping an ``InferenceContext`` to a ``BeamletResult``, or
        ``None`` to skip the beamlet.
    """
    scale = config.scale
    gamma_params = config.gamma_params

    def per_sample_fn(ctx: InferenceContext):
        energy_mev = denormalize_energy(ctx.energy.item(), scale)

        # ── Zero-flux guard ─────────────────────────────────────────────
        flux_channel = ctx.x[1]  # (D, H, W)
        if torch.abs(flux_channel).max().item() < 1e-9:
            logger.warning(
                f"Skipping sample {ctx.sample_id}: flux channel is all zeros"
            )
            return None

        # ── Energy threshold guard ──────────────────────────────────────
        if energy_mev > config.max_energy_mev:
            logger.debug(
                f"Skipping sample {ctx.sample_id}: "
                f"energy {energy_mev:.1f} MeV > threshold {config.max_energy_mev:.1f} MeV"
            )
            return None

        # ── De-normalise to physical units ──────────────────────────────
        y_np, y_pred_np = ctx.denorm(scale)

        ct_norm = ctx.x[0].cpu().numpy()  # (D, H, W) – CT channel
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])

        # ── Bragg-peak location (from GT dose) ──────────────────────────
        dose_3d = y_np.squeeze()  # (D, H, W)
        bp_idx = estimate_bragg_peak(dose_3d)  # (depth, y, x)

        # ── BP-neighbourhood heterogeneity ──────────────────────────────
        sigma_hu = compute_bp_sigma_hu(
            ct_hu, bp_idx, config.bp_radius_mm, config.resolution
        )
        tv = compute_bp_tv(ct_hu, bp_idx, config.bp_radius_mm, config.resolution)
        cv = compute_bp_cv(ct_hu, bp_idx, config.bp_radius_mm, config.resolution)
        label = (
            "interface" if sigma_hu > config.sigma_hu_threshold else "homogeneous"
        )

        # ── Dose-prediction metrics ─────────────────────────────────────
        rmse = calculate_rmse(to_gy(y_pred_np), to_gy(y_np))

        # MAPE at the 10% ground-truth-dose threshold. Canonical run_model.py
        # form: mask on the ground truth, calculate_pure_mape(predicted, reference).
        mask = y_np > 0.1 * np.max(y_np)
        mape = calculate_pure_mape(y_pred_np[mask], y_np[mask])

        rde = calculate_relative_dose_error(to_gy(y_pred_np), to_gy(y_np))

        # Gamma pass rate (optionally skipped for speed)
        if compute_gpr:
            scale_gpr = {"y_min": scale["min_ds"], "y_max": scale["max_ds"]}
            gpr_result = gamma_index_torch(
                ctx.y.unsqueeze(0),
                ctx.y_pred,
                scale=scale_gpr,
                gamma_params=gamma_params,
                resolution=config.resolution,
            )
            gpr = gpr_result[1][0] * 100
        else:
            gpr = float("nan")

        return BeamletResult(
            sample_id=ctx.sample_id,
            energy_mev=energy_mev,
            bp_idx=bp_idx,
            sigma_hu=sigma_hu,
            tv=tv,
            cv=cv,
            label=label,
            gpr=gpr,
            rmse=rmse,
            mape=mape,
            rde=rde,
            calc_time=ctx.calc_time,
        )

    return per_sample_fn


def evaluate_samples(
    model: DoTA3D_v3,
    record_ids: list,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
    device: torch.device,
    show_progress: bool = True,
    compute_gpr: bool = True,
) -> list:
    """Evaluate all beamlets via the shared evaluation engine.

    Args:
        model: The loaded DoTA model.
        record_ids: List of record IDs aligned with the dataset indices.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.
        device: Target device for computation.
        show_progress: Whether to show a progress bar.
        compute_gpr: Whether to compute gamma pass rate per beamlet.

    Returns:
        List of BeamletResult objects.
    """
    source = H5Source(dataset, record_ids)
    per_sample_fn = _make_per_sample_fn(config, compute_gpr)
    results = evaluate(
        model,
        source,
        device=device,
        per_sample_fn=per_sample_fn,
        show_progress=show_progress,
        desc="Analysing beamlets",
        postfix_fn=lambda r: {
            "sigma": f"{r.sigma_hu:.0f}",
            "label": r.label[:4],
            "gpr": f"{r.gpr:.1f}%",
        },
    )

    n_samples = len(source)
    n_skipped = n_samples - len(results)
    if n_skipped > 0:
        logger.info(
            f"Skipped {n_skipped}/{n_samples} samples "
            f"(zero flux or energy > {config.max_energy_mev:.0f} MeV) "
            f"({100.0 * n_skipped / n_samples:.1f}%)"
        )

    return results


# ── Results CSV ─────────────────────────────────────────────────────────────


def save_results_csv(results: list, output_path: Path) -> None:
    """Save per-beamlet results to a CSV file.

    Args:
        results: List of BeamletResult objects.
        output_path: Path to the output CSV file.
    """
    columns = [
        CsvColumn("sample_id", lambda r: r.sample_id),
        CsvColumn("energy_mev", lambda r: f"{r.energy_mev:.2f}"),
        CsvColumn("bp_depth", lambda r: r.bp_idx[0]),
        CsvColumn("bp_y", lambda r: r.bp_idx[1]),
        CsvColumn("bp_x", lambda r: r.bp_idx[2]),
        CsvColumn("sigma_hu", lambda r: f"{r.sigma_hu:.4f}", lambda r: r.sigma_hu, ".4f"),
        CsvColumn("tv", lambda r: f"{r.tv:.4f}", lambda r: r.tv, ".4f"),
        CsvColumn("cv", lambda r: f"{r.cv:.6f}", lambda r: r.cv, ".6f"),
        CsvColumn("label", lambda r: r.label),
        CsvColumn("gpr_pct", lambda r: f"{r.gpr:.2f}", lambda r: r.gpr, ".2f"),
        CsvColumn("rmse_gy", lambda r: f"{r.rmse:.9f}", lambda r: r.rmse, ".9f"),
        CsvColumn("mape_pct", lambda r: f"{r.mape:.4f}", lambda r: r.mape, ".4f"),
        CsvColumn("rde_pct", lambda r: f"{r.rde:.4f}", lambda r: r.rde, ".4f"),
        CsvColumn(
            "calc_time_s", lambda r: f"{r.calc_time:.4f}", lambda r: r.calc_time, ".4f"
        ),
    ]
    save_csv(
        results,
        output_path,
        columns,
        sort_key=lambda r: r.energy_mev,
        label_column="sample_id",
        logger=logger,
    )


# ── Prevalence report ──────────────────────────────────────────────────────


def report_prevalence(results: list, config: AnalysisConfig) -> None:
    """Log prevalence statistics for interface vs homogeneous beamlets.

    Args:
        results: List of BeamletResult objects.
        config: Analysis configuration (for threshold info).
    """
    n_total = len(results)
    n_interface = sum(1 for r in results if r.label == "interface")
    n_homogeneous = n_total - n_interface
    pct_interface = 100.0 * n_interface / n_total if n_total > 0 else 0.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("PREVALENCE REPORT")
    logger.info("=" * 60)
    logger.info(
        f"Threshold: σ_HU > {config.sigma_hu_threshold:.1f} HU  "
        f"(sphere radius = {config.bp_radius_mm:.1f} mm)"
    )
    logger.info(f"Total beamlets:       {n_total}")
    logger.info(
        f"  Homogeneous:        {n_homogeneous}  ({100.0 - pct_interface:.1f}%)"
    )
    logger.info(f"  Interface:          {n_interface}  ({pct_interface:.1f}%)")
    logger.info("=" * 60)

    # Per-group metric summaries
    for group_label in ("homogeneous", "interface"):
        group = [r for r in results if r.label == group_label]
        if not group:
            logger.info(f"  [{group_label}] – no samples")
            continue
        gprs = [r.gpr for r in group]
        mapes = [r.mape for r in group]
        rmses = [r.rmse for r in group]
        rdes = [r.rde for r in group]
        tvs = [r.tv for r in group]
        cvs = [r.cv for r in group]
        logger.info(
            f"  [{group_label}] N={len(group)}: "
            f"GPR={np.mean(gprs):.2f}\u00b1{np.std(gprs):.2f}%, "
            f"MAPE={np.mean(mapes):.2f}\u00b1{np.std(mapes):.2f}%, "
            f"RMSE={np.mean(rmses):.9f}\u00b1{np.std(rmses):.9f} Gy, "
            f"RDE={np.mean(rdes):.4f}\u00b1{np.std(rdes):.4f}%"
        )
        logger.info(
            f"           TV={np.mean(tvs):.2f}\u00b1{np.std(tvs):.2f} HU, "
            f"CV={np.mean(cvs):.4f}\u00b1{np.std(cvs):.4f}"
        )


# ── Figures ─────────────────────────────────────────────────────────────────


def generate_publication_figures(
    model: DoTA3D_v3,
    results: list,
    record_ids: list,
    dataset: H5PYGenerator,
    output_dir: Path,
    config: AnalysisConfig,
    device: torch.device,
    compute_gpr: bool = True,
    n_worst: Optional[int] = None,
) -> None:
    """Generate publication figures for selected cases.

    **Default mode** (``n_worst`` is ``None``): for each tissue category
    (homogeneous and interface) the function selects three representative
    beamlets based on their **relative dose error** (RDE): best (lowest),
    worst (highest), and closest to the group mean.

    **N-worst mode** (``n_worst`` is set): for each tissue category the
    function selects the *N* beamlets with the highest RDE (worst
    predictions).  Best and closest-to-mean figures are **not** generated.

    For each selected sample the data is reloaded from the HDF5 dataset,
    inference is re-run, and a full publication figure is produced via
    ``publication_figure``.

    Args:
        model: The loaded DoTA model.
        results: List of BeamletResult objects.
        record_ids: Ordered list of record IDs matching the dataset.
        dataset: The H5PYGenerator dataset.
        output_dir: Directory where figures will be saved.
        config: AnalysisConfig (scale, gamma_params, resolution).
        device: Target device for computation.
        compute_gpr: Whether GPR was computed (used for figure labelling).
        n_worst: If set, generate figures for the N worst beamlets per
            group (ranked by RDE) instead of best/worst/mean.
    """
    scale = config.scale
    gamma_params = config.gamma_params
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build sample_id → dataset index mapping
    id_to_idx: dict[str, int] = {rid: i for i, rid in enumerate(record_ids)}

    for group_label in ("homogeneous", "interface"):
        group = [r for r in results if r.label == group_label]
        if not group:
            logger.warning(
                f"No {group_label} samples – skipping publication figures "
                f"for this group"
            )
            continue

        tag = group_label.capitalize()

        if n_worst is not None:
            # ── N-worst mode: select N samples with highest RDE ─────
            sorted_by_rde = sorted(group, key=lambda r: r.rde, reverse=True)
            n_pick = min(n_worst, len(sorted_by_rde))
            selected = sorted_by_rde[:n_pick]

            cases: list[tuple[str, BeamletResult]] = [
                (f"Worst_{rank+1:03d}", r) for rank, r in enumerate(selected)
            ]

            logger.info(
                f"Publication figures for {tag} group – " f"{n_pick} worst by RDE:"
            )
            for rank, r in enumerate(selected):
                logger.info(f"  #{rank+1}: {r.sample_id}  RDE = {r.rde:.4f}%")
        else:
            # ── Default mode: best / worst / closest-to-mean ────────
            rdes = [r.rde for r in group]
            mean_rde = float(np.mean(rdes))

            best_result = min(group, key=lambda r: r.rde)
            worst_result = max(group, key=lambda r: r.rde)
            closest_result = min(group, key=lambda r: abs(r.rde - mean_rde))

            cases = [
                ("Best", best_result),
                ("Worst", worst_result),
                ("Closest_to_Mean", closest_result),
            ]

            logger.info(f"Publication figures for {tag} group (ranked by RDE):")
            logger.info(
                f"  Best RDE:  {best_result.sample_id}  "
                f"RDE = {best_result.rde:.4f}%"
            )
            logger.info(
                f"  Worst RDE: {worst_result.sample_id}  "
                f"RDE = {worst_result.rde:.4f}%"
            )
            logger.info(
                f"  Mean RDE:  {closest_result.sample_id}  "
                f"RDE = {closest_result.rde:.4f}%  "
                f"(group mean = {mean_rde:.4f}%)"
            )

        for desc, result in cases:
            sample_idx = id_to_idx.get(result.sample_id)
            if sample_idx is None:
                logger.warning(
                    f"Sample {result.sample_id} not found in dataset – skipping"
                )
                continue

            logger.info(
                f"Generating {tag}/{desc} figure for {result.sample_id} "
                f"(E = {result.energy_mev:.1f} MeV, σ_HU = {result.sigma_hu:.1f} HU)"
            )

            # Reload data from dataset
            x, energy, y = dataset[sample_idx]
            x = x.to(device)
            energy = energy.to(device)
            y = y.to(device)

            # Run inference
            with torch.no_grad():
                y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]

            # De-normalise to physical units
            # Keep raw 2-channel (CT+flux) for beamlet_shape display
            x_input = x.squeeze().cpu().numpy()  # (2, D, H, W)
            gt = inverse_minmax(
                y.squeeze().cpu().numpy(),
                scale["min_ds"],
                scale["max_ds"],
            )
            pred = inverse_minmax(
                y_pred.squeeze().cpu().numpy(),
                scale["min_ds"],
                scale["max_ds"],
            )

            # Metrics for annotation
            rmse = calculate_rmse(to_gy(pred), to_gy(gt))
            mask = gt > 0.1 * np.max(gt)
            mape = calculate_pure_mape(pred[mask], gt[mask])

            if compute_gpr:
                scale_gpr = {"y_min": scale["min_ds"], "y_max": scale["max_ds"]}
                gpr_result = gamma_index_torch(
                    y.unsqueeze(0),
                    y_pred,
                    scale=scale_gpr,
                    gamma_params=gamma_params,
                    resolution=config.resolution,
                )
                gpr = gpr_result[1][0] * 100
            else:
                gpr = result.gpr  # NaN when GPR was skipped

            logger.info(
                f"  {desc} – RMSE: {rmse:.6f}, MAPE: {mape:.2f}%, "
                f"RDE: {result.rde:.4f}%"
                + (f", GPR: {gpr:.2f}%" if not np.isnan(gpr) else "")
            )

            figure_path = output_dir / f"{tag}_{desc}_E{result.energy_mev:.2f}MeV.svg"

            publication_figure(
                x_input,
                result.energy_mev,
                gt,
                pred,
                str(figure_path),
                rmse,
                mape,
                gpr if not np.isnan(gpr) else 0.0,
                gamma_params=gamma_params,
                beamlet_shape=True,
            )

    logger.info("Publication figures generation complete")


def generate_violin_plots(
    results: list,
    output_dir: Path,
    config: AnalysisConfig,
    compute_gpr: bool = True,
) -> None:
    """Generate violin + box plots of per-beamlet metrics split by group.

    Produces a 1×N figure where N depends on whether GPR was computed:
        With GPR:    (a) GPR  (b) MAPE  (c) RDE  (d) RMSE
        Without GPR: (a) MAPE  (b) RDE  (c) RMSE

    Args:
        results: List of BeamletResult objects.
        output_dir: Directory where the figure will be saved.
        config: Analysis configuration (for labelling).
        compute_gpr: Whether GPR values are available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    homo = [r for r in results if r.label == "homogeneous"]
    intf = [r for r in results if r.label == "interface"]

    if not homo or not intf:
        logger.warning("One group is empty – skipping violin plots")
        return

    # Build metric list dynamically based on available data
    metrics: list[tuple[str, str, str]] = []
    letter = ord("a")
    if compute_gpr:
        gpr_label = (
            f"GPR ({config.gamma_params['dose_percent_threshold']}%, "
            f"{config.gamma_params['distance_mm_threshold']}mm, "
            f"{config.gamma_params['lower_percent_dose_cutoff']}%) [%]"
        )
        metrics.append(("gpr", gpr_label, chr(letter)))
        letter += 1
    metrics.append(("mape", "MAPE [%]", chr(letter)))
    letter += 1
    metrics.append(("rde", "RDE [%]", chr(letter)))
    letter += 1
    metrics.append(("rmse", "RMSE [Gy]", chr(letter)))

    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 6), dpi=300)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"Performance by tissue-interface category  "
        f"(σ_HU threshold = {config.sigma_hu_threshold:.0f} HU, "
        f"r = {config.bp_radius_mm:.0f} mm, "
        f"N_homo = {len(homo)}, N_intf = {len(intf)})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    colors = ["#4C72B0", "#DD8452"]  # blue=homogeneous, orange=interface

    for ax, (attr, ylabel, letter) in zip(axes, metrics):
        data_homo = [getattr(r, attr) for r in homo]
        data_intf = [getattr(r, attr) for r in intf]
        data = [data_homo, data_intf]

        parts = ax.violinplot(
            data,
            positions=[0, 1],
            showmedians=False,
            showextrema=False,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(0.6)

        # Overlay box plots for quartiles
        bp = ax.boxplot(
            data,
            positions=[0, 1],
            widths=0.15,
            patch_artist=True,
            showfliers=False,
        )
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.8)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color("black")
                line.set_linewidth(1.0)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Homogeneous", "Interface"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter}) {ylabel.split('[')[0].strip()}")
        ax.grid(axis="y", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(
        output_dir / "interface_vs_homogeneous_violin.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        f"Violin plot saved to " f"{output_dir / 'interface_vs_homogeneous_violin.png'}"
    )


def generate_scatter_plot(
    results: list,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Scatter plot: σ_HU (x) vs per-beamlet GPR (y), coloured by group.

    Args:
        results: List of BeamletResult objects.
        output_dir: Directory where the figure will be saved.
        config: Analysis configuration (for labelling).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) < 3:
        logger.info("Skipping scatter plot – fewer than 3 samples")
        return

    sigma_arr = np.array([r.sigma_hu for r in results])
    gpr_arr = np.array([r.gpr for r in results])
    labels = np.array([r.label for r in results])

    homo_mask = labels == "homogeneous"
    intf_mask = labels == "interface"

    r_val, p_val = pearsonr(sigma_arr, gpr_arr)

    gpr_label = (
        f"GPR ({config.gamma_params['dose_percent_threshold']}%, "
        f"{config.gamma_params['distance_mm_threshold']}mm, "
        f"{config.gamma_params['lower_percent_dose_cutoff']}%) [%]"
    )

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    ax.scatter(
        sigma_arr[homo_mask],
        gpr_arr[homo_mask],
        s=12,
        alpha=0.5,
        color="#4C72B0",
        edgecolors="k",
        linewidths=0.2,
        label=f"Homogeneous (N={homo_mask.sum()})",
    )
    ax.scatter(
        sigma_arr[intf_mask],
        gpr_arr[intf_mask],
        s=12,
        alpha=0.5,
        color="#DD8452",
        edgecolors="k",
        linewidths=0.2,
        label=f"Interface (N={intf_mask.sum()})",
    )

    # Linear fit (all data)
    z = np.polyfit(sigma_arr, gpr_arr, 1)
    x_fit = np.linspace(sigma_arr.min(), sigma_arr.max(), 200)
    ax.plot(
        x_fit,
        np.polyval(z, x_fit),
        "r--",
        linewidth=1.2,
        label=f"fit  (r = {r_val:.3f}, p = {p_val:.2e})",
    )

    # Threshold line
    ax.axvline(
        x=config.sigma_hu_threshold,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"threshold σ_HU = {config.sigma_hu_threshold:.0f} HU",
    )

    ax.set_xlabel("σ_HU at Bragg peak [HU]")
    ax.set_ylabel(gpr_label)
    ax.set_title(
        f"Tissue heterogeneity at Bragg peak vs Gamma pass rate  (N = {len(results)})"
    )
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(
        output_dir / "sigma_hu_vs_gpr_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"Scatter plot saved to {output_dir / 'sigma_hu_vs_gpr_scatter.png'}")

    logger.info(f"Pearson r(σ_HU, GPR) = {r_val:.4f}  (p = {p_val:.4e})")


def generate_tv_vs_rde_scatter(
    results: list,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Scatter plot: Total Variation (x) vs Relative Dose Error (y).

    Homogeneous beamlets are shown in blue, interface beamlets in red.

    Args:
        results: List of BeamletResult objects.
        output_dir: Directory where the figure will be saved.
        config: Analysis configuration (for labelling).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) < 3:
        logger.info("Skipping TV vs RDE scatter plot – fewer than 3 samples")
        return

    tv_arr = np.array([r.tv for r in results])
    rde_arr = np.array([r.rde for r in results])
    labels = np.array([r.label for r in results])

    homo_mask = labels == "homogeneous"
    intf_mask = labels == "interface"

    r_val, p_val = pearsonr(tv_arr, rde_arr)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    ax.scatter(
        tv_arr[homo_mask],
        rde_arr[homo_mask],
        s=12,
        alpha=0.5,
        color="#4C72B0",
        edgecolors="k",
        linewidths=0.2,
        label=f"Homogeneous (N={homo_mask.sum()})",
    )
    ax.scatter(
        tv_arr[intf_mask],
        rde_arr[intf_mask],
        s=12,
        alpha=0.5,
        color="#C44E52",
        edgecolors="k",
        linewidths=0.2,
        label=f"Interface (N={intf_mask.sum()})",
    )

    # Linear fit (all data)
    z = np.polyfit(tv_arr, rde_arr, 1)
    x_fit = np.linspace(tv_arr.min(), tv_arr.max(), 200)
    ax.plot(
        x_fit,
        np.polyval(z, x_fit),
        "r--",
        linewidth=1.2,
        label=f"fit  (r = {r_val:.3f}, p = {p_val:.2e})",
    )

    ax.set_xlabel(
        f"Total Variation of HU at Bragg peak "
        f"(r = {config.bp_radius_mm:.0f} mm) [HU]"
    )
    ax.set_ylabel("Relative Dose Error [%]")
    ax.set_title(f"Total Variation vs Relative Dose Error  (N = {len(results)})")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(
        output_dir / "tv_vs_rde_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        f"TV vs RDE scatter plot saved to {output_dir / 'tv_vs_rde_scatter.png'}"
    )
    logger.info(f"Pearson r(TV, RDE) = {r_val:.4f}  (p = {p_val:.4e})")


def generate_tv_vs_gpr_scatter(
    results: list,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Scatter plot: Total Variation (x) vs Gamma Pass Rate (y).

    Homogeneous beamlets are shown in blue, interface beamlets in red.

    Args:
        results: List of BeamletResult objects.
        output_dir: Directory where the figure will be saved.
        config: Analysis configuration (for labelling).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) < 3:
        logger.info("Skipping TV vs GPR scatter plot \u2013 fewer than 3 samples")
        return

    tv_arr = np.array([r.tv for r in results])
    gpr_arr = np.array([r.gpr for r in results])
    labels = np.array([r.label for r in results])

    homo_mask = labels == "homogeneous"
    intf_mask = labels == "interface"

    r_val, p_val = pearsonr(tv_arr, gpr_arr)

    gpr_label = (
        f"GPR ({config.gamma_params['dose_percent_threshold']}%, "
        f"{config.gamma_params['distance_mm_threshold']}mm, "
        f"{config.gamma_params['lower_percent_dose_cutoff']}%) [%]"
    )

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    ax.scatter(
        tv_arr[homo_mask],
        gpr_arr[homo_mask],
        s=12,
        alpha=0.5,
        color="#4C72B0",
        edgecolors="k",
        linewidths=0.2,
        label=f"Homogeneous (N={homo_mask.sum()})",
    )
    ax.scatter(
        tv_arr[intf_mask],
        gpr_arr[intf_mask],
        s=12,
        alpha=0.5,
        color="#C44E52",
        edgecolors="k",
        linewidths=0.2,
        label=f"Interface (N={intf_mask.sum()})",
    )

    # Linear fit (all data)
    z = np.polyfit(tv_arr, gpr_arr, 1)
    x_fit = np.linspace(tv_arr.min(), tv_arr.max(), 200)
    ax.plot(
        x_fit,
        np.polyval(z, x_fit),
        "r--",
        linewidth=1.2,
        label=f"fit  (r = {r_val:.3f}, p = {p_val:.2e})",
    )

    ax.set_xlabel(
        f"Total Variation of HU at Bragg peak "
        f"(r = {config.bp_radius_mm:.0f} mm) [HU]"
    )
    ax.set_ylabel(gpr_label)
    ax.set_title(f"Total Variation vs Gamma Pass Rate  (N = {len(results)})")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(
        output_dir / "tv_vs_gpr_scatter.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        f"TV vs GPR scatter plot saved to {output_dir / 'tv_vs_gpr_scatter.png'}"
    )
    logger.info(f"Pearson r(TV, GPR) = {r_val:.4f}  (p = {p_val:.4e})")


def generate_sigma_hu_histogram(
    results: list,
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Histogram of σ_HU values across all beamlets, with threshold annotated.

    Args:
        results: List of BeamletResult objects.
        output_dir: Directory where the figure will be saved.
        config: Analysis configuration.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    sigma_arr = np.array([r.sigma_hu for r in results])

    n_interface = np.sum(sigma_arr > config.sigma_hu_threshold)
    n_total = len(results)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.hist(
        sigma_arr,
        bins=80,
        color="steelblue",
        edgecolor="k",
        linewidth=0.4,
        alpha=0.85,
    )
    ax.axvline(
        x=config.sigma_hu_threshold,
        color="red",
        linewidth=1.5,
        linestyle="--",
        label=(
            f"threshold = {config.sigma_hu_threshold:.0f} HU  "
            f"({n_interface}/{n_total} = {100.0 * n_interface / n_total:.1f}% interface)"
        ),
    )
    ax.set_xlabel("σ_HU at Bragg peak [HU]")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of BP-neighbourhood heterogeneity  "
        f"(r = {config.bp_radius_mm:.0f} mm, N = {n_total})"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(
        output_dir / "sigma_hu_histogram.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"σ_HU histogram saved to {output_dir / 'sigma_hu_histogram.png'}")


# ── Summary ─────────────────────────────────────────────────────────────────


def print_summary(results: list, total_time: float) -> None:
    """Print evaluation summary to the logger.

    Args:
        results: List of BeamletResult objects.
        total_time: Total evaluation time in seconds.
    """
    calc_times = [r.calc_time for r in results]
    gprs = [r.gpr for r in results]
    mapes = [r.mape for r in results]
    rmses = [r.rmse for r in results]
    rdes = [r.rde for r in results]
    tvs = [r.tv for r in results]
    cvs = [r.cv for r in results]

    logger.info(f"Total elapsed time: {total_time:.2f}s")
    logger.info(f"Average time per sample: {np.mean(calc_times):.4f}s")

    logger.info(
        f"GPR   \u2013 mean: {np.mean(gprs):.2f}%, " f"std: {np.std(gprs):.2f}%"
    )
    logger.info(
        f"RMSE  \u2013 mean: {np.mean(rmses):.9f} Gy, " f"std: {np.std(rmses):.9f} Gy"
    )
    logger.info(
        f"MAPE  \u2013 mean: {np.mean(mapes):.4f}%, " f"std: {np.std(mapes):.4f}%"
    )
    logger.info(
        f"RDE   \u2013 mean: {np.mean(rdes):.4f}%, " f"std: {np.std(rdes):.4f}%"
    )
    logger.info(
        f"TV    \u2013 mean: {np.mean(tvs):.2f} HU, " f"std: {np.std(tvs):.2f} HU"
    )
    logger.info(f"CV    \u2013 mean: {np.mean(cvs):.4f}, " f"std: {np.std(cvs):.4f}")

    worst_idx = np.argmin(gprs)
    best_idx = np.argmax(gprs)
    worst = results[worst_idx]
    best = results[best_idx]

    logger.info(
        f"Best case (highest GPR): {best.sample_id}, "
        f"E = {best.energy_mev:.2f} MeV, "
        f"GPR = {best.gpr:.2f}%, σ_HU = {best.sigma_hu:.1f} HU ({best.label})"
    )
    logger.info(
        f"Worst case (lowest GPR): {worst.sample_id}, "
        f"E = {worst.energy_mev:.2f} MeV, "
        f"GPR = {worst.gpr:.2f}%, σ_HU = {worst.sigma_hu:.1f} HU ({worst.label})"
    )


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
    sigma_hu_threshold: Annotated[
        Optional[float],
        typer.Option(help="σ_HU threshold to classify interface beamlets [HU]"),
    ] = None,
    bp_radius_mm: Annotated[
        Optional[float],
        typer.Option(help="Radius of spherical neighbourhood at BP [mm]"),
    ] = None,
    max_energy_mev: Annotated[
        Optional[float],
        typer.Option(
            help="Skip beamlets with initial energy above this threshold [MeV]"
        ),
    ] = None,
    n_worst_figures: Annotated[
        Optional[int],
        typer.Option(
            help="Generate publication figures for the N worst samples per group "
            "(by RDE). When set, best and closest-to-mean figures are skipped."
        ),
    ] = None,
    n_samples: Annotated[
        Optional[int],
        typer.Option(help="Limit evaluation to the first N samples (default: all)"),
    ] = None,
    no_gpr: Annotated[
        Optional[bool],
        typer.Option(help="Skip gamma pass rate calculation (faster experiments)"),
    ] = None,
    no_progress: Annotated[
        Optional[bool], typer.Option(help="Disable progress bar")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose output")
    ] = None,
) -> None:
    """Analyse tissue-interface beamlet prevalence and ADoTA performance.

    Iterates over every beamlet in the HDF5 training set, computes the
    CT heterogeneity (σ_HU) at the Bragg peak, classifies each beamlet
    as "interface" or "homogeneous", runs ADoTA inference, and reports
    prevalence + performance statistics with figures.

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
    sigma_hu_threshold = (
        sigma_hu_threshold
        if sigma_hu_threshold is not None
        else yaml_config.get("sigma_hu_threshold", 150.0)
    )
    bp_radius_mm = (
        bp_radius_mm
        if bp_radius_mm is not None
        else yaml_config.get("bp_radius_mm", 5.0)
    )
    max_energy_mev = (
        max_energy_mev
        if max_energy_mev is not None
        else yaml_config.get("max_energy_mev", 250.0)
    )
    n_worst_figures = (
        n_worst_figures
        if n_worst_figures is not None
        else yaml_config.get("n_worst_figures")
    )
    n_samples = n_samples if n_samples is not None else yaml_config.get("n_samples")
    no_gpr = no_gpr if no_gpr is not None else yaml_config.get("no_gpr", False)
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
    logger.info(f"σ_HU threshold: {sigma_hu_threshold} HU")
    logger.info(f"BP radius: {bp_radius_mm} mm")
    logger.info(f"Max energy: {max_energy_mev} MeV")
    logger.info(f"N samples: {n_samples if n_samples is not None else 'all'}")
    logger.info(
        f"N worst figures: "
        f"{n_worst_figures if n_worst_figures is not None else 'off (best/worst/mean)'}"
    )
    logger.info(f"Compute GPR: {'no (skipped)' if no_gpr else 'yes'}")
    logger.info("=" * 60)

    # ── Setup analysis configuration ────────────────────────────────────
    analysis_config = AnalysisConfig(
        sigma_hu_threshold=sigma_hu_threshold,
        bp_radius_mm=bp_radius_mm,
        max_energy_mev=max_energy_mev,
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

    # ── Discover samples in HDF5 ───────────────────────────────────────
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

    # ── Setup device & load model ──────────────────────────────────────
    device = resolve_device(device_index)
    logger.info(f"Using device: {device}")

    model = load_model(model_path, hyperparams_path, device)

    # ── Run evaluation ─────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING ANALYSIS")
    logger.info("=" * 60)

    start_time = perf_counter()
    results = evaluate_samples(
        model=model,
        record_ids=record_ids,
        dataset=dataset,
        config=analysis_config,
        device=device,
        show_progress=not no_progress,
        compute_gpr=not no_gpr,
    )
    total_time = perf_counter() - start_time

    # ── Save results CSV ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    save_results_csv(results, run_dir / "results.csv")

    # ── Prevalence report ──────────────────────────────────────────────
    report_prevalence(results, analysis_config)

    # ── Print summary ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    print_summary(results, total_time)

    # ── Generate figures ───────────────────────────────────────────────
    figures_dir = run_dir / "figures"

    generate_sigma_hu_histogram(
        results=results,
        output_dir=figures_dir,
        config=analysis_config,
    )

    generate_violin_plots(
        results=results,
        output_dir=figures_dir,
        config=analysis_config,
        compute_gpr=not no_gpr,
    )

    generate_tv_vs_rde_scatter(
        results=results,
        output_dir=figures_dir,
        config=analysis_config,
    )

    if no_gpr:
        logger.info(
            "GPR was skipped – scatter plots (σ_HU vs GPR, TV vs GPR) not generated"
        )
    else:
        generate_scatter_plot(
            results=results,
            output_dir=figures_dir,
            config=analysis_config,
        )
        generate_tv_vs_gpr_scatter(
            results=results,
            output_dir=figures_dir,
            config=analysis_config,
        )

    generate_publication_figures(
        model=model,
        results=results,
        record_ids=record_ids,
        dataset=dataset,
        output_dir=figures_dir / "publication",
        config=analysis_config,
        device=device,
        compute_gpr=not no_gpr,
        n_worst=n_worst_figures,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Analysis complete! Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
