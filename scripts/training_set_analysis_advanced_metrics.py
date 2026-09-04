"""
Training Set Analysis – Advanced Metrics

Iterates over all samples in the training set, extracts per-beamlet
data (CT, flux / fast beamlet-shape projection, initial energy, and
ground-truth dose) and logs basic statistics.

No model inference or figure generation is performed — this script is
the skeleton for future advanced-metric experiments.
"""

import logging
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from src.acquisition.features import analyse_density_regions, compute_advanced_metrics

# Add project root to path
from src.adota.config import (
    DEFAULT_GAMMA_PARAMS,
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
from src.evaluation.cli import resolve_device
from src.evaluation.engine import InferenceContext, evaluate
from src.evaluation.sources import H5Source
from src.figures.advanced_metrics import (
    generate_beam_angle_figures,
    generate_figures_for_selection,
)
from src.figures.ct_visualizations import smooth_ct
from src.loaders.generator import H5PYGenerator
from src.loaders.utils import validate_inputs
from src.metrics.classic import calculate_relative_dose_error
from src.metrics.gamma_pass_rate import gamma_index_torch
from src.metrics.range_metrics import (
    compute_range_metrics,
    integrated_depth_dose,
    range_metric_deltas,
)
from src.metrics.sobel import (
    compute_sobel_metrics,
    compute_sobel_metrics_sphere,
    compute_structure_tensor_metrics_sphere,
)
from src.processing.interface_severity import interface_severity
from src.processing.pflugfelder_hi import pflugfelder_hi
from src.schemas.configs import AdvancedAnalysisConfig as AnalysisConfig
from src.schemas.results import SampleRecord
from src.utils.dose_grid_utils import estimate_bp_range
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

PROJECT_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)
app = typer.Typer(help="Training-set advanced-metrics analysis")

# ── Helpers ─────────────────────────────────────────────────────────────────


# ── Per-sample extraction ─────────────────────────────────────────────────────


def denorm_ctx(
    ctx: InferenceContext,
    config: AnalysisConfig,
) -> Optional[tuple[SampleRecord, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]]:
    """De-normalize one inference context into physical-unit arrays + a base record.

    Consumes the shared evaluation engine's :class:`InferenceContext` (which
    already carries the loaded input, ground truth and model prediction), so the
    dataset is read once per sample instead of twice.

    Args:
        ctx: The per-sample inference context from :func:`evaluate`.
        config: Analysis configuration.

    Returns:
        Tuple ``(SampleRecord, ct_hu, flux, energy_mev, gt_dose, pred_dose)``:

        - *ct_hu* – 3-D CT volume in Hounsfield Units ``(D, H, W)``.
        - *flux* – fast beamlet-shape projection ``(D, H, W)``.
        - *energy_mev* – initial beam energy in MeV.
        - *gt_dose* – ground-truth dose grid ``(D, H, W)`` in physical units.
        - *pred_dose* – ADoTA predicted dose ``(D, H, W)`` in physical units.

        Returns ``None`` if the sample is skipped (zero flux or energy above
        threshold).
    """
    scale = config.scale
    record_id = ctx.sample_id

    energy_mev = denormalize_energy(ctx.energy.item(), scale)

    # ── Zero-flux guard ─────────────────────────────────────────────────
    flux_channel = ctx.x[1]  # (D, H, W)
    if torch.abs(flux_channel).max().item() < 1e-9:
        logger.warning(f"Skipping sample {record_id}: flux channel is all zeros")
        return None

    # ── Energy threshold guard ──────────────────────────────────────────
    if energy_mev > config.max_energy_mev:
        logger.debug(
            f"Skipping sample {record_id}: "
            f"energy {energy_mev:.1f} MeV > threshold {config.max_energy_mev:.1f} MeV"
        )
        return None

    # ── De-normalise to physical units ──────────────────────────────────
    ct_norm = ctx.x[0].cpu().numpy()  # (D, H, W) – CT channel
    ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])

    flux_np = ctx.x[1].cpu().numpy()  # (D, H, W) – flux / fast beamlet shape

    # denorm() returns (1, 1, D, H, W) arrays in physical dose for GT and pred.
    y_np, y_pred_np = ctx.denorm(scale)
    gt_dose = np.squeeze(y_np)
    pred_dose = np.squeeze(y_pred_np)

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
        extract_time=ctx.calc_time,
    )
    return record, ct_hu, flux_np, energy_mev, gt_dose, pred_dose


# ── Density region analysis ─────────────────────────────────────────────────


def _make_per_sample_fn(
    config: AnalysisConfig,
    device: torch.device,
    idd_store: Optional[dict] = None,
    angle_map: Optional[dict] = None,
):
    """Build the per-beamlet metric callback for the shared evaluation engine.

    The returned function maps one :class:`InferenceContext` to a fully
    populated :class:`SampleRecord` (heterogeneity metrics, model-accuracy
    metrics, and range-fidelity metrics), or ``None`` to skip the beamlet. When
    ``idd_store`` is provided, the MC and predicted IDD curves are cached under
    the sample id for the worst-outlier overlay diagnostics. ``angle_map`` maps
    ``sample_id -> (beamlet_angle_0, beamlet_angle_1, gantry_angle)`` in degrees.
    """
    dz_mm = float(config.resolution[0])

    def per_sample_fn(ctx: InferenceContext) -> Optional[SampleRecord]:
        out = denorm_ctx(ctx, config)
        if out is None:
            return None
        record, ct_hu, flux, energy_mev, gt_dose, pred_dose = out

        # ── Beamlet direction (from the dataset attrs, if available) ────────
        if angle_map is not None:
            angles = angle_map.get(record.sample_id)
            if angles is not None:
                record.beamlet_angle_0_deg = float(angles[0])
                record.beamlet_angle_1_deg = float(angles[1])
                record.gantry_angle_deg = float(angles[2])

        # ── Bragg-peak range estimation ─────────────────────────────
        z_min, z_max = estimate_bp_range(
            ct_hu,
            gt_dose,
            proximal_fraction=config.proximal_fraction,
            fall_fraction=config.fall_fraction,
        )
        record.bp_range_min_mm = z_min * dz_mm
        record.bp_range_max_mm = z_max * dz_mm

        # Depth of maximum IDD gradient
        idd = gt_dose.sum(axis=(1, 2))
        idd_grad = np.gradient(idd)
        record.max_grad_depth_mm = float(int(np.argmax(idd_grad))) * dz_mm

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

        # ── Structure-tensor Sobel metrics (DW and TH) ──────────────────
        st = compute_structure_tensor_metrics_sphere(
            ct_for_sobel,
            gt_dose,
            radius_mm=config.sphere_radius_mm,
            resolution=config.resolution,
        )
        record.sobel_dw_mean = st["sobel_dw_mean"]
        record.sobel_dw_anisotropy = st["sobel_dw_anisotropy"]
        record.sobel_dw_beam_angle = st["sobel_dw_beam_angle"]
        record.sobel_dw_edge_energy = st["sobel_dw_edge_energy"]
        record.sobel_th_mean = st["sobel_th_mean"]
        record.sobel_th_anisotropy = st["sobel_th_anisotropy"]
        record.sobel_th_beam_angle = st["sobel_th_beam_angle"]
        record.sobel_th_edge_energy = st["sobel_th_edge_energy"]

        # Orientation-weighted edge energy: emphasize edges parallel to the beam
        # (theta ~ 90 deg), the lateral interfaces that dilute the distal range.
        theta_rad = np.radians(st["sobel_dw_beam_angle"])
        record.lateral_edge_energy = float(
            st["sobel_dw_edge_energy"] * np.sin(theta_rad) ** 2
        )

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

        # ── Model-accuracy metrics (single inference, from the engine) ──
        # RDE from the de-normalised physical-unit doses.
        record.rde = calculate_relative_dose_error(to_gy(pred_dose), to_gy(gt_dose))

        # GPR from the normalised tensors still on-device (matches run_model.py).
        scale = config.scale
        gpr_result = gamma_index_torch(
            ctx.y.unsqueeze(0),
            ctx.y_pred,
            scale={"y_min": scale["min_ds"], "y_max": scale["max_ds"]},
            gamma_params=config.gamma_params,
            resolution=config.resolution,
        )
        record.gpr = gpr_result[1][0] * 100

        # ── Range-fidelity metrics (MC vs ADoTA IDD) ────────────────
        mc_idd = integrated_depth_dose(gt_dose)
        pred_idd = integrated_depth_dose(pred_dose)
        mc_rm = compute_range_metrics(
            mc_idd, dz_mm, oversample=config.range_oversample
        )
        pred_rm = compute_range_metrics(
            pred_idd, dz_mm, oversample=config.range_oversample
        )
        record.mc_r100_mm = mc_rm.r100_mm
        record.mc_r80_mm = mc_rm.r80_mm
        record.mc_r20_mm = mc_rm.r20_mm
        record.mc_dfw_mm = mc_rm.dfw_mm
        # Each delta is recorded independently and only when both its endpoints
        # are defined, so the robust Bragg-peak error (ΔR100) is kept even for
        # plateau/air-channel beamlets whose distal R80/R20 are undefined (NaN).
        deltas = range_metric_deltas(pred_rm, mc_rm)
        for key in ("r100_delta_mm", "r80_delta_mm", "r20_delta_mm", "dfw_delta_mm"):
            val = deltas[key]
            if np.isfinite(val):
                setattr(record, key, val)
        if (
            idd_store is not None
            and config.n_worst_idd_figures > 0
            and np.isfinite(record.r80_delta_mm)
        ):
            idd_store[record.sample_id] = (mc_idd, pred_idd, dz_mm)

        logger.info(
            f"Sample {record.sample_id}: "
            f"BP range = [{record.bp_range_min_mm:.1f}, {record.bp_range_max_mm:.1f}] mm, "
            f"density regions = {n_regions}, GPR = {record.gpr:.2f}%, "
            f"RDE = {record.rde:.4f}%, dR80 = {record.r80_delta_mm:+.2f} mm, "
            f"E = {energy_mev:.1f} MeV"
        )
        return record

    return per_sample_fn


def read_angle_map(h5_path: Path, record_ids: list) -> dict:
    """Read per-beamlet ``(ba0, ba1, gantry)`` angles (deg) from the H5 attrs.

    Only record attributes are touched (no CT/dose/flux array reads), so this is
    cheap even for the full training set. Records missing angle metadata are
    silently skipped; the per-sample callback then leaves the angle fields NaN.
    """
    angle_map: dict = {}
    with h5py.File(h5_path, "r") as f:
        for rid in record_ids:
            if rid not in f:
                continue
            attrs = f[rid].attrs
            if "beamlet_angles" not in attrs:
                continue
            ba = np.asarray(attrs["beamlet_angles"], dtype=float).ravel()
            if ba.size < 2:
                continue
            gantry = float(attrs.get("gantry_angle", float("nan")))
            angle_map[rid] = (float(ba[0]), float(ba[1]), gantry)
    logger.info(
        "Loaded beamlet angles for %d/%d records", len(angle_map), len(record_ids)
    )
    return angle_map


def extract_all_samples(
    model: DoTA3D_v3,
    dataset: H5PYGenerator,
    record_ids: list,
    config: AnalysisConfig,
    device: torch.device,
    idd_store: Optional[dict] = None,
    angle_map: Optional[dict] = None,
    show_progress: bool = True,
) -> list[SampleRecord]:
    """Run inference + metric extraction over every beamlet via the engine.

    Delegates the load / device-move / forward / timing / skip protocol to the
    shared :func:`src.evaluation.engine.evaluate`, so each record is read from
    HDF5 exactly once (the previous implementation loaded twice).

    Args:
        model: The loaded DoTA model.
        dataset: The H5PYGenerator dataset.
        record_ids: Record IDs aligned with ``dataset`` order.
        config: Analysis configuration.
        device: Target device for computation.
        idd_store: Optional dict populated with ``{sample_id: (mc_idd, pred_idd,
            dz_mm)}`` for the worst-outlier IDD overlays.
        angle_map: Optional ``{sample_id: (ba0, ba1, gantry)}`` in degrees.
        show_progress: Whether to show a progress bar.

    Returns:
        List of SampleRecord objects.
    """
    source = H5Source(dataset, record_ids=record_ids)
    per_sample_fn = _make_per_sample_fn(
        config, device, idd_store=idd_store, angle_map=angle_map
    )
    results = evaluate(
        model,
        source,
        device=device,
        per_sample_fn=per_sample_fn,
        show_progress=show_progress,
        desc="Extracting samples",
        postfix_fn=lambda r: {"E": f"{r.energy_mev:.0f}", "GPR": f"{r.gpr:.1f}"},
    )
    n_skipped = len(dataset) - len(results)
    if n_skipped > 0:
        logger.info(
            f"Skipped {n_skipped}/{len(dataset)} samples "
            f"(zero flux or energy > {config.max_energy_mev:.0f} MeV) "
            f"({100.0 * n_skipped / len(dataset):.1f}%)"
        )
    return results


# ── Results CSV ─────────────────────────────────────────────────────────────


def results_to_dataframe(results: list[SampleRecord]) -> pd.DataFrame:
    """Flatten SampleRecords into a per-beamlet DataFrame (energy-sorted).

    Uses the dataclass field names directly (via :func:`asdict`), so the CSV and
    the correlation/stratification analyses share one column vocabulary and new
    :class:`SampleRecord` fields appear everywhere without further edits.
    """
    df = pd.DataFrame([asdict(r) for r in results])
    if "energy_mev" in df.columns:
        df = df.sort_values("energy_mev").reset_index(drop=True)
    return df


def save_results_csv(results: list[SampleRecord], output_path: Path) -> None:
    """Save per-beamlet results to a CSV file (dataclass-driven columns)."""
    results_to_dataframe(results).to_csv(output_path, index=False)
    logger.info(f"Results CSV saved to {output_path}")


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


# ── Metric / target vocabulary for the correlation analysis ─────────────────
# Predictor metrics (input-space heterogeneity + edge descriptors) as
# (column, human label). Columns are SampleRecord field names or derived
# columns added by :func:`_add_derived_columns`.
METRIC_LABELS: list[tuple[str, str]] = [
    ("n_density_regions", "Number of density regions"),
    ("total_hu_change", "Total HU change [HU]"),
    ("max_hu_jump", "Max single HU jump [HU]"),
    ("sigma_hu_bp", r"$\sigma_{HU}$ in BP zone [HU]"),
    ("max_hu_gradient", "Max axial HU gradient [HU/slice]"),
    ("lateral_hu_var_bp", r"Lateral HU variance at BP [HU$^2$]"),
    ("hetero_fraction", "Hetero fraction (non-dominant class)"),
    ("mean_sobel_axial", "Mean Sobel axial (flux-weighted) [HU/vox]"),
    ("p95_sobel_bp", "P95 Sobel gradient magnitude [HU/vox]"),
    ("sum_sobel_bp", "Sum Sobel gradient magnitude [HU/vox]"),
    ("interface_bp_distance", "Interface-to-BP distance [slices]"),
    ("pflugfelder_hi", "Pflugfelder HI (WEPL CV)"),
    ("wepl_mean", "WEPL mean [mm]"),
    ("wepl_std", r"WEPL $\sigma$ [mm]"),
    ("isi_sum", r"ISI sum $(\Delta\mathrm{RSP})^2$"),
    ("isi_max", r"ISI max $(\Delta\mathrm{RSP})^2$"),
    ("isi_mean", r"ISI mean $(\Delta\mathrm{RSP})^2$"),
    ("isi_axial_sum", r"ISI axial sum $(\Delta\mathrm{RSP})^2$"),
    ("sobel_dw_mean", r"DW mean Sobel $|\mathbf{g}|$ [HU/vox]"),
    ("sobel_dw_anisotropy", r"DW edge anisotropy $A$"),
    ("sobel_dw_beam_angle", r"DW beam-edge angle $\theta$ [$^\circ$]"),
    ("sobel_dw_edge_energy", r"DW edge energy tr$(J_{dw})$"),
    ("sobel_th_mean", r"TH mean Sobel $|\mathbf{g}|$ [HU/vox]"),
    ("sobel_th_anisotropy", r"TH edge anisotropy $A$"),
    ("sobel_th_beam_angle", r"TH beam-edge angle $\theta$ [$^\circ$]"),
    ("sobel_th_edge_energy", r"TH edge energy tr$(J_{th})$"),
    ("lateral_edge_energy", r"Lateral edge energy tr$(J_{dw})\sin^2\theta$"),
    ("hu_change_per_region", "HU change per region [HU]"),
    ("hu_change_per_mm", "HU change per mm [HU/mm]"),
    ("bp_range_mm", "BP range [mm]"),
]


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived predictor + |range-error| target columns to the DataFrame."""
    df = df.copy()
    n_regions = df["n_density_regions"].astype(float).clip(lower=1)
    df["bp_range_mm"] = df["bp_range_max_mm"] - df["bp_range_min_mm"]
    df["hu_change_per_region"] = df["total_hu_change"] / n_regions
    df["hu_change_per_mm"] = df["total_hu_change"] / df["bp_range_mm"].clip(lower=1)
    # Absolute range-error magnitudes, the targets for the edge-vs-range study.
    for base in ("r80_delta_mm", "r100_delta_mm", "r20_delta_mm", "dfw_delta_mm"):
        if base in df.columns:
            df[f"abs_{base}"] = df[base].abs()
    return df


def _target_specs(config: AnalysisConfig) -> list[tuple[str, str]]:
    """Correlation targets as (column, human label), model error + range error."""
    gpr_label = (
        f"GPR ({config.gamma_params['dose_percent_threshold']}%/"
        f"{config.gamma_params['distance_mm_threshold']}mm, "
        f"{config.gamma_params['lower_percent_dose_cutoff']}% cutoff) [%]"
    )
    return [
        ("gpr", gpr_label),
        ("rde", "RDE [%]"),
        ("abs_r80_delta_mm", "|ΔR80| clinical range error [mm]"),
        ("abs_r100_delta_mm", "|ΔR100| Bragg-peak error [mm]"),
        ("abs_dfw_delta_mm", "|ΔDFW| distal-width mismatch [mm]"),
    ]


def _correlate_target(
    df: pd.DataFrame,
    target_col: str,
    target_label: str,
    out_dir: Path,
    *,
    top_n: int,
) -> Optional[pd.DataFrame]:
    """Correlate every predictor in ``METRIC_LABELS`` against one target.

    Writes ``correlation_summary.csv`` + ``correlation_ranking.png`` for the
    target, and scatter plots for the ``top_n`` strongest metrics (by
    |Spearman|). Returns the correlation table (or ``None`` if too little data).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if target_col not in df.columns or df[target_col].notna().sum() < 3:
        logger.info("Skipping target %s (insufficient data)", target_col)
        return None

    rows: list[dict] = []
    for col, label in METRIC_LABELS:
        if col not in df.columns:
            continue
        sub = df[[col, target_col]].dropna()
        if len(sub) < 3 or sub[col].std() == 0 or sub[target_col].std() == 0:
            continue
        x, y = sub[col].to_numpy(float), sub[target_col].to_numpy(float)
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)
        rows.append(
            dict(metric=col, label=label, n=len(sub), pearson_r=r_p,
                 pearson_p=p_p, spearman_r=r_s, spearman_p=p_s)
        )
    if not rows:
        return None

    corr = pd.DataFrame(rows).sort_values(
        "spearman_r", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)
    corr.to_csv(out_dir / "correlation_summary.csv", index=False)

    logger.info("")
    logger.info("=" * 72)
    logger.info("CORRELATION vs %s (sorted by |Spearman r|)", target_col)
    logger.info("=" * 72)
    for _, row in corr.iterrows():
        logger.info(
            f"{row['metric']:24s}  Pearson r={row['pearson_r']:+.3f}  "
            f"Spearman r={row['spearman_r']:+.3f}  (n={int(row['n'])})"
        )

    # Ranking bar chart across all metrics.
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x_pos = np.arange(len(corr))
    bar_w = 0.4
    ax.barh(x_pos - bar_w / 2, corr["pearson_r"].abs(), bar_w,
            label="|Pearson r|", color="#4C72B0")
    ax.barh(x_pos + bar_w / 2, corr["spearman_r"].abs(), bar_w,
            label="|Spearman r|", color="#DD8452")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(corr["metric"], fontsize=8)
    ax.set_xlabel(f"Absolute correlation with {target_col}")
    ax.set_title(f"Metric correlation ranking vs {target_label}  (N = {len(df)})")
    ax.legend()
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Scatter plots for the strongest top_n predictors.
    label_by_col = dict(METRIC_LABELS)
    for _, row in corr.head(top_n).iterrows():
        col = row["metric"]
        sub = df[[col, target_col]].dropna()
        x, y = sub[col].to_numpy(float), sub[target_col].to_numpy(float)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.scatter(x, y, s=16, alpha=0.6, color="#4C72B0",
                   edgecolors="k", linewidths=0.3)
        coef = np.polyfit(x, y, 1)
        xf = np.linspace(x.min(), x.max(), 200)
        ax.plot(xf, np.polyval(coef, xf), "r--", lw=1.2,
                label=(f"Pearson r = {row['pearson_r']:.3f}\n"
                       f"Spearman r = {row['spearman_r']:.3f}"))
        ax.set_xlabel(label_by_col.get(col, col))
        ax.set_ylabel(target_label)
        ax.set_title(f"{col}  vs  {target_col}  (N = {len(sub)})")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}_vs_{target_col}.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    return corr


def generate_correlation_analysis(
    results: list[SampleRecord],
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Correlate heterogeneity/edge metrics against model- and range-error targets.

    One subdirectory per target under ``output_dir/correlations/`` (GPR, RDE,
    |ΔR80|, |ΔR100|, |ΔDFW|). A combined ``spearman_by_target.csv`` pivot makes
    the edge-vs-range hypothesis directly readable.
    """
    if len(results) < 3:
        logger.info("Skipping correlation analysis -- fewer than 3 samples")
        return

    df = _add_derived_columns(results_to_dataframe(results))
    corr_root = output_dir / "correlations"

    combined: list[pd.DataFrame] = []
    for target_col, target_label in _target_specs(config):
        corr = _correlate_target(
            df, target_col, target_label,
            corr_root / target_col, top_n=config.scatter_top_n,
        )
        if corr is not None:
            combined.append(
                corr[["metric", "spearman_r"]].rename(
                    columns={"spearman_r": target_col}
                ).set_index("metric")
            )

    if combined:
        pivot = pd.concat(combined, axis=1)
        pivot.to_csv(corr_root / "spearman_by_target.csv")
        logger.info("Combined Spearman-by-target table: %s",
                    corr_root / "spearman_by_target.csv")


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
    target_col: str = "gpr",
    target_label: str = "GPR",
) -> None:
    """Produce energy-stratified correlation analysis for one target.

    ``target_col`` selects the quantity every metric is correlated against
    (``gpr`` for model accuracy, ``abs_r80_delta_mm`` for the clinical range
    error, etc.). Outputs go to ``output_dir/energy_stratified/<target_col>/``.
    """
    from scipy.stats import pearsonr, spearmanr

    strat_dir = output_dir / "energy_stratified" / target_col
    strat_dir.mkdir(parents=True, exist_ok=True)

    df = _add_derived_columns(results_to_dataframe(results))
    if target_col not in df.columns or df[target_col].isna().all():
        logger.warning(
            "No %s data – skipping energy-stratified analysis.", target_col
        )
        return

    # Metric columns to analyse (shared predictor vocabulary).
    metrics = [
        c for c, _ in METRIC_LABELS if c in df.columns and df[c].notna().sum() > 5
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
        sub = df[df["energy_bin"] == label].dropna(subset=[target_col])
        for m in metrics:
            valid = sub.dropna(subset=[m])
            n = len(valid)
            count_mat.loc[label, m] = n
            # Need >=5 points and non-constant columns for a defined correlation.
            if n >= 5 and valid[m].std() > 0 and valid[target_col].std() > 0:
                spearman_mat.loc[label, m] = spearmanr(
                    valid[m], valid[target_col]
                ).correlation
                pearson_mat.loc[label, m] = pearsonr(
                    valid[m], valid[target_col]
                ).statistic
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
        ax.set_title(
            "{} correlation with {} by energy bin".format(
                name.capitalize(), target_label
            )
        )
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
        valid = df.dropna(subset=[m, target_col])
        if len(valid) >= 5 and valid[m].std() > 0 and valid[target_col].std() > 0:
            corr = spearmanr(valid[m], valid[target_col]).correlation
            if np.isfinite(corr):
                overall_corr[m] = abs(corr)
    top_metrics = sorted(overall_corr, key=overall_corr.get, reverse=True)[:6]

    for m in top_metrics:
        sub = df.dropna(subset=[m, target_col, "energy_mev"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            sub[m],
            sub[target_col],
            c=sub["energy_mev"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="none",
            s=20,
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Energy (MeV)")
        ax.set_xlabel(m)
        ax.set_ylabel(target_label)
        ax.set_title("{} vs {} (coloured by energy)".format(m, target_label))
        fig.tight_layout()
        fig.savefig(
            strat_dir / "{}_vs_{}_energy_coloured.png".format(m, target_col),
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
        valid = df.dropna(subset=[m, target_col, "energy_mev"])
        if len(valid) >= 10 and valid[m].std() > 0 and valid[target_col].std() > 0:
            raw_spearman = spearmanr(valid[m], valid[target_col]).correlation
            partial_spearman = _partial_correlation(valid, m, target_col, "energy_mev")
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
        ax.set_ylabel(f"Spearman correlation with {target_label}")
        ax.set_title(
            f"Raw vs Partial Spearman vs {target_label} (controlling for energy)"
        )
        ax.legend()
        ax.axhline(0, color="grey", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(
            strat_dir / "partial_vs_raw_correlation.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
    logger.info("Partial correlation analysis saved.")


# ── Range-error diagnostics ─────────────────────────────────────────────────


def plot_worst_idd_overlays(
    results: list[SampleRecord],
    idd_store: dict,
    output_dir: Path,
    n_worst: int,
) -> None:
    """MC-vs-ADoTA IDD overlays for the beamlets with the largest |ΔR80|.

    ``idd_store`` maps ``sample_id -> (mc_idd, pred_idd, dz_mm)`` for beamlets
    that produced a valid range delta. Selects the ``n_worst`` records by
    |ΔR80| and overlays the two integrated depth-dose curves with their R80
    markers, the direct visual test of the range-error hypothesis.
    """
    if n_worst <= 0 or not idd_store:
        return
    ranked = sorted(
        (r for r in results if r.sample_id in idd_store
         and np.isfinite(r.r80_delta_mm)),
        key=lambda r: abs(r.r80_delta_mm),
        reverse=True,
    )
    if not ranked:
        return
    fig_dir = output_dir / "range_diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for rank, r in enumerate(ranked[:n_worst], start=1):
        mc_idd, pred_idd, dz_mm = idd_store[r.sample_id]
        z = np.arange(len(mc_idd)) * dz_mm
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(z, mc_idd, label="MC (GT)", color="tab:orange")
        ax.plot(z, pred_idd, label="ADoTA", color="tab:green")
        ax.axvline(r.mc_r80_mm, color="tab:orange", ls="--", lw=0.8,
                   label=f"MC R80={r.mc_r80_mm:.1f}")
        ax.axvline(r.mc_r80_mm + r.r80_delta_mm, color="tab:green", ls="--", lw=0.8,
                   label=f"ADoTA R80={r.mc_r80_mm + r.r80_delta_mm:.1f}")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Integrated depth dose")
        ax.set_title(
            f"#{rank} |ΔR80|={abs(r.r80_delta_mm):.2f} mm  "
            f"({r.sample_id}, E={r.energy_mev:.1f} MeV, "
            f"lateral edge={r.lateral_edge_energy:.2e})"
        )
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(fig_dir / f"worst_idd_{rank:02d}_{r.sample_id}.png", dpi=150)
        plt.close(fig)
    logger.info("Saved %d worst-|ΔR80| IDD overlays to %s",
                min(n_worst, len(ranked)), fig_dir)


# ── Beamlet-angle performance maps ──────────────────────────────────────────


def generate_angle_performance_analysis(
    results: list[SampleRecord],
    output_dir: Path,
    config: AnalysisConfig,
) -> None:
    """Map model performance over beamlet steering direction (ba0, ba1).

    The beamlet angles are the per-spot deflection = the rotation applied to the
    input before inference, so this probes rotation/resampling degradation. For
    each target (GPR, RDE, |ΔR100|, |ΔR80|) it writes:

      * a 2D binned mean-metric heatmap over (ba0, ba1) with low-count bins
        masked and smoothed iso-lines overlaid (honest binning, not
        interpolated contour), plus a companion beamlet-count map;
      * a radial plot of the metric vs total obliquity ``|angle| =
        sqrt(ba0^2 + ba1^2)`` -- the signature of a rotation artifact.
    """
    out_dir = output_dir / "beamlet_angle"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _add_derived_columns(results_to_dataframe(results))
    df = df[df["beamlet_angle_0_deg"].notna() & df["beamlet_angle_1_deg"].notna()]
    if len(df) < 50:
        logger.info("Skipping angle analysis -- %d beamlets with angles", len(df))
        return

    ba0 = df["beamlet_angle_0_deg"].to_numpy(float)
    ba1 = df["beamlet_angle_1_deg"].to_numpy(float)
    df["angle_mag_deg"] = np.hypot(ba0, ba1)

    targets = [
        ("gpr", "GPR [%]", "viridis"),
        ("rde", "RDE [%]", "magma"),
        ("abs_r100_delta_mm", "|ΔR100| [mm]", "magma"),
        ("abs_r80_delta_mm", "|ΔR80| [mm]", "magma"),
    ]
    n_bins = 24
    lim = float(np.nanpercentile(np.abs(np.r_[ba0, ba1]), 99.5))
    edges = np.linspace(-lim, lim, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _, _ = np.histogram2d(ba0, ba1, bins=[edges, edges])
    min_count = 5

    # Shared beamlet-count map (context for every metric panel).
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(edges, edges, counts.T, cmap="Greys", shading="flat")
    fig.colorbar(im, ax=ax, label="beamlets per bin")
    ax.set_xlabel("Beamlet angle 0 [deg]")
    ax.set_ylabel("Beamlet angle 1 [deg]")
    ax.set_title(f"Beamlet sampling density (N={len(df)})")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "angle_count_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for col, label, cmap in targets:
        if col not in df.columns or df[col].notna().sum() < 50:
            continue
        vals = df[col].to_numpy(float)
        finite = np.isfinite(vals)
        sums, _, _ = np.histogram2d(
            ba0[finite], ba1[finite], bins=[edges, edges], weights=vals[finite]
        )
        cnt, _, _ = np.histogram2d(ba0[finite], ba1[finite], bins=[edges, edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / cnt
        mean_masked = np.ma.masked_where(cnt < min_count, mean)

        fig, ax = plt.subplots(figsize=(7.5, 6))
        im = ax.pcolormesh(edges, edges, mean_masked.T, cmap=cmap, shading="flat")
        fig.colorbar(im, ax=ax, label=f"mean {label}")
        # Smoothed iso-lines over the filled (count-weighted) mean field.
        filled = np.where(cnt >= min_count, np.nan_to_num(mean), np.nan)
        valid = np.isfinite(filled)
        if valid.sum() > 10:
            sm = filled.copy()
            sm[~valid] = np.nanmean(filled[valid])
            sm = gaussian_filter(sm, sigma=1.0)
            try:
                cs = ax.contour(
                    centers, centers, sm.T, colors="k", linewidths=0.6, alpha=0.5
                )
                ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
            except ValueError:
                pass
        ax.set_xlabel("Beamlet angle 0 [deg]")
        ax.set_ylabel("Beamlet angle 1 [deg]")
        ax.set_title(f"{label} vs beamlet direction  (bins <{min_count} masked)")
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_dir / f"angle_map_{col}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Radial: metric vs total obliquity (rotation-artifact signature).
        sub = df[["angle_mag_deg", col]].dropna()
        r = sub["angle_mag_deg"].to_numpy()
        rbins = np.linspace(0, np.nanpercentile(r, 99), 12)
        idx = np.digitize(r, rbins)
        rc, rm, rse = [], [], []
        for b in range(1, len(rbins)):
            m = sub[col].to_numpy()[idx == b]
            if len(m) >= 10:
                rc.append(0.5 * (rbins[b - 1] + rbins[b]))
                rm.append(float(np.mean(m)))
                rse.append(float(np.std(m) / np.sqrt(len(m))))
        if len(rc) >= 3:
            rho, p = spearmanr(sub["angle_mag_deg"], sub[col])
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.errorbar(rc, rm, yerr=rse, marker="o", lw=1.5, capsize=3)
            ax.set_xlabel(r"Total obliquity $|angle| = \sqrt{ba_0^2 + ba_1^2}$ [deg]")
            ax.set_ylabel(f"mean {label}")
            ax.set_title(f"{label} vs beamlet obliquity  "
                         f"(Spearman r={rho:+.3f}, p={p:.1e})")
            ax.grid(linestyle="--", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(out_dir / f"angle_radial_{col}.png", dpi=200,
                        bbox_inches="tight")
            plt.close(fig)

    logger.info("Beamlet-angle performance maps saved to %s", out_dir)


# ── Metric redundancy: hierarchical clustering ──────────────────────────────


def generate_metric_clustermap(
    results: list[SampleRecord],
    output_dir: Path,
    target_col: str = "abs_r100_delta_mm",
) -> None:
    """Cluster the predictor metrics by intercorrelation to expose redundancy.

    Distance is ``1 - |Spearman|``; average-linkage hierarchical clustering feeds
    both a dendrogram-ordered correlation clustermap and a flat clustering at
    |corr| >= 0.8. For each cluster the member most correlated with ``target_col``
    is proposed as the representative, yielding a parsimonious, de-collinearised
    feature basis for the acquisition function (written to CSV).
    """
    out_dir = output_dir / "metric_clustering"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _add_derived_columns(results_to_dataframe(results))
    metrics = [c for c, _ in METRIC_LABELS if c in df.columns and df[c].notna().sum() > 50]
    if len(metrics) < 3:
        logger.info("Skipping metric clustering -- too few metrics")
        return

    X = df[metrics].fillna(df[metrics].median())
    corr = X.corr("spearman")
    dist = 1.0 - corr.abs()
    Z = linkage(squareform(dist.values, checks=False), method="average")

    # Flat clusters at |corr| >= 0.8 (distance <= 0.2); representative = member
    # most correlated with the target error.
    cl = fcluster(Z, t=0.2, criterion="distance")
    tgt = df[target_col] if target_col in df.columns else None
    rows = []
    for c in sorted(set(cl)):
        members = [metrics[i] for i in range(len(metrics)) if cl[i] == c]
        if tgt is not None:
            scores = {
                m: abs(spearmanr(X[m], tgt, nan_policy="omit").correlation)
                for m in members
            }
            rep = max(scores, key=scores.get)
            rep_score = scores[rep]
        else:
            rep, rep_score = members[0], float("nan")
        rows.append({
            "cluster": c, "representative": rep,
            f"rep_abs_spearman_vs_{target_col}": round(rep_score, 4),
            "n_members": len(members), "members": "; ".join(members),
        })
    basis = pd.DataFrame(rows).sort_values(
        f"rep_abs_spearman_vs_{target_col}", ascending=False
    )
    basis.to_csv(out_dir / "parsimonious_basis.csv", index=False)
    logger.info("Metric clustering: %d metrics -> %d non-redundant clusters",
                len(metrics), len(set(cl)))
    for _, r in basis.iterrows():
        logger.info("  keep %-22s (|r|=%.2f vs %s) | cluster of %d",
                    r["representative"], r[f"rep_abs_spearman_vs_{target_col}"],
                    target_col, r["n_members"])

    # Dendrogram-ordered correlation clustermap.
    try:
        g = sns.clustermap(
            corr, row_linkage=Z, col_linkage=Z, cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, figsize=(13, 13),
            cbar_kws={"label": "Spearman r"},
        )
        g.fig.suptitle(f"Metric intercorrelation clustering (N={len(df)} beamlets)",
                       y=1.01)
        g.savefig(out_dir / "metric_clustermap.png", dpi=150, bbox_inches="tight")
        plt.close(g.fig)
    except Exception as exc:  # clustermap is finicky on degenerate matrices
        logger.warning("clustermap failed (%s); skipping figure", exc)
    logger.info("Metric clustering outputs saved to %s", out_dir)


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
    # Artifacts default to <project>/runs but can be redirected (e.g. to
    # /scratch) via the ``runs_dir`` YAML key to keep heavy outputs off /home.
    runs_dir = Path(yaml_config.get("runs_dir", PROJECT_ROOT / "runs"))
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
    scatter_top_n = int(yaml_config.get("scatter_top_n", 6))
    n_worst_idd_figures = int(yaml_config.get("n_worst_idd_figures", 8))
    range_oversample = int(yaml_config.get("range_oversample", 20))

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
        scatter_top_n=scatter_top_n,
        n_worst_idd_figures=n_worst_idd_figures,
        range_oversample=range_oversample,
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
    device = resolve_device(device_index)
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
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Per-beamlet steering + gantry angles (cheap: attrs only, no array reads).
    angle_map = read_angle_map(h5_path, record_ids)

    start_time = perf_counter()
    idd_store: dict = {}
    results = extract_all_samples(
        model=model,
        dataset=dataset,
        record_ids=record_ids,
        config=analysis_config,
        device=device,
        idd_store=idd_store,
        angle_map=angle_map,
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

    # ── Generate figures for extreme beam-angle cases ──────────────
    generate_beam_angle_figures(
        model=model,
        results=results,
        record_ids=record_ids,
        dataset=dataset,
        config=analysis_config,
        device=device,
        figures_dir=figures_dir,
    )

    # ── Correlation analysis (multi-target: GPR, RDE, range errors) ─
    generate_correlation_analysis(
        results=results,
        output_dir=figures_dir,
        config=analysis_config,
    )

    # ── Energy-stratified analysis (model accuracy + clinical range) ─
    for target_col, target_label in (
        ("gpr", "GPR"),
        ("abs_r80_delta_mm", "|ΔR80| [mm]"),
    ):
        generate_energy_stratified_analysis(
            results=results,
            output_dir=figures_dir,
            config=analysis_config,
            target_col=target_col,
            target_label=target_label,
        )

    # ── Range-error IDD overlays (largest |ΔR80|) ───────────────────
    plot_worst_idd_overlays(
        results=results,
        idd_store=idd_store,
        output_dir=figures_dir,
        n_worst=analysis_config.n_worst_idd_figures,
    )

    # ── Beamlet-angle performance maps (rotation-degradation probe) ──
    generate_angle_performance_analysis(
        results=results,
        output_dir=figures_dir,
        config=analysis_config,
    )

    # ── Metric redundancy clustering (acquisition-function feature basis) ─
    generate_metric_clustermap(
        results=results,
        output_dir=figures_dir,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Analysis complete! Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
