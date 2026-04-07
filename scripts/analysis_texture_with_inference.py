"""
CT Texture & Model Inference Analysis

Combines DoTA model evaluation with CT heterogeneity / texture metrics
and computes Pearson correlations between model performance (MAPE, GPR)
and CT complexity scores (G_phi, R, H_phi, GLCM homogeneity).

The script:
  1. Loads the DoTA model and evaluates every sample in test_data.
  2. For each sample, loads the *original* (un-normalised) CT grid and the
     fluence (phi) array, then computes heterogeneity and GLCM metrics.
  3. Prints a combined results table.
  4. Computes and reports Pearson correlations.
  5. Saves everything (JSON + CSV + correlation summary) in a timestamped
     run directory.

Usage:
    uv run python scripts/analysis_texture_with_inference.py \\
        --config scripts/config_analysis_texture_with_inference.yaml
"""

import csv
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
import yaml
from scipy import stats
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.models import DoTA3D_v3
from src.adota.utils import count_parameters_per_block, count_total_parameters
from src.image_processing.heterogeneity import (
    beam_aligned_global_heterogeneity,
    beam_axis_roughness,
    beam_weighted_gradient,
    gradient_magnitude_3d,
)
from src.image_processing.homogeneity_scores import glcm_homogeneity_idm
from src.image_processing.intensity_heterogeneity import (
    GlobalIntensityHeterogeneity,
    global_intensity_heterogeneity,
)
from src.loaders.dir_based import get_single_record, save_prediction
from src.metrics.classic import (
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from src.metrics.gamma_pass_rate import gamma_index_torch
from src.figures.single_beam import publication_figure
from src.tables.results import print_results_table
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="CT Texture & Model Inference Analysis — correlate model performance "
    "with CT heterogeneity metrics.",
)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

# Default scaling parameters
DEFAULT_SCALE = {
    "min_ds": 0.0,
    "max_ds": 25277028.0,
    "min_ct": -1024,
    "max_ct": 3071,
    "min_energy": 70.0,
    "max_energy": 270.0,
}

DEFAULT_GAMMA_PARAMS = {
    "dose_percent_threshold": 2,
    "distance_mm_threshold": 2,
    "interp_fraction": 10,
    "max_gamma": 2,
    "lower_percent_dose_cutoff": 10,
    "random_subset": None,
    "local_gamma": False,
    "quiet": True,
}


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())
    gamma_params: dict = field(default_factory=lambda: DEFAULT_GAMMA_PARAMS.copy())
    normalize_flux: bool = True
    resolution: tuple = (2.0, 2.0, 2.0)


@dataclass
class MetricsConfig:
    """Toggle flags for the three texture-metric families.

    Each flag can be set via CLI (``--enable-heterogeneity`` etc.) or in the
    YAML config under the ``metrics`` key.
    """

    heterogeneity: bool = True
    glcm: bool = True
    intensity: bool = True


@dataclass
class SampleResult:
    """Container for one sample's combined inference + texture results."""

    sample_id: str
    energy_mev: float
    beamlet_angles: tuple

    # Model performance metrics
    gpr: float
    rmse: float
    mape: float
    rde: float
    calc_time: float

    # Beam-aligned heterogeneity metrics (optional)
    g_phi: Optional[float] = None
    r_roughness: Optional[float] = None
    h_phi: Optional[float] = None

    # GLCM homogeneity — volume-level mean (optional)
    glcm_homogeneity_mean: Optional[float] = None
    glcm_homogeneity_std: Optional[float] = None

    # Global intensity heterogeneity (optional)
    intensity_mean: Optional[float] = None
    intensity_std: Optional[float] = None
    intensity_iqr: Optional[float] = None
    intensity_mad: Optional[float] = None
    intensity_entropy: Optional[float] = None
    intensity_uniformity: Optional[float] = None
    intensity_skewness: Optional[float] = None
    intensity_kurtosis: Optional[float] = None

    # Cached tensors for publication figures
    prediction: Optional[torch.Tensor] = field(default=None, repr=False)
    ground_truth: Optional[torch.Tensor] = field(default=None, repr=False)
    input_data: Optional[torch.Tensor] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Helpers (same as in run_model.py)
# ---------------------------------------------------------------------------


def setup_run_directory(runs_dir: Path) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"analysis_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool = False) -> Path:
    """Configure logging to both console and file."""
    log_file = run_dir / "analysis.log"
    log_level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return log_file


def denormalize_energy(energy_normalized: float, scale: dict) -> float:
    """Convert normalized energy back to MeV."""
    return (
        energy_normalized * (scale["max_energy"] - scale["min_energy"])
        + scale["min_energy"]
    )


def get_device(device_index: int) -> torch.device:
    """Get the appropriate torch device."""
    if torch.cuda.is_available() and device_index >= 0:
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def load_model(
    model_path: Path,
    hyperparams_path: Path,
    device: torch.device,
) -> DoTA3D_v3:
    """Load and configure the DoTA model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not hyperparams_path.exists():
        raise FileNotFoundError(f"Hyperparams file not found: {hyperparams_path}")

    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)

    model = DoTA3D_v3(**hyperparams)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    logger.info("Model loaded from %s", model_path)
    return model


def load_yaml_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise typer.BadParameter(f"Failed to parse YAML config: {e}")
    return cfg if cfg is not None else {}


# ---------------------------------------------------------------------------
# CT texture metrics computation
# ---------------------------------------------------------------------------


def compute_heterogeneity_metrics(
    ct_hu: np.ndarray,
    phi: np.ndarray,
    spacing_zyx: tuple,
    beam_axis: int = 0,
    mask_threshold: Optional[float] = None,
) -> dict:
    """Compute beam-aligned heterogeneity metrics on the original CT grid.

    Parameters
    ----------
    ct_hu : np.ndarray
        Original CT volume in HU, shape (D, H, W).
    phi : np.ndarray
        Fluence / beamlet projection weights, same shape as *ct_hu*.
    spacing_zyx : tuple
        Voxel spacing (z, y, x) in mm.
    beam_axis : int
        Depth axis (default 0).
    mask_threshold : float or None
        If set, only voxels with HU > *mask_threshold* contribute.

    Returns
    -------
    dict
        Keys: ``G_phi``, ``R``, ``H_phi``.
    """
    mask = None
    if mask_threshold is not None:
        mask = ct_hu > mask_threshold

    return beam_aligned_global_heterogeneity(
        ct_hu=ct_hu,
        phi=phi,
        spacing_zyx=tuple(float(s) for s in spacing_zyx),
        axis=beam_axis,
        mask=mask,
    )


def compute_glcm_metrics(
    ct_hu: np.ndarray,
    glcm_cfg: dict,
) -> dict:
    """Compute GLCM homogeneity on sampled axial slices.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``min``, ``max``, ``per_slice``.
    """
    n_slices = ct_hu.shape[0]

    levels = int(glcm_cfg.get("levels", 64))
    variant = str(glcm_cfg.get("variant", "idm"))
    distances = tuple(int(d) for d in glcm_cfg.get("distances", [1]))
    angles_deg = glcm_cfg.get("angles_deg", [0, 45, 90, 135])
    angles = tuple(float(a) * np.pi / 180.0 for a in angles_deg)
    symmetric = bool(glcm_cfg.get("symmetric", True))
    vr = glcm_cfg.get("value_range", None)
    value_range = tuple(float(v) for v in vr) if vr is not None else None

    n_sample = min(int(glcm_cfg.get("n_sample_slices", 10)), n_slices)
    if n_sample >= n_slices:
        slice_indices = list(range(n_slices))
    else:
        slice_indices = np.linspace(0, n_slices - 1, n_sample, dtype=int).tolist()

    per_slice = {}
    for idx in slice_indices:
        score = glcm_homogeneity_idm(
            ct_hu[idx],
            levels=levels,
            value_range=value_range,
            distances=distances,
            angles=angles,
            symmetric=symmetric,
            variant=variant,
        )
        per_slice[idx] = score

    scores = list(per_slice.values())
    return {
        "variant": variant,
        "levels": levels,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def compute_intensity_metrics(
    ct_hu: np.ndarray,
    intensity_cfg: dict,
    mask_threshold: Optional[float] = None,
) -> GlobalIntensityHeterogeneity:
    """Compute global intensity heterogeneity on the original CT grid.

    Parameters
    ----------
    ct_hu : np.ndarray
        Original CT volume in HU, shape (D, H, W).
    intensity_cfg : dict
        Configuration keys: ``hu_range``, ``bin_width_hu``.
    mask_threshold : float or None
        If set, only voxels with HU > *mask_threshold* contribute.

    Returns
    -------
    GlobalIntensityHeterogeneity
        Frozen dataclass with all intensity heterogeneity metrics.
    """
    mask = None
    if mask_threshold is not None:
        mask = ct_hu > mask_threshold

    hu_range_raw = intensity_cfg.get("hu_range", None)
    hu_range = (
        tuple(float(v) for v in hu_range_raw) if hu_range_raw is not None else None
    )
    bin_width_hu = float(intensity_cfg.get("bin_width_hu", 25.0))

    return global_intensity_heterogeneity(
        ct_hu=ct_hu,
        mask=mask,
        hu_range=hu_range,
        bin_width_hu=bin_width_hu,
    )


# ---------------------------------------------------------------------------
# Single-sample evaluation
# ---------------------------------------------------------------------------


def evaluate_single_sample(
    model: DoTA3D_v3,
    sample_id: str,
    test_data_path: Path,
    config: EvaluationConfig,
    device: torch.device,
    downsampling_method: str,
    heterogeneity_cfg: dict,
    glcm_cfg: dict,
    intensity_cfg: dict,
    metrics_config: MetricsConfig,
) -> SampleResult:
    """Evaluate a single sample: run inference, compute model metrics and
    the enabled CT texture / heterogeneity metrics.
    """
    scale = config.scale
    gamma_params = config.gamma_params

    start_time = perf_counter()

    # Load data for model inference
    x, energy, y, ba = get_single_record(
        sample_id,
        test_data_path,
        scale=scale,
        normalize_flux=config.normalize_flux,
        downsampling_method=downsampling_method,
        beamlet_angle=True,
    )

    x, energy, y = x.to(device), energy.to(device), y.to(device)
    energy_mev = denormalize_energy(energy.item(), scale)

    # Run inference
    with torch.no_grad():
        y_pred_tuple = model(x.unsqueeze(0), energy.unsqueeze(0))
        y_pred = y_pred_tuple[0]

    calc_time = perf_counter() - start_time

    # Convert to numpy for model metrics
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

    # Model performance metrics
    rmse = calculate_rmse(to_gy(y_pred_np), to_gy(y_np))
    mask_mape = y_pred_np > 0.1 * np.max(y_pred_np)
    mape = calculate_pure_mape(y_np[mask_mape], y_pred_np[mask_mape])
    rde = calculate_relative_dose_error(to_gy(y_pred_np), to_gy(y_np))

    scale_gpr = {"y_min": scale["min_ds"], "y_max": scale["max_ds"]}
    gpr_result = gamma_index_torch(
        y.unsqueeze(0),
        y_pred,
        scale=scale_gpr,
        gamma_params=gamma_params,
        resolution=config.resolution,
    )
    gpr = gpr_result[1][0] * 100

    # ------------------------------------------------------------------
    # CT texture / heterogeneity metrics (on the *original* un-normalised
    # CT and fluence grids, before any downsampling).
    # Only load raw arrays if at least one texture metric is enabled.
    # ------------------------------------------------------------------
    needs_raw_ct = (
        metrics_config.heterogeneity or metrics_config.glcm or metrics_config.intensity
    )

    ct_hu = None
    mask_threshold = None
    if needs_raw_ct:
        ct_raw = np.load(os.path.join(test_data_path, f"{sample_id}_ct.npy"))
        # The stored arrays are (H, W, D) → transpose to (D, H, W)
        ct_hu = ct_raw.transpose(2, 0, 1).astype(np.float32)
        mask_threshold = heterogeneity_cfg.get("mask_threshold", None)
        if mask_threshold is not None:
            mask_threshold = float(mask_threshold)

    # Beam-aligned heterogeneity
    het_kwargs: dict = {}
    if metrics_config.heterogeneity and ct_hu is not None:
        flux_raw = np.load(os.path.join(test_data_path, f"{sample_id}_flux.npy"))
        phi = np.abs(flux_raw.transpose(2, 0, 1).astype(np.float32))

        spacing_zyx = tuple(
            float(s) for s in heterogeneity_cfg.get("spacing_zyx", [2.0, 2.0, 2.0])
        )
        beam_axis = int(heterogeneity_cfg.get("beam_axis", 0))

        het_metrics = compute_heterogeneity_metrics(
            ct_hu=ct_hu,
            phi=phi,
            spacing_zyx=spacing_zyx,
            beam_axis=beam_axis,
            mask_threshold=mask_threshold,
        )
        het_kwargs = {
            "g_phi": het_metrics["G_phi"],
            "r_roughness": het_metrics["R"],
            "h_phi": het_metrics["H_phi"],
        }

    # GLCM homogeneity
    glcm_kwargs: dict = {}
    if metrics_config.glcm and ct_hu is not None:
        glcm_result = compute_glcm_metrics(ct_hu, glcm_cfg)
        glcm_kwargs = {
            "glcm_homogeneity_mean": glcm_result["mean"],
            "glcm_homogeneity_std": glcm_result["std"],
        }

    # Global intensity heterogeneity
    intensity_kwargs: dict = {}
    if metrics_config.intensity and ct_hu is not None:
        gi = compute_intensity_metrics(ct_hu, intensity_cfg, mask_threshold)
        intensity_kwargs = {
            "intensity_mean": gi.mean,
            "intensity_std": gi.std,
            "intensity_iqr": gi.iqr,
            "intensity_mad": gi.mad,
            "intensity_entropy": gi.entropy,
            "intensity_uniformity": gi.uniformity,
            "intensity_skewness": gi.skewness,
            "intensity_kurtosis": gi.kurtosis_excess,
        }

    return SampleResult(
        sample_id=sample_id,
        energy_mev=energy_mev,
        beamlet_angles=tuple(ba) if isinstance(ba, list) else ba,
        gpr=gpr,
        rmse=rmse,
        mape=mape,
        rde=rde,
        calc_time=calc_time,
        **het_kwargs,
        **glcm_kwargs,
        **intensity_kwargs,
        prediction=y_pred.cpu(),
        ground_truth=y.cpu(),
        input_data=x.cpu(),
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def evaluate_all_samples(
    model: DoTA3D_v3,
    sample_ids: list,
    test_data_path: Path,
    config: EvaluationConfig,
    device: torch.device,
    downsampling_method: str,
    heterogeneity_cfg: dict,
    glcm_cfg: dict,
    intensity_cfg: dict,
    metrics_config: MetricsConfig,
    show_progress: bool = True,
) -> list:
    """Evaluate all samples and return a list of SampleResult."""
    results = []
    iterator = (
        tqdm(sample_ids, desc="Evaluating samples") if show_progress else sample_ids
    )

    for sample_id in iterator:
        try:
            result = evaluate_single_sample(
                model=model,
                sample_id=sample_id,
                test_data_path=test_data_path,
                config=config,
                device=device,
                downsampling_method=downsampling_method,
                heterogeneity_cfg=heterogeneity_cfg,
                glcm_cfg=glcm_cfg,
                intensity_cfg=intensity_cfg,
                metrics_config=metrics_config,
            )
            results.append(result)

            if show_progress and hasattr(iterator, "set_postfix"):
                postfix = {
                    "E": f"{result.energy_mev:.0f}",
                    "GPR": f"{result.gpr:.1f}%",
                }
                if result.g_phi is not None:
                    postfix["G"] = f"{result.g_phi:.2f}"
                iterator.set_postfix(**postfix)
        except Exception:
            logger.exception("Failed to process sample %s", sample_id)

    return results


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def compute_correlations(results: list, metrics_config: MetricsConfig) -> dict:
    """Compute Pearson correlations between model performance and the
    enabled texture metrics.

    Returns
    -------
    dict
        Nested dict: ``{performance_metric: {texture_metric: {"r": ..., "p": ...}}}``.
    """
    if len(results) < 3:
        logger.warning(
            "Only %d samples — Pearson correlations may be unreliable.", len(results)
        )

    mapes = np.array([r.mape for r in results])
    gprs = np.array([r.gpr for r in results])

    performance = {"MAPE": mapes, "GPR": gprs}

    # Dynamically collect only enabled texture arrays
    texture: dict[str, np.ndarray] = {}
    if metrics_config.heterogeneity and results[0].g_phi is not None:
        texture["G_phi"] = np.array([r.g_phi for r in results])
        texture["R"] = np.array([r.r_roughness for r in results])
        texture["H_phi"] = np.array([r.h_phi for r in results])
    if metrics_config.glcm and results[0].glcm_homogeneity_mean is not None:
        texture["GLCM_homogeneity"] = np.array(
            [r.glcm_homogeneity_mean for r in results]
        )
    if metrics_config.intensity and results[0].intensity_entropy is not None:
        texture["entropy"] = np.array([r.intensity_entropy for r in results])
        texture["uniformity"] = np.array([r.intensity_uniformity for r in results])
        texture["IQR"] = np.array([r.intensity_iqr for r in results])
        texture["MAD"] = np.array([r.intensity_mad for r in results])

    if not texture:
        logger.warning("No texture metrics enabled — skipping correlations.")
        return {}

    correlations: dict = {}
    for perf_name, perf_vals in performance.items():
        correlations[perf_name] = {}
        for tex_name, tex_vals in texture.items():
            # Guard against constant arrays
            if np.std(perf_vals) < 1e-12 or np.std(tex_vals) < 1e-12:
                correlations[perf_name][tex_name] = {
                    "r": float("nan"),
                    "p": float("nan"),
                }
                continue
            r_val, p_val = stats.pearsonr(perf_vals, tex_vals)
            correlations[perf_name][tex_name] = {
                "r": float(r_val),
                "p": float(p_val),
            }

    return correlations


def print_correlation_table(correlations: dict) -> None:
    """Pretty-print the correlation matrix to logger."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("PEARSON CORRELATIONS:  performance metric  vs  texture metric")
    logger.info("=" * 80)
    logger.info(
        f"{'Performance':<12} {'Texture':<20} {'r':>10} {'p-value':>12} {'Signif.':>8}"
    )
    logger.info("-" * 80)
    for perf_name, tex_dict in correlations.items():
        for tex_name, vals in tex_dict.items():
            r_val = vals["r"]
            p_val = vals["p"]
            sig = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            )
            logger.info(
                f"{perf_name:<12} {tex_name:<20} {r_val:>10.4f} {p_val:>12.6f} {sig:>8}"
            )
    logger.info("=" * 80)
    logger.info("Significance levels:  * p<0.05  ** p<0.01  *** p<0.001")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def print_combined_results_table(
    results: list, gamma_params: dict, metrics_config: MetricsConfig
) -> None:
    """Print a combined table with model + enabled texture metrics."""
    logger.info("")
    logger.info("=" * 160)
    logger.info("COMBINED RESULTS TABLE")
    logger.info("=" * 160)

    # Build dynamic header
    header = f"{'ID':<40} {'E[MeV]':>8} "
    header += f"{'GPR[%]':>8} {'RMSE[Gy]':>14} {'MAPE[%]':>10} {'RDE[%]':>10}"
    if metrics_config.heterogeneity:
        header += f" {'G_phi':>10} {'R':>10} {'H_phi':>12}"
    if metrics_config.glcm:
        header += f" {'GLCM_h':>10}"
    if metrics_config.intensity:
        header += f" {'entropy':>10} {'uniform':>10} {'IQR':>10} {'MAD':>10}"

    logger.info(header)
    logger.info("-" * 160)

    for r in sorted(results, key=lambda x: x.energy_mev):
        line = (
            f"{r.sample_id:<40} {r.energy_mev:>8.1f} "
            f"{r.gpr:>8.2f} {r.rmse:>14.9f} {r.mape:>10.4f} {r.rde:>10.4f}"
        )
        if metrics_config.heterogeneity and r.g_phi is not None:
            line += f" {r.g_phi:>10.4f} {r.r_roughness:>10.4f} {r.h_phi:>12.4f}"
        if metrics_config.glcm and r.glcm_homogeneity_mean is not None:
            line += f" {r.glcm_homogeneity_mean:>10.6f}"
        if metrics_config.intensity and r.intensity_entropy is not None:
            line += (
                f" {r.intensity_entropy:>10.4f} {r.intensity_uniformity:>10.6f}"
                f" {r.intensity_iqr:>10.4f} {r.intensity_mad:>10.4f}"
            )
        logger.info(line)

    logger.info("=" * 160)

    # Summary statistics
    gprs = [r.gpr for r in results]
    mapes = [r.mape for r in results]
    logger.info(f"Mean GPR : {np.mean(gprs):.2f}% ± {np.std(gprs):.2f}%")
    logger.info(f"Mean MAPE: {np.mean(mapes):.4f}% ± {np.std(mapes):.4f}%")

    if metrics_config.heterogeneity and results[0].g_phi is not None:
        g_phis = [r.g_phi for r in results]
        h_phis = [r.h_phi for r in results]
        logger.info(f"Mean G_φ : {np.mean(g_phis):.4f} ± {np.std(g_phis):.4f}")
        logger.info(f"Mean H_φ : {np.mean(h_phis):.4f} ± {np.std(h_phis):.4f}")
    if metrics_config.intensity and results[0].intensity_entropy is not None:
        entropies = [r.intensity_entropy for r in results]
        logger.info(f"Mean entropy: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")


# ---------------------------------------------------------------------------
# Scatter plots
# ---------------------------------------------------------------------------


def generate_correlation_plots(
    results: list,
    correlations: dict,
    output_dir: Path,
    metrics_config: MetricsConfig,
) -> None:
    """Generate scatter plots of model performance vs enabled texture metrics."""
    if len(results) < 2:
        logger.info("Not enough samples for scatter plots — skipping.")
        return

    mapes = np.array([r.mape for r in results])
    gprs = np.array([r.gpr for r in results])

    # Dynamically collect enabled texture data
    texture_data: dict[str, np.ndarray] = {}
    if metrics_config.heterogeneity and results[0].g_phi is not None:
        texture_data["G_phi"] = np.array([r.g_phi for r in results])
        texture_data["R"] = np.array([r.r_roughness for r in results])
        texture_data["H_phi"] = np.array([r.h_phi for r in results])
    if metrics_config.glcm and results[0].glcm_homogeneity_mean is not None:
        texture_data["GLCM_homogeneity"] = np.array(
            [r.glcm_homogeneity_mean for r in results]
        )
    if metrics_config.intensity and results[0].intensity_entropy is not None:
        texture_data["entropy"] = np.array([r.intensity_entropy for r in results])
        texture_data["uniformity"] = np.array([r.intensity_uniformity for r in results])
        texture_data["IQR"] = np.array([r.intensity_iqr for r in results])
        texture_data["MAD"] = np.array([r.intensity_mad for r in results])

    if not texture_data:
        logger.info("No texture metrics enabled — skipping scatter plots.")
        return

    n_tex = len(texture_data)
    performance_data = {"MAPE [%]": mapes, "GPR [%]": gprs}

    for perf_name, perf_vals in performance_data.items():
        fig, axes = plt.subplots(1, n_tex, figsize=(5 * n_tex, 5), squeeze=False)
        fig.suptitle(f"{perf_name} vs CT Texture Metrics", fontsize=14)

        for ax, (tex_name, tex_vals) in zip(axes[0], texture_data.items()):
            ax.scatter(tex_vals, perf_vals, alpha=0.6, edgecolors="k", linewidths=0.5)

            # Add trend line
            if np.std(tex_vals) > 1e-12 and np.std(perf_vals) > 1e-12:
                z = np.polyfit(tex_vals, perf_vals, 1)
                p = np.poly1d(z)
                x_line = np.linspace(tex_vals.min(), tex_vals.max(), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.7)

                # Annotate with correlation
                perf_key = perf_name.split(" ")[0]  # "MAPE" or "GPR"
                corr_info = correlations.get(perf_key, {}).get(tex_name, {})
                r_val = corr_info.get("r", float("nan"))
                p_val = corr_info.get("p", float("nan"))
                ax.annotate(
                    f"r={r_val:.3f}\np={p_val:.4f}",
                    xy=(0.05, 0.95),
                    xycoords="axes fraction",
                    verticalalignment="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
                )

            ax.set_xlabel(tex_name)
            ax.set_ylabel(perf_name)
            ax.grid(linestyle="--", linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        safe_name = perf_name.replace(" ", "_").replace("[", "").replace("]", "")
        fig_path = output_dir / f"correlation_{safe_name}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved correlation plot: %s", fig_path)


# ---------------------------------------------------------------------------
# Model evaluation results & figures (same as run_model.py)
# ---------------------------------------------------------------------------


def print_summary(results: list, total_time: float) -> None:
    """Print evaluation summary."""
    calc_times = [r.calc_time for r in results]
    gprs = [r.gpr for r in results]

    logger.info("Total elapsed time: %.2fs", total_time)
    logger.info("Average time per beamlet: %.4fs", np.mean(calc_times))

    worst_idx = int(np.argmin(gprs))
    best_idx = int(np.argmax(gprs))

    worst = results[worst_idx]
    best = results[best_idx]

    logger.info(
        "Worst case: Energy: %.2f MeV, "
        "Beamlet angles: (%.2f, %.2f) degrees, GPR: %.2f%%",
        worst.energy_mev,
        worst.beamlet_angles[0],
        worst.beamlet_angles[1],
        worst.gpr,
    )
    logger.info(
        "Best case: Energy: %.2f MeV, "
        "Beamlet angles: (%.2f, %.2f) degrees, GPR: %.2f%%",
        best.energy_mev,
        best.beamlet_angles[0],
        best.beamlet_angles[1],
        best.gpr,
    )


def generate_gpr_plot(
    results: list,
    output_path: Path,
    gamma_params: dict,
) -> None:
    """Generate and save GPR vs Energy plot."""
    if len(results) <= 1:
        logger.info("Skipping GPR plot - need more than one energy level")
        return

    energies = [r.energy_mev for r in results]
    gprs = [r.gpr for r in results]

    fig, ax = plt.subplots(dpi=300)
    ax.plot(energies, gprs, "o")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel(
        f"Gamma pass rate ({gamma_params['dose_percent_threshold']}%, "
        f"{gamma_params['distance_mm_threshold']}mm, "
        f"{gamma_params['lower_percent_dose_cutoff']}%) [%]"
    )
    ax.set_title("Gamma pass rate vs energy")
    ax.grid(linestyle="--", linewidth=0.5)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("GPR plot saved to %s", output_path)


def generate_publication_figures(
    model: DoTA3D_v3,
    results: list,
    test_data_path: Path,
    output_dir: Path,
    config: EvaluationConfig,
    device: torch.device,
) -> None:
    """Generate publication figures for best, worst, and mean cases."""
    scale = config.scale
    gamma_params = config.gamma_params

    gprs = [r.gpr for r in results]
    mean_gpr = np.mean(gprs)

    best_result = max(results, key=lambda r: r.gpr)
    worst_result = min(results, key=lambda r: r.gpr)
    closest_result = min(results, key=lambda r: abs(r.gpr - mean_gpr))

    cases = {
        "Best": best_result,
        "Worst": worst_result,
        "Closest_to_Mean": closest_result,
    }

    logger.info("Best GPR: %s with GPR: %.2f%%", best_result.sample_id, best_result.gpr)
    logger.info(
        "Worst GPR: %s with GPR: %.2f%%", worst_result.sample_id, worst_result.gpr
    )
    logger.info(
        "Closest to mean GPR: %s with GPR: %.2f%%",
        closest_result.sample_id,
        closest_result.gpr,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for desc, result in cases.items():
        logger.info(
            "Generating publication figure for %s case, id: %s", desc, result.sample_id
        )

        # Use cached data if available, otherwise reload
        if (
            result.input_data is not None
            and result.ground_truth is not None
            and result.prediction is not None
        ):
            x = result.input_data.to(device)
            y = result.ground_truth.to(device)
            y_pred = result.prediction.to(device)
            energy_mev = result.energy_mev
        else:
            x, energy, y = get_single_record(
                result.sample_id,
                test_data_path,
                scale=scale,
                normalize_flux=config.normalize_flux,
            )
            x, energy, y = x.to(device), energy.to(device), y.to(device)
            energy_mev = denormalize_energy(energy.item(), scale)

            with torch.no_grad():
                y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]

        # Prepare data for figure
        x_input = inverse_minmax(
            x.squeeze().cpu().numpy(),
            scale["min_ct"],
            scale["max_ct"],
        )
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

        # Calculate metrics
        rmse = calculate_rmse(to_gy(pred), to_gy(gt))
        mask = gt > 0.1 * np.max(gt)
        mape = calculate_pure_mape(pred[mask], gt[mask])

        scale_gpr = {"y_min": scale["min_ds"], "y_max": scale["max_ds"]}
        gpr_result = gamma_index_torch(
            y.unsqueeze(0),
            y_pred,
            scale=scale_gpr,
            gamma_params=gamma_params,
            resolution=config.resolution,
        )
        gpr = gpr_result[1][0] * 100

        logger.info(
            "%s case - RMSE: %.6f, MAPE: %.2f%%, GPR: %.2f%%", desc, rmse, mape, gpr
        )

        figure_path = output_dir / f"{desc}_E{energy_mev:.2f}MeV.svg"

        publication_figure(
            x_input,
            energy_mev,
            gt,
            pred,
            str(figure_path),
            rmse,
            mape,
            gpr,
            gamma_params=gamma_params,
        )


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------


def save_results_csv(
    results: list, output_path: Path, metrics_config: MetricsConfig
) -> None:
    """Save combined results to CSV for easy downstream analysis."""
    fieldnames = [
        "sample_id",
        "energy_mev",
        "beamlet_angle_0",
        "beamlet_angle_1",
        "gpr",
        "rmse",
        "mape",
        "rde",
    ]
    if metrics_config.heterogeneity:
        fieldnames += ["G_phi", "R", "H_phi"]
    if metrics_config.glcm:
        fieldnames += ["glcm_homogeneity_mean", "glcm_homogeneity_std"]
    if metrics_config.intensity:
        fieldnames += [
            "intensity_mean",
            "intensity_std",
            "intensity_iqr",
            "intensity_mad",
            "intensity_entropy",
            "intensity_uniformity",
            "intensity_skewness",
            "intensity_kurtosis",
        ]
    fieldnames.append("calc_time")

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            ba = (
                r.beamlet_angles
                if isinstance(r.beamlet_angles, (list, tuple))
                else (0.0, 0.0)
            )
            row: dict = {
                "sample_id": r.sample_id,
                "energy_mev": f"{r.energy_mev:.2f}",
                "beamlet_angle_0": f"{ba[0]:.2f}" if len(ba) > 0 else "0.0",
                "beamlet_angle_1": f"{ba[1]:.2f}" if len(ba) > 1 else "0.0",
                "gpr": f"{r.gpr:.4f}",
                "rmse": f"{r.rmse:.9f}",
                "mape": f"{r.mape:.4f}",
                "rde": f"{r.rde:.4f}",
                "calc_time": f"{r.calc_time:.4f}",
            }
            if metrics_config.heterogeneity and r.g_phi is not None:
                row["G_phi"] = f"{r.g_phi:.6f}"
                row["R"] = f"{r.r_roughness:.6f}"
                row["H_phi"] = f"{r.h_phi:.6f}"
            if metrics_config.glcm and r.glcm_homogeneity_mean is not None:
                row["glcm_homogeneity_mean"] = f"{r.glcm_homogeneity_mean:.6f}"
                row["glcm_homogeneity_std"] = f"{r.glcm_homogeneity_std:.6f}"
            if metrics_config.intensity and r.intensity_entropy is not None:
                row["intensity_mean"] = f"{r.intensity_mean:.6f}"
                row["intensity_std"] = f"{r.intensity_std:.6f}"
                row["intensity_iqr"] = f"{r.intensity_iqr:.6f}"
                row["intensity_mad"] = f"{r.intensity_mad:.6f}"
                row["intensity_entropy"] = f"{r.intensity_entropy:.6f}"
                row["intensity_uniformity"] = f"{r.intensity_uniformity:.6f}"
                row["intensity_skewness"] = f"{r.intensity_skewness:.6f}"
                row["intensity_kurtosis"] = f"{r.intensity_kurtosis:.6f}"
            writer.writerow(row)

    logger.info("Results CSV saved to %s", output_path)


def save_results_json(
    results: list,
    correlations: dict,
    output_path: Path,
    metrics_config: MetricsConfig,
) -> None:
    """Save detailed results and correlations to JSON."""
    payload: dict = {
        "metrics_enabled": {
            "heterogeneity": metrics_config.heterogeneity,
            "glcm": metrics_config.glcm,
            "intensity": metrics_config.intensity,
        },
        "samples": [],
        "correlations": correlations,
        "summary": {},
    }

    for r in results:
        ba = (
            r.beamlet_angles
            if isinstance(r.beamlet_angles, (list, tuple))
            else (0.0, 0.0)
        )
        sample_dict: dict = {
            "sample_id": r.sample_id,
            "energy_mev": float(r.energy_mev),
            "beamlet_angles": [float(a) for a in ba],
            "gpr": float(r.gpr),
            "rmse": float(r.rmse),
            "mape": float(r.mape),
            "rde": float(r.rde),
            "calc_time": float(r.calc_time),
        }
        if metrics_config.heterogeneity and r.g_phi is not None:
            sample_dict["G_phi"] = float(r.g_phi)
            sample_dict["R"] = float(r.r_roughness)
            sample_dict["H_phi"] = float(r.h_phi)
        if metrics_config.glcm and r.glcm_homogeneity_mean is not None:
            sample_dict["glcm_homogeneity_mean"] = float(r.glcm_homogeneity_mean)
            sample_dict["glcm_homogeneity_std"] = float(r.glcm_homogeneity_std)
        if metrics_config.intensity and r.intensity_entropy is not None:
            sample_dict["intensity_mean"] = float(r.intensity_mean)
            sample_dict["intensity_std"] = float(r.intensity_std)
            sample_dict["intensity_iqr"] = float(r.intensity_iqr)
            sample_dict["intensity_mad"] = float(r.intensity_mad)
            sample_dict["intensity_entropy"] = float(r.intensity_entropy)
            sample_dict["intensity_uniformity"] = float(r.intensity_uniformity)
            sample_dict["intensity_skewness"] = float(r.intensity_skewness)
            sample_dict["intensity_kurtosis"] = float(r.intensity_kurtosis)
        payload["samples"].append(sample_dict)

    # Summary statistics
    gprs = [r.gpr for r in results]
    mapes = [r.mape for r in results]
    payload["summary"] = {
        "n_samples": len(results),
        "mean_gpr": float(np.mean(gprs)),
        "std_gpr": float(np.std(gprs)),
        "mean_mape": float(np.mean(mapes)),
        "std_mape": float(np.std(mapes)),
    }

    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Results JSON saved to %s", output_path)


def generate_metrics_description(
    output_path: Path,
    metrics_config: MetricsConfig,
    gamma_params: dict,
) -> None:
    """Write a Markdown reference document describing every computed metric.

    The file uses LaTeX maths (compatible with GitHub / VS Code / Pandoc
    renderers) and is saved in the timestamped run directory for
    reproducibility.
    """
    lines: list[str] = []
    a = lines.append  # shorthand

    a("# Metrics Reference")
    a("")
    a("Mathematical definitions of every metric produced by this run.")
    a("Notation: $I(\\mathbf{r})$ denotes CT Hounsfield-unit intensity")
    a("at voxel position $\\mathbf{r}$; $\\hat{D}$ and $D$ are the")
    a("predicted and reference dose distributions, respectively.")
    a("")

    # ------------------------------------------------------------------
    # 1. Model performance metrics (always computed)
    # ------------------------------------------------------------------
    a("## 1 Model Performance Metrics")
    a("")

    a("### 1.1 Gamma Pass Rate (GPR)")
    a("")
    a("The gamma index at evaluation point $\\mathbf{r}_e$ is")
    a("")
    a("$$")
    a("\\Gamma(\\mathbf{r}_e) = \\min_{\\mathbf{r}_r}")
    a("\\sqrt{\\frac{|\\mathbf{r}_r - \\mathbf{r}_e|^2}" "{\\Delta d^2}")
    a("      + \\frac{[D(\\mathbf{r}_r) - \\hat{D}(\\mathbf{r}_e)]^2}" "{\\Delta D^2}}")
    a("$$")
    a("")
    dd = gamma_params.get("distance_mm_threshold", 2)
    dp = gamma_params.get("dose_percent_threshold", 2)
    lc = gamma_params.get("lower_percent_dose_cutoff", 10)
    a(f"where $\\Delta d = {dd}$ mm (distance-to-agreement criterion)")
    a(f"and $\\Delta D = {dp}\\%$ of the reference maximum dose")
    a(f"(dose-difference criterion), with a lower dose cutoff of ${lc}\\%$.")
    a("")
    a("The pass rate is the fraction of evaluated voxels satisfying $\\Gamma \\le 1$:")
    a("")
    a("$$")
    a("\\mathrm{GPR} = \\frac{1}{N}\\sum_{i=1}^{N}")
    a("\\mathbb{1}[\\Gamma(\\mathbf{r}_i) \\le 1]")
    a("\\times 100\\%")
    a("$$")
    a("")

    a("### 1.2 Root Mean Square Error (RMSE)")
    a("")
    a("$$")
    a("\\mathrm{RMSE} = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}")
    a("\\bigl(\\hat{D}_i - D_i\\bigr)^2}")
    a("$$")
    a("")
    a("Computed on dose values converted to Gray.")
    a("")

    a("### 1.3 Mean Absolute Percentage Error (MAPE)")
    a("")
    a("$$")
    a("\\mathrm{MAPE} = \\frac{100\\%}{N}\\sum_{i=1}^{N}")
    a("\\frac{|\\hat{D}_i - D_i|}{|D_i|}")
    a("$$")
    a("")
    a("Evaluated only on voxels where $\\hat{D}_i > 0.1 \\, \\max(\\hat{D})$")
    a("to suppress low-dose noise.")
    a("")

    a("### 1.4 Relative Dose Error (RDE)")
    a("")
    a("$$")
    a("\\mathrm{RDE} = \\frac{100\\%}{N}")
    a("\\;\\frac{\\|\\hat{D} - D\\|_1}{D_{\\max}}")
    a("$$")
    a("")
    a("where $\\|\\cdot\\|_1$ is the $L^1$ norm over all $N$ voxels.")
    a("")

    # ------------------------------------------------------------------
    # 2. Beam-aligned heterogeneity
    # ------------------------------------------------------------------
    if metrics_config.heterogeneity:
        a("## 2 Beam-Aligned Heterogeneity Metrics")
        a("")
        a("These metrics quantify the structural complexity of the CT volume")
        a("as *seen by the proton beamlet* through its fluence map $\\varphi$.")
        a("")

        a("### 2.1 Beam-Weighted Gradient Magnitude ($G_\\varphi$)")
        a("")
        a("$$")
        a("G_\\varphi = \\sum_{\\mathbf{r}}")
        a("w(\\mathbf{r})\\;\\|\\nabla I(\\mathbf{r})\\|_2")
        a("$$")
        a("")
        a("where")
        a("")
        a("$$")
        a("w(\\mathbf{r}) = \\frac{\\varphi(\\mathbf{r})}")
        a("{\\sum_{\\mathbf{r'}} \\varphi(\\mathbf{r'})}")
        a("$$")
        a("")
        a("are the normalised beamlet fluence weights. The gradient")
        a("$\\nabla I$ is computed via spacing-aware second-order central")
        a("finite differences in 3-D.")
        a("")

        a("### 2.2 Beam-Axis Roughness ($R$)")
        a("")
        a("$$")
        a("R = \\frac{1}{K-1}\\sum_{k=1}^{K-1}")
        a("|\\bar{I}_{k+1} - \\bar{I}_k|")
        a("$$")
        a("")
        a("where $\\bar{I}_k$ is the lateral-mean HU at depth slice $k$")
        a("along the beam axis. $R$ captures tissue-composition changes")
        a("along the proton path.")
        a("")

        a("### 2.3 Combined Heterogeneity ($H_\\varphi$)")
        a("")
        a("$$")
        a("H_\\varphi = G_\\varphi \\times R")
        a("$$")
        a("")
        a("A single scalar combining gradient-based and roughness-based")
        a("information about beamlet-path complexity.")
        a("")

    # ------------------------------------------------------------------
    # 3. GLCM homogeneity
    # ------------------------------------------------------------------
    if metrics_config.glcm:
        a("## 3 GLCM Homogeneity")
        a("")
        a("The Gray-Level Co-occurrence Matrix $P(i,j \\mid d, \\theta)$")
        a("records the joint frequency of grey-level pairs $(i, j)$ at")
        a("pixel offset $(d, \\theta)$. From the normalised matrix")
        a("$p(i,j) = P(i,j) / \\sum P$ the **Inverse Difference Moment**")
        a("(IDM) is computed as")
        a("")
        a("$$")
        a("\\mathrm{IDM} = \\sum_{i,j}")
        a("\\frac{p(i,j)}{1 + (i - j)^2}")
        a("$$")
        a("")
        a("The score is averaged over all configured offsets")
        a("$(d, \\theta)$ and over evenly sampled axial slices of the")
        a("CT volume. Higher IDM values indicate greater local")
        a("homogeneity (neighbouring voxels have similar HU values).")
        a("")

    # ------------------------------------------------------------------
    # 4. Global intensity heterogeneity
    # ------------------------------------------------------------------
    if metrics_config.intensity:
        a("## 4 Global Intensity Heterogeneity")
        a("")
        a("These metrics characterise the *intensity distribution* of the")
        a("CT volume within the ROI mask, without regard for spatial")
        a("arrangement. Let $\\{I_i\\}_{i=1}^{N}$ denote the $N$ masked")
        a("voxel intensities.")
        a("")

        a("### 4.1 Descriptive Statistics")
        a("")
        a("$$")
        a("\\mu = \\frac{1}{N}\\sum_i I_i,")
        a("\\qquad")
        a("\\sigma = \\sqrt{\\frac{1}{N}\\sum_i (I_i - \\mu)^2}")
        a("$$")
        a("")

        a("### 4.2 Inter-Quartile Range (IQR)")
        a("")
        a("$$")
        a("\\mathrm{IQR} = Q_{75} - Q_{25}")
        a("$$")
        a("")
        a("where $Q_p$ is the $p$-th percentile of $\\{I_i\\}$.")
        a("")

        a("### 4.3 Median Absolute Deviation (MAD)")
        a("")
        a("$$")
        a("\\mathrm{MAD} = \\mathrm{median}_i\\,|I_i - \\tilde{I}|")
        a("$$")
        a("")
        a("where $\\tilde{I} = \\mathrm{median}(\\{I_i\\})$.")
        a("A robust measure of dispersion, less sensitive to outliers")
        a("than $\\sigma$.")
        a("")

        a("### 4.4 Shannon Entropy")
        a("")
        a("The HU histogram is binned with a fixed bin width")
        a("$\\Delta_{\\mathrm{HU}}$ and normalised to a probability")
        a("distribution $\\{p_b\\}$:")
        a("")
        a("$$")
        a("H = -\\sum_{b} p_b \\ln p_b")
        a("$$")
        a("")
        a("Higher entropy indicates greater diversity in tissue")
        a("composition within the ROI.")
        a("")

        a("### 4.5 Uniformity (Energy)")
        a("")
        a("$$")
        a("U = \\sum_{b} p_b^{\\,2}")
        a("$$")
        a("")
        a("The sum of squared bin probabilities. $U = 1$ for a")
        a("perfectly uniform volume; lower values indicate")
        a("heterogeneous tissue composition.")
        a("")

        a("### 4.6 Skewness")
        a("")
        a("$$")
        a("\\gamma_1 = \\frac{m_3}{\\sigma^3}")
        a("$$")
        a("")
        a("where $m_3 = \\frac{1}{N}\\sum_i (I_i - \\mu)^3$ is the")
        a("third central moment. Positive skewness indicates a tail")
        a("towards higher HU (e.g. bone), negative towards lower HU.")
        a("")

        a("### 4.7 Excess Kurtosis")
        a("")
        a("$$")
        a("\\kappa = \\frac{m_4}{\\sigma^4} - 3")
        a("$$")
        a("")
        a("where $m_4 = \\frac{1}{N}\\sum_i (I_i - \\mu)^4$ is the")
        a("fourth central moment. Gaussian distributions yield")
        a("$\\kappa = 0$; positive excess kurtosis signals heavy tails")
        a("(e.g. high-density implants or metal artefacts).")
        a("")

    # ------------------------------------------------------------------
    # 5. Correlation analysis
    # ------------------------------------------------------------------
    a("## 5 Correlation Analysis")
    a("")
    a("Pearson product-moment correlation coefficient between each")
    a("enabled texture metric $T$ and each model performance metric $P$:")
    a("")
    a("$$")
    a("r_{PT} = \\frac{\\sum_{s}(P_s - \\bar{P})(T_s - \\bar{T})}")
    a("{\\sqrt{\\sum_{s}(P_s - \\bar{P})^2}")
    a("\\;\\sqrt{\\sum_{s}(T_s - \\bar{T})^2}}")
    a("$$")
    a("")
    a("where $s$ indexes samples. Two-tailed $p$-values are reported")
    a("with significance thresholds $*\\; p<0.05$,")
    a("$**\\; p<0.01$, $***\\; p<0.001$.")
    a("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Metrics description saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    model_name: Annotated[
        Optional[str], typer.Argument(help="Name of the model directory")
    ] = None,
    test_data: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the directory with input data to evaluate"),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to YAML configuration file"),
    ] = None,
    downsampling_method: Annotated[
        Optional[str],
        typer.Option(help="Downsampling method: 'interpolation' or 'avg_pooling'"),
    ] = None,
    model_fname: Annotated[Optional[str], typer.Option(help="Model filename")] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index (-1 for CPU)")
    ] = None,
    enable_heterogeneity: Annotated[
        Optional[bool],
        typer.Option(
            help="Enable beam-aligned heterogeneity metrics (G_phi, R, H_phi)"
        ),
    ] = None,
    enable_glcm: Annotated[
        Optional[bool],
        typer.Option(help="Enable GLCM homogeneity metrics"),
    ] = None,
    enable_intensity: Annotated[
        Optional[bool],
        typer.Option(
            help="Enable global intensity heterogeneity metrics "
            "(entropy, uniformity, IQR, MAD, skewness, kurtosis)"
        ),
    ] = None,
    no_progress: Annotated[
        Optional[bool], typer.Option(help="Disable progress bar")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option("--verbose", "-v", help="Enable verbose output")
    ] = None,
) -> None:
    """Run DoTA model evaluation combined with CT texture / heterogeneity
    analysis, then compute Pearson correlations between model performance
    and CT complexity scores.

    Three texture-metric families can be toggled independently via CLI flags
    (``--enable-heterogeneity``, ``--enable-glcm``, ``--enable-intensity``)
    or through the ``metrics`` section in the YAML config.  All three are
    enabled by default.
    """
    # ---- Load YAML config ----
    yaml_cfg: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_cfg = load_yaml_config(config_path)

    # ---- Merge CLI + YAML (CLI wins) ----
    model_name = model_name or yaml_cfg.get("model_name")
    test_data = test_data or (
        Path(yaml_cfg["test_data"]) if "test_data" in yaml_cfg else None
    )
    downsampling_method = downsampling_method or yaml_cfg.get(
        "downsampling_method", "interpolation"
    )
    model_fname = model_fname or yaml_cfg.get("model_fname", "best_model.pth")
    device_index = (
        device_index if device_index is not None else yaml_cfg.get("device_index", 0)
    )
    no_progress = (
        no_progress if no_progress is not None else yaml_cfg.get("no_progress", False)
    )
    verbose = verbose if verbose is not None else yaml_cfg.get("verbose", False)

    # ---- Build metrics toggle (CLI > YAML > default=True) ----
    metrics_yaml = yaml_cfg.get("metrics", {})
    metrics_config = MetricsConfig(
        heterogeneity=(
            enable_heterogeneity
            if enable_heterogeneity is not None
            else metrics_yaml.get("heterogeneity", True)
        ),
        glcm=(
            enable_glcm if enable_glcm is not None else metrics_yaml.get("glcm", True)
        ),
        intensity=(
            enable_intensity
            if enable_intensity is not None
            else metrics_yaml.get("intensity", True)
        ),
    )

    # Validate required arguments
    if model_name is None:
        raise typer.BadParameter(
            "MODEL_NAME is required (via CLI argument or YAML config)"
        )
    if test_data is None:
        raise typer.BadParameter(
            "TEST_DATA is required (via CLI argument or YAML config)"
        )

    # ---- Setup run directory & logging ----
    runs_dir = Path(yaml_cfg.get("runs_dir", "runs"))
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir
    run_dir = setup_run_directory(runs_dir)
    log_file = setup_logging(run_dir, verbose=verbose)

    # Copy config for reproducibility
    if config_path is not None:
        shutil.copy2(config_path, run_dir / config_path.name)
        logger.info("Config file copied to %s", run_dir / config_path.name)

    logger.info("Run directory: %s", run_dir)
    logger.info("Log file: %s", log_file)

    # ---- Validate downsampling method ----
    if downsampling_method not in ("interpolation", "avg_pooling"):
        raise typer.BadParameter(
            f"Invalid downsampling method: {downsampling_method}. "
            "Must be 'interpolation' or 'avg_pooling'."
        )

    # ---- Setup paths ----
    model_hub = PROJECT_ROOT / "models"
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"

    if not test_data.is_absolute():
        test_data = PROJECT_ROOT / test_data

    if not test_data.exists():
        raise typer.BadParameter(f"Test data directory not found: {test_data}")
    if not model_path.exists():
        raise typer.BadParameter(f"Model file not found: {model_path}")
    if not hyperparams_path.exists():
        raise typer.BadParameter(f"Hyperparams file not found: {hyperparams_path}")

    # ---- Sub-configs ----
    scale_cfg = yaml_cfg.get("scale", DEFAULT_SCALE.copy())
    gamma_cfg = yaml_cfg.get("gamma", DEFAULT_GAMMA_PARAMS.copy())
    heterogeneity_cfg = yaml_cfg.get("heterogeneity", {})
    glcm_cfg = yaml_cfg.get("glcm", {})
    intensity_cfg = yaml_cfg.get("intensity", {})

    # ---- Log run configuration ----
    logger.info("=" * 60)
    logger.info("RUN CONFIGURATION")
    logger.info("=" * 60)
    logger.info("Model name          : %s", model_name)
    logger.info("Model file          : %s", model_fname)
    logger.info("Test data           : %s", test_data)
    logger.info("Downsampling method : %s", downsampling_method)
    logger.info("Gamma params        : %s", gamma_cfg)
    logger.info(
        "Enabled metrics     : heterogeneity=%s  glcm=%s  intensity=%s",
        metrics_config.heterogeneity,
        metrics_config.glcm,
        metrics_config.intensity,
    )
    if metrics_config.heterogeneity:
        logger.info("Heterogeneity cfg   : %s", heterogeneity_cfg)
    if metrics_config.glcm:
        logger.info("GLCM cfg            : %s", glcm_cfg)
    if metrics_config.intensity:
        logger.info("Intensity cfg       : %s", intensity_cfg)
    logger.info("=" * 60)

    # ---- Setup device & model ----
    device = get_device(device_index)
    logger.info("Using device: %s", device)

    model = load_model(model_path, hyperparams_path, device)

    # ---- Build evaluation config ----
    eval_config = EvaluationConfig()
    eval_config.scale = scale_cfg
    eval_config.gamma_params = gamma_cfg

    # ---- Discover samples ----
    files = os.listdir(test_data)
    sample_ids = np.unique([f.split("_")[0] for f in files if "_" in f]).tolist()
    logger.info("Found %d unique samples for evaluation", len(sample_ids))

    if not sample_ids:
        logger.error("No samples found in test data directory")
        raise typer.Exit(code=1)

    # ---- Evaluate all samples ----
    total_start = perf_counter()
    results = evaluate_all_samples(
        model=model,
        sample_ids=sample_ids,
        test_data_path=test_data,
        config=eval_config,
        device=device,
        downsampling_method=downsampling_method,
        heterogeneity_cfg=heterogeneity_cfg,
        glcm_cfg=glcm_cfg,
        intensity_cfg=intensity_cfg,
        metrics_config=metrics_config,
        show_progress=not no_progress,
    )
    total_time = perf_counter() - total_start

    if not results:
        logger.error("No samples were successfully evaluated")
        raise typer.Exit(code=1)

    # ---- Standard results table (same as run_model.py) ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS TABLE")
    logger.info("=" * 60)
    print_results_table(
        energies=[r.energy_mev for r in results],
        gprs_calc=[r.gpr for r in results],
        rmses=[r.rmse for r in results],
        mapes=[r.mape for r in results],
        rdes=[r.rde for r in results],
        gamma_params=eval_config.gamma_params,
        beamlet_angles=[list(r.beamlet_angles) for r in results],
        logger=logger,
    )

    # ---- Combined results table (with texture metrics) ----
    print_combined_results_table(results, gamma_cfg, metrics_config)

    # ---- Correlation analysis ----
    correlations = compute_correlations(results, metrics_config)
    print_correlation_table(correlations)

    # ---- Save outputs ----
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    save_results_csv(results, run_dir / "results.csv", metrics_config)
    save_results_json(results, correlations, run_dir / "results.json", metrics_config)
    generate_metrics_description(
        run_dir / "metrics_reference.md", metrics_config, eval_config.gamma_params
    )
    generate_correlation_plots(results, correlations, figures_dir, metrics_config)

    generate_gpr_plot(
        results=results,
        output_path=figures_dir / "gpr_vs_energy.png",
        gamma_params=eval_config.gamma_params,
    )

    generate_publication_figures(
        model=model,
        results=results,
        test_data_path=test_data,
        output_dir=figures_dir,
        config=eval_config,
        device=device,
    )

    # ---- Summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    print_summary(results, total_time)
    logger.info("Total elapsed time : %.2fs", total_time)
    logger.info("Samples evaluated  : %d / %d", len(results), len(sample_ids))
    logger.info(
        "Avg time per sample: %.4fs",
        np.mean([r.calc_time for r in results]),
    )
    logger.info("")
    logger.info("Evaluation complete!  Results saved to: %s", run_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
