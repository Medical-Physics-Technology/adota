"""
DoTA Model Evaluation Script

A command-line tool for running DoTA model inference on dose prediction tasks.
"""

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from scipy.stats import pearsonr
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
from src.adota.utils import count_parameters_per_block, count_total_parameters, load_model
from src.figures.single_beam import publication_figure
from src.loaders.dir_based import get_single_record, save_prediction
from src.loaders.utils import validate_inputs
from src.metrics.classic import (
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from src.metrics.gamma_pass_rate import gamma_index_torch
from src.tables.results import print_results_table
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)

from src.schemas.configs import EvaluationConfig
from src.schemas.results import EvaluationResult

app = typer.Typer(help="DoTA Model Evaluation Tool")


def evaluate_single_sample(
    model: DoTA3D_v3,
    sample_id: str,
    test_data_path: Path,
    config: EvaluationConfig,
    device: torch.device,
    downsampling_method: str,
    save_predictions: bool = True,
) -> EvaluationResult:
    """Evaluate a single sample and compute metrics.

    Args:
        model: The loaded DoTA model.
        sample_id: Unique identifier for the sample.
        test_data_path: Path to the test data directory.
        config: Evaluation configuration.
        device: Target device for computation.
        downsampling_method: Method for downsampling.
        save_predictions: Whether to save prediction to disk.

    Returns:
        EvaluationResult containing all metrics and optionally cached tensors.
    """
    scale = config.scale
    gamma_params = config.gamma_params

    start_time = perf_counter()

    # Load data
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

    # Save prediction if requested
    if save_predictions:
        save_prediction(y_pred, sample_id, test_data_path, scale)

    calc_time = perf_counter() - start_time

    # Convert to numpy for metrics
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

    # Calculate metrics
    rmse = calculate_rmse(to_gy(y_pred_np), to_gy(y_np))

    # MAPE with threshold mask
    mask = y_pred_np > 0.1 * np.max(y_pred_np)
    mape = calculate_pure_mape(y_np[mask], y_pred_np[mask])

    rde = calculate_relative_dose_error(to_gy(y_pred_np), to_gy(y_np))

    # Gamma pass rate
    scale_gpr = {"y_min": scale["min_ds"], "y_max": scale["max_ds"]}
    gpr_result = gamma_index_torch(
        y.unsqueeze(0),
        y_pred,
        scale=scale_gpr,
        gamma_params=gamma_params,
        resolution=config.resolution,
    )
    gpr = gpr_result[1][0] * 100

    return EvaluationResult(
        sample_id=sample_id,
        energy_mev=energy_mev,
        beamlet_angles=tuple(ba) if isinstance(ba, list) else ba,
        gpr=gpr,
        rmse=rmse,
        mape=mape,
        rde=rde,
        calc_time=calc_time,
        prediction=y_pred.cpu(),
        ground_truth=y.cpu(),
        input_data=x.cpu(),
    )


def evaluate_samples(
    model: DoTA3D_v3,
    sample_ids: list,
    test_data_path: Path,
    config: EvaluationConfig,
    device: torch.device,
    downsampling_method: str,
    show_progress: bool = True,
) -> list:
    """Evaluate multiple samples.

    Args:
        model: The loaded DoTA model.
        sample_ids: List of sample IDs to evaluate.
        test_data_path: Path to the test data directory.
        config: Evaluation configuration.
        device: Target device for computation.
        downsampling_method: Method for downsampling.
        show_progress: Whether to show a progress bar.

    Returns:
        List of EvaluationResult objects.
    """
    results = []
    iterator = (
        tqdm(sample_ids, desc="Evaluating samples") if show_progress else sample_ids
    )

    for i, sample_id in enumerate(iterator):
        if i == 169:
            logger.debug(f"Debug breakpoint: sample_id={sample_id}")
            continue

        result = evaluate_single_sample(
            model=model,
            sample_id=sample_id,
            test_data_path=test_data_path,
            config=config,
            device=device,
            downsampling_method=downsampling_method,
        )
        results.append(result)

        if show_progress:
            iterator.set_postfix(
                energy=f"{result.energy_mev:.1f}MeV",
                gpr=f"{result.gpr:.1f}%",
            )

    return results


def generate_gpr_plot(
    results: list,
    output_path: Path,
    gamma_params: dict,
) -> None:
    """Generate and save GPR vs Energy plot.

    Args:
        results: List of evaluation results.
        output_path: Path to save the plot.
        gamma_params: Gamma parameters for labeling.
    """
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
    logger.info(f"GPR plot saved to {output_path}")


def density_variability_vs_gpr(
    results: list,
    output_dir: Path,
    config: EvaluationConfig,
    gamma_params: dict,
) -> None:
    """Study the correlation between CT density-profile variability and GPR.

    Two variability metrics are computed for the average CT density profile
    (mean HU per depth slice) of every sample:

    1. **Total Variation (TV)** – sum of absolute differences between
       consecutive depth slices:
       ``TV = Σ|ρ̄_{k+1} − ρ̄_k|``.
       Captures the roughness / heterogeneity of the tissue composition
       along the beam path.

    2. **Coefficient of Variation (CV)** – ratio of the standard deviation
       to the absolute mean of the density profile:
       ``CV = σ(ρ̄) / |μ(ρ̄)|``.
       Captures the overall relative spread of density values.

    The function produces a single figure with two subplots:
    (a) TV vs GPR and (b) CV vs GPR, each annotated with the Pearson
    correlation coefficient.

    Args:
        results: List of EvaluationResult objects (with cached tensors).
        output_dir: Directory where the figure will be stored.
        config: EvaluationConfig (used for de-normalisation scale).
        gamma_params: Gamma parameters (for axis labelling).
    """
    scale = config.scale
    output_dir.mkdir(parents=True, exist_ok=True)

    tv_values: list[float] = []
    cv_values: list[float] = []
    gpr_values: list[float] = []
    sample_ids: list[str] = []

    for result in results:
        if result.input_data is None:
            continue

        ct_norm = result.input_data[0].numpy()  # (D, H, W)
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])
        avg_density = ct_hu.mean(axis=(1, 2))  # (D,)

        # Metric 1: Total Variation
        tv = float(np.sum(np.abs(np.diff(avg_density))))

        # Metric 2: Coefficient of Variation
        mu = np.mean(avg_density)
        sigma = np.std(avg_density)
        cv = float(sigma / np.abs(mu)) if np.abs(mu) > 1e-9 else 0.0

        tv_values.append(tv)
        cv_values.append(cv)
        gpr_values.append(result.gpr)
        sample_ids.append(result.sample_id)

    if len(tv_values) < 3:
        logger.info("Skipping density-variability study – fewer than 3 samples")
        return

    tv_arr = np.array(tv_values)
    cv_arr = np.array(cv_values)
    gpr_arr = np.array(gpr_values)

    # Pearson correlations
    r_tv, p_tv = pearsonr(tv_arr, gpr_arr)
    r_cv, p_cv = pearsonr(cv_arr, gpr_arr)

    logger.info(
        f"Density variability vs GPR – "
        f"TV: r = {r_tv:.4f} (p = {p_tv:.4e}), "
        f"CV: r = {r_cv:.4f} (p = {p_cv:.4e})"
    )

    gpr_label = (
        f"GPR ({gamma_params['dose_percent_threshold']}%, "
        f"{gamma_params['distance_mm_threshold']}mm, "
        f"{gamma_params['lower_percent_dose_cutoff']}%) [%]"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # (a) TV vs GPR
    ax = axes[0]
    ax.scatter(tv_arr, gpr_arr, s=20, alpha=0.7, edgecolors="k", linewidths=0.3)
    # linear fit
    z = np.polyfit(tv_arr, gpr_arr, 1)
    x_fit = np.linspace(tv_arr.min(), tv_arr.max(), 100)
    ax.plot(
        x_fit,
        np.polyval(z, x_fit),
        "r--",
        linewidth=1.0,
        label=f"fit (r = {r_tv:.3f}, p = {p_tv:.2e})",
    )
    ax.set_xlabel("Total Variation of density profile [HU]")
    ax.set_ylabel(gpr_label)
    ax.set_title("(a) Total Variation vs GPR")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5)

    # (b) CV vs GPR
    ax = axes[1]
    ax.scatter(cv_arr, gpr_arr, s=20, alpha=0.7, edgecolors="k", linewidths=0.3)
    z = np.polyfit(cv_arr, gpr_arr, 1)
    x_fit = np.linspace(cv_arr.min(), cv_arr.max(), 100)
    ax.plot(
        x_fit,
        np.polyval(z, x_fit),
        "r--",
        linewidth=1.0,
        label=f"fit (r = {r_cv:.3f}, p = {p_cv:.2e})",
    )
    ax.set_xlabel("Coefficient of Variation of density profile")
    ax.set_ylabel(gpr_label)
    ax.set_title("(b) Coefficient of Variation vs GPR")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(
        output_dir / "density_variability_vs_gpr.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        f"Density variability vs GPR figure saved to "
        f"{output_dir / 'density_variability_vs_gpr.png'}"
    )


def advanced_metrics_and_figures(
    results: list,
    output_dir: Path,
    config: EvaluationConfig,
) -> dict:
    """Calculate IDD-based advanced metrics and generate depth-profile figures.

    For every sample the function computes:
    - Integrated Depth Dose (IDD) for ground truth and prediction,
    - Average CT density per depth slice,
    - Slice-wise IDD difference (predicted − ground truth),
    - Pearson correlation between the average CT density profile
      and the IDD difference profile.

    A per-sample figure and an aggregate summary figure are saved
    to ``output_dir``.

    Args:
        results: List of EvaluationResult objects (with cached tensors).
        output_dir: Directory where figures will be stored.
        config: EvaluationConfig (used for de-normalisation scale).

    Returns:
        Dictionary mapping sample_id → Pearson-r value.
    """
    scale = config.scale
    output_dir.mkdir(parents=True, exist_ok=True)

    correlations: dict[str, float] = {}

    # Per-sample aggregates for the summary figure
    summary_gpr: list[float] = []
    summary_r: list[float] = []
    summary_tv: list[float] = []
    summary_bp_grad: list[float] = []  # density gradient near BP
    summary_bp_idd_rel_err: list[float] = []  # mean |rel IDD error| near BP

    BP_HALF_WINDOW = 5  # slices each side of Bragg peak

    for result in results:
        if (
            result.input_data is None
            or result.ground_truth is None
            or result.prediction is None
        ):
            logger.warning(
                f"Skipping {result.sample_id} – cached tensors not available"
            )
            continue

        # --- de-normalise to physical units ---------------------------------
        ct_norm = result.input_data[0].numpy()  # (D, H, W), normalised
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])  # HU values

        gt = inverse_minmax(
            result.ground_truth.numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )  # dose grid – keep original shape
        pred = inverse_minmax(
            result.prediction.numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )  # prediction – keep original shape

        # Squeeze to 3-D (D, H, W) regardless of leading singleton dims
        gt = gt.squeeze()
        pred = pred.squeeze()

        logger.debug(f"Calculating advanced metrics for sample {result.sample_id}...")
        logger.debug(
            "Shapes: CT: {}, GT dose: {}, Pred dose: {}".format(
                ct_hu.shape, gt.shape, pred.shape
            )
        )
        # --- IDD: sum over lateral dimensions per depth slice ---------------
        idd_gt = gt.sum(axis=(1, 2))  # (D,)
        idd_pred = pred.sum(axis=(1, 2))  # (D,)

        # --- average CT density per depth slice -----------------------------
        avg_density = ct_hu.mean(axis=(1, 2))  # (D,)

        # --- IDD difference (pred − gt) -------------------------------------
        idd_diff = idd_pred - idd_gt  # (D,)

        # --- Pearson correlation: avg density  vs  IDD difference -----------
        if np.std(avg_density) > 0 and np.std(idd_diff) > 0:
            r, p = pearsonr(avg_density, idd_diff)
        else:
            r, p = 0.0, 1.0

        correlations[result.sample_id] = r
        logger.info(
            f"Sample {result.sample_id}: "
            f"Pearson r(avg_density, IDD_diff) = {r:.4f} (p = {p:.4e})"
        )

        # --- normalise for plotting -----------------------------------------
        idd_gt_max = np.max(np.abs(idd_gt)) if np.max(np.abs(idd_gt)) > 0 else 1.0
        idd_gt_norm = idd_gt / idd_gt_max
        idd_pred_norm = idd_pred / idd_gt_max  # same reference for comparability

        density_max = (
            np.max(np.abs(avg_density)) if np.max(np.abs(avg_density)) > 0 else 1.0
        )
        density_norm = avg_density / density_max

        bp_idx = int(np.argmax(idd_gt))

        # --- per-sample metrics for summary figure --------------------------
        n_slices = len(avg_density)
        lo = max(0, bp_idx - BP_HALF_WINDOW)
        hi = min(n_slices, bp_idx + BP_HALF_WINDOW + 1)

        # Total Variation of full density profile
        tv = float(np.sum(np.abs(np.diff(avg_density))))

        # Density gradient magnitude in BP neighbourhood
        bp_density = avg_density[lo:hi]
        bp_grad = (
            float(np.sum(np.abs(np.diff(bp_density)))) if len(bp_density) > 1 else 0.0
        )

        # Mean |relative IDD error| in BP neighbourhood
        idd_gt_bp = idd_gt[lo:hi]
        idd_diff_bp = idd_diff[lo:hi]
        # avoid division by zero: only where GT IDD is appreciable
        bp_mask = np.abs(idd_gt_bp) > 1e-6 * np.max(np.abs(idd_gt))
        if np.any(bp_mask):
            bp_idd_rel_err = float(
                np.mean(np.abs(idd_diff_bp[bp_mask] / idd_gt_bp[bp_mask]))
            )
        else:
            bp_idd_rel_err = 0.0

        summary_gpr.append(result.gpr)
        summary_r.append(r)
        summary_tv.append(tv)
        summary_bp_grad.append(bp_grad)
        summary_bp_idd_rel_err.append(bp_idd_rel_err)

        # --- per-sample figure ----------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

        # top: depth profiles
        ax = axes[0]
        ax.plot(density_norm, label="CT (GT), normalized", color="tab:blue")
        ax.plot(idd_gt_norm, label="Dose (GT), normalized", color="tab:orange")
        ax.plot(idd_pred_norm, label="Dose (Pred), normalized", color="tab:green")
        ax.axvline(
            x=bp_idx,
            color="red",
            linestyle="--",
            label=f"BP (slice {bp_idx})",
        )
        ax.set_ylabel("Normalized value")
        ax.set_title(
            f"Depth profiles – {result.sample_id}\n"
            f"E = {result.energy_mev:.1f} MeV, "
            f"GPR = {result.gpr:.1f}%"
        )
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)

        # bottom: IDD difference vs average density
        ax = axes[1]
        ax.plot(
            idd_diff / idd_gt_max,
            label="IDD diff (Pred − GT), normalized",
            color="tab:red",
        )
        ax.plot(
            density_norm,
            label="Avg CT density, normalized",
            color="tab:blue",
            alpha=0.6,
        )
        ax.set_xlabel("Depth [voxels]")
        ax.set_ylabel("Normalized value")
        ax.set_title(f"Pearson r = {r:.4f}")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"depth_profile_{result.sample_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # --- aggregate summary figure (2×2) ------------------------------------
    if len(summary_gpr) >= 3:
        gpr_arr = np.array(summary_gpr)
        r_arr = np.array(summary_r)
        tv_arr = np.array(summary_tv)
        bp_grad_arr = np.array(summary_bp_grad)
        bp_rel_err_arr = np.array(summary_bp_idd_rel_err)

        fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
        fig.suptitle(
            f"Tissue heterogeneity vs model performance  "
            f"(N = {len(gpr_arr)},  "
            f"γ: {config.gamma_params['dose_percent_threshold']}%/"
            f"{config.gamma_params['distance_mm_threshold']}mm)",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        gpr_label = (
            f"GPR ({config.gamma_params['dose_percent_threshold']}%, "
            f"{config.gamma_params['distance_mm_threshold']}mm, "
            f"{config.gamma_params['lower_percent_dose_cutoff']}%) [%]"
        )

        # ---- (a) Histogram of Pearson r(density, IDD error) ----------------
        ax = axes[0, 0]
        ax.hist(
            r_arr, bins=30, color="steelblue", edgecolor="k", linewidth=0.4, alpha=0.85
        )
        ax.axvline(x=0, color="k", linewidth=0.8, linestyle="-")
        ax.axvline(
            x=np.mean(r_arr),
            color="red",
            linewidth=1.2,
            linestyle="--",
            label=f"mean = {np.mean(r_arr):.3f}",
        )
        ax.axvline(
            x=np.median(r_arr),
            color="orange",
            linewidth=1.2,
            linestyle="-.",
            label=f"median = {np.median(r_arr):.3f}",
        )
        ax.set_xlabel("Pearson r  (avg CT density vs IDD error)")
        ax.set_ylabel("Count")
        ax.set_title("(a) Distribution of density–error correlation")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)

        # ---- (b) GPR vs Total Variation of density profile -----------------
        ax = axes[0, 1]
        r_tv, p_tv = pearsonr(tv_arr, gpr_arr)
        ax.scatter(tv_arr, gpr_arr, s=18, alpha=0.6, edgecolors="k", linewidths=0.3)
        z = np.polyfit(tv_arr, gpr_arr, 1)
        x_fit = np.linspace(tv_arr.min(), tv_arr.max(), 100)
        ax.plot(
            x_fit,
            np.polyval(z, x_fit),
            "r--",
            linewidth=1.0,
            label=f"fit  (r = {r_tv:.3f}, p = {p_tv:.2e})",
        )
        ax.set_xlabel("Total Variation of density profile [HU]")
        ax.set_ylabel(gpr_label)
        ax.set_title("(b) Full-path heterogeneity vs GPR")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)

        # ---- (c) GPR vs local density gradient at Bragg peak ---------------
        ax = axes[1, 0]
        r_bg, p_bg = pearsonr(bp_grad_arr, gpr_arr)
        ax.scatter(
            bp_grad_arr, gpr_arr, s=18, alpha=0.6, edgecolors="k", linewidths=0.3
        )
        z = np.polyfit(bp_grad_arr, gpr_arr, 1)
        x_fit = np.linspace(bp_grad_arr.min(), bp_grad_arr.max(), 100)
        ax.plot(
            x_fit,
            np.polyval(z, x_fit),
            "r--",
            linewidth=1.0,
            label=f"fit  (r = {r_bg:.3f}, p = {p_bg:.2e})",
        )
        ax.set_xlabel(
            f"Density gradient near Bragg peak "
            f"(TV in ±{BP_HALF_WINDOW} slices) [HU]"
        )
        ax.set_ylabel(gpr_label)
        ax.set_title("(c) Bragg-peak-local heterogeneity vs GPR")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)

        # ---- (d) GPR vs mean |rel IDD error| near BP, coloured by grad ----
        ax = axes[1, 1]
        r_be, p_be = pearsonr(bp_rel_err_arr, gpr_arr)
        sc = ax.scatter(
            bp_rel_err_arr * 100,
            gpr_arr,
            c=bp_grad_arr,
            cmap="plasma",
            s=20,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.3,
        )
        z = np.polyfit(bp_rel_err_arr * 100, gpr_arr, 1)
        x_fit = np.linspace(bp_rel_err_arr.min() * 100, bp_rel_err_arr.max() * 100, 100)
        ax.plot(
            x_fit,
            np.polyval(z, x_fit),
            "r--",
            linewidth=1.0,
            label=f"fit  (r = {r_be:.3f}, p = {p_be:.2e})",
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(
            f"BP-local density gradient [HU]",
            fontsize=9,
        )
        ax.set_xlabel(
            f"Mean |relative IDD error| near BP " f"(±{BP_HALF_WINDOW} slices) [%]"
        )
        ax.set_ylabel(gpr_label)
        ax.set_title("(d) Bragg-peak IDD accuracy vs GPR")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(
            output_dir / "density_heterogeneity_vs_performance_summary.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        logger.info(
            f"Summary figure saved. "
            f"Pearson r(density, IDD_err): mean={np.mean(r_arr):.4f}, "
            f"median={np.median(r_arr):.4f}. "
            f"TV vs GPR: r={r_tv:.4f}. "
            f"BP-grad vs GPR: r={r_bg:.4f}. "
            f"BP-IDD-err vs GPR: r={r_be:.4f}."
        )
    else:
        logger.info("Skipping summary figure – fewer than 3 samples with data")

    return correlations


def generate_publication_figures(
    model: DoTA3D_v3,
    results: list,
    test_data_path: Path,
    output_dir: Path,
    config: EvaluationConfig,
    device: torch.device,
) -> None:
    """Generate publication figures for best, worst, and mean cases.

    Args:
        model: The loaded DoTA model.
        results: List of evaluation results.
        test_data_path: Path to the test data directory.
        output_dir: Directory to save figures.
        config: Evaluation configuration.
        device: Target device for computation.
    """
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

    logger.info(f"Best GPR: {best_result.sample_id} with GPR: {best_result.gpr:.2f}%")
    logger.info(
        f"Worst GPR: {worst_result.sample_id} with GPR: {worst_result.gpr:.2f}%"
    )
    logger.info(
        f"Closest to mean GPR: {closest_result.sample_id} with GPR: {closest_result.gpr:.2f}%"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    for desc, result in cases.items():
        logger.info(
            f"Generating publication figure for {desc} case, id: {result.sample_id}"
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
            f"{desc} case - RMSE: {rmse:.6f}, MAPE: {mape:.2f}%, GPR: {gpr:.2f}%"
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


def print_summary(results: list, total_time: float) -> None:
    """Print evaluation summary.

    Args:
        results: List of evaluation results.
        total_time: Total evaluation time in seconds.
    """
    calc_times = [r.calc_time for r in results]
    gprs = [r.gpr for r in results]

    logger.info(f"Total elapsed time: {total_time:.2f}s")
    logger.info(f"Average time per beamlet: {np.mean(calc_times):.4f}s")

    worst_idx = np.argmin(gprs)
    best_idx = np.argmax(gprs)

    worst = results[worst_idx]
    best = results[best_idx]

    logger.info(
        f"Worst case: Energy: {worst.energy_mev:.2f} MeV, "
        f"Beamlet angles: ({worst.beamlet_angles[0]:.2f}, {worst.beamlet_angles[1]:.2f}) degrees, "
        f"GPR: {worst.gpr:.2f}%"
    )
    logger.info(
        f"Best case: Energy: {best.energy_mev:.2f} MeV, "
        f"Beamlet angles: ({best.beamlet_angles[0]:.2f}, {best.beamlet_angles[1]:.2f}) degrees, "
        f"GPR: {best.gpr:.2f}%"
    )


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
        typer.Option(help="Path to YAML configuration file"),
    ] = None,
    downsampling_method: Annotated[
        Optional[str],
        typer.Option(help="Downsampling method: 'interpolation' or 'avg_pooling'"),
    ] = None,
    model_fname: Annotated[Optional[str], typer.Option(help="Model filename")] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index (-1 for CPU)")
    ] = None,
    dose_threshold: Annotated[
        Optional[float], typer.Option(help="Dose percent threshold for gamma")
    ] = None,
    distance_threshold: Annotated[
        Optional[float], typer.Option(help="Distance threshold (mm) for gamma")
    ] = None,
    no_progress: Annotated[
        Optional[bool], typer.Option(help="Disable progress bar")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose output")
    ] = None,
) -> None:
    """Run the DoTA model for dose prediction.

    Evaluates the model on all samples in the test data directory,
    computes metrics (GPR, RMSE, MAPE, RDE), and generates publication figures.

    Can be configured via CLI arguments, a YAML config file (--config), or both.
    CLI arguments take precedence over YAML values.
    """
    # Load YAML config if provided
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    # Merge: CLI args override YAML values, YAML overrides defaults
    model_name = model_name or yaml_config.get("model_name")
    test_data = test_data or (
        Path(yaml_config["test_data"]) if "test_data" in yaml_config else None
    )
    downsampling_method = downsampling_method or yaml_config.get(
        "downsampling_method", "interpolation"
    )
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )
    dose_threshold = (
        dose_threshold
        if dose_threshold is not None
        else yaml_config.get("dose_threshold", 2.0)
    )
    distance_threshold = (
        distance_threshold
        if distance_threshold is not None
        else yaml_config.get("distance_threshold", 2.0)
    )
    no_progress = (
        no_progress
        if no_progress is not None
        else yaml_config.get("no_progress", False)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    # Validate required arguments
    if model_name is None:
        raise typer.BadParameter(
            "MODEL_NAME is required (via CLI argument or YAML config)"
        )
    if test_data is None:
        raise typer.BadParameter(
            "TEST_DATA is required (via CLI argument or YAML config)"
        )

    # Setup run directory and logging first
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir)
    log_file = setup_logging(run_dir, verbose=verbose)

    # Copy YAML config to run directory for reproducibility
    if config_path is not None:
        shutil.copy2(config_path, run_dir / config_path.name)
        logger.info(f"Config file copied to {run_dir / config_path.name}")

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    # Validate downsampling method
    if downsampling_method not in ("interpolation", "avg_pooling"):
        raise typer.BadParameter(
            f"Invalid downsampling method: {downsampling_method}. "
            "Must be 'interpolation' or 'avg_pooling'."
        )

    # Setup paths
    model_hub = PROJECT_ROOT / "models"
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"

    # Convert test_data to absolute path if relative
    if not test_data.is_absolute():
        test_data = PROJECT_ROOT / test_data

    # Validate inputs
    validate_inputs(test_data, model_path, hyperparams_path)

    # Log run configuration
    logger.info("=" * 60)
    logger.info("RUN CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model file: {model_fname}")
    logger.info(f"Test data: {test_data}")
    logger.info(f"Downsampling method: {downsampling_method}")
    logger.info(f"Dose threshold: {dose_threshold}%")
    logger.info(f"Distance threshold: {distance_threshold}mm")
    logger.info("=" * 60)

    # Setup device and model
    device = get_device(device_index)
    logger.info(f"Using device: {device}")

    model = load_model(model_path, hyperparams_path, device)

    # Setup configuration
    config = EvaluationConfig()
    config.gamma_params["dose_percent_threshold"] = dose_threshold
    config.gamma_params["distance_mm_threshold"] = distance_threshold

    # Get sample IDs
    files = os.listdir(test_data)
    sample_ids = np.unique([f.split("_")[0] for f in files if "_" in f]).tolist()
    logger.info(f"Found {len(sample_ids)} unique samples for evaluation")

    if not sample_ids:
        logger.error("No samples found in test data directory")
        raise typer.Exit(code=1)

    # Run evaluation
    start_time = perf_counter()
    results = evaluate_samples(
        model=model,
        sample_ids=sample_ids,
        test_data_path=test_data,
        config=config,
        device=device,
        downsampling_method=downsampling_method,
        show_progress=not no_progress,
    )
    total_time = perf_counter() - start_time

    # Print results table (goes to both console and log file)
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
        gamma_params=config.gamma_params,
        beamlet_angles=[list(r.beamlet_angles) for r in results],
        logger=logger,
    )

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    print_summary(results, total_time)

    # Generate plots and figures in run directory
    figures_dir = run_dir / "figures"

    generate_gpr_plot(
        results=results,
        output_path=figures_dir / "gpr_vs_energy.png",
        gamma_params=config.gamma_params,
    )

    generate_publication_figures(
        model=model,
        results=results,
        test_data_path=test_data,
        output_dir=figures_dir,
        config=config,
        device=device,
    )

    advanced_metrics_and_figures(
        results=results,
        output_dir=figures_dir / "depth_profiles",
        config=config,
    )

    density_variability_vs_gpr(
        results=results,
        output_dir=figures_dir,
        config=config,
        gamma_params=config.gamma_params,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Evaluation complete! Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
