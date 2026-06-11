"""
DoTA Model Evaluation Script (HDF5-based)

A command-line tool for running DoTA model inference on samples stored
in an HDF5 dataset via H5PYGenerator.
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
from src.figures.single_beam import publication_figure
from src.loaders.generator import H5PYGenerator
from src.loaders.utils import validate_inputs
from src.metrics.classic import (
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)

app = typer.Typer(help="DoTA Model Evaluation Tool (HDF5)")


from src.schemas.configs import EvaluationConfig
from src.schemas.results import H5EvaluationResult as EvaluationResult
from src.evaluation.cli import resolve_device
from src.evaluation.engine import InferenceContext, evaluate
from src.evaluation.outputs import CsvColumn, save_results_csv as save_csv
from src.evaluation.sources import H5Source

# ── Per-sample evaluation ───────────────────────────────────────────────────


def _make_per_sample_fn(config: EvaluationConfig):
    """Build the per-sample callback for the evaluation engine.

    Reproduces the original metric block: RMSE, MAPE at the 10% ground-truth
    threshold (canonical form), RDE, and CT-channel TV / CV. No gamma pass rate.

    Args:
        config: Evaluation configuration.

    Returns:
        A callable mapping an ``InferenceContext`` to an ``H5EvaluationResult``.
    """
    scale = config.scale

    def per_sample_fn(ctx: InferenceContext) -> EvaluationResult:
        energy_mev = denormalize_energy(ctx.energy.item(), scale)

        # Convert to numpy for metrics
        y_np, y_pred_np = ctx.denorm(scale)

        # ── Metrics ─────────────────────────────────────────────────────

        # RMSE
        rmse = calculate_rmse(to_gy(y_pred_np), to_gy(y_np))

        # MAPE at the 10% ground-truth-dose threshold. Canonical run_model.py
        # form: mask on the ground truth, calculate_pure_mape(predicted, reference).
        mask = y_np > 0.1 * np.max(y_np)
        mape = calculate_pure_mape(y_pred_np[mask], y_np[mask])

        # Relative Dose Error
        rde = calculate_relative_dose_error(to_gy(y_pred_np), to_gy(y_np))

        # ── CT-based variability metrics ────────────────────────────────
        ct_norm = ctx.x[0].cpu().numpy()  # (D, H, W) – CT channel
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])
        avg_density = ct_hu.mean(axis=(1, 2))  # (D,)

        # Total Variation
        tv = float(np.sum(np.abs(np.diff(avg_density))))

        # Coefficient of Variation
        mu = np.mean(avg_density)
        sigma = np.std(avg_density)
        cv = float(sigma / np.abs(mu)) if np.abs(mu) > 1e-9 else 0.0

        return EvaluationResult(
            sample_id=ctx.sample_id,
            energy_mev=energy_mev,
            rmse=rmse,
            mape=mape,
            rde=rde,
            tv=tv,
            cv=cv,
            calc_time=ctx.calc_time,
            prediction=ctx.y_pred.cpu(),
            ground_truth=ctx.y.cpu(),
            input_data=ctx.x.cpu(),
        )

    return per_sample_fn


def evaluate_samples(
    model: DoTA3D_v3,
    record_ids: list,
    dataset: H5PYGenerator,
    config: EvaluationConfig,
    device: torch.device,
    show_progress: bool = True,
) -> list:
    """Evaluate every sample in the dataset via the shared evaluation engine.

    Args:
        model: The loaded DoTA model.
        record_ids: List of record IDs aligned with the dataset indices.
        dataset: The H5PYGenerator dataset.
        config: Evaluation configuration.
        device: Target device for computation.
        show_progress: Whether to show a progress bar.

    Returns:
        List of H5EvaluationResult objects.
    """
    source = H5Source(dataset, record_ids)
    per_sample_fn = _make_per_sample_fn(config)
    return evaluate(
        model,
        source,
        device=device,
        per_sample_fn=per_sample_fn,
        show_progress=show_progress,
        desc="Evaluating samples",
        postfix_fn=lambda r: {
            "energy": f"{r.energy_mev:.1f}MeV",
            "mape": f"{r.mape:.2f}%",
        },
    )


# ── Results table (CSV) ─────────────────────────────────────────────────────


def save_results_csv(results: list, output_path: Path) -> None:
    """Save per-sample results to a CSV file.

    Columns are consistent with the table produced by run_model.py,
    with the addition of TV and CV metrics.

    Args:
        results: List of H5EvaluationResult objects.
        output_path: Path to the output CSV file.
    """
    columns = [
        CsvColumn("sample_id", lambda r: r.sample_id),
        CsvColumn("energy_mev", lambda r: f"{r.energy_mev:.2f}"),
        CsvColumn("rmse_gy", lambda r: f"{r.rmse:.9f}", lambda r: r.rmse, ".9f"),
        CsvColumn("mape_pct", lambda r: f"{r.mape:.4f}", lambda r: r.mape, ".4f"),
        CsvColumn("rde_pct", lambda r: f"{r.rde:.4f}", lambda r: r.rde, ".4f"),
        CsvColumn("tv_hu", lambda r: f"{r.tv:.4f}", lambda r: r.tv, ".4f"),
        CsvColumn("cv", lambda r: f"{r.cv:.6f}", lambda r: r.cv, ".6f"),
        CsvColumn(
            "calc_time_s", lambda r: f"{r.calc_time:.4f}", lambda r: r.calc_time, ".4f"
        ),
    ]
    # Sort by energy for consistency with run_model.py
    save_csv(
        results,
        output_path,
        columns,
        sort_key=lambda r: r.energy_mev,
        label_column="sample_id",
        logger=logger,
    )


# ── Publication figures (best / worst / closest-to-mean) ────────────────────


def generate_publication_figures(
    results: list,
    output_dir: Path,
    config: EvaluationConfig,
) -> None:
    """Generate publication figures for 3 best, 3 worst, and 3 closest-to-mean.

    Ranking is based on MAPE (lower is better).

    Args:
        results: List of EvaluationResult objects (with cached tensors).
        output_dir: Directory to save figures.
        config: Evaluation configuration.
    """
    scale = config.scale
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) < 3:
        logger.warning("Fewer than 3 results – skipping publication figures")
        return

    sorted_by_mape = sorted(results, key=lambda r: r.mape)
    mean_mape = np.mean([r.mape for r in results])

    best_3 = sorted_by_mape[:3]
    worst_3 = sorted_by_mape[-3:]
    closest_3 = sorted(results, key=lambda r: abs(r.mape - mean_mape))[:3]

    cases = {}
    for i, r in enumerate(best_3):
        cases[f"Best_{i + 1}"] = r
    for i, r in enumerate(worst_3):
        cases[f"Worst_{i + 1}"] = r
    for i, r in enumerate(closest_3):
        cases[f"ClosestToMean_{i + 1}"] = r

    for desc, result in cases.items():
        if (
            result.input_data is None
            or result.ground_truth is None
            or result.prediction is None
        ):
            logger.warning(
                f"Skipping {desc} ({result.sample_id}) – cached tensors not available"
            )
            continue

        logger.info(
            f"Generating publication figure for {desc} case, "
            f"id: {result.sample_id}, MAPE: {result.mape:.2f}%"
        )

        # Prepare data for figure
        x_input = inverse_minmax(
            result.input_data.squeeze().cpu().numpy(),
            scale["min_ct"],
            scale["max_ct"],
        )
        gt = inverse_minmax(
            result.ground_truth.squeeze().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )
        pred = inverse_minmax(
            result.prediction.squeeze().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )

        figure_path = (
            output_dir / f"{desc}_E{result.energy_mev:.2f}MeV_MAPE{result.mape:.2f}.svg"
        )

        publication_figure(
            x_input,
            result.energy_mev,
            gt,
            pred,
            str(figure_path),
            result.rmse,
            result.mape,
            gpr=0.0,  # GPR not computed in this script
        )

        logger.info(f"  → saved {figure_path.name}")


# ── Correlation figures (TV/CV vs MAPE/RDE/RMSE) ───────────────────────────


def generate_correlation_figures(
    results: list,
    output_dir: Path,
) -> None:
    """Generate correlation scatter plots between CT variability and dose metrics.

    Produces 6 panels in a single 2×3 figure:
        (a) TV vs MAPE    (b) TV vs RDE    (c) TV vs RMSE
        (d) CV vs MAPE    (e) CV vs RDE    (f) CV vs RMSE

    Each panel includes a linear fit and the Pearson correlation.

    Args:
        results: List of EvaluationResult objects.
        output_dir: Directory where the figure will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) < 3:
        logger.info("Skipping correlation figures – fewer than 3 samples")
        return

    tv_arr = np.array([r.tv for r in results])
    cv_arr = np.array([r.cv for r in results])
    mape_arr = np.array([r.mape for r in results])
    rde_arr = np.array([r.rde for r in results])
    rmse_arr = np.array([r.rmse for r in results])

    panels = [
        # (row, col, x_data, y_data, xlabel, ylabel, title_letter)
        (0, 0, tv_arr, mape_arr, "Total Variation [HU]", "MAPE [%]", "a"),
        (0, 1, tv_arr, rde_arr, "Total Variation [HU]", "RDE [%]", "b"),
        (0, 2, tv_arr, rmse_arr, "Total Variation [HU]", "RMSE [Gy]", "c"),
        (1, 0, cv_arr, mape_arr, "Coefficient of Variation", "MAPE [%]", "d"),
        (1, 1, cv_arr, rde_arr, "Coefficient of Variation", "RDE [%]", "e"),
        (1, 2, cv_arr, rmse_arr, "Coefficient of Variation", "RMSE [Gy]", "f"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(21, 11), dpi=300)
    fig.suptitle(
        f"CT density variability vs dose-prediction accuracy  (N = {len(results)})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for row, col, x_data, y_data, xlabel, ylabel, letter in panels:
        ax = axes[row, col]
        r_val, p_val = pearsonr(x_data, y_data)

        ax.scatter(
            x_data,
            y_data,
            s=18,
            alpha=0.6,
            edgecolors="k",
            linewidths=0.3,
        )

        # Linear fit
        z = np.polyfit(x_data, y_data, 1)
        x_fit = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(
            x_fit,
            np.polyval(z, x_fit),
            "r--",
            linewidth=1.0,
            label=f"fit  (r = {r_val:.3f}, p = {p_val:.2e})",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"({letter}) {xlabel.split('[')[0].strip()} vs {ylabel.split('[')[0].strip()}"
        )
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", linewidth=0.5)

        logger.info(
            f"Correlation ({letter}): {xlabel} vs {ylabel} – "
            f"r = {r_val:.4f} (p = {p_val:.4e})"
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        output_dir / "variability_vs_accuracy_correlations.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(
        f"Correlation figure saved to "
        f"{output_dir / 'variability_vs_accuracy_correlations.png'}"
    )


# ── Summary ─────────────────────────────────────────────────────────────────


def print_summary(results: list, total_time: float) -> None:
    """Print evaluation summary to the logger.

    Args:
        results: List of evaluation results.
        total_time: Total evaluation time in seconds.
    """
    calc_times = [r.calc_time for r in results]
    mapes = [r.mape for r in results]
    rmses = [r.rmse for r in results]
    rdes = [r.rde for r in results]

    logger.info(f"Total elapsed time: {total_time:.2f}s")
    logger.info(f"Average time per sample: {np.mean(calc_times):.4f}s")

    logger.info(
        f"RMSE  – mean: {np.mean(rmses):.9f} Gy, " f"std: {np.std(rmses):.9f} Gy"
    )
    logger.info(f"MAPE  – mean: {np.mean(mapes):.4f}%, " f"std: {np.std(mapes):.4f}%")
    logger.info(f"RDE   – mean: {np.mean(rdes):.4f}%, " f"std: {np.std(rdes):.4f}%")

    worst_idx = np.argmax(mapes)
    best_idx = np.argmin(mapes)
    worst = results[worst_idx]
    best = results[best_idx]

    logger.info(
        f"Best case (lowest MAPE): {best.sample_id}, "
        f"E = {best.energy_mev:.2f} MeV, "
        f"MAPE = {best.mape:.2f}%, RMSE = {best.rmse:.9f} Gy"
    )
    logger.info(
        f"Worst case (highest MAPE): {worst.sample_id}, "
        f"E = {worst.energy_mev:.2f} MeV, "
        f"MAPE = {worst.mape:.2f}%, RMSE = {worst.rmse:.9f} Gy"
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
    no_progress: Annotated[
        Optional[bool], typer.Option(help="Disable progress bar")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose output")
    ] = None,
) -> None:
    """Run the DoTA model inference on an HDF5 dataset.

    Evaluates the model on every sample in the HDF5 file, computes metrics
    (RMSE, MAPE, RDE, TV, CV), saves results to CSV, generates publication
    figures for best/worst/closest-to-mean cases, and produces correlation
    plots between tissue heterogeneity metrics and dose-prediction accuracy.

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
    logger.info("=" * 60)

    # ── Setup configuration ─────────────────────────────────────────────
    eval_config = EvaluationConfig()

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
    logger.info("STARTING EVALUATION")
    logger.info("=" * 60)

    start_time = perf_counter()
    results = evaluate_samples(
        model=model,
        record_ids=record_ids,
        dataset=dataset,
        config=eval_config,
        device=device,
        show_progress=not no_progress,
    )
    total_time = perf_counter() - start_time

    # ── Save results CSV ───────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    save_results_csv(results, run_dir / "results.csv")

    # ── Print summary ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    print_summary(results, total_time)

    # ── Generate figures ───────────────────────────────────────────────
    figures_dir = run_dir / "figures"

    generate_publication_figures(
        results=results,
        output_dir=figures_dir,
        config=eval_config,
    )

    generate_correlation_figures(
        results=results,
        output_dir=figures_dir,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Evaluation complete! Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
