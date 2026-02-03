"""
DoTA Model Evaluation Script

A command-line tool for running DoTA model inference on dose prediction tasks.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.models import DoTA3D_v3
from src.adota.utils import count_parameters_per_block, count_total_parameters
from src.figures.single_beam import publication_figure
from src.loaders.dir_based import get_single_record, save_prediction
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

app = typer.Typer(help="DoTA Model Evaluation Tool")


def setup_run_directory(runs_dir: Path) -> Path:
    """Create a timestamped run directory.

    Args:
        runs_dir: Base directory for all runs.

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool = False) -> Path:
    """Configure logging to both console and file.

    Args:
        run_dir: Directory where log file will be stored.
        verbose: Whether to enable debug level logging.

    Returns:
        Path to the log file.
    """
    log_file = run_dir / "evaluation.log"
    log_level = logging.DEBUG if verbose else logging.INFO

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    return log_file


# Default scaling parameters
DEFAULT_SCALE = {
    "min_ds": 0.0,
    "max_ds": 25277028.0,
    "min_ct": -1024,
    "max_ct": 3071,
    "min_energy": 70.0,
    "max_energy": 270.0,
}

# Default gamma parameters
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
class EvaluationResult:
    """Container for a single sample's evaluation results."""

    sample_id: str
    energy_mev: float
    beamlet_angles: tuple
    gpr: float
    rmse: float
    mape: float
    rde: float
    calc_time: float
    prediction: Optional[torch.Tensor] = field(default=None, repr=False)
    ground_truth: Optional[torch.Tensor] = field(default=None, repr=False)
    input_data: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())
    gamma_params: dict = field(default_factory=lambda: DEFAULT_GAMMA_PARAMS.copy())
    normalize_flux: bool = True
    resolution: tuple = (2.0, 2.0, 2.0)


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
    """Load and configure the DoTA model.

    Args:
        model_path: Path to the model weights file.
        hyperparams_path: Path to the hyperparameters JSON file.
        device: Target device for the model.

    Returns:
        Configured DoTA3D_v3 model in eval mode.

    Raises:
        FileNotFoundError: If model or hyperparams files don't exist.
    """
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

    logger.info(f"Model loaded from {model_path}")
    return model


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

    for sample_id in iterator:
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


def validate_inputs(test_data: Path, model_path: Path, hyperparams_path: Path) -> None:
    """Validate that required input files exist.

    Args:
        test_data: Path to test data directory.
        model_path: Path to model weights.
        hyperparams_path: Path to hyperparameters file.

    Raises:
        typer.BadParameter: If any required file is missing.
    """
    if not test_data.exists():
        raise typer.BadParameter(f"Test data directory not found: {test_data}")
    if not model_path.exists():
        raise typer.BadParameter(f"Model file not found: {model_path}")
    if not hyperparams_path.exists():
        raise typer.BadParameter(f"Hyperparams file not found: {hyperparams_path}")


@app.command()
def main(
    model_name: Annotated[str, typer.Argument(help="Name of the model directory")],
    test_data: Annotated[
        Path, typer.Argument(help="Path to the directory with input data to evaluate")
    ],
    downsampling_method: Annotated[
        str, typer.Option(help="Downsampling method: 'interpolation' or 'avg_pooling'")
    ] = "interpolation",
    model_fname: Annotated[str, typer.Option(help="Model filename")] = "best_model.pth",
    device_index: Annotated[
        int, typer.Option(help="CUDA device index (-1 for CPU)")
    ] = 0,
    dose_threshold: Annotated[
        float, typer.Option(help="Dose percent threshold for gamma")
    ] = 2.0,
    distance_threshold: Annotated[
        float, typer.Option(help="Distance threshold (mm) for gamma")
    ] = 2.0,
    no_progress: Annotated[bool, typer.Option(help="Disable progress bar")] = False,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
) -> None:
    """Run the DoTA model for dose prediction.

    Evaluates the model on all samples in the test data directory,
    computes metrics (GPR, RMSE, MAPE, RDE), and generates publication figures.
    """
    # Setup run directory and logging first
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir)
    log_file = setup_logging(run_dir, verbose=verbose)

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

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Evaluation complete! Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
