"""
Threshold Sensitivity Sweep: σ_HU Threshold vs Performance Gap

Sweeps the classification threshold τ over a user-defined range and
plots the GPR (and optionally RDE / MAPE / RMSE) gap between interface
and homogeneous groups as a function of τ.

Two modes of operation:
  • **CSV mode** (default): reads a pre-computed ``results.csv`` produced
    by ``training_set_analysis.py``.  Re-classifies each beamlet at every
    threshold value without re-running model inference (< 1 min).
  • **Inference mode**: if no CSV is provided, loads the model and
    HDF5 dataset, runs ADoTA inference, and then performs the sweep.

Usage:
    uv run python scripts/threshold_sweep.py --config scripts/config_threshold_sweep.yaml
"""

import csv
import logging
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

app = typer.Typer(help="σ_HU threshold sensitivity sweep")


# ── Data container ──────────────────────────────────────────────────────────


@dataclass
class BeamletRecord:
    """Lightweight container for a single beamlet read from CSV."""

    sample_id: str
    energy_mev: float
    sigma_hu: float
    tv: float
    cv: float
    gpr: float  # GPR [%]
    rmse: float  # RMSE [Gy]
    mape: float  # MAPE [%]
    rde: float  # RDE [%]


@dataclass
class SweepPoint:
    """Metrics at a single threshold value."""

    tau: float
    n_homo: int
    n_intf: int
    pct_intf: float
    gpr_homo: float
    gpr_intf: float
    gpr_gap: float  # homo − intf (positive = interface is worse)
    rde_homo: float
    rde_intf: float
    rde_gap: float
    mape_homo: float
    mape_intf: float
    mape_gap: float
    rmse_homo: float
    rmse_intf: float
    rmse_gap: float


# ── YAML helper ─────────────────────────────────────────────────────────────


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise typer.BadParameter(f"Failed to parse YAML config: {e}")
    return config if config is not None else {}


# ── Run directory & logging ─────────────────────────────────────────────────


def setup_run_directory(runs_dir: Path) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool = False) -> Path:
    """Configure logging to both console and file."""
    log_file = run_dir / "threshold_sweep.log"
    log_level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(console_format)
    root_logger.addHandler(file_handler)

    return log_file


# ── CSV loading ─────────────────────────────────────────────────────────────


def load_results_csv(csv_path: Path) -> list[BeamletRecord]:
    """Load beamlet records from a ``results.csv`` file.

    Skips the summary rows at the bottom (those with a non-UUID
    ``sample_id`` like 'mean', 'std', etc.) and rows with missing data.

    Args:
        csv_path: Path to the CSV file produced by training_set_analysis.py.

    Returns:
        List of :class:`BeamletRecord` objects.
    """
    records: list[BeamletRecord] = []
    summary_ids = {"mean", "std", "min", "max", ""}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("sample_id", "").strip()
            if sid in summary_ids:
                continue
            # Skip rows with empty numeric fields
            try:
                records.append(
                    BeamletRecord(
                        sample_id=sid,
                        energy_mev=float(row["energy_mev"]),
                        sigma_hu=float(row["sigma_hu"]),
                        tv=float(row["tv"]),
                        cv=float(row["cv"]),
                        gpr=float(row["gpr_pct"]),
                        rmse=float(row["rmse_gy"]),
                        mape=float(row["mape_pct"]),
                        rde=float(row["rde_pct"]),
                    )
                )
            except (ValueError, KeyError):
                continue

    return records


# ── Inference-mode loading ──────────────────────────────────────────────────


def load_records_from_inference(
    model_name: str,
    h5_path: Path,
    model_fname: str,
    excluded_indexes_file: Optional[Path],
    device_index: int,
    bp_radius_mm: float,
    max_energy_mev: float,
    n_samples: Optional[int],
) -> list[BeamletRecord]:
    """Run ADoTA inference and return beamlet records.

    This is the slow path — used only when no pre-computed CSV is given.
    Imports heavy dependencies (torch, h5py, model code) only when needed.
    """
    import h5py
    import torch
    from tqdm import tqdm

    from src.adota.models import DoTA3D_v3
    from src.adota.utils import load_model
    from src.loaders.generator import H5PYGenerator
    from src.loaders.utils import validate_inputs
    from src.metrics.classic import (
        calculate_pure_mape,
        calculate_relative_dose_error,
        calculate_rmse,
    )
    from src.metrics.gamma_pass_rate import gamma_index_torch
    from src.utils.dose_grid_utils import estimate_bragg_peak
    from src.utils.scallers import inverse_minmax
    from src.utils.unit_conversions import to_gy

    # Re-use constants and helpers from training_set_analysis
    from scripts.training_set_analysis import (
        DEFAULT_GAMMA_PARAMS,
        DEFAULT_SCALE,
        AnalysisConfig,
        compute_bp_cv,
        compute_bp_sigma_hu,
        compute_bp_tv,
        denormalize_energy,
        get_device,
    )

    scale = DEFAULT_SCALE.copy()
    gamma_params = DEFAULT_GAMMA_PARAMS.copy()

    config = AnalysisConfig(
        scale=scale,
        gamma_params=gamma_params,
        bp_radius_mm=bp_radius_mm,
        max_energy_mev=max_energy_mev,
    )

    # Resolve paths
    model_hub = PROJECT_ROOT / "models"
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"

    if not h5_path.is_absolute():
        h5_path = PROJECT_ROOT / h5_path

    validate_inputs(h5_path, model_path, hyperparams_path)

    # Load excluded indexes
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

    # Discover samples
    with h5py.File(h5_path, "r") as ds:
        all_record_ids = list(ds.keys())

    record_ids = [rid for rid in all_record_ids if rid not in excluded_indexes]
    if n_samples is not None:
        record_ids = record_ids[:n_samples]

    logger.info(f"Evaluating {len(record_ids)} samples (inference mode)")

    dataset = H5PYGenerator(
        file_path=str(h5_path),
        indexes=record_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )

    device = get_device(device_index)
    model = load_model(model_path, hyperparams_path, device)

    records: list[BeamletRecord] = []
    n_skipped = 0

    for i in tqdm(range(len(dataset)), desc="Running inference"):
        x, energy, y = dataset[i]
        x = x.to(device)
        energy = energy.to(device)
        y = y.to(device)

        energy_mev = denormalize_energy(energy.item(), scale)

        # Guards
        flux_channel = x[1]
        if torch.abs(flux_channel).max().item() < 1e-9:
            n_skipped += 1
            continue
        if energy_mev > max_energy_mev:
            n_skipped += 1
            continue

        with torch.no_grad():
            y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]

        # De-normalise
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
        ct_norm = x[0].cpu().numpy()
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])

        dose_3d = y_np.squeeze()
        bp_idx = estimate_bragg_peak(dose_3d)

        sigma_hu = compute_bp_sigma_hu(
            ct_hu, bp_idx, config.bp_radius_mm, config.resolution
        )
        tv = compute_bp_tv(ct_hu, bp_idx, config.bp_radius_mm, config.resolution)
        cv = compute_bp_cv(ct_hu, bp_idx, config.bp_radius_mm, config.resolution)

        rmse = calculate_rmse(to_gy(y_pred_np), to_gy(y_np))
        mask = y_pred_np > 0.1 * np.max(y_pred_np)
        mape = calculate_pure_mape(y_np[mask], y_pred_np[mask])
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

        records.append(
            BeamletRecord(
                sample_id=record_ids[i],
                energy_mev=energy_mev,
                sigma_hu=sigma_hu,
                tv=tv,
                cv=cv,
                gpr=gpr,
                rmse=rmse,
                mape=mape,
                rde=rde,
            )
        )

    if n_skipped > 0:
        logger.info(f"Skipped {n_skipped} samples (zero flux or energy filter)")

    return records


# ── Sweep logic ─────────────────────────────────────────────────────────────


def run_sweep(
    records: list[BeamletRecord],
    tau_values: np.ndarray,
) -> list[SweepPoint]:
    """Classify beamlets at each threshold and compute per-group metrics.

    Args:
        records: Pre-computed beamlet records with σ_HU and metrics.
        tau_values: 1-D array of threshold values to sweep.

    Returns:
        List of :class:`SweepPoint` objects, one per threshold.
    """
    sigma_arr = np.array([r.sigma_hu for r in records])
    gpr_arr = np.array([r.gpr for r in records])
    rde_arr = np.array([r.rde for r in records])
    mape_arr = np.array([r.mape for r in records])
    rmse_arr = np.array([r.rmse for r in records])

    sweep_results: list[SweepPoint] = []

    for tau in tau_values:
        intf_mask = sigma_arr > tau
        homo_mask = ~intf_mask

        n_intf = int(intf_mask.sum())
        n_homo = int(homo_mask.sum())

        if n_intf == 0 or n_homo == 0:
            logger.warning(
                f"τ = {tau:.1f} HU: one group is empty "
                f"(homo={n_homo}, intf={n_intf}) – skipping"
            )
            continue

        pct_intf = 100.0 * n_intf / len(records)

        gpr_homo = float(np.mean(gpr_arr[homo_mask]))
        gpr_intf = float(np.mean(gpr_arr[intf_mask]))
        rde_homo = float(np.mean(rde_arr[homo_mask]))
        rde_intf = float(np.mean(rde_arr[intf_mask]))
        mape_homo = float(np.mean(mape_arr[homo_mask]))
        mape_intf = float(np.mean(mape_arr[intf_mask]))
        rmse_homo = float(np.mean(rmse_arr[homo_mask]))
        rmse_intf = float(np.mean(rmse_arr[intf_mask]))

        sweep_results.append(
            SweepPoint(
                tau=tau,
                n_homo=n_homo,
                n_intf=n_intf,
                pct_intf=pct_intf,
                gpr_homo=gpr_homo,
                gpr_intf=gpr_intf,
                gpr_gap=gpr_homo - gpr_intf,
                rde_homo=rde_homo,
                rde_intf=rde_intf,
                rde_gap=rde_intf - rde_homo,
                mape_homo=mape_homo,
                mape_intf=mape_intf,
                mape_gap=mape_intf - mape_homo,
                rmse_homo=rmse_homo,
                rmse_intf=rmse_intf,
                rmse_gap=rmse_intf - rmse_homo,
            )
        )

    return sweep_results


# ── Results CSV ─────────────────────────────────────────────────────────────


def save_sweep_csv(sweep_results: list[SweepPoint], output_path: Path) -> None:
    """Save the sweep results to a CSV file."""
    fieldnames = [
        "tau_hu",
        "n_homo",
        "n_intf",
        "pct_intf",
        "gpr_homo",
        "gpr_intf",
        "gpr_gap",
        "rde_homo",
        "rde_intf",
        "rde_gap",
        "mape_homo",
        "mape_intf",
        "mape_gap",
        "rmse_homo",
        "rmse_intf",
        "rmse_gap",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sp in sweep_results:
            writer.writerow(
                {
                    "tau_hu": f"{sp.tau:.1f}",
                    "n_homo": sp.n_homo,
                    "n_intf": sp.n_intf,
                    "pct_intf": f"{sp.pct_intf:.2f}",
                    "gpr_homo": f"{sp.gpr_homo:.4f}",
                    "gpr_intf": f"{sp.gpr_intf:.4f}",
                    "gpr_gap": f"{sp.gpr_gap:.4f}",
                    "rde_homo": f"{sp.rde_homo:.6f}",
                    "rde_intf": f"{sp.rde_intf:.6f}",
                    "rde_gap": f"{sp.rde_gap:.6f}",
                    "mape_homo": f"{sp.mape_homo:.4f}",
                    "mape_intf": f"{sp.mape_intf:.4f}",
                    "mape_gap": f"{sp.mape_gap:.4f}",
                    "rmse_homo": f"{sp.rmse_homo:.12f}",
                    "rmse_intf": f"{sp.rmse_intf:.12f}",
                    "rmse_gap": f"{sp.rmse_gap:.12f}",
                }
            )

    logger.info(f"Sweep results CSV saved to {output_path}")


# ── Figures ─────────────────────────────────────────────────────────────────


def plot_gpr_gap(
    sweep_results: list[SweepPoint],
    output_dir: Path,
) -> None:
    """Plot the GPR gap (homogeneous − interface) as a function of τ.

    Also overlays the interface prevalence on a secondary y-axis.

    Args:
        sweep_results: List of SweepPoint objects from the sweep.
        output_dir: Directory where the figure will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tau = np.array([sp.tau for sp in sweep_results])
    gap = np.array([sp.gpr_gap for sp in sweep_results])
    pct = np.array([sp.pct_intf for sp in sweep_results])

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # GPR gap
    color_gap = "#C44E52"
    ax1.plot(
        tau, gap, "o-", color=color_gap, linewidth=2, markersize=4, label="GPR gap"
    )
    ax1.set_xlabel("σ_HU threshold τ [HU]", fontsize=12)
    ax1.set_ylabel("GPR gap (homo − intf) [pp]", fontsize=12, color=color_gap)
    ax1.tick_params(axis="y", labelcolor=color_gap)
    ax1.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Interface prevalence on secondary axis
    ax2 = ax1.twinx()
    color_pct = "#4C72B0"
    ax2.plot(
        tau,
        pct,
        "s--",
        color=color_pct,
        linewidth=1.5,
        markersize=3,
        alpha=0.7,
        label="Interface %",
    )
    ax2.set_ylabel("Interface beamlets [%]", fontsize=12, color=color_pct)
    ax2.tick_params(axis="y", labelcolor=color_pct)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title(
        f"GPR Gap vs σ_HU Threshold  (N = {sweep_results[0].n_homo + sweep_results[0].n_intf})",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()
    fig_path = output_dir / "gpr_gap_vs_tau.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"GPR gap plot saved to {fig_path}")


def plot_all_metric_gaps(
    sweep_results: list[SweepPoint],
    output_dir: Path,
) -> None:
    """Plot GPR, RDE, MAPE, and RMSE gaps as a 2×2 panel figure.

    Each panel shows the metric gap (interface − homogeneous, or
    homogeneous − interface for GPR) as a function of τ, with
    interface prevalence on a secondary axis.

    Args:
        sweep_results: List of SweepPoint objects from the sweep.
        output_dir: Directory where the figure will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tau = np.array([sp.tau for sp in sweep_results])
    pct = np.array([sp.pct_intf for sp in sweep_results])

    metrics = [
        ("gpr_gap", "GPR gap (homo − intf) [pp]", "#C44E52"),
        ("rde_gap", "RDE gap (intf − homo) [pp]", "#DD8452"),
        ("mape_gap", "MAPE gap (intf − homo) [pp]", "#55A868"),
        ("rmse_gap", "RMSE gap (intf − homo) [Gy]", "#8172B2"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    n_total = sweep_results[0].n_homo + sweep_results[0].n_intf

    for ax, (attr, ylabel, color) in zip(axes, metrics):
        gap = np.array([getattr(sp, attr) for sp in sweep_results])

        ax.plot(tau, gap, "o-", color=color, linewidth=2, markersize=4)
        ax.set_xlabel("σ_HU threshold τ [HU]")
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Prevalence on secondary axis
        ax2 = ax.twinx()
        ax2.plot(
            tau,
            pct,
            "s--",
            color="#4C72B0",
            linewidth=1,
            markersize=2,
            alpha=0.5,
        )
        ax2.set_ylabel("Interface [%]", color="#4C72B0", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#4C72B0", labelsize=8)

        title = ylabel.split("(")[0].strip()
        ax.set_title(title, fontweight="bold")

    fig.suptitle(
        f"Metric Gaps vs σ_HU Threshold  (N = {n_total})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig_path = output_dir / "all_metric_gaps_vs_tau.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"All-metric gap plot saved to {fig_path}")


def plot_group_means(
    sweep_results: list[SweepPoint],
    output_dir: Path,
) -> None:
    """Plot absolute GPR means for both groups as a function of τ.

    Args:
        sweep_results: List of SweepPoint objects from the sweep.
        output_dir: Directory where the figure will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tau = np.array([sp.tau for sp in sweep_results])
    gpr_homo = np.array([sp.gpr_homo for sp in sweep_results])
    gpr_intf = np.array([sp.gpr_intf for sp in sweep_results])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(
        tau,
        gpr_homo,
        "o-",
        color="#4C72B0",
        linewidth=2,
        markersize=4,
        label="Homogeneous (mean GPR)",
    )
    ax.plot(
        tau,
        gpr_intf,
        "s-",
        color="#DD8452",
        linewidth=2,
        markersize=4,
        label="Interface (mean GPR)",
    )

    ax.fill_between(tau, gpr_intf, gpr_homo, alpha=0.15, color="#C44E52", label="Gap")

    ax.set_xlabel("σ_HU threshold τ [HU]", fontsize=12)
    ax.set_ylabel("Mean GPR [%]", fontsize=12)
    ax.set_title(
        "Group GPR vs σ_HU Threshold",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig_path = output_dir / "group_gpr_vs_tau.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Group GPR plot saved to {fig_path}")


# ── Main CLI ────────────────────────────────────────────────────────────────


@app.command()
def main(
    results_csv: Annotated[
        Optional[Path],
        typer.Argument(help="Path to a results.csv from training_set_analysis.py"),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML configuration file"),
    ] = None,
    # ── Sweep parameters ────────────────────────────────────────────────
    tau_min: Annotated[
        Optional[float],
        typer.Option(help="Lower bound of τ sweep [HU]"),
    ] = None,
    tau_max: Annotated[
        Optional[float],
        typer.Option(help="Upper bound of τ sweep [HU]"),
    ] = None,
    tau_step: Annotated[
        Optional[float],
        typer.Option(help="Step size for τ sweep [HU]"),
    ] = None,
    # ── Inference-mode parameters (only when no CSV) ────────────────────
    model_name: Annotated[
        Optional[str],
        typer.Option(help="Model directory name (inference mode)"),
    ] = None,
    h5_path: Annotated[
        Optional[Path],
        typer.Option(help="HDF5 dataset path (inference mode)"),
    ] = None,
    model_fname: Annotated[
        Optional[str],
        typer.Option(help="Model filename (inference mode)"),
    ] = None,
    excluded_indexes_file: Annotated[
        Optional[Path],
        typer.Option(help="Excluded indexes file (inference mode)"),
    ] = None,
    device_index: Annotated[
        Optional[int],
        typer.Option(help="CUDA device index (inference mode)"),
    ] = None,
    bp_radius_mm: Annotated[
        Optional[float],
        typer.Option(help="BP sphere radius [mm] (inference mode)"),
    ] = None,
    max_energy_mev: Annotated[
        Optional[float],
        typer.Option(help="Max energy threshold [MeV] (inference mode)"),
    ] = None,
    n_samples: Annotated[
        Optional[int],
        typer.Option(help="Limit to first N samples (inference mode)"),
    ] = None,
    # ── Flags ───────────────────────────────────────────────────────────
    plot_all_metrics: Annotated[
        Optional[bool],
        typer.Option(help="Also plot RDE, MAPE, RMSE gap curves"),
    ] = None,
    verbose: Annotated[
        Optional[bool],
        typer.Option(help="Enable verbose output"),
    ] = None,
) -> None:
    """Sweep σ_HU threshold τ and plot the performance gap.

    Reads a pre-computed results.csv (fast) or runs ADoTA inference
    from scratch (slow).  Produces GPR-gap and metric-gap plots as
    a function of the classification threshold τ.
    """
    # ── Load & merge config ─────────────────────────────────────────────
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    results_csv = results_csv or (
        Path(yaml_config["results_csv"]) if "results_csv" in yaml_config else None
    )
    tau_min = tau_min if tau_min is not None else yaml_config.get("tau_min", 50.0)
    tau_max = tau_max if tau_max is not None else yaml_config.get("tau_max", 400.0)
    tau_step = tau_step if tau_step is not None else yaml_config.get("tau_step", 10.0)
    plot_all_metrics = (
        plot_all_metrics
        if plot_all_metrics is not None
        else yaml_config.get("plot_all_metrics", True)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    # Inference-mode parameters
    model_name = model_name or yaml_config.get("model_name")
    h5_path = h5_path or (
        Path(yaml_config["h5_path"]) if "h5_path" in yaml_config else None
    )
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    excluded_indexes_file = excluded_indexes_file or (
        Path(yaml_config["excluded_indexes_file"])
        if "excluded_indexes_file" in yaml_config
        else None
    )
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )
    bp_radius_mm = (
        bp_radius_mm
        if bp_radius_mm is not None
        else yaml_config.get("bp_radius_mm", 10.0)
    )
    max_energy_mev = (
        max_energy_mev
        if max_energy_mev is not None
        else yaml_config.get("max_energy_mev", 250.0)
    )
    n_samples = n_samples if n_samples is not None else yaml_config.get("n_samples")

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

    # ── Log run configuration ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("THRESHOLD SWEEP CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"τ range: [{tau_min:.1f}, {tau_max:.1f}] HU, step = {tau_step:.1f} HU")
    n_steps = int(np.round((tau_max - tau_min) / tau_step)) + 1
    logger.info(f"Number of threshold values: {n_steps}")
    if results_csv is not None:
        logger.info(f"Mode: CSV (fast) — reading from {results_csv}")
    else:
        logger.info("Mode: Inference (slow) — running ADoTA on the fly")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  HDF5: {h5_path}")
        logger.info(f"  BP radius: {bp_radius_mm} mm")
        logger.info(f"  Max energy: {max_energy_mev} MeV")
    logger.info(f"Plot all metrics: {plot_all_metrics}")
    logger.info("=" * 60)

    # ── Load beamlet records ────────────────────────────────────────────
    start_time = perf_counter()

    if results_csv is not None:
        csv_path = (
            results_csv if results_csv.is_absolute() else PROJECT_ROOT / results_csv
        )
        if not csv_path.exists():
            raise typer.BadParameter(f"Results CSV not found: {csv_path}")
        records = load_results_csv(csv_path)
        logger.info(f"Loaded {len(records)} beamlet records from {csv_path}")
    else:
        if model_name is None or h5_path is None:
            raise typer.BadParameter(
                "Either results_csv or (model_name + h5_path) must be provided "
                "(via CLI or YAML config)"
            )
        records = load_records_from_inference(
            model_name=model_name,
            h5_path=h5_path,
            model_fname=model_fname,
            excluded_indexes_file=excluded_indexes_file,
            device_index=device_index,
            bp_radius_mm=bp_radius_mm,
            max_energy_mev=max_energy_mev,
            n_samples=n_samples,
        )
        logger.info(f"Computed {len(records)} beamlet records from inference")

    if not records:
        logger.error("No beamlet records – nothing to sweep")
        raise typer.Exit(code=1)

    load_time = perf_counter() - start_time
    logger.info(f"Data loading took {load_time:.2f}s")

    # ── Run sweep ───────────────────────────────────────────────────────
    tau_values = np.arange(tau_min, tau_max + tau_step / 2, tau_step)
    logger.info(f"Sweeping τ over {len(tau_values)} values...")

    sweep_results = run_sweep(records, tau_values)
    logger.info(f"Sweep complete: {len(sweep_results)} valid threshold points")

    # ── Save results CSV ────────────────────────────────────────────────
    save_sweep_csv(sweep_results, run_dir / "sweep_results.csv")

    # ── Log key points ──────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SWEEP SUMMARY")
    logger.info("=" * 60)

    # Find τ with maximum GPR gap
    max_gap_sp = max(sweep_results, key=lambda sp: sp.gpr_gap)
    logger.info(
        f"Maximum GPR gap: {max_gap_sp.gpr_gap:.4f} pp  at τ = {max_gap_sp.tau:.1f} HU  "
        f"(intf = {max_gap_sp.pct_intf:.1f}%)"
    )

    # Find τ with minimum GPR gap
    min_gap_sp = min(sweep_results, key=lambda sp: sp.gpr_gap)
    logger.info(
        f"Minimum GPR gap: {min_gap_sp.gpr_gap:.4f} pp  at τ = {min_gap_sp.tau:.1f} HU  "
        f"(intf = {min_gap_sp.pct_intf:.1f}%)"
    )

    # Report at some reference thresholds
    for ref_tau in [100.0, 150.0, 200.0, 250.0, 300.0]:
        matching = [sp for sp in sweep_results if abs(sp.tau - ref_tau) < 0.5]
        if matching:
            sp = matching[0]
            logger.info(
                f"  τ = {sp.tau:6.1f} HU: "
                f"GPR gap = {sp.gpr_gap:.4f} pp, "
                f"RDE gap = {sp.rde_gap:.6f} pp, "
                f"intf = {sp.pct_intf:.1f}%"
            )

    logger.info("=" * 60)

    # ── Generate figures ────────────────────────────────────────────────
    figures_dir = run_dir / "figures"

    plot_gpr_gap(sweep_results, figures_dir)
    plot_group_means(sweep_results, figures_dir)

    if plot_all_metrics:
        plot_all_metric_gaps(sweep_results, figures_dir)

    total_time = perf_counter() - start_time
    logger.info("")
    logger.info(f"Total elapsed time: {total_time:.2f}s")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    app()
