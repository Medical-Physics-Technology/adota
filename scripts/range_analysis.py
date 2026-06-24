"""
Beamlet-Level Range-Fidelity Analysis (MC vs ADoTA)

Runs ADoTA inference over a directory-based test set and, for every beamlet,
extracts range-related metrics from the integrated depth-dose (IDD) of both the
Monte-Carlo ground truth and the ADoTA prediction:

  * R100 – Bragg-peak depth
  * R90 / R80 / R50 / R20 – distal fall-off depths (R80 = clinical range)
  * DFW  – distal fall-off width (R20 - R80), a steepness measure

It then reports the signed (ADoTA - MC) differences: the clinical range error
(ΔR80), the Bragg-peak location error (ΔR100), and the distal fall-off width
mismatch (ΔDFW, which tests whether ADoTA produces a smoother distal edge).

Output: runs/range_{timestamp}/ with a per-beamlet CSV, summary statistics
(overall + energy-stratified), error histograms, an MC-vs-ADoTA R80 scatter,
and diagnostic IDD overlays for the largest range outliers.

Usage:
    uv run python scripts/range_analysis.py --config scripts/config_range_analysis.yaml

CLI arguments take precedence over YAML values. A copy of the config is saved to
the run directory for reproducibility.
"""

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from scipy.stats import pearsonr

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
from src.adota.utils import load_model
from src.evaluation.cli import resolve_device
from src.evaluation.engine import InferenceContext, evaluate
from src.evaluation.sources import DirSource
from src.loaders.utils import validate_inputs
from src.metrics.range_metrics import (
    compute_range_metrics,
    integrated_depth_dose,
    range_metric_deltas,
)
from src.schemas.configs import EvaluationConfig
from src.schemas.results import RangeRecord

logger = logging.getLogger(__name__)

app = typer.Typer(help="Beamlet-level range-fidelity analysis (MC vs ADoTA)")


# ═══════════════════════════════════════════════════════════════════════════
#  Test-data discovery (mirrors run_model.py conventions)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TestDataset:
    """A labeled directory-based test dataset."""

    label: str
    path: Path


def _resolve_data_path(path_value: Union[str, Path]) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_test_data_config(test_data_value: Any) -> list[TestDataset]:
    """Normalize single-path and multi-site YAML formats to labeled datasets."""
    if isinstance(test_data_value, (str, Path)):
        path = _resolve_data_path(test_data_value)
        return [TestDataset(label=path.name, path=path)]
    if not isinstance(test_data_value, list):
        raise typer.BadParameter(
            "test_data must be a path or a list of entries with 'label' and 'path'"
        )
    datasets: list[TestDataset] = []
    for index, entry in enumerate(test_data_value, start=1):
        if isinstance(entry, (str, Path)):
            path = _resolve_data_path(entry)
            datasets.append(TestDataset(label=path.name, path=path))
            continue
        if not isinstance(entry, dict):
            raise typer.BadParameter(
                f"Invalid test_data entry #{index}: expected path or mapping"
            )
        label = entry.get("label")
        path_value = entry.get("path")
        if not label or not path_value:
            raise typer.BadParameter(
                f"Invalid test_data entry #{index}: 'label' and 'path' are required"
            )
        datasets.append(
            TestDataset(label=str(label), path=_resolve_data_path(path_value))
        )
    if not datasets:
        raise typer.BadParameter("test_data must contain at least one dataset")
    return datasets


def discover_sample_ids(test_data_path: Path) -> list[str]:
    """Discover sample IDs using the same filename convention as run_model.py."""
    files = os.listdir(test_data_path)
    return np.unique([f.split("_")[0] for f in files if "_" in f]).tolist()


# ═══════════════════════════════════════════════════════════════════════════
#  Per-sample callback
# ═══════════════════════════════════════════════════════════════════════════


def _make_per_sample_fn(
    config: EvaluationConfig,
    anatomical_site: str,
    oversample: int,
    min_peak_dose_frac: float,
    max_energy_mev: float,
    keep_idd: bool,
):
    """Build the per-beamlet range-metric callback for the evaluation engine."""
    scale = config.scale
    dz_mm = float(config.resolution[0])

    def per_sample_fn(ctx: InferenceContext) -> Optional[RangeRecord]:
        energy_mev = denormalize_energy(ctx.energy.item(), scale)
        if energy_mev > max_energy_mev:
            return None

        y_np, y_pred_np = ctx.denorm(scale)  # (1, 1, D, H, W) each, physical units
        mc_dose = np.squeeze(y_np)
        pred_dose = np.squeeze(y_pred_np)

        mc_idd = integrated_depth_dose(mc_dose)
        pred_idd = integrated_depth_dose(pred_dose)

        # Skip empty / zero-flux beamlets: the MC peak must be meaningfully > 0.
        peak_floor = min_peak_dose_frac * float(np.max(mc_idd)) if mc_idd.size else 0.0
        if float(np.max(mc_idd)) <= 0.0:
            return None

        mc_metrics = compute_range_metrics(
            mc_idd, dz_mm, oversample=oversample, min_peak_dose=peak_floor
        )
        pred_metrics = compute_range_metrics(
            pred_idd, dz_mm, oversample=oversample, min_peak_dose=0.0
        )
        if np.isnan(mc_metrics.r80_mm) or np.isnan(pred_metrics.r80_mm):
            return None

        deltas = range_metric_deltas(pred_metrics, mc_metrics)
        ba = ctx.extra.get("beamlet_angles") or (float("nan"), float("nan"))

        return RangeRecord(
            sample_id=ctx.sample_id,
            energy_mev=energy_mev,
            beamlet_angles=tuple(ba),
            anatomical_site=anatomical_site,
            mc=mc_metrics,
            pred=pred_metrics,
            deltas=deltas,
            calc_time=ctx.calc_time,
            dz_mm=dz_mm,
            mc_idd=mc_idd if keep_idd else None,
            pred_idd=pred_idd if keep_idd else None,
        )

    return per_sample_fn


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregation, CSV, figures
# ═══════════════════════════════════════════════════════════════════════════


def records_to_dataframe(records: list[RangeRecord]) -> pd.DataFrame:
    """Flatten range records into a per-beamlet DataFrame."""
    rows = []
    for r in records:
        row = {
            "anatomical_site": r.anatomical_site,
            "sample_id": r.sample_id,
            "energy_mev": r.energy_mev,
            "beamlet_angle_0_deg": r.beamlet_angles[0],
            "beamlet_angle_1_deg": r.beamlet_angles[1],
            "calc_time_s": r.calc_time,
        }
        row.update(r.mc.as_dict(prefix="mc_"))
        row.update(r.pred.as_dict(prefix="pred_"))
        row.update(r.deltas)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric summary stats over the signed-difference columns."""
    delta_cols = [c for c in df.columns if c.endswith("_delta_mm")]
    rows = []
    for col in delta_cols:
        vals = df[col].dropna().to_numpy()
        if vals.size == 0:
            continue
        abs_vals = np.abs(vals)
        rows.append(
            {
                "metric": col,
                "n": int(vals.size),
                "mean_mm": round(float(vals.mean()), 3),
                "std_mm": round(float(vals.std()), 3),
                "mae_mm": round(float(abs_vals.mean()), 3),
                "median_mm": round(float(np.median(vals)), 3),
                "p95_abs_mm": round(float(np.percentile(abs_vals, 95)), 3),
                "within_1mm_pct": round(float(100.0 * np.mean(abs_vals <= 1.0)), 2),
                "within_2mm_pct": round(float(100.0 * np.mean(abs_vals <= 2.0)), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_delta_histogram(df: pd.DataFrame, col: str, label: str, out: Path) -> None:
    vals = df[col].dropna().to_numpy()
    if vals.size < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=60, edgecolor="black", linewidth=0.3, alpha=0.75)
    ax.axvline(0, color="red", ls="--", lw=0.8)
    ax.set_xlabel(f"{label}  (ADoTA - MC) [mm]")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} distribution")
    ax.text(
        0.97,
        0.95,
        f"mean={vals.mean():.2f} mm\nstd={vals.std():.2f} mm\nMAE={np.abs(vals).mean():.2f} mm\nn={vals.size}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_r80_scatter(df: pd.DataFrame, out: Path) -> None:
    valid = df[["mc_r80_mm", "pred_r80_mm"]].dropna()
    if len(valid) < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(valid["mc_r80_mm"], valid["pred_r80_mm"], alpha=0.15, s=8, edgecolors="none")
    lims = [
        min(valid["mc_r80_mm"].min(), valid["pred_r80_mm"].min()) - 5,
        max(valid["mc_r80_mm"].max(), valid["pred_r80_mm"].max()) + 5,
    ]
    ax.plot(lims, lims, "k--", lw=0.8, label="identity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("MC R80 [mm]")
    ax.set_ylabel("ADoTA R80 [mm]")
    ax.set_title("Distal range (R80): ADoTA vs MC")
    r, _ = pearsonr(valid["mc_r80_mm"], valid["pred_r80_mm"])
    mae = np.mean(np.abs(valid["pred_r80_mm"] - valid["mc_r80_mm"]))
    ax.text(
        0.05,
        0.95,
        f"r={r:.3f}\nMAE={mae:.2f} mm\nn={len(valid)}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_energy_stratified(
    df: pd.DataFrame, energy_bins: list[float], run_dir: Path
) -> None:
    """Energy-binned MAE bar chart for the key range-error metrics."""
    delta_cols = ["r80_delta_mm", "r100_delta_mm", "dfw_delta_mm"]
    delta_cols = [c for c in delta_cols if c in df.columns]
    if not delta_cols:
        return
    df = df.copy()
    df["energy_bin"] = pd.cut(
        df["energy_mev"],
        bins=energy_bins,
        right=False,
        labels=[f"{energy_bins[i]}-{energy_bins[i+1]}" for i in range(len(energy_bins) - 1)],
    )
    rows = []
    for col in delta_cols:
        for bin_label, grp in df.groupby("energy_bin", observed=True):
            vals = grp[col].dropna().to_numpy()
            if vals.size < 2:
                continue
            rows.append(
                {
                    "metric": col,
                    "energy_bin": bin_label,
                    "MAE_mm": float(np.abs(vals).mean()),
                    "bias_mm": float(vals.mean()),
                    "n": int(vals.size),
                }
            )
    if not rows:
        return
    strat = pd.DataFrame(rows)
    strat.to_csv(run_dir / "energy_stratified_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=strat, x="energy_bin", y="MAE_mm", hue="metric", ax=ax)
    ax.set_xlabel("Energy bin [MeV]")
    ax.set_ylabel("MAE [mm]")
    ax.set_title("Energy-stratified range-metric error")
    ax.legend(title="metric", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "figures" / "energy_stratified_range_mae.png", dpi=150)
    plt.close(fig)


def plot_worst_idd_overlays(
    records: list[RangeRecord], n_worst: int, run_dir: Path
) -> None:
    """Diagnostic MC-vs-ADoTA IDD overlays for the largest |ΔR80| beamlets."""
    with_idd = [
        r
        for r in records
        if r.mc_idd is not None
        and r.pred_idd is not None
        and np.isfinite(r.deltas.get("r80_delta_mm", np.nan))
    ]
    if not with_idd:
        return
    with_idd.sort(key=lambda r: abs(r.deltas["r80_delta_mm"]), reverse=True)
    for rank, r in enumerate(with_idd[:n_worst], start=1):
        z = np.arange(len(r.mc_idd)) * r.dz_mm
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(z, r.mc_idd, label="MC (GT)", color="tab:orange")
        ax.plot(z, r.pred_idd, label="ADoTA", color="tab:green")
        ax.axvline(r.mc.r80_mm, color="tab:orange", ls="--", lw=0.8, label=f"MC R80={r.mc.r80_mm:.1f}")
        ax.axvline(r.pred.r80_mm, color="tab:green", ls="--", lw=0.8, label=f"ADoTA R80={r.pred.r80_mm:.1f}")
        ax.set_xlabel("Depth [mm]")
        ax.set_ylabel("Integrated depth dose")
        ax.set_title(
            f"#{rank} |ΔR80|={abs(r.deltas['r80_delta_mm']):.2f} mm  "
            f"({r.anatomical_site}, {r.sample_id}, E={r.energy_mev:.1f} MeV)"
        )
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(run_dir / "figures" / f"worst_idd_{rank:02d}_{r.sample_id}.png", dpi=150)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════


@app.command()
def main(
    config: Annotated[
        Optional[Path], typer.Option(help="Path to YAML configuration file")
    ] = None,
    model_name: Annotated[
        Optional[str], typer.Option(help="Name of the model directory in models/")
    ] = None,
    model_fname: Annotated[Optional[str], typer.Option(help="Model filename")] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index (-1 for CPU)")
    ] = None,
    downsampling_method: Annotated[
        Optional[str], typer.Option(help="'interpolation' or 'avg_pooling'")
    ] = None,
    max_energy_mev: Annotated[
        Optional[float], typer.Option(help="Skip beamlets above this energy [MeV]")
    ] = None,
    no_progress: Annotated[Optional[bool], typer.Option(help="Disable progress bar")] = None,
    verbose: Annotated[Optional[bool], typer.Option(help="Enable verbose output")] = None,
) -> None:
    """Run beamlet-level range-fidelity analysis over a directory test set."""
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    model_name = model_name or yaml_config.get("model_name")
    test_data_config = yaml_config.get("test_data")
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )
    downsampling_method = downsampling_method or yaml_config.get(
        "downsampling_method", "interpolation"
    )
    max_energy_mev = (
        max_energy_mev
        if max_energy_mev is not None
        else yaml_config.get("max_energy_mev", 250.0)
    )
    no_progress = (
        no_progress if no_progress is not None else yaml_config.get("no_progress", False)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    resolution = tuple(yaml_config.get("resolution", [2.0, 2.0, 2.0]))
    oversample = int(yaml_config.get("oversample", 20))
    min_peak_dose_frac = float(yaml_config.get("min_peak_dose_frac", 0.0))
    energy_bins = yaml_config.get("energy_bins", [70, 100, 130, 160, 190, 220, 250])
    n_worst_figures = int(yaml_config.get("n_worst_figures", 10))
    normalize_flux = bool(yaml_config.get("normalize_flux", True))

    if model_name is None:
        raise typer.BadParameter("model_name is required (CLI or YAML)")
    if test_data_config is None:
        raise typer.BadParameter("test_data is required (YAML)")

    test_datasets = normalize_test_data_config(test_data_config)

    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir, prefix="range_")
    log_file = setup_logging(run_dir, verbose=verbose, log_filename="range_analysis.log")
    if config_path is not None:
        shutil.copy2(config_path, run_dir / config_path.name)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file: {log_file}")

    if downsampling_method not in ("interpolation", "avg_pooling"):
        raise typer.BadParameter(f"Invalid downsampling method: {downsampling_method}")

    model_hub = PROJECT_ROOT / "models"
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"
    for dataset in test_datasets:
        validate_inputs(dataset.path, model_path, hyperparams_path)

    device = resolve_device(device_index)
    logger.info(f"Using device: {device}")
    model = load_model(model_path, hyperparams_path, device)

    config = EvaluationConfig()
    config.scale = DEFAULT_SCALE.copy()
    config.resolution = resolution
    config.normalize_flux = normalize_flux

    keep_idd = n_worst_figures > 0
    start_time = perf_counter()
    all_records: list[RangeRecord] = []
    for dataset in test_datasets:
        sample_ids = discover_sample_ids(dataset.path)
        logger.info(f"[{dataset.label}] {len(sample_ids)} samples")
        if not sample_ids:
            logger.error(f"No samples found in {dataset.path}")
            raise typer.Exit(code=1)

        source = DirSource(
            dataset.path,
            sample_ids,
            scale=config.scale,
            normalize_flux=config.normalize_flux,
            downsampling_method=downsampling_method,
            beamlet_angle=True,
        )
        per_sample_fn = _make_per_sample_fn(
            config=config,
            anatomical_site=dataset.label,
            oversample=oversample,
            min_peak_dose_frac=min_peak_dose_frac,
            max_energy_mev=max_energy_mev,
            keep_idd=keep_idd,
        )
        records = evaluate(
            model,
            source,
            device=device,
            per_sample_fn=per_sample_fn,
            show_progress=not no_progress,
            desc=f"Range [{dataset.label}]",
            postfix_fn=lambda r: {"dR80": f"{r.deltas['r80_delta_mm']:+.1f}mm"},
        )
        all_records.extend(records)

    elapsed = perf_counter() - start_time
    logger.info(f"Processed {len(all_records)} beamlets in {elapsed:.1f}s")

    if not all_records:
        logger.error("No valid beamlets produced range metrics.")
        raise typer.Exit(code=1)

    # ── CSV + summaries ──────────────────────────────────────────────────
    df = records_to_dataframe(all_records)
    csv_path = run_dir / "range_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-beamlet results: {csv_path}")

    summary = summarize(df)
    summary.to_csv(run_dir / "range_summary.csv", index=False)
    logger.info("Range-error summary (ADoTA - MC):")
    for _, row in summary.iterrows():
        logger.info(
            f"  {row['metric']:18s}  bias={row['mean_mm']:+.2f}  MAE={row['mae_mm']:.2f}  "
            f"std={row['std_mm']:.2f}  <=1mm={row['within_1mm_pct']:.0f}%  "
            f"<=2mm={row['within_2mm_pct']:.0f}%  (n={row['n']})"
        )

    # ── Figures ──────────────────────────────────────────────────────────
    fig_dir = run_dir / "figures"
    plot_delta_histogram(df, "r80_delta_mm", "Range error ΔR80", fig_dir / "hist_r80_delta.png")
    plot_delta_histogram(df, "r100_delta_mm", "Bragg-peak error ΔR100", fig_dir / "hist_r100_delta.png")
    plot_delta_histogram(df, "dfw_delta_mm", "Distal fall-off width mismatch ΔDFW", fig_dir / "hist_dfw_delta.png")
    plot_r80_scatter(df, fig_dir / "scatter_r80.png")
    plot_energy_stratified(df, energy_bins, run_dir)
    plot_worst_idd_overlays(all_records, n_worst_figures, run_dir)

    logger.info(f"All outputs in {run_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    app()
