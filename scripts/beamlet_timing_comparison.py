from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import typer
from matplotlib import pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adota.config import denormalize_energy, get_device
from src.adota.utils import load_model
from src.figures.single_beam import compare_two_inputs
from src.image_processing.rotation import rotate_lateral_axes_sequential
from src.loaders.dir_based import get_single_record
from src.schemas.configs import EvaluationConfig

app = typer.Typer(
    help="Smoke timing study for ADoTA beamlet-angle rotations around lateral axes."
)

DEFAULT_TEST_DATA = [
    "thoracic=/scratch/mstryja/DoTA_dataset_v2/testset_Lung-PET-CT-Dx_v3",
    "pelvic_abdominal=/scratch/mstryja/DoTA_dataset_v2/pelvis_abdominal_testset_paper",
]


@dataclass(frozen=True)
class TestDataset:
    label: str
    path: Path


@dataclass
class SampleTimingRow:
    dataset_label: str
    dataset_path: str
    sample_id: str
    device: str
    rotation_backend: str
    repeats: int
    batch_size: int
    raw_shape_hwd: str
    full_shape_dhw: str
    model_shape_dhw: str
    energy_mev: Optional[float]
    ba0_deg: Optional[float]
    ba1_deg: Optional[float]
    rotation_y_deg: Optional[float]
    rotation_x_deg: Optional[float]
    load_record_s: Optional[float]
    ct_load_s: Optional[float]
    flux_load_s: Optional[float]
    dose_load_s: Optional[float]
    raw_load_total_s: Optional[float]
    ct_rotation_y_s: Optional[float]
    ct_rotation_x_s: Optional[float]
    ct_rotation_total_s: Optional[float]
    flux_rotation_y_s: Optional[float]
    flux_rotation_x_s: Optional[float]
    flux_rotation_total_s: Optional[float]
    dose_rotation_y_s: Optional[float]
    dose_rotation_x_s: Optional[float]
    dose_rotation_total_s: Optional[float]
    all_rotation_total_s: Optional[float]
    rotated_preview_npz: str
    comparison_figure_png: str
    status: str
    error: str = ""


def setup_run_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        run_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT_DIR / "runs" / f"beamlet_timing_rotvsproj_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rotated_previews").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool) -> Path:
    log_path = run_dir / "run.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def parse_test_dataset(entry: str) -> TestDataset:
    if "=" not in entry:
        path = Path(entry)
        resolved = path if path.is_absolute() else ROOT_DIR / path
        return TestDataset(label=resolved.name, path=resolved)
    label, path_value = entry.split("=", 1)
    path = Path(path_value)
    resolved = path if path.is_absolute() else ROOT_DIR / path
    return TestDataset(label=label.strip(), path=resolved)


def discover_sample_ids(test_data_path: Path) -> list[str]:
    if not test_data_path.is_dir():
        raise FileNotFoundError(f"Test data directory does not exist: {test_data_path}")
    sample_ids: set[str] = set()
    for path in test_data_path.glob("*_ct.npy"):
        sample_ids.add(path.name.removesuffix("_ct.npy"))
    return sorted(sample_ids)


def has_required_files(sample_id: str, dataset_path: Path) -> tuple[bool, str]:
    required_suffixes = ("ct.npy", "flux.npy", "ds.npy", "sim_res.json")
    missing = [
        f"{sample_id}_{suffix}"
        for suffix in required_suffixes
        if not (dataset_path / f"{sample_id}_{suffix}").exists()
    ]
    if missing:
        return False, f"missing required files: {missing}"
    return True, ""


def select_sample_ids(
    all_sample_ids: list[str], max_samples: Optional[int], sample_seed: int
) -> list[str]:
    if max_samples is None or max_samples >= len(all_sample_ids):
        return all_sample_ids
    rng = random.Random(sample_seed)
    selected = list(all_sample_ids)
    rng.shuffle(selected)
    return sorted(selected[:max_samples])


def load_raw_volume_dhw(dataset_path: Path, sample_id: str, channel: str) -> tuple[np.ndarray, float]:
    start = perf_counter()
    raw = np.load(dataset_path / f"{sample_id}_{channel}.npy")
    load_seconds = perf_counter() - start
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3-D raw {channel} volume, got shape {raw.shape}")
    return np.ascontiguousarray(np.transpose(raw, (2, 0, 1)).astype(np.float32, copy=False)), load_seconds


def write_config(
    run_dir: Path,
    *,
    model_name: str,
    model_fname: str,
    model_path: Path,
    hyperparams_path: Path,
    datasets: list[TestDataset],
    device: torch.device,
    rotation_backend: str,
    downsampling_method: str,
    max_samples: Optional[int],
    sample_seed: int,
    repeats: int,
    batch_size: int,
    rotation_volume: str,
    projection_time_s: float,
    cropping_time_s: float,
    inference_time_ms: float,
    generate_sample_figures: bool,
    full_run: bool,
) -> None:
    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_fname": model_fname,
        "model_path": str(model_path),
        "hyperparams_path": str(hyperparams_path),
        "test_datasets": [
            {"label": dataset.label, "path": str(dataset.path)} for dataset in datasets
        ],
        "device": str(device),
        "rotation_backend": rotation_backend,
        "downsampling_method": downsampling_method,
        "max_samples": max_samples,
        "sample_seed": sample_seed,
        "repeats": repeats,
        "batch_size": batch_size,
        "rotation_volume": rotation_volume,
        "rotated_volumes": ["ct", "flux", "dose"],
        "projection_time_s": projection_time_s,
        "cropping_time_s": cropping_time_s,
        "inference_time_ms": inference_time_ms,
        "generate_sample_figures": generate_sample_figures,
        "full_run": full_run,
        "shape_conventions": {
            "raw_npy": "(H, W, D)",
            "internal_rotation": "(D, H, W)",
            "model_input": "(D, H, W)",
        },
        "axis_conventions": {
            "ba1": "rotation around lateral Y/H axis; applied as -ba1 first",
            "ba0": "rotation around lateral X/W axis; applied as ba0 second because of output-to-input affine convention",
            "depth_axis": "no depth-axis rotation is performed",
        },
    }
    with (run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)


def write_rows(run_dir: Path, rows: list[SampleTimingRow]) -> None:
    csv_path = run_dir / "per_sample_rotation_timing.csv"
    with csv_path.open("w", newline="") as file:
        fieldnames = list(asdict(rows[0]).keys()) if rows else list(SampleTimingRow.__dataclass_fields__.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _float_or_nan(value: Optional[float]) -> float:
    return float(value) if value is not None else float("nan")


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if clean.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "std": float(np.std(clean)),
    }


def write_timing_summary(
    run_dir: Path,
    rows: list[SampleTimingRow],
    *,
    projection_time_s: float,
    cropping_time_s: float,
    inference_time_ms: float,
) -> None:
    ok_rows = [row for row in rows if row.status == "ok"]
    if not ok_rows:
        return

    comparison_rows: list[dict[str, str | float]] = []
    for row in ok_rows:
        rotate_all_s = _float_or_nan(row.all_rotation_total_s)
        ct_rotation_s = _float_or_nan(row.ct_rotation_total_s)
        flux_rotation_s = _float_or_nan(row.flux_rotation_total_s)
        dose_rotation_s = _float_or_nan(row.dose_rotation_total_s)
        ct_to_bev_rotation_s = ct_rotation_s
        dose_back_rotation_s = dose_rotation_s
        dota_bev_reinterpolation_s = ct_to_bev_rotation_s + dose_back_rotation_s
        inference_time_s = inference_time_ms / 1000.0
        adota_total_s = projection_time_s + cropping_time_s + inference_time_s
        dota_total_s = dota_bev_reinterpolation_s + cropping_time_s + inference_time_s
        comparison_rows.append(
            {
                "dataset_label": row.dataset_label,
                "sample_id": row.sample_id,
                "projection_time_s": projection_time_s,
                "cropping_time_s": cropping_time_s,
                "inference_time_s": inference_time_s,
                "adota_projection_branch_total_s": adota_total_s,
                "rotation_reinterpolation_branch_total_s": dota_total_s,
                "dota_bev_reinterpolation_total_s": dota_bev_reinterpolation_s,
                "ct_to_bev_rotation_s": ct_to_bev_rotation_s,
                "dose_back_rotation_s": dose_back_rotation_s,
                "all_rotation_total_s": rotate_all_s,
                "ct_rotation_total_s": ct_rotation_s,
                "flux_rotation_total_s": flux_rotation_s,
                "dose_rotation_total_s": dose_rotation_s,
                "rotation_minus_projection_s": dota_bev_reinterpolation_s - projection_time_s,
                "rotation_over_projection_ratio": dota_bev_reinterpolation_s / projection_time_s,
                "branch_speedup_ratio": dota_total_s / adota_total_s,
            }
        )

    comparison_path = run_dir / "per_sample_branch_comparison.csv"
    with comparison_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    groups = ["overall"] + sorted({row.dataset_label for row in ok_rows})
    metrics = [
        "adota_projection_branch_total_s",
        "rotation_reinterpolation_branch_total_s",
        "dota_bev_reinterpolation_total_s",
        "ct_to_bev_rotation_s",
        "dose_back_rotation_s",
        "all_rotation_total_s",
        "ct_rotation_total_s",
        "flux_rotation_total_s",
        "dose_rotation_total_s",
        "rotation_over_projection_ratio",
        "branch_speedup_ratio",
    ]
    summary_rows: list[dict[str, str | float | int]] = []
    summary_json: dict[str, dict[str, dict[str, float | int]]] = {}
    for group in groups:
        group_rows = comparison_rows if group == "overall" else [
            comparison_row
            for comparison_row in comparison_rows
            if comparison_row["dataset_label"] == group
        ]
        summary_json[group] = {}
        for metric in metrics:
            stats = _summary_stats([float(comparison_row[metric]) for comparison_row in group_rows])
            summary_json[group][metric] = stats
            summary_rows.append({"group": group, "metric": metric, **stats})

    summary_path = run_dir / "timing_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["group", "metric", "count", "mean", "median", "min", "max", "std"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (run_dir / "timing_summary.json").open("w") as file:
        json.dump(
            {
                "projection_time_s": projection_time_s,
                "cropping_time_s": cropping_time_s,
                "inference_time_ms": inference_time_ms,
                "groups": summary_json,
            },
            file,
            indent=2,
        )

    labels = groups
    adota_means = [summary_json[group]["adota_projection_branch_total_s"]["mean"] for group in labels]
    rotation_means = [summary_json[group]["rotation_reinterpolation_branch_total_s"]["mean"] for group in labels]
    x_positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.bar(x_positions - width / 2, adota_means, width, label="ADoTA projection + crop")
    ax.bar(x_positions + width / 2, rotation_means, width, label="DoTA BEV rotations + crop")
    ax.set_ylabel("Mean time [s]")
    ax.set_title("Branch timing comparison")
    ax.set_xticks(x_positions, labels, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    fig.savefig(run_dir / "figures" / "timing_branch_comparison.png", bbox_inches="tight")
    plt.close(fig)

    component_labels = ["CT to BEV", "Dose back"]
    component_means = [
        summary_json["overall"]["ct_to_bev_rotation_s"]["mean"],
        summary_json["overall"]["dose_back_rotation_s"]["mean"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    ax.bar(component_labels, component_means, color=["#4c78a8", "#54a24b"])
    ax.set_ylabel("Mean BEV reorientation time [s]")
    ax.set_title("DoTA BEV reorientation components")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.subplots_adjust(bottom=0.28, wspace=0.35)
    fig.savefig(run_dir / "figures" / "timing_volume_breakdown.png", bbox_inches="tight")
    plt.close(fig)

    write_publication_preprocessing_figure(
        run_dir,
        projection_time_s=projection_time_s,
        cropping_time_s=cropping_time_s,
        inference_time_ms=inference_time_ms,
        ct_rotation_time_s=float(summary_json["overall"]["ct_rotation_total_s"]["mean"]),
        dose_rotation_time_s=float(summary_json["overall"]["dose_rotation_total_s"]["mean"]),
    )


def write_publication_preprocessing_figure(
    run_dir: Path,
    *,
    projection_time_s: float,
    cropping_time_s: float,
    inference_time_ms: float,
    ct_rotation_time_s: float,
    dose_rotation_time_s: float,
) -> None:
    colors = ["#00CECE", "#6441CD", "#C90000", "#077312"]
    label_fontsize = 13
    title_fontsize = 15
    tick_fontsize = 12
    annotation_fontsize = 11
    legend_fontsize = 11
    inference_time_s = inference_time_ms / 1000.0
    ct_rotation_to_bev_s = ct_rotation_time_s
    dose_rotation_from_bev_s = dose_rotation_time_s
    dota_bev_reinterpolation_s = ct_rotation_to_bev_s + dose_rotation_from_bev_s
    x = np.arange(1)
    width = 0.16

    fig = plt.figure(figsize=(14, 6.5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1], hspace=0.3, wspace=0.38)
    ax = fig.add_subplot(gs[0])

    rects_projection = ax.bar(
        x - 1.5 * width,
        [projection_time_s],
        width,
        label="ADoTA: analytical beamlet projection",
        color=colors[1],
        edgecolor="black",
        linewidth=0.5,
    )
    rects_ct_forward = ax.bar(
        x - 0.5 * width,
        [ct_rotation_to_bev_s],
        width,
        label="DoTA: rotate CT subvolume to BEV",
        color=colors[0],
        edgecolor="black",
        linewidth=0.5,
    )
    rects_dose_back = ax.bar(
        x + 0.5 * width,
        [dose_rotation_from_bev_s],
        width,
        label="DoTA: rotate predicted dose back",
        color=colors[3],
        edgecolor="black",
        linewidth=0.5,
    )
    rects_dota_total = ax.bar(
        x + 1.5 * width,
        [dota_bev_reinterpolation_s],
        width,
        label="DoTA: two rotations total",
        color=colors[2],
        edgecolor="black",
        linewidth=0.5,
    )

    for bar_container in [rects_projection, rects_ct_forward, rects_dose_back, rects_dota_total]:
        for bar in bar_container:
            height = bar.get_height()
            if height > 0.001:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_fontsize,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="gray",
                        alpha=0.7,
                        linewidth=0.5,
                    ),
                )

    ax.set_ylabel("Beamlet-specific step time [s]", fontsize=label_fontsize, weight="bold")
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=1.1)
    ax.set_xlabel("Mean across tested beamlets", fontsize=label_fontsize, weight="bold")
    ax.set_title("Beamlet-specific preprocessing", fontsize=title_fontsize, weight="bold")
    ax.set_xticklabels([], fontsize=tick_fontsize, weight="bold")
    ax.set_xticks([])
    ax.grid(linestyle="--", which="both", axis="y", alpha=0.3, linewidth=0.8, zorder=0)
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=legend_fontsize,
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor="black",
        framealpha=0.95,
        columnspacing=1.0,
        handlelength=2.0,
    )

    ax2 = fig.add_subplot(gs[1])
    x2 = np.arange(2)
    dota_time = dota_bev_reinterpolation_s
    adota_time = projection_time_s

    rects_with = ax2.bar(
        x2[0],
        dota_time,
        0.6,
        label="DoTA regime",
        color="#FF0000",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )
    rects_without = ax2.bar(
        x2[1],
        adota_time,
        0.6,
        label="ADoTA regime",
        color="#6441CD",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.85,
    )

    ax2.set_ylabel("Beamlet-specific step time [s]", fontsize=label_fontsize, weight="bold")
    ax2.tick_params(axis="both", labelsize=tick_fontsize, width=1.1)
    ax2.set_xticks(x2)
    ax2.grid(linestyle="--", which="both", axis="y", alpha=0.3, linewidth=0.8, zorder=0)
    ax2.set_title("Direct replacement comparison", fontsize=title_fontsize, weight="bold")
    ax2.set_xticklabels(
        ["DoTA\nBEV rotations", "ADoTA\nprojection"],
        fontsize=tick_fontsize,
        weight="bold",
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.2)
    ax2.spines["bottom"].set_linewidth(1.2)

    for bar_container in [rects_with, rects_without]:
        for bar in bar_container:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}s",
                ha="center",
                va="bottom",
                fontsize=annotation_fontsize,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8,
                ),
            )

    ax2.text(
        0.5,
        -0.18,
        "DoTA BEV step = CT to BEV + dose back",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85,
        ),
    )

    fig.subplots_adjust(bottom=0.32, left=0.08, right=0.98, top=0.88, wspace=0.38)
    output_stem = run_dir / "figures" / "preprocessing_times_per_beamlet_rotvsproj"
    fig.savefig(output_stem.with_suffix(".png"), format="png", bbox_inches="tight", dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


@app.command()
def main(
    model_name: str = typer.Option(
        "DoTA_v3_grid_search_v11",
        "--model-name",
        help="Model directory name under --model-hub.",
    ),
    model_fname: str = typer.Option(
        "best_model.pth",
        "--model-fname",
        help="Model weights filename inside the model directory.",
    ),
    model_hub: Path = typer.Option(
        ROOT_DIR / "models",
        "--model-hub",
        help="Directory containing model subdirectories.",
    ),
    test_data: list[str] = typer.Option(
        DEFAULT_TEST_DATA,
        "--test-data",
        help="Dataset as LABEL=PATH. Repeatable.",
    ),
    device_index: int = typer.Option(
        0,
        "--device-index",
        help="CUDA device index, or -1 for CPU.",
    ),
    downsampling_method: str = typer.Option(
        "interpolation",
        "--downsampling-method",
        help="Downsampling method passed to get_single_record.",
    ),
    max_samples: Optional[int] = typer.Option(
        5,
        "--max-samples",
        min=1,
        help="Maximum samples per dataset for this smoke timing run.",
    ),
    sample_seed: int = typer.Option(
        0,
        "--sample-seed",
        help="Random seed used when selecting a subset of samples.",
    ),
    repeats: int = typer.Option(
        1,
        "--repeats",
        min=1,
        help="Timed repeats for each axis rotation.",
    ),
    rotation_backend: str = typer.Option(
        "scipy",
        "--rotation-backend",
        help="Rotation backend for the smoke test: scipy or torch.",
    ),
    rotation_volume: str = typer.Option(
        "ct",
        "--rotation-volume",
        help="Deprecated; all raw volumes are rotated and timed separately.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        min=1,
        help="Batch size metadata stored with timing outputs. Inference batching is not implemented in this smoke stage.",
    ),
    projection_time_s: float = typer.Option(
        0.039,
        "--projection-time-s",
        help="Fixed CPU beamlet-shape construction time stored in config.",
    ),
    cropping_time_s: float = typer.Option(
        0.152,
        "--cropping-time-s",
        help="Fixed ray-tracer/cropping time stored in config.",
    ),
    inference_time_ms: float = typer.Option(
        1.5,
        "--inference-time-ms",
        help="Fixed single-beamlet inference time in milliseconds used in summary figures.",
    ),
    generate_sample_figures: bool = typer.Option(
        True,
        "--sample-figures/--no-sample-figures",
        help="Generate per-sample CT/flux/dose validation figures. Summary figures are always generated.",
    ),
    full_run: bool = typer.Option(
        False,
        "--full",
        help="Process all samples from each dataset and skip per-sample validation figures.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Output directory. Defaults to runs/beamlet_timing_rotvsproj_YYYYMMDD_HHMMSS.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging."),
) -> None:
    """Load model/data and time sequential Y-then-X beamlet rotations."""
    if rotation_backend not in {"scipy", "torch"}:
        raise typer.BadParameter("--rotation-backend must be 'scipy' or 'torch'.")
    if rotation_volume not in {"ct", "flux", "ds", "dose"}:
        raise typer.BadParameter("--rotation-volume is deprecated, but must be one of ct, flux, ds, or dose if provided.")
    if downsampling_method not in {"interpolation", "avg_pooling"}:
        raise typer.BadParameter("--downsampling-method must be 'interpolation' or 'avg_pooling'.")

    if full_run:
        max_samples = None
        generate_sample_figures = False

    run_dir = setup_run_dir(output_dir)
    log_path = setup_logging(run_dir, verbose)
    logger = logging.getLogger(__name__)

    datasets = [parse_test_dataset(entry) for entry in test_data]
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not hyperparams_path.exists():
        raise FileNotFoundError(f"Model hyperparameters not found: {hyperparams_path}")

    device = get_device(device_index)
    logger.info("Run directory: %s", run_dir)
    logger.info("Log file: %s", log_path)
    logger.info("Loading model: %s", model_path)
    _model = load_model(model_path, hyperparams_path, device)
    logger.info("Model loaded on %s", device)

    write_config(
        run_dir,
        model_name=model_name,
        model_fname=model_fname,
        model_path=model_path,
        hyperparams_path=hyperparams_path,
        datasets=datasets,
        device=device,
        rotation_backend=rotation_backend,
        downsampling_method=downsampling_method,
        max_samples=max_samples,
        sample_seed=sample_seed,
        repeats=repeats,
        batch_size=batch_size,
        rotation_volume=rotation_volume,
        projection_time_s=projection_time_s,
        cropping_time_s=cropping_time_s,
        inference_time_ms=inference_time_ms,
        generate_sample_figures=generate_sample_figures,
        full_run=full_run,
    )

    config = EvaluationConfig()
    rows: list[SampleTimingRow] = []
    total_selected = 0

    for dataset in datasets:
        sample_ids = discover_sample_ids(dataset.path)
        selected_ids = select_sample_ids(sample_ids, max_samples, sample_seed)
        total_selected += len(selected_ids)
        logger.info(
            "Dataset %s: discovered %d samples, selected %d",
            dataset.label,
            len(sample_ids),
            len(selected_ids),
        )

        for sample_id in selected_ids:
            preview_path = run_dir / "rotated_previews" / f"{dataset.label}_{sample_id}_{rotation_backend}_all_volumes.npz"
            figure_path = run_dir / "figures" / f"{dataset.label}_{sample_id}_{rotation_backend}_ct_flux_comparison.png"
            base_row = dict(
                dataset_label=dataset.label,
                dataset_path=str(dataset.path),
                sample_id=sample_id,
                device=str(device),
                rotation_backend=rotation_backend,
                repeats=repeats,
                batch_size=batch_size,
                raw_shape_hwd="",
                full_shape_dhw="",
                model_shape_dhw="",
                energy_mev=None,
                ba0_deg=None,
                ba1_deg=None,
                rotation_y_deg=None,
                rotation_x_deg=None,
                load_record_s=None,
                ct_load_s=None,
                flux_load_s=None,
                dose_load_s=None,
                raw_load_total_s=None,
                ct_rotation_y_s=None,
                ct_rotation_x_s=None,
                ct_rotation_total_s=None,
                flux_rotation_y_s=None,
                flux_rotation_x_s=None,
                flux_rotation_total_s=None,
                dose_rotation_y_s=None,
                dose_rotation_x_s=None,
                dose_rotation_total_s=None,
                all_rotation_total_s=None,
                rotated_preview_npz="",
                comparison_figure_png="",
            )
            try:
                ok, reason = has_required_files(sample_id, dataset.path)
                if not ok:
                    rows.append(SampleTimingRow(**base_row, status="skipped", error=reason))
                    logger.warning("Skipping %s/%s: %s", dataset.label, sample_id, reason)
                    continue

                start = perf_counter()
                x, energy, _y, beamlet_angles = get_single_record(
                    sample_id,
                    str(dataset.path),
                    scale=config.scale,
                    normalize_flux=config.normalize_flux,
                    downsampling_method=downsampling_method,
                    beamlet_angle=True,
                )
                load_record_s = perf_counter() - start

                if beamlet_angles is None or len(beamlet_angles) < 2:
                    raise ValueError(f"Missing two beamlet angles: {beamlet_angles!r}")
                ba0 = float(beamlet_angles[0])
                ba1 = float(beamlet_angles[1])
                rotation_y_deg = -ba1
                rotation_x_deg = ba0
                energy_mev = denormalize_energy(float(energy.item()), config.scale)

                raw_hwd = np.load(dataset.path / f"{sample_id}_ct.npy", mmap_mode="r")
                ct_dhw, ct_load_s = load_raw_volume_dhw(dataset.path, sample_id, "ct")
                flux_dhw, flux_load_s = load_raw_volume_dhw(dataset.path, sample_id, "flux")
                dose_dhw, dose_load_s = load_raw_volume_dhw(dataset.path, sample_id, "ds")
                raw_load_total_s = ct_load_s + flux_load_s + dose_load_s

                def rotate_and_time(volume_dhw: np.ndarray) -> tuple[np.ndarray, float, float, float]:
                    rotated, first, second = rotate_lateral_axes_sequential(
                        volume_dhw,
                        angle_y_deg=rotation_y_deg,
                        angle_x_deg=rotation_x_deg,
                        repeats=repeats,
                        backend=rotation_backend,
                        device=device,
                    )
                    total_s = first.pure_median + second.pure_median
                    return rotated, first.pure_median, second.pure_median, total_s

                rotated_ct, ct_rotation_y_s, ct_rotation_x_s, ct_rotation_total_s = rotate_and_time(ct_dhw)
                rotated_flux, flux_rotation_y_s, flux_rotation_x_s, flux_rotation_total_s = rotate_and_time(flux_dhw)
                rotated_dose, dose_rotation_y_s, dose_rotation_x_s, dose_rotation_total_s = rotate_and_time(dose_dhw)
                all_rotation_total_s = ct_rotation_total_s + flux_rotation_total_s + dose_rotation_total_s

                comparison_figure_png = ""
                if generate_sample_figures:
                    compare_two_inputs(
                        original_input=np.stack((ct_dhw, flux_dhw), axis=0),
                        rotated_input=np.stack((rotated_ct, rotated_flux), axis=0),
                        original_dose=dose_dhw,
                        rotated_dose=rotated_dose,
                        initial_energy=energy_mev,
                        beamlet_angles=(ba0, ba1),
                        rotation_angles=(rotation_y_deg, rotation_x_deg),
                        figure_path=str(figure_path),
                    )
                    comparison_figure_png = str(figure_path)

                np.savez_compressed(
                    preview_path,
                    original_ct=ct_dhw,
                    original_flux=flux_dhw,
                    original_dose=dose_dhw,
                    rotated_ct=rotated_ct,
                    rotated_flux=rotated_flux,
                    rotated_dose=rotated_dose,
                    ba0_deg=ba0,
                    ba1_deg=ba1,
                    rotation_y_deg=rotation_y_deg,
                    rotation_x_deg=rotation_x_deg,
                )

                row_data = base_row.copy()
                row_data.update(
                    raw_shape_hwd=str(tuple(raw_hwd.shape)),
                    full_shape_dhw=str(tuple(ct_dhw.shape)),
                    model_shape_dhw=str(tuple(x.shape[1:])),
                    energy_mev=energy_mev,
                    ba0_deg=ba0,
                    ba1_deg=ba1,
                    rotation_y_deg=rotation_y_deg,
                    rotation_x_deg=rotation_x_deg,
                    load_record_s=load_record_s,
                    ct_load_s=ct_load_s,
                    flux_load_s=flux_load_s,
                    dose_load_s=dose_load_s,
                    raw_load_total_s=raw_load_total_s,
                    ct_rotation_y_s=ct_rotation_y_s,
                    ct_rotation_x_s=ct_rotation_x_s,
                    ct_rotation_total_s=ct_rotation_total_s,
                    flux_rotation_y_s=flux_rotation_y_s,
                    flux_rotation_x_s=flux_rotation_x_s,
                    flux_rotation_total_s=flux_rotation_total_s,
                    dose_rotation_y_s=dose_rotation_y_s,
                    dose_rotation_x_s=dose_rotation_x_s,
                    dose_rotation_total_s=dose_rotation_total_s,
                    all_rotation_total_s=all_rotation_total_s,
                    rotated_preview_npz=str(preview_path),
                    comparison_figure_png=comparison_figure_png,
                )
                rows.append(SampleTimingRow(**row_data, status="ok"))
                logger.info(
                    "%s/%s rotated all volumes: ba=(%.3f, %.3f), ct=%.4fs flux=%.4fs dose=%.4fs total=%.4fs",
                    dataset.label,
                    sample_id,
                    ba0,
                    ba1,
                    ct_rotation_total_s,
                    flux_rotation_total_s,
                    dose_rotation_total_s,
                    all_rotation_total_s,
                )
            except Exception as exc:
                error = str(exc) + "\n" + traceback.format_exc()
                rows.append(SampleTimingRow(**base_row, status="error", error=error))
                logger.exception("Failed %s/%s", dataset.label, sample_id)

    write_rows(run_dir, rows)
    write_timing_summary(
        run_dir,
        rows,
        projection_time_s=projection_time_s,
        cropping_time_s=cropping_time_s,
        inference_time_ms=inference_time_ms,
    )
    ok_rows = [row for row in rows if row.status == "ok"]
    logger.info("Processed %d selected samples; %d successful rotations", total_selected, len(ok_rows))
    logger.info("Per-sample timings: %s", run_dir / "per_sample_rotation_timing.csv")
    logger.info("Per-sample branch comparison: %s", run_dir / "per_sample_branch_comparison.csv")
    logger.info("Timing summary: %s", run_dir / "timing_summary.csv")
    logger.info("Rotated previews: %s", run_dir / "rotated_previews")


if __name__ == "__main__":
    app()
