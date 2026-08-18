"""Fair optimized-GPU reinterpretation benchmark (batched flux vs batched BEV sampling).

Answers reviewer round-2: the old study compared CPU analytical flux vs CPU
unoptimized rotation and called it "end-to-end". This benchmark optimizes BOTH
sides symmetrically and reports the methodology the reviewer asked for.

It produces, per swept ``batch_size``:
  * a 3x2 timing matrix -- {CPU, GPU per-item, GPU batched} x {ADoTA flux (x1 per
    spot), DoTA rotation (2 resamples per spot)} -- in fp32, each cell with full
    variability (median/mean/std/IQR/CI/min/max) and, for GPU, compute-only vs
    compute+transfer (H2D + D2H);
  * throughput curves (beamlets/s vs batch size);
  * two MEASURED plan runs through the streaming harness (reinterpretation_mode
    adota_flux and dota_rotation), reported as compute-only (excludes disk I/O).

Interpolation: DoTA CPU = scipy.ndimage.affine_transform order=1 (trilinear);
GPU = grid_sample bilinear/align_corners. ADoTA flux = analytic, no interpolation.
Threading and hardware are pinned and logged.

Usage:
  uv run python scripts/reinterp_gpu_benchmark.py --config scripts/config_reinterp_gpu_benchmark.yaml
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adota.config import load_yaml_config, setup_logging, setup_run_directory
from src.adota.utils import load_model
from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.flux import (
    flux_projection,
    flux_projection_gpu,
    flux_projection_gpu_batched,
    flux_spatial_spread,
)
from src.beamlets.streaming import StreamingConfig, run_streaming_pipeline
from src.evaluation.cli import resolve_device
from src.image_processing.rotation import (
    rotate_beamlet_crop,
    rotate_beamlet_crops_batched,
)
from src.loaders.dir_based import DEFAULT_SCALE
from src.loaders.plan_directory import load_plan_directory

# Reuse the trusted staged CPU extraction from the existing study.
from scripts.beamlet_bev_rotation_timing import _load_beamlet, run_staged_cpu_extraction

logger = logging.getLogger(__name__)
app = typer.Typer(help="Fair optimized-GPU reinterpretation benchmark.")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _stats_ms(seconds: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Summarise a list of per-unit seconds as milliseconds with variability."""
    if not seconds:
        return {"n": 0}
    a = np.asarray(seconds, dtype=float) * 1000.0
    rng = np.random.default_rng(seed)
    if a.size > 1:
        means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
        ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])
    else:
        ci_lo = ci_hi = float(a[0])
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "iqr_lo": float(np.percentile(a, 25)),
        "iqr_hi": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
    }


# --------------------------------------------------------------------------- #
# Load beamlets into memory as timing items
# --------------------------------------------------------------------------- #
def _load_items(beamlets_dir: Path, bdl: BeamDataLibrary, grid_factor: int,
                n_benchmark: Optional[int], seed: int) -> list[dict]:
    spot_ids = sorted(p.name.removesuffix("_sim_res.json")
                      for p in beamlets_dir.glob("*_sim_res.json"))
    if not spot_ids:
        raise FileNotFoundError(f"No beamlets under {beamlets_dir}")
    if n_benchmark is not None and n_benchmark < len(spot_ids):
        spot_ids = sorted(random.Random(seed).sample(spot_ids, n_benchmark))
    items: list[dict] = []
    for sid in spot_ids:
        ct, _flux, sim_res = _load_beamlet(beamlets_dir, sid)
        energy = float(sim_res["simulation_log"]["energy"][0])
        items.append({
            "ct": ct,
            "angles": tuple(float(a) for a in sim_res["simulation_log"]["beamlet_angles"]),
            "re_proj": sim_res["rays_entrence_point_proj"],
            "sigmas": flux_spatial_spread(bdl, energy),
            "energy": energy,
        })
    logger.info("Loaded %d beamlet items from %s", len(items), beamlets_dir)
    return items


def _batches(items: list[dict], bs: int) -> list[list[dict]]:
    return [items[i:i + bs] for i in range(0, len(items), bs)]


# --------------------------------------------------------------------------- #
# Timing protocol (uniform across cells)
# --------------------------------------------------------------------------- #
# Every cell reports one value per UNIT (a beamlet for the CPU baselines, a batch
# for the GPU-batched cells), each already reduced to the MEDIAN of ``repeats``
# timed calls after one discarded warm-up. Cross-unit statistics (median/IQR/CI)
# are then taken over those per-unit values. GPU per-item = the batch-size-1 point
# of the sweep (same batched kernel, batch of one), so there is no separate,
# inconsistently-measured per-item tier.
# --------------------------------------------------------------------------- #
def _pair_scipy_time(ct: np.ndarray, angles, repeats: int) -> float:
    """Median wall time of a forward+inverse scipy rotation pair (one 'DoTA' spot)."""
    rotate_beamlet_crop(ct, angles, backend="scipy", repeats=1)  # warmup
    rotate_beamlet_crop(ct, angles, inverse=True, backend="scipy", repeats=1)
    ts = []
    for _ in range(repeats):
        t0 = perf_counter()
        rotate_beamlet_crop(ct, angles, backend="scipy", repeats=1)
        rotate_beamlet_crop(ct, angles, inverse=True, backend="scipy", repeats=1)
        ts.append(perf_counter() - t0)
    return float(np.median(ts))


def _time_cpu_baselines(items: list[dict], repeats: int, gf: int) -> dict:
    """Batch-independent CPU baselines (timed once over ALL beamlets).

    ADoTA flux (analytic, x1 per spot) and DoTA rotation (2 scipy resamples per
    spot). Per-beamlet value = median of ``repeats`` after one warm-up.
    """
    spacing = np.asarray([gf, gf, gf], dtype=np.float32)
    flux_s: list[float] = []
    rot_s: list[float] = []
    for it in items:
        shape = it["ct"].shape
        flux_projection(it["re_proj"], it["angles"], it["sigmas"], shape, spacing=spacing)  # warmup
        ts = []
        for _ in range(repeats):
            t0 = perf_counter()
            flux_projection(it["re_proj"], it["angles"], it["sigmas"], shape, spacing=spacing)
            ts.append(perf_counter() - t0)
        flux_s.append(float(np.median(ts)))
        rot_s.append(_pair_scipy_time(it["ct"], it["angles"], repeats))
    n = len(items)
    out = {}
    for name, arr in (("flux_cpu", flux_s), ("rot_cpu", rot_s)):
        total = float(sum(arr))
        out[name] = {
            "stats_ms": _stats_ms(arr),
            "throughput_beamlets_per_s": (n / total if total > 0 else float("nan")),
            "raw_ms": [x * 1000.0 for x in arr],
        }
    return out


def _median_batched_call(fn, repeats: int, on_cuda: bool, device) -> float:
    """Median of ``repeats`` timed calls of a batched GPU op, after one warm-up."""
    fn()  # warmup
    if on_cuda:
        torch.cuda.synchronize(device)
    ts = []
    for _ in range(repeats):
        if on_cuda:
            torch.cuda.synchronize(device)
        t0 = perf_counter()
        fn()
        if on_cuda:
            torch.cuda.synchronize(device)
        ts.append(perf_counter() - t0)
    return float(np.median(ts))


def _time_gpu_batched_for_batch(items: list[dict], bs: int, repeats: int,
                                device: torch.device, gf: int) -> dict:
    """Time the four GPU-batched cells at one batch size (bs=1 => per-item GPU)."""
    spacing = np.asarray([gf, gf, gf], dtype=np.float32)
    dev = str(device)
    on_cuda = device.type == "cuda"
    cells = ("flux_gpu_batched_pure", "flux_gpu_batched_practical",
             "rot_gpu_batched_pure", "rot_gpu_batched_practical")
    samples: dict[str, list[float]] = {k: [] for k in cells}
    totals: dict[str, float] = {k: 0.0 for k in cells}

    batch_list = _batches(items, bs)
    for batch in batch_list:
        n = len(batch)
        shape = batch[0]["ct"].shape
        ents = [it["re_proj"] for it in batch]
        angs = [it["angles"] for it in batch]
        sigs = [it["sigmas"] for it in batch]
        crops = [it["ct"] for it in batch]

        # ADoTA flux batched: pure (resident) vs practical (+ D2H).
        fpure = _median_batched_call(
            lambda: flux_projection_gpu_batched(ents, angs, sigs, shape, spacing=spacing,
                                                device=dev, return_numpy=False),
            repeats, on_cuda, device)
        fprac = _median_batched_call(
            lambda: flux_projection_gpu_batched(ents, angs, sigs, shape, spacing=spacing,
                                                device=dev, return_numpy=True),
            repeats, on_cuda, device)
        samples["flux_gpu_batched_pure"].append(fpure / n); totals["flux_gpu_batched_pure"] += fpure
        samples["flux_gpu_batched_practical"].append(fprac / n); totals["flux_gpu_batched_practical"] += fprac

        # DoTA rotation batched: 2 resamples (fwd + inverse); kernels return medians.
        fwd = rotate_beamlet_crops_batched(crops, angs, inverse=False, device=dev,
                                           dtype=torch.float32, repeats=repeats, return_numpy=True)
        inv = rotate_beamlet_crops_batched(crops, angs, inverse=True, device=dev,
                                           dtype=torch.float32, repeats=repeats, return_numpy=True)
        rpure = fwd.pure_median + inv.pure_median
        rprac = fwd.practical_median + inv.practical_median
        samples["rot_gpu_batched_pure"].append(rpure / n); totals["rot_gpu_batched_pure"] += rpure
        samples["rot_gpu_batched_practical"].append(rprac / n); totals["rot_gpu_batched_practical"] += rprac

    n_spots = len(items)
    throughput = {k: (n_spots / v if v > 0 else float("nan")) for k, v in totals.items()}
    return {
        "batch_size": bs,
        "n_spots": n_spots,
        "n_batches": len(batch_list),
        "stats_ms": {k: _stats_ms(v) for k, v in samples.items()},
        "throughput_beamlets_per_s": throughput,
        "raw_ms": {k: [x * 1000.0 for x in v] for k, v in samples.items()},
    }


# --------------------------------------------------------------------------- #
# Measured plan runs (adota_flux + dota_rotation) through the streaming harness
# --------------------------------------------------------------------------- #
def _run_measured_plans(plan_directory, model, device, out_dir: Path,
                        grid_factor: int, batch_size: int) -> dict:
    results = {}
    for mode in ("adota_flux", "dota_rotation"):
        # Fairness: the ADoTA plan gets BATCHED GPU flux, the counterpart to the
        # batched GPU BEV rotation the DoTA plan gets.
        cfg = StreamingConfig(
            grid_factor=grid_factor, batch_size=batch_size, flux_on_gpu=True,
            flux_batched=(mode == "adota_flux"), flux_device=str(device),
            precision="fp32", reinterpretation_mode=mode,
        )
        summary = run_streaming_pipeline(
            plan_directory, model, device, out_dir / f"Dose_{mode}.mhd", cfg
        )
        timing = summary["timing"]
        compute = {k: float(v) for k, v in timing.items() if k != "write"}
        results[mode] = {
            "n_spots": int(summary["n_spots"]),
            "n_fields": int(summary["n_fields"]),
            "compute_s_by_step": compute,
            "compute_total_s": float(sum(compute.values())),
            "write_s_excluded": float(timing.get("write", 0.0)),
        }
    a = results["adota_flux"]["compute_total_s"]
    d = results["dota_rotation"]["compute_total_s"]
    results["speedup_dota_over_adota"] = d / a if a > 0 else float("nan")
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _hardware_info(device: torch.device, num_threads: int) -> dict:
    info = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "requested_num_threads": num_threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "precision": "fp32",
        "interpolation": {
            "dota_cpu": "scipy.ndimage.affine_transform order=1 (trilinear), mode=constant",
            "dota_gpu": "grid_sample bilinear, align_corners=True, padding=zeros",
            "adota_flux": "analytic Gaussian projection (no interpolation)",
        },
    }
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def _row(bs, lab, s, tp, divider=False):
    return ([str(bs), lab, f"{s['median']:.4f}", f"{s['mean']:.4f}±{s['std']:.4f}",
             f"[{s['iqr_lo']:.4f},{s['iqr_hi']:.4f}]",
             f"[{s['ci95_lo']:.4f},{s['ci95_hi']:.4f}]", f"{tp:.0f}"], divider)


def _format_matrix_table(cpu_baseline: dict, sweep: list[dict]) -> str:
    from prettytable import PrettyTable
    t = PrettyTable()
    t.field_names = ["batch", "cell", "median ms/spot", "mean±std", "IQR",
                     "95% CI", "beamlets/s"]
    for col in t.field_names:
        t.align[col] = "r"
    t.align["cell"] = "l"
    # CPU baselines (batch-independent), timed once over all beamlets.
    for key, lab in (("flux_cpu", "ADoTA flux  CPU (baseline)"),
                     ("rot_cpu", "DoTA rot    CPU 2x (baseline)")):
        c = cpu_baseline[key]
        row, div = _row("all", lab, c["stats_ms"], c["throughput_beamlets_per_s"],
                        divider=(key == "rot_cpu"))
        t.add_row(row, divider=div)
    # GPU-batched sweep (bs=1 => per-item GPU).
    labels = [
        ("flux_gpu_batched_pure", "ADoTA flux  GPU batch (compute)"),
        ("flux_gpu_batched_practical", "ADoTA flux  GPU batch (+xfer)"),
        ("rot_gpu_batched_pure", "DoTA rot    GPU batch (compute)"),
        ("rot_gpu_batched_practical", "DoTA rot    GPU batch (+xfer)"),
    ]
    for entry in sweep:
        bs = entry["batch_size"]
        st = entry["stats_ms"]
        tp = entry["throughput_beamlets_per_s"]
        for key, lab in labels:
            s = st.get(key, {})
            if not s or s.get("n", 0) == 0:
                continue
            row, div = _row(bs, lab, s, tp[key],
                            divider=(key == "rot_gpu_batched_practical"))
            t.add_row(row, divider=div)
    return t.get_string()


def _throughput_figure(sweep: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    series = [
        ("flux_gpu_batched_pure", "ADoTA flux GPU-batched (compute)", "#2a78d6", "-o"),
        ("flux_gpu_batched_practical", "ADoTA flux GPU-batched (+xfer)", "#2a78d6", "--o"),
        ("rot_gpu_batched_pure", "DoTA rot GPU-batched (compute)", "#e34948", "-s"),
        ("rot_gpu_batched_practical", "DoTA rot GPU-batched (+xfer)", "#e34948", "--s"),
    ]
    xs = [e["batch_size"] for e in sweep]
    for key, lab, color, style in series:
        ys = [e["throughput_beamlets_per_s"].get(key, float("nan")) for e in sweep]
        ax.plot(xs, ys, style, color=color, label=lab, lw=2, ms=7)
    ax.set_xlabel("Batch size", fontsize=14)
    ax.set_ylabel("Throughput [beamlets / s]", fontsize=14)
    ax.set_title("Reinterpretation throughput vs batch size (GPU-batched, fp32)", fontsize=13)
    ax.grid(True, ls=":", lw=0.6)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@app.command()
def main(
    config: Annotated[Optional[Path], typer.Option(help="YAML config.")] = None,
    plan_dir: Annotated[Optional[Path], typer.Option(help="OpenTPS plan dir.")] = None,
    model_name: Annotated[Optional[str], typer.Option()] = None,
    model_fname: Annotated[Optional[str], typer.Option()] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA index (-1=CPU).")] = None,
    runs_dir: Annotated[Optional[Path], typer.Option()] = None,
    grid_factor: Annotated[Optional[int], typer.Option()] = None,
    batch_sizes: Annotated[Optional[str], typer.Option(help="Comma-separated sweep, e.g. 8,16,32,56.")] = None,
    repeats: Annotated[Optional[int], typer.Option()] = None,
    n_benchmark: Annotated[Optional[int], typer.Option(help="Spots for the matrix (null=all).")] = None,
    num_threads: Annotated[Optional[int], typer.Option(help="CPU threads (pinned + logged).")] = None,
    n_spots: Annotated[Optional[int], typer.Option(help="Extraction subset.")] = None,
    seed: Annotated[Optional[int], typer.Option()] = None,
    run_measured_plan: Annotated[Optional[bool], typer.Option()] = None,
    overwrite: Annotated[Optional[bool], typer.Option()] = None,
    verbose: Annotated[Optional[bool], typer.Option()] = None,
) -> None:
    cfg = load_yaml_config(config) if config is not None else {}

    def pick(cli, key, default=None):
        return cli if cli is not None else cfg.get(key, default)

    plan_dir = Path(pick(plan_dir, "plan_dir"))
    model_name = pick(model_name, "model_name", "DoTA_v3_grid_search_v11")
    model_fname = pick(model_fname, "model_fname", "best_model.pth")
    device_index = pick(device_index, "device_index", 0)
    runs_dir = Path(pick(runs_dir, "runs_dir", ROOT_DIR / "runs"))
    grid_factor = int(pick(grid_factor, "grid_factor", 2))
    bs_raw = pick(batch_sizes, "batch_sizes", "1,8,16,32,56,112")
    batch_sizes_list = [int(b) for b in str(bs_raw).split(",") if str(b).strip()]
    repeats = int(pick(repeats, "repeats", 5))
    n_benchmark = pick(n_benchmark, "n_benchmark", None)  # None = all beamlets
    n_benchmark = int(n_benchmark) if n_benchmark is not None else None
    num_threads = int(pick(num_threads, "num_threads", 1))
    n_spots = pick(n_spots, "n_spots")
    seed = int(pick(seed, "seed", 0))
    run_measured_plan = bool(pick(run_measured_plan, "run_measured_plan", True))
    overwrite = bool(pick(overwrite, "overwrite", True))
    verbose = bool(pick(verbose, "verbose", False))

    # Pin threads BEFORE any heavy tensor op.
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)

    run_dir = setup_run_directory(runs_dir, prefix="reinterp_gpu_bench_", subdirs=("figures",))
    setup_logging(run_dir, verbose=verbose, log_filename="run.log")
    device = resolve_device(device_index)
    beamlets_dir = run_dir / "beamlets"

    logger.info("Run dir: %s | plan: %s | device: %s", run_dir, plan_dir, device)
    plan_directory = load_plan_directory(plan_dir, bdl_path=None)
    bdl = BeamDataLibrary.from_file(plan_directory.bdl_path)

    # Stage 1: staged CPU extraction (reuse the trusted path; stores every beamlet).
    run_staged_cpu_extraction(
        plan_directory, beamlets_dir, bdl_path=None, grid_factor=grid_factor,
        n_spots=n_spots, beams=None, overwrite=overwrite,
    )
    items = _load_items(beamlets_dir, bdl, grid_factor, n_benchmark, seed)

    hardware = _hardware_info(device, num_threads)
    logger.info("Hardware/threads: %s", json.dumps(hardware, indent=2))

    # Stage 2a: CPU baselines, timed ONCE over all beamlets (batch-independent).
    logger.info("Timing CPU baselines over %d beamlets ...", len(items))
    cpu_baseline = _time_cpu_baselines(items, repeats, grid_factor)

    # Stage 2b: the GPU-batched sweep (bs=1 gives the per-item GPU point).
    sweep = []
    for bs in batch_sizes_list:
        logger.info("Timing GPU-batched cells for batch_size=%d ...", bs)
        sweep.append(_time_gpu_batched_for_batch(items, bs, repeats, device, grid_factor))

    # Stage 3: measured plan runs.
    measured = None
    if run_measured_plan:
        logger.info("Loading model for measured plan runs ...")
        model = load_model(
            ROOT_DIR / "models" / model_name / model_fname,
            ROOT_DIR / "models" / model_name / "hyperparams.json", device,
        )
        measured = _run_measured_plans(
            plan_directory, model, device, run_dir, grid_factor,
            batch_size=max(batch_sizes_list),
        )

    report = {
        "hardware": hardware,
        "grid_factor": grid_factor,
        "repeats": repeats,
        "n_benchmark": len(items),
        "n_beamlets_all": len(items),
        "batch_sizes": batch_sizes_list,
        "cpu_baseline": {k: {kk: vv for kk, vv in c.items() if kk != "raw_ms"}
                         for k, c in cpu_baseline.items()},
        "matrix_sweep": [{k: v for k, v in e.items() if k != "raw_ms"} for e in sweep],
        "measured_plan": measured,
    }
    (run_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))
    # raw samples saved separately (large).
    (run_dir / "raw_samples.json").write_text(json.dumps({
        "cpu_baseline": {k: c["raw_ms"] for k, c in cpu_baseline.items()},
        "gpu_batched": {e["batch_size"]: e["raw_ms"] for e in sweep},
    }))

    logger.info("\n%s", _format_matrix_table(cpu_baseline, sweep))
    _throughput_figure(sweep, run_dir / "figures" / "throughput_vs_batch.png")
    if measured:
        logger.info(
            "MEASURED PLAN (compute-only, excl. I/O): ADoTA=%.2fs  DoTA=%.2fs  (DoTA/ADoTA=%.2fx)",
            measured["adota_flux"]["compute_total_s"],
            measured["dota_rotation"]["compute_total_s"],
            measured["speedup_dota_over_adota"],
        )
    logger.info("Report: %s", run_dir / "benchmark_report.json")
    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    app()
