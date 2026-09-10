"""Profile where a beamlet-scale GPU gamma evaluation spends its time (EXP-0008, E).

The beamlet benchmarks show single and double precision taking the same time,
which is consistent with fixed overhead dominating, but consistency is not
attribution. This script runs the torch backend under the PyTorch profiler on
representative beamlets and splits the wall time into what the profiler can
see: device kernel time by category, host-side CUDA runtime calls (launches,
synchronisations, memory copies, allocations) and the remainder, which is
Python executing the host-side convergence loop.

Profiling adds overhead of its own, so the wall times here are for attribution
only; headline performance comes from experiment A.

Example::

    uv run python scripts/gamma_profile_beamlet.py \\
        --pairs /scratch/mstryja/<run>/inputs/pairs.npz \\
        --sweep /scratch/mstryja/<run>/A/beamlet_matched_array.json \\
        --out-dir /scratch/mstryja/<run>/E --device cuda:0
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.benchmark_provenance import write_manifest  # noqa: E402
from src.metrics.gamma_beamlet_benchmark import RUNGS, GammaCase, build_gamma_call, environment_stamp  # noqa: E402
from src.metrics.gamma_beamlet_pairs import load_pairs  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(help="Profile beamlet-scale GPU gamma with the PyTorch profiler.", add_completion=False)

# How a profiler event is attributed. Order matters: the first match wins.
CATEGORIES = (
    ("memcpy", ("Memcpy", "cudaMemcpy")),
    ("memset", ("Memset", "cudaMemset")),
    ("synchronise", ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")),
    ("launch", ("cudaLaunchKernel",)),
    ("allocate", ("cudaMalloc", "cudaFree", "cudaHostAlloc")),
)


def _category(name: str) -> str:
    for label, needles in CATEGORIES:
        if any(needle in name for needle in needles):
            return label
    return "other"


def representative_beamlets(sweep_rows: List[dict], criterion: str) -> Dict[str, str]:
    """Easy, median and hard beamlets by the reference rung's median time."""
    times = {
        row["sample_id"]: row["seconds_median"]
        for row in sweep_rows
        if row["rung"] == 1 and row["path"] == "array" and row["criterion"] == criterion
    }
    ordered = sorted(times, key=times.get)
    return {"easy": ordered[0], "median": ordered[len(ordered) // 2], "hard": ordered[-1]}


def profile_one(pair, case: GammaCase, rung, scale: dict, device: str, iterations: int, trace_path: Path) -> dict:
    """Profile ``iterations`` warmed-up calls and summarise the events."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    prepared = build_gamma_call(pair, case, rung, scale, device=device, path="array")
    target = torch.device(prepared.target)
    prepared.call()
    torch.cuda.synchronize(target)

    torch.cuda.synchronize(target)
    plain_start = perf_counter()
    for _ in range(iterations):
        prepared.call()
    torch.cuda.synchronize(target)
    plain_wall = (perf_counter() - plain_start) / iterations

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        torch.cuda.synchronize(target)
        start = perf_counter()
        for _ in range(iterations):
            prepared.call()
        torch.cuda.synchronize(target)
        profiled_wall = (perf_counter() - start) / iterations
    prof.export_chrome_trace(str(trace_path))

    device_us = defaultdict(float)
    host_us = defaultdict(float)
    counts = defaultdict(int)
    kernel_names = defaultdict(float)
    for event in prof.key_averages():
        label = _category(event.key)
        is_device = str(getattr(event, "device_type", "")).endswith("CUDA")
        if is_device:
            device_us[label] += float(event.self_device_time_total)
            if label == "other":
                kernel_names[event.key] += float(event.self_device_time_total)
        else:
            host_us[label] += float(event.self_cpu_time_total)
        counts[f"{'device' if is_device else 'host'}:{label}"] += int(event.count)

    per_iter = lambda microseconds: microseconds / iterations / 1e6  # noqa: E731
    device_kernel_s = per_iter(device_us["other"])
    host_runtime_s = per_iter(sum(v for k, v in host_us.items() if k != "other"))
    return {
        "sample_id": pair.sample_id,
        "dtype": rung.dtype,
        "iterations": iterations,
        "wall_unprofiled_s": plain_wall,
        "wall_profiled_s": profiled_wall,
        "device_kernel_s": device_kernel_s,
        "device_memcpy_s": per_iter(device_us["memcpy"]),
        "device_memset_s": per_iter(device_us["memset"]),
        "host_launch_s": per_iter(host_us["launch"]),
        "host_synchronise_s": per_iter(host_us["synchronise"]),
        "host_memcpy_call_s": per_iter(host_us["memcpy"]),
        "host_allocate_s": per_iter(host_us["allocate"]),
        "host_other_cpu_s": per_iter(host_us["other"]),
        "kernel_launches_per_call": counts["host:launch"] / iterations,
        "device_kernels_per_call": counts["device:other"] / iterations,
        "synchronise_calls_per_call": counts["host:synchronise"] / iterations,
        "memcpy_calls_per_call": counts["host:memcpy"] / iterations,
        "host_runtime_total_s": host_runtime_s,
        "iterations_in_loop": prepared.stats.get("iterations"),
        "interp_samples": prepared.stats.get("interp_samples"),
        "top_kernels_us_per_call": {
            name: value / iterations for name, value in sorted(kernel_names.items(), key=lambda kv: -kv[1])[:8]
        },
        "trace": str(trace_path),
    }


@app.command()
def main(
    pairs_path: Path = typer.Option(..., "--pairs"),
    sweep: Path = typer.Option(..., help="Experiment A array-path sweep JSON, to pick easy/median/hard beamlets."),
    out_dir: Path = typer.Option(...),
    device: str = typer.Option("cuda:0"),
    criterion: str = typer.Option("1%/1mm/10%", help="Criterion whose reference times rank the beamlets."),
    iterations: int = typer.Option(5),
    provenance: bool = typer.Option(False),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Profile easy, median and hard beamlets in float64 and float32."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    pairs, scale = load_pairs(pairs_path)
    by_id = {pair.sample_id: pair for pair in pairs}
    rows = json.loads(sweep.read_text())["rows"]
    chosen = representative_beamlets(rows, criterion)
    percent, _, rest = criterion.partition("%/")
    millimetres, _, cutoff = rest.partition("mm/")
    case = GammaCase(float(percent), float(millimetres), float(cutoff.rstrip("%")), 10, 2.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    if provenance:
        write_manifest(out_dir, device=device, repos={"adota": PROJECT_ROOT, "reports": PROJECT_ROOT / "reports"},
                       inputs={"pairs": pairs_path, "sweep": sweep},
                       extra={"chosen": chosen, "criterion": criterion, "iterations": iterations})

    results: List[dict] = []
    for role, sample_id in chosen.items():
        for rung_name in ("rung3", "rung4"):
            rung = RUNGS[rung_name]
            trace = out_dir / f"trace_{role}_{rung.dtype}.json"
            result = profile_one(by_id[sample_id], case, rung, scale, device, iterations, trace)
            result["role"] = role
            results.append(result)
            logger.info("%s %s: wall %.4f s, kernels %.4f s, launches %.0f/call, host other %.4f s",
                        role, rung.dtype, result["wall_unprofiled_s"], result["device_kernel_s"],
                        result["kernel_launches_per_call"], result["host_other_cpu_s"])

    (out_dir / "profile.json").write_text(
        json.dumps({"environment": environment_stamp(), "criterion": criterion, "results": results}, indent=1) + "\n"
    )
    fields = [k for k in results[0] if k not in ("top_kernels_us_per_call",)]
    with (out_dir / "profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    typer.echo(f"Wrote {len(results)} profiles to {out_dir}")


if __name__ == "__main__":
    app()
