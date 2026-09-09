# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Reduce the EXP-0008 raw results into the rows the report tables are built from.

One function per experiment. Each takes the parsed JSON a harness wrote and
returns a list of flat dicts, one per table row, with nothing computed that the
raw rows do not support: every number here can be traced to a raw file by the
identifiers the row carries. Rendering to LaTeX and CSV is a separate module,
:mod:`src.metrics.gamma_evidence_latex`, so the numbers can be tested without
the formatting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.metrics.gamma_beamlet_report import MatchedSetError, paired_speedups

__all__ = [
    "SHARED_CRITERIA",
    "RUNG_NAMES",
    "load_payload",
    "summarise_matched",
    "summarise_sweep_pass_rates",
    "summarise_beamlet_maps",
    "summarise_plan_maps",
    "summarise_pools",
    "projection_errors",
    "summarise_scaling",
    "summarise_profile",
    "validate_same_environment",
]

SHARED_CRITERIA: Tuple[str, ...] = ("1%/1mm/10%", "2%/2mm/10%", "3%/3mm/10%")

# Descriptive backend labels, so a table never has to say "rung 4".
RUNG_NAMES: Dict[int, str] = {
    1: "PyMedPhys, CPU",
    2: "PyTorch, CPU, float64",
    3: "PyTorch, GPU, float64",
    4: "PyTorch, GPU, float32",
}


def load_payload(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def validate_same_environment(
    payloads: Iterable[Dict[str, Any]], keys: Sequence[str] = ("gpu", "torch", "pymedphys")
) -> None:
    """Refuse to combine runs taken under different software or hardware.

    Raises:
        ValueError: If any of ``keys`` differs between the payloads' environments.
    """
    seen: Dict[str, set] = defaultdict(set)
    for payload in payloads:
        environment = payload.get("environment", {})
        for key in keys:
            seen[key].add(str(environment.get(key)))
    conflicts = {key: values for key, values in seen.items() if len(values) > 1}
    if conflicts:
        raise ValueError(f"runs were taken under different environments: {conflicts}")


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(array)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
        "sum": float(array.sum()),
    }


# ── Experiment A ────────────────────────────────────────────────────────────


def summarise_matched(rows: Sequence[Dict[str, Any]], tested_rungs: Sequence[int] = (2, 3, 4)) -> List[Dict[str, Any]]:
    """Paired speed-ups per (path, criterion, rung) on matched case sets.

    Rungs absent from a path are skipped; rungs present on a different case set
    than the reference raise, because that is the error this exists to catch.
    """
    summary: List[Dict[str, Any]] = []
    paths = sorted({row["path"] for row in rows})
    for path in paths:
        for criterion in SHARED_CRITERIA:
            for rung in tested_rungs:
                if not any(r["rung"] == rung and r["path"] == path and r["criterion"] == criterion for r in rows):
                    continue
                stats = paired_speedups(rows, rung, criterion=criterion, path=path)
                stats.pop("ratios")
                stats.pop("sample_ids")
                stats["tested_label"] = RUNG_NAMES[rung]
                summary.append(stats)
    return summary


def summarise_sweep_pass_rates(rows: Sequence[Dict[str, Any]], path: str = "array") -> List[Dict[str, Any]]:
    """Maximum and mean absolute pass-rate deviation from rung 1, per criterion and rung."""
    reference = {
        (r["criterion"], r["sample_id"]): r["pass_rate_pct"] for r in rows if r["rung"] == 1 and r["path"] == path
    }
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in rows:
        if row["rung"] == 1 or row["path"] != path:
            continue
        key = (row["criterion"], row["sample_id"])
        if key in reference:
            grouped[(row["criterion"], row["rung"])].append(abs(row["pass_rate_pct"] - reference[key]))
    return [
        {"criterion": criterion, "rung": rung, "label": RUNG_NAMES[rung], "n_cases": len(deltas),
         "max_abs_delta_pp": float(max(deltas)), "mean_abs_delta_pp": float(np.mean(deltas))}
        for (criterion, rung), deltas in sorted(grouped.items())
    ]


# ── Experiment B ────────────────────────────────────────────────────────────


def _aggregate_map_rows(rows: Sequence[Dict[str, Any]], key_fields: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[f] for f in key_fields)].append(row)
    summary = []
    for key, group in sorted(grouped.items(), key=lambda kv: tuple(str(k) for k in kv[0])):
        record = dict(zip(key_fields, key))
        record.update(
            {
                "n_comparisons": len(group),
                "other_dtype": group[0]["other_dtype"],
                "tested_label": RUNG_NAMES[group[0]["tested_rung"]],
                "baseline_label": RUNG_NAMES[group[0]["baseline_rung"]],
                "n_evaluated_total": int(sum(r["n_evaluated_both"] for r in group)),
                "max_abs_delta": float(max(r["max_abs_delta"] for r in group)),
                "mean_abs_delta": float(np.mean([r["mean_abs_delta"] for r in group])),
                "max_p99.99_abs_delta": float(max(r["p99.99_abs_delta"] for r in group)),
                "mask_disagreements": int(sum(r["mask_disagreements"] for r in group)),
                "boundary_crossings": int(sum(r["boundary_crossings"] for r in group)),
                "max_abs_pass_rate_delta_pp": float(max(abs(r["pass_rate_delta_pp"]) for r in group)),
                "n_bitwise_identical": int(sum(1 for r in group if r.get("bitwise_identical"))),
                "all_gates_passed": bool(all(r["passed"] for r in group)),
            }
        )
        summary.append(record)
    return summary


def summarise_beamlet_maps(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (tested, baseline, criterion) aggregate over the beamlets."""
    return _aggregate_map_rows(rows, ("tested_rung", "baseline_rung", "criterion"))


def summarise_plan_maps(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per plan, criterion and comparison, as measured."""
    return _aggregate_map_rows(rows, ("plan", "criterion", "tested_rung", "baseline_rung"))


# ── Experiment C ────────────────────────────────────────────────────────────


def _pool_config_row(pool: str, result: Dict[str, Any], kind: str) -> Dict[str, Any]:
    passes = result["passes"]
    walls = [p["wall_s"] for p in passes]
    per_key = "per_beamlet_s" if kind == "cached" else "per_beamlet_gamma_s"
    per_beamlet = [t for p in passes for t in p[per_key]]
    stats = _quantiles(per_beamlet)
    row = {
        "pool": pool,
        "kind": kind,
        "rung": result["rung"],
        "label": RUNG_NAMES[result["rung"]],
        "path": result["path"],
        "criterion": result["criterion"],
        "n_beamlets": result["n_beamlets"],
        "passes": len(passes),
        "wall_median_s": float(np.median(walls)),
        "wall_min_s": float(min(walls)),
        "wall_max_s": float(max(walls)),
        "beamlets_per_s": result["n_beamlets"] / float(np.median(walls)),
        "per_beamlet_median_s": stats["median"],
        "per_beamlet_p95_s": stats["p95"],
        "per_beamlet_p99_s": stats["p99"],
        "per_beamlet_max_s": stats["max"],
        "peak_gpu_bytes": max((p["peak_gpu_bytes"] or 0) for p in passes) or None,
        "mean_pass_rate_pct": float(np.mean(result["pass_rates_pct"])),
    }
    if kind == "integrated":
        row["inference_sum_median_s"] = float(np.median([p["sum_inference_s"] for p in passes]))
        row["gamma_sum_median_s"] = float(np.median([p["sum_gamma_s"] for p in passes]))
        inference_times = [t for p in passes for t in p["per_beamlet_inference_s"]]
        row["per_beamlet_inference_median_s"] = float(np.median(inference_times))
    return row


def summarise_pools(
    cached: Dict[str, Dict[str, Any]], integrated: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Per (pool, kind, configuration) totals and per-beamlet distributions.

    Args:
        cached: ``{pool name: payload}`` from ``gamma_pool_benchmark.py time``.
        integrated: ``{pool name: payload}`` from ``... integrated``.
    """
    summary = []
    for pool, payload in cached.items():
        for result in payload["results"]:
            summary.append(_pool_config_row(pool, result, "cached"))
    for pool, payload in integrated.items():
        for result in payload["results"]:
            summary.append(_pool_config_row(pool, result, "integrated"))
    return summary


def projection_errors(
    pool_rows: Sequence[Dict[str, Any]], projections: Dict[Tuple[str, int, str], float]
) -> List[Dict[str, Any]]:
    """Relative error of an earlier projection against the measured pool wall time.

    Args:
        pool_rows: Output of :func:`summarise_pools`.
        projections: ``{(pool, rung, path): projected seconds}``.
    """
    out = []
    for row in pool_rows:
        key = (row["pool"], row["rung"], row["path"])
        if row["kind"] != "cached" or key not in projections:
            continue
        projected = projections[key]
        out.append(
            {
                **{k: row[k] for k in ("pool", "rung", "label", "path", "n_beamlets", "wall_median_s")},
                "projected_s": projected,
                "relative_error": (projected - row["wall_median_s"]) / row["wall_median_s"],
            }
        )
    return out


# ── Experiment D ────────────────────────────────────────────────────────────


def summarise_scaling(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (plan, crop) times and paired speed-ups, with the search counters."""
    grouped: Dict[Tuple[str, str, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["plan"], row["criterion"], row["voxels"])][row["rung"]] = row
    summary = []
    for (plan, criterion, voxels), by_rung in sorted(grouped.items()):
        if 1 not in by_rung:
            raise MatchedSetError(f"no reference rung for {plan} {criterion} at {voxels} voxels")
        reference = by_rung[1]
        record: Dict[str, Any] = {
            "plan": plan,
            "criterion": criterion,
            "voxels": voxels,
            "shape_zyx": reference["shape_zyx"],
            "n_evaluated": reference["n_evaluated"],
            "n_above_cutoff": reference.get("n_above_cutoff"),
            "repeats": reference["repeats"],
            "rung1_median_s": reference["seconds_median"],
            "rung1_q1_s": reference["seconds_q1"],
            "rung1_q3_s": reference["seconds_q3"],
        }
        for rung, row in by_rung.items():
            if rung == 1:
                continue
            reference_all = np.array(reference["seconds_all"], dtype=float)
            ratios = reference_all[:, None] / np.array(row["seconds_all"], dtype=float)[None, :]
            record.update(
                {
                    f"rung{rung}_median_s": row["seconds_median"],
                    f"rung{rung}_q1_s": row["seconds_q1"],
                    f"rung{rung}_q3_s": row["seconds_q3"],
                    f"rung{rung}_speedup": reference["seconds_median"] / row["seconds_median"],
                    f"rung{rung}_speedup_min": float(ratios.min()),
                    f"rung{rung}_speedup_max": float(ratios.max()),
                    f"rung{rung}_iterations": row.get("iterations"),
                    f"rung{rung}_shell_points": row.get("shell_points"),
                    f"rung{rung}_interp_samples": row.get("interp_samples"),
                    f"rung{rung}_samples_per_s": row.get("samples_per_s"),
                    f"rung{rung}_peak_gpu_bytes": row.get("peak_gpu_bytes"),
                }
            )
        summary.append(record)
    return summary


# ── Experiment E ────────────────────────────────────────────────────────────


PROFILE_FIELDS = (
    "role", "sample_id", "dtype", "iterations", "wall_unprofiled_s", "wall_profiled_s", "device_kernel_s",
    "device_memcpy_s", "host_launch_s", "host_synchronise_s", "host_allocate_s", "host_other_cpu_s",
    "kernel_launches_per_call", "synchronise_calls_per_call", "iterations_in_loop", "interp_samples",
)


def summarise_profile(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attribution of the profiled wall time per (role, dtype), as fractions too."""
    summary = []
    for result in results:
        wall = result["wall_profiled_s"]
        accounted = (
            result["device_kernel_s"] + result["device_memcpy_s"] + result["device_memset_s"]
        )
        summary.append(
            {
                **{k: result[k] for k in PROFILE_FIELDS},
                "device_busy_fraction": accounted / wall if wall else None,
                "host_other_fraction": result["host_other_cpu_s"] / wall if wall else None,
                "profiling_overhead": (wall - result["wall_unprofiled_s"]) / result["wall_unprofiled_s"],
            }
        )
    return summary


def optional(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a payload if the path exists, else ``None`` (an experiment not run)."""
    if path is None or not Path(path).is_file():
        return None
    return load_payload(path)
