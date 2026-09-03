"""Assemble the gamma deviation ladder into JSON + markdown tables.

Consumes the per-rung JSONs written by :func:`src.metrics.gamma_benchmark.write_rung_json`
and emits three tables:

* **deviation** -- ``pass_rate_pct`` at every rung for each (plan, criterion),
  with the rung-to-rung deltas and a verdict against the acceptance gates.
* **performance** -- wall time, peak GPU memory and interpolation throughput per
  (plan, criterion), plus the per-plan totals and the aggregate.
* **voxel parity** -- for the plans whose gamma maps were persisted: max and
  99.9th-percentile ``|dgamma|``, the fraction of evaluated voxels that cross the
  gamma = 1 boundary, and the NaN-disagreement count.

Rung 0 (the pass rates recorded in each plan's own ``gamma_metrics.json``) is
carried inside every rung JSON, so no extra input is needed for it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ACCEPTANCE", "build_report", "write_report", "render_markdown"]

# (later rung, baseline rung) -> max tolerated |delta| in percentage points.
# Rung 1 vs 0 is reported, not gated: that shift is pymedphys's interpolator
# change, not this implementation's.
ACCEPTANCE: Dict[Tuple[str, str], float] = {
    ("2", "1"): 0.01,
    ("3", "2"): 0.001,
    ("4", "2"): 0.1,
    # Not one of the brief's gates. Rungs 2 and 3 run the same code at the same
    # precision, so where rung 2 was not measured this carries the same evidence
    # as the rung-2 correctness gate, and it is held to the same tolerance.
    ("3", "1"): 0.01,
}


def _load_rungs(paths: Sequence[Path]) -> Dict[str, dict]:
    """Read the rung JSONs, keyed by their ``rung`` label.

    Several JSONs may carry the same rung label -- a long baseline is often run
    as a few batches of plans -- in which case their ``plans`` are merged. A plan
    appearing twice under one rung is an error, since the two runs would
    disagree about which measurement is the record.

    Raises:
        ValueError: If one plan appears twice under the same rung.
    """
    rungs: Dict[str, dict] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        label = str(payload["rung"])
        payload["_source"] = str(path)
        if label not in rungs:
            rungs[label] = payload
            continue
        existing = rungs[label]
        clashes = set(existing["plans"]) & set(payload["plans"])
        if clashes:
            raise ValueError(
                f"Rung {label} has {sorted(clashes)} in both {existing['_source']} "
                f"and {path}."
            )
        existing["plans"].update(payload["plans"])
        existing["_source"] += f", {path}"
    return rungs


def _entry_index(payload: dict) -> Dict[Tuple[str, str], dict]:
    """``{(plan, criterion label): entry}`` for one rung."""
    index: Dict[Tuple[str, str], dict] = {}
    for plan, plan_result in payload["plans"].items():
        for entry in plan_result["criteria"]:
            index[(plan, entry["label"])] = entry
    return index


def _rung0_index(rungs: Dict[str, dict]) -> Dict[Tuple[str, str], float]:
    """Recorded rung-0 pass rates, taken from whichever rung JSON carries them."""
    recorded: Dict[Tuple[str, str], float] = {}
    for payload in rungs.values():
        for plan, plan_result in payload["plans"].items():
            for label, value in plan_result.get("recorded_pass_rate_pct", {}).items():
                recorded.setdefault((plan, label), float(value))
    return recorded


def build_report(rung_paths: Sequence[Path]) -> dict:
    """Build the full comparison from a set of rung JSONs.

    Args:
        rung_paths: Paths to the per-rung JSONs (any subset of rungs 1-4).

    Returns:
        ``{"rungs", "deviation", "performance", "voxel_parity", "acceptance"}``.

    Raises:
        ValueError: If two inputs claim the same rung label.
    """
    rungs = _load_rungs(rung_paths)
    indices = {label: _entry_index(payload) for label, payload in rungs.items()}
    recorded = _rung0_index(rungs)

    keys = sorted({key for index in indices.values() for key in index})
    # Preserve corpus order rather than alphabetical, so plans read as they do
    # in the brief's table.
    plan_order: List[str] = []
    for payload in rungs.values():
        for plan in payload["plans"]:
            if plan not in plan_order:
                plan_order.append(plan)
    keys.sort(key=lambda key: (plan_order.index(key[0]), key[1]))

    deviation = [_deviation_row(key, indices, recorded) for key in keys]
    performance = [_performance_row(key, indices, rungs) for key in keys]

    return {
        "rungs": {
            label: {
                "source": payload["_source"],
                "backend": payload["backend"],
                "environment": payload["environment"],
            }
            for label, payload in rungs.items()
        },
        "acceptance": _acceptance_summary(deviation),
        "deviation": deviation,
        "performance": performance,
        "voxel_parity": _voxel_rows(keys, indices),
        "totals": _totals(rungs),
    }


def _deviation_row(key, indices, recorded) -> dict:
    """One (plan, criterion) row: pass rate at each rung plus the deltas."""
    plan, label = key
    row: dict = {"plan": plan, "criterion": label}
    if key in recorded:
        row["rung0"] = recorded[key]
    for rung_label, index in indices.items():
        if key in index:
            row[f"rung{rung_label}"] = index[key]["pass_rate_pct"]

    for later, baseline in [("1", "0"), ("2", "1"), ("3", "2"), ("4", "2"), ("3", "1")]:
        a, b = f"rung{later}", f"rung{baseline}"
        if a in row and b in row:
            row[f"d{later}_{baseline}"] = row[a] - row[b]
    return row


def _performance_row(key, indices, rungs) -> dict:
    """One (plan, criterion) row of wall time, memory and throughput."""
    plan, label = key
    row: dict = {"plan": plan, "criterion": label}
    for rung_label, index in indices.items():
        entry = index.get(key)
        if entry is None:
            continue
        row[f"rung{rung_label}_s"] = entry["elapsed_s"]
        if entry.get("peak_gpu_mem_bytes") is not None:
            row[f"rung{rung_label}_peak_gib"] = entry["peak_gpu_mem_bytes"] / 2**30
        if entry.get("interp_samples"):
            row[f"rung{rung_label}_samples"] = entry["interp_samples"]
            if entry["elapsed_s"] > 0:
                row[f"rung{rung_label}_samples_per_s"] = (
                    entry["interp_samples"] / entry["elapsed_s"]
                )
    for payload in rungs.values():
        plan_result = payload["plans"].get(plan)
        if plan_result is not None:
            row["n_voxels"] = plan_result["n_voxels"]
            row["recorded_elapsed_s"] = plan_result.get("recorded_elapsed_s")
            break
    return row


def _voxel_rows(keys, indices) -> List[dict]:
    """Voxel-level parity for every (plan, criterion) with two persisted maps."""
    from src.metrics.gamma_benchmark import voxel_parity

    baseline_label = "1" if "1" in indices else None
    if baseline_label is None:
        return []

    rows: List[dict] = []
    for key in keys:
        baseline_entry = indices[baseline_label].get(key)
        if baseline_entry is None or not baseline_entry.get("gamma_map_path"):
            continue
        baseline_path = Path(baseline_entry["gamma_map_path"])
        if not baseline_path.is_file():
            logger.warning("Missing persisted gamma map %s", baseline_path)
            continue
        baseline_map = np.load(baseline_path)
        for rung_label, index in indices.items():
            entry = index.get(key)
            if rung_label == baseline_label or entry is None:
                continue
            path = entry.get("gamma_map_path")
            if not path or not Path(path).is_file():
                continue
            stats = voxel_parity(baseline_map, np.load(path))
            rows.append(
                {
                    "plan": key[0],
                    "criterion": key[1],
                    "baseline_rung": baseline_label,
                    "rung": rung_label,
                    **stats,
                }
            )
        del baseline_map
    return rows


def _acceptance_summary(deviation: List[dict]) -> List[dict]:
    """Worst |delta| per gated rung pair and whether it clears the threshold."""
    summary: List[dict] = []
    for (later, baseline), tolerance in ACCEPTANCE.items():
        field = f"d{later}_{baseline}"
        deltas = [
            (abs(row[field]), row["plan"], row["criterion"])
            for row in deviation
            if field in row
        ]
        if not deltas:
            continue
        worst, plan, criterion = max(deltas)
        summary.append(
            {
                "comparison": f"rung {later} vs rung {baseline}",
                "tolerance_pp": tolerance,
                "n_pairs": len(deltas),
                "max_abs_delta_pp": worst,
                "worst_plan": plan,
                "worst_criterion": criterion,
                "passed": bool(worst <= tolerance),
            }
        )
    return summary


def _totals(rungs: Dict[str, dict]) -> Dict[str, dict]:
    """Aggregate wall time per rung, plus the rung-0 recorded total."""
    totals: Dict[str, dict] = {}
    for label, payload in rungs.items():
        total = sum(
            plan_result["total_elapsed_s"] for plan_result in payload["plans"].values()
        )
        totals[f"rung{label}"] = {
            "total_elapsed_s": total,
            "n_plans": len(payload["plans"]),
        }
    for payload in rungs.values():
        recorded = [
            plan_result.get("recorded_elapsed_s")
            for plan_result in payload["plans"].values()
        ]
        recorded = [value for value in recorded if value is not None]
        if recorded:
            totals["rung0"] = {
                "total_elapsed_s": float(sum(recorded)),
                "n_plans": len(recorded),
                "note": "JSON provenance: another machine, pymedphys 0.40",
            }
        break
    return totals


def _fmt(value: Optional[float], digits: int = 4) -> str:
    """Fixed-width number for a markdown cell; empty for a missing value."""
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a github-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    """Render the whole report as one markdown document."""
    parts: List[str] = ["# GPU gamma index -- results", ""]

    parts.append("## Rungs measured")
    parts.append("")
    rung_rows = []
    for label in sorted(report["rungs"]):
        info = report["rungs"][label]
        env = info["environment"]
        rung_rows.append(
            [
                label,
                info["backend"],
                env.get("device", "--"),
                env.get("dtype") or "--",
                env.get("gpu_name", "--"),
                f"py{env.get('python')} / pymedphys {env.get('pymedphys')} / "
                f"torch {env.get('torch')}",
            ]
        )
    parts.append(
        _table(
            ["rung", "backend", "device", "dtype", "gpu", "environment"], rung_rows
        )
    )
    parts.append("")

    parts.append("## Acceptance")
    parts.append("")
    parts.append(
        _table(
            ["comparison", "pairs", "tolerance (pp)", "max |delta| (pp)", "worst", "verdict"],
            [
                [
                    item["comparison"],
                    str(item["n_pairs"]),
                    _fmt(item["tolerance_pp"], 3),
                    _fmt(item["max_abs_delta_pp"], 6),
                    f"{item['worst_plan']} {item['worst_criterion']}",
                    "PASS" if item["passed"] else "FAIL",
                ]
                for item in report["acceptance"]
            ],
        )
    )
    parts.append("")

    parts.append("## Deviation table (`pass_rate_pct`)")
    parts.append("")
    rung_keys = [key for key in ("rung0", "rung1", "rung2", "rung3", "rung4")]
    delta_keys = ["d1_0", "d2_1", "d3_1", "d3_2", "d4_2"]
    header = ["plan", "criterion"] + rung_keys + [f"D {k}" for k in delta_keys]
    parts.append(
        _table(
            header,
            [
                [row["plan"], row["criterion"]]
                + [_fmt(row.get(key)) for key in rung_keys]
                + [_fmt(row.get(key), 6) for key in delta_keys]
                for row in report["deviation"]
            ],
        )
    )
    parts.append("")

    parts.append("## Performance")
    parts.append("")
    perf_rungs = sorted(
        {
            key[4]
            for row in report["performance"]
            for key in row
            if key.startswith("rung") and key.endswith("_s")
        }
    )
    header = ["plan", "criterion", "voxels"]
    for label in perf_rungs:
        header += [f"rung{label} (s)"]
    header += ["peak GiB", "Msamples/s"]
    perf_rows = []
    for row in report["performance"]:
        cells = [row["plan"], row["criterion"], f"{row.get('n_voxels', 0) / 1e6:.1f}M"]
        for label in perf_rungs:
            cells.append(_fmt(row.get(f"rung{label}_s"), 1))
        peak = next(
            (row[f"rung{label}_peak_gib"] for label in perf_rungs
             if f"rung{label}_peak_gib" in row),
            None,
        )
        rate = next(
            (row[f"rung{label}_samples_per_s"] for label in perf_rungs
             if f"rung{label}_samples_per_s" in row),
            None,
        )
        cells.append(_fmt(peak, 2))
        cells.append(_fmt(rate / 1e6 if rate is not None else None, 1))
        perf_rows.append(cells)
    parts.append(_table(header, perf_rows))
    parts.append("")

    totals = report["totals"]
    parts.append("### Totals")
    parts.append("")
    parts.append(
        _table(
            ["rung", "plans", "total wall time (s)", "note"],
            [
                [
                    label,
                    str(info["n_plans"]),
                    _fmt(info["total_elapsed_s"], 1),
                    info.get("note", ""),
                ]
                for label, info in sorted(totals.items())
            ],
        )
    )
    parts.append("")

    if report["voxel_parity"]:
        parts.append("## Voxel-level parity")
        parts.append("")
        parts.append(
            _table(
                [
                    "plan", "criterion", "rungs", "evaluated",
                    "max |dg|", "p99.9 |dg|", "g=1 crossings", "crossing frac",
                    "eval-mask disagreements",
                ],
                [
                    [
                        row["plan"],
                        row["criterion"],
                        f"{row['rung']} vs {row['baseline_rung']}",
                        f"{row['n_evaluated'] / 1e6:.1f}M",
                        f"{row['max_abs_delta']:.3e}",
                        f"{row['p999_abs_delta']:.3e}",
                        str(row["n_boundary_cross"]),
                        f"{row['boundary_cross_frac']:.3e}",
                        str(row["n_evaluated_disagree"]),
                    ]
                    for row in report["voxel_parity"]
                ],
            )
        )
        parts.append("")

    return "\n".join(parts) + "\n"


def write_report(report: dict, out_dir: Path) -> List[Path]:
    """Write ``gamma_gpu_results.json`` and ``gamma_gpu_results.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gamma_gpu_results.json"
    md_path = out_dir / "gamma_gpu_results.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(render_markdown(report))
    return [json_path, md_path]
