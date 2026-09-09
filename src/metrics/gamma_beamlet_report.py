# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Aggregation of the beamlet gamma sweep into parity and timing tables.

Splits from :mod:`src.metrics.gamma_beamlet_benchmark` by role, as the plan-level
harness splits :mod:`src.metrics.gamma_report` from
:mod:`src.metrics.gamma_benchmark`: that module measures, this one reduces.

Three tables come out, and each answers one question:

``parity``
    Does a rung reproduce the pymedphys pass rate? Reported per criterion as the
    maximum and mean absolute deviation in percentage points over the beamlets,
    with the worst beamlet named.
``timing``
    What does one beamlet cost on each rung, and what is the speed-up over
    pymedphys? Reported as the median over beamlets of the per-beamlet best time,
    so one slow beamlet cannot carry the headline number.
``throughput``
    How many beamlets per second, and how long a gamma pass over a validation
    pool of a given size takes. This is the number the training loop is
    budgeted against.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "REFERENCE_RUNG",
    "build_report",
    "write_report",
]

# The rung every other rung is compared against for correctness. pymedphys is
# the published implementation, so it defines the pass rate rather than merely
# participating in an average.
REFERENCE_RUNG = 1


def _key(row: Dict[str, Any]) -> Tuple:
    """Identity of a measurement, ignoring which rung produced it."""
    return (row["criterion"], row["interp_fraction"], row["path"], row["sample_id"])


def _group(rows: Iterable[Dict[str, Any]], *fields: str) -> Dict[Tuple, List[Dict[str, Any]]]:
    """Bucket rows by the given field values."""
    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return grouped


def _rung_label(row: Dict[str, Any]) -> str:
    """Column label for a rung, from any of its rows."""
    if row["backend"] == "pymedphys":
        return "pymedphys cpu"
    return f"torch {row['device']} {row['dtype']}"


def parity_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (criterion, rung) deviation of the pass rate from pymedphys.

    Args:
        rows: Sweep rows, as :func:`src.metrics.gamma_beamlet_benchmark.sweep`
            returns them.

    Returns:
        One record per criterion and rung, carrying the maximum and mean
        absolute deviation in percentage points and the beamlet that produced
        the maximum.
    """
    reference = {
        _key(row): row["pass_rate_pct"] for row in rows if row["rung"] == REFERENCE_RUNG
    }
    table: List[Dict[str, Any]] = []
    for (criterion, interp, path, rung), group in _group(
        rows, "criterion", "interp_fraction", "path", "rung"
    ).items():
        if rung == REFERENCE_RUNG:
            continue
        deltas = [
            (abs(row["pass_rate_pct"] - reference[_key(row)]), row["sample_id"])
            for row in group
            if _key(row) in reference
        ]
        if not deltas:
            continue
        worst_delta, worst_id = max(deltas)
        table.append(
            {
                "criterion": criterion,
                "interp_fraction": interp,
                "path": path,
                "rung": rung,
                "label": _rung_label(group[0]),
                "beamlets": len(deltas),
                "max_abs_delta_pp": worst_delta,
                "mean_abs_delta_pp": float(np.mean([d for d, _ in deltas])),
                "worst_sample_id": worst_id,
                "mean_pass_rate_pct": float(np.mean([row["pass_rate_pct"] for row in group])),
            }
        )
    return sorted(table, key=lambda item: (item["path"], item["criterion"], item["rung"]))


def timing_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (criterion, rung) beamlet timing and speed-up over pymedphys.

    The per-beamlet statistic is the *median* of the per-beamlet best times.
    A mean would be pulled by the few beamlets whose search converges late, and
    the question here is what a typical beamlet costs.

    Args:
        rows: Sweep rows.

    Returns:
        One record per criterion, path and rung.
    """
    medians: Dict[Tuple, float] = {}
    table: List[Dict[str, Any]] = []
    for (criterion, interp, path, rung), group in _group(
        rows, "criterion", "interp_fraction", "path", "rung"
    ).items():
        times = np.array([row["seconds_best"] for row in group], dtype=np.float64)
        medians[(criterion, interp, path, rung)] = float(np.median(times))
        table.append(
            {
                "criterion": criterion,
                "interp_fraction": interp,
                "path": path,
                "rung": rung,
                "label": _rung_label(group[0]),
                "beamlets": len(group),
                "voxels": int(group[0]["voxels"]),
                "median_s": float(np.median(times)),
                "min_s": float(times.min()),
                "max_s": float(times.max()),
                "total_s": float(times.sum()),
                "mean_pass_rate_pct": float(
                    np.mean([row["pass_rate_pct"] for row in group])
                ),
            }
        )
    totals = {
        (record["criterion"], record["interp_fraction"], record["path"], record["rung"]):
        (record["total_s"], record["beamlets"])
        for record in table
    }
    for record in table:
        index = (record["criterion"], record["interp_fraction"], record["path"])
        baseline = medians.get(index + (REFERENCE_RUNG,))
        record["speedup"] = (
            float(baseline / record["median_s"]) if baseline and record["median_s"] > 0 else None
        )
        record["beamlets_per_s"] = float(1.0 / record["median_s"]) if record["median_s"] > 0 else None
        # The aggregate speed-up over the whole draw, which is a different number
        # from the median one and the more relevant of the two for a validation
        # pool: the CPU baseline's cost is dominated by its few hard beamlets,
        # and the GPU's is not, so the pool speeds up by more than the median
        # beamlet does.
        # Only when the two rungs covered the same draw: rung 2 is measured on a
        # subset, and dividing its total by the baseline's would report a
        # speed-up that is really a difference in the number of beamlets.
        baseline_total, baseline_beamlets = totals.get(index + (REFERENCE_RUNG,), (None, None))
        record["total_speedup"] = (
            float(baseline_total / record["total_s"])
            if baseline_total
            and record["total_s"] > 0
            and baseline_beamlets == record["beamlets"]
            else None
        )
    return sorted(table, key=lambda item: (item["path"], item["criterion"], item["rung"]))


def throughput_table(
    rows: Sequence[Dict[str, Any]], pool_sizes: Sequence[int] = (20, 200, 2000)
) -> List[Dict[str, Any]]:
    """Wall time of one gamma pass over a validation pool of each size.

    Args:
        rows: Sweep rows.
        pool_sizes: Pool sizes to extrapolate to. The default covers the frozen
            comparable subset, a realistic pool, and a whole validation split.

    Returns:
        One record per criterion, path, rung and pool size.
    """
    table: List[Dict[str, Any]] = []
    for record in timing_table(rows):
        for size in pool_sizes:
            table.append(
                {
                    "criterion": record["criterion"],
                    "path": record["path"],
                    "rung": record["rung"],
                    "label": record["label"],
                    "pool_size": int(size),
                    "pass_seconds": float(record["median_s"] * size),
                }
            )
    return table


def build_report(
    rows: Sequence[Dict[str, Any]],
    environment: Dict[str, Any],
    pool_sizes: Sequence[int] = (20, 200, 2000),
) -> Dict[str, Any]:
    """Assemble every table into one JSON-serialisable record."""
    return {
        "environment": environment,
        "beamlets": sorted({row["sample_id"] for row in rows}),
        "parity": parity_table(rows),
        "timing": timing_table(rows),
        "throughput": throughput_table(rows, pool_sizes),
        "rows": list(rows),
    }


def _markdown_table(headers: Sequence[str], records: Sequence[Sequence[str]]) -> List[str]:
    """One markdown table, as a list of lines."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(record) + " |" for record in records)
    lines.append("")
    return lines


def _format_report(report: Dict[str, Any]) -> str:
    """Render the report as markdown."""
    environment = report["environment"]
    lines = [
        "# Beamlet-scale gamma index -- results",
        "",
        f"{len(report['beamlets'])} beamlets, "
        f"{environment.get('gpu') or 'no GPU'}, "
        f"python {environment.get('python')} / pymedphys {environment.get('pymedphys')} / "
        f"torch {environment.get('torch')}.",
        "",
        "## Accuracy against pymedphys",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["path", "criterion", "rung", "backend", "beamlets", "max |d| (pp)", "mean |d| (pp)", "worst beamlet"],
            [
                [
                    record["path"],
                    record["criterion"],
                    str(record["rung"]),
                    record["label"],
                    str(record["beamlets"]),
                    f"{record['max_abs_delta_pp']:.6f}",
                    f"{record['mean_abs_delta_pp']:.6f}",
                    record["worst_sample_id"][:8],
                ]
                for record in report["parity"]
            ],
        )
    )
    lines.extend(["## Time per beamlet", ""])
    lines.extend(
        _markdown_table(
            ["path", "criterion", "rung", "backend", "median (s)", "min (s)", "max (s)", "speed-up", "beamlets/s"],
            [
                [
                    record["path"],
                    record["criterion"],
                    str(record["rung"]),
                    record["label"],
                    f"{record['median_s']:.4f}",
                    f"{record['min_s']:.4f}",
                    f"{record['max_s']:.4f}",
                    "--" if record["speedup"] is None else f"{record['speedup']:.1f}x",
                    f"{record['beamlets_per_s']:.1f}",
                ]
                for record in report["timing"]
            ],
        )
    )
    lines.extend(["## One gamma pass over a validation pool", ""])
    lines.extend(
        _markdown_table(
            ["path", "criterion", "rung", "backend", "pool", "pass time (s)"],
            [
                [
                    record["path"],
                    record["criterion"],
                    str(record["rung"]),
                    record["label"],
                    str(record["pool_size"]),
                    f"{record['pass_seconds']:.1f}",
                ]
                for record in report["throughput"]
            ],
        )
    )
    return "\n".join(lines)


def write_report(report: Dict[str, Any], out_dir: Path, stem: str = "gamma_beamlet_results") -> List[Path]:
    """Write the report as JSON, markdown and a flat per-measurement CSV.

    The CSV exists because CLAUDE.md requires the numbers behind any reported
    figure to sit beside it in machine-readable form.

    Args:
        report: The record :func:`build_report` returns.
        out_dir: Directory to write into; created if absent.
        stem: Basename shared by the three files.

    Returns:
        The paths written, in JSON, markdown, CSV order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    csv_path = out_dir / f"{stem}.csv"

    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_format_report(report) + "\n")

    rows = report["rows"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return [json_path, md_path, csv_path]
