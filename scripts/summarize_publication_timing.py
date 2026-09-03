"""Collect the publication plans' pipeline_timing.json into one table.

Reads each plan's freshly written ``pipeline_timing.json`` (the ``streaming``
stage plus the model/plan load and figure stages), prints a per-plan and a
per-step summary, and writes ``run_logs/publication_timing_summary.json``.

    uv run python scripts/summarize_publication_timing.py
"""
from __future__ import annotations

import json
from pathlib import Path

from prettytable import PrettyTable

PLAN_ROOT = Path("/scratch/mstryja/opentps_plans")
LOAD_TSV = Path("run_logs/pubtiming_load.tsv")
PLANS = [
    "LUNG1-062_Publication_Plan_1",
    "LUNG1-195_Publication_Plan_2",
    "LUNG1-250_Publication_Plan_3",
    "LUNG1-364_Publication_Plan_5",
    "Prostate-AEC-004_Publication_Plan_1",
    "Prostate-AEC-069_Publication_Plan_2",
    "Prostate-AEC-006_Publication_Plan_3",
    "Prostate-AEC-007_Publication_Plan_4",
]
# The streaming sub-steps, in execution order.
STEPS = [
    ("rotation", "rotate CT (per field)"),
    ("crop", "CT cropping"),
    ("flux", "flux projection"),
    ("prep", "input prep (H->D)"),
    ("forward", "ADoTA forward"),
    ("post", "postprocess (D->H)"),
    ("deposit", "deposit"),
    ("derotate", "de-rotate (per field)"),
    ("write", "write Dose_ADoTA.mhd"),
]


def _load_averages() -> dict:
    """Per-plan 1-minute load average recorded around each run (shared machine).

    The GPU steps are insensitive to it; the CPU-side ones (CT cropping, deposit,
    de-rotation, write) are not, so it is reported alongside the timings rather
    than left out.
    """
    if not LOAD_TSV.exists():
        return {}
    out: dict = {}
    for line in LOAD_TSV.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        plan, when, l1 = parts[0], parts[1], float(parts[2])
        out.setdefault(plan, {})[when] = l1
    return out


LOAD: dict = {}


def main() -> None:
    global LOAD
    LOAD = _load_averages()
    rows, missing = [], []
    for name in PLANS:
        path = PLAN_ROOT / name / "pipeline_timing.json"
        if not path.exists():
            missing.append(name)
            continue
        report = json.loads(path.read_text())
        stream = report.get("stages", {}).get("streaming")
        if stream is None:
            missing.append(f"{name} (no streaming stage)")
            continue
        rows.append({
            "plan": name,
            "n_spots": int(stream["n_spots"]),
            "n_fields": int(stream["n_fields"]),
            "batch_size": stream.get("batch_size"),
            "mode": stream.get("grid_mode"),
            "precision": stream.get("precision"),
            "batched_io": bool(stream.get("flux_batched") or stream.get("batched_prep")),
            "stream_s": float(stream["total_s"]),
            "ms_per_beamlet": float(stream["ms_per_beamlet"]),
            "model_load_s": float(report["stages"].get("model_load", {}).get("total_s", 0.0)),
            "plan_load_s": float(report["stages"].get("plan_load", {}).get("total_s", 0.0)),
            "figures_s": float(
                report["stages"].get("comparison_figures", {}).get("total_s", 0.0)
            ),
            "load": LOAD.get(name, {}),
            "total_s": float(report["total_s"]),
            "steps": {k: float(v["total_s"]) for k, v in stream.get("steps", {}).items()},
        })

    if not rows:
        print("No timing reports found.")
        return

    per_plan = PrettyTable()
    per_plan.field_names = ["Plan", "spots", "fields", "stream [s]", "ms/beamlet",
                            "TOTAL [s]", "load1"]
    per_plan.align["Plan"] = "l"
    for col in per_plan.field_names[1:]:
        per_plan.align[col] = "r"
    for r in rows:
        load = r["load"].get("before")
        per_plan.add_row([r["plan"], f"{r['n_spots']:,}", r["n_fields"],
                          f"{r['stream_s']:.2f}", f"{r['ms_per_beamlet']:.3f}",
                          f"{r['total_s']:.2f}",
                          f"{load:.1f}" if load is not None else "-"])
    n_spots = sum(r["n_spots"] for r in rows)
    stream_s = sum(r["stream_s"] for r in rows)
    loads = [r["load"]["before"] for r in rows if "before" in r["load"]]
    per_plan.add_row(["ALL PLANS", f"{n_spots:,}", sum(r["n_fields"] for r in rows),
                      f"{stream_s:.2f}", f"{stream_s / max(n_spots, 1) * 1000:.3f}",
                      f"{sum(r['total_s'] for r in rows):.2f}",
                      f"{sum(loads)/len(loads):.1f}" if loads else "-"], divider=True)

    per_step = PrettyTable()
    per_step.field_names = ["Streaming step", "total [s]", "ms/beamlet", "% of stream"]
    per_step.align["Streaming step"] = "l"
    for col in per_step.field_names[1:]:
        per_step.align[col] = "r"
    for key, label in STEPS:
        total = sum(r["steps"].get(key, 0.0) for r in rows)
        if total <= 0:
            continue
        per_step.add_row([label, f"{total:.2f}", f"{total / n_spots * 1000:.3f}",
                          f"{100 * total / stream_s:.1f}"])

    head = rows[0]
    print(f"\nPUBLICATION PLAN TIMING  ({head['mode']}, {head['precision']}, "
          f"batch {head['batch_size']}, batched I/O {head['batched_io']})")
    print(per_plan.get_string())
    print("\nWHERE THE STREAM TIME GOES (summed over all plans)")
    print(per_step.get_string())
    if missing:
        print("\nMissing / incomplete:", ", ".join(missing))

    out = Path("run_logs/publication_timing_summary.json")
    out.parent.mkdir(exist_ok=True)
    import os
    out.write_text(json.dumps({"plans": rows, "n_spots_total": n_spots,
                               "stream_s_total": stream_s,
                               "n_cores": os.cpu_count(),
                               "note": ("shared machine: load1 is the 1-minute load "
                                        "average when each plan started; figures were "
                                        "skipped (--no-figures)")}, indent=2) + "\n")
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
