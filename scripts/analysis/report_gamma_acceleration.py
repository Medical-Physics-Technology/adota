"""Assemble the gamma-acceleration technical report's tables, figures and data.

Reads the recorded benchmarks and writes, into one report directory:

* ``tables/*.tex``   LaTeX ``tabular`` fragments, one per table in the report;
* ``tables/numbers.tex``  ``\\newcommand`` macros for every headline number the
  prose quotes, so the text cannot drift from the tables;
* ``figures/*``      through the ``src/figures`` publication layer;
* ``data/*.csv``     the numbers behind every table and panel.

Inputs, all optional except the plan ladder, so the report can be rebuilt from
whichever experiments have run:

* ``--plan-json``     the CHG-0005 plan-level ladder (``docs/gamma_gpu/``);
* ``--exp7-json``     the EXP-0007 beamlet report, for the search-resolution
  table and the projections experiment C is checked against;
* ``--evidence-dir``  an EXP-0008 run directory with ``A/``, ``B_beamlet/``,
  ``B_plan/``, ``C/``, ``D/`` and ``E/``.

Nothing is computed here that a harness did not measure: this script selects,
reduces through :mod:`src.metrics.gamma_evidence_summaries`, and formats
through :mod:`src.metrics.gamma_evidence_latex`.

Example::

    uv run python scripts/analysis/report_gamma_acceleration.py \\
        --plan-json docs/gamma_gpu/gamma_gpu_results.json \\
        --exp7-json docs/gamma_beamlet/gamma_beamlet_results.json \\
        --evidence-dir /scratch/mstryja/gamma_evidence/exp0008_<stamp> \\
        --out-dir reports/technical-reports/gamma-pass-rate
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.figures.gamma_backend_performance import gamma_backend_performance_figure  # noqa: E402
from src.figures.gamma_scaling import gamma_scaling_figure  # noqa: E402
from src.metrics import gamma_evidence_latex as latex  # noqa: E402
from src.metrics import gamma_evidence_summaries as summaries  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(help="Build the tables, figures and data files of the gamma-acceleration report.",
                  add_completion=False)

CRITERIA = summaries.SHARED_CRITERIA
RUNG_LABELS = {1: "pymedphys cpu", 2: "torch cpu float64", 3: "torch cuda float64", 4: "torch cuda float32"}
POOL_NAMES = {
    "test200": "Held-out test, subset",
    "testall": "Held-out test, complete",
    "valsplit2000": "Training-run validation split, subset",
}


# ── CHG-0005 plan ladder ────────────────────────────────────────────────────


def plan_summary(plan_json: Dict) -> List[Dict[str, Any]]:
    """Median plan time and median paired speed-up per criterion, from the recorded ladder."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in plan_json["performance"]:
        grouped[record["criterion"]].append(record)
    rows = []
    for criterion, records in grouped.items():
        times = {rung: np.array([r[f"rung{rung}_s"] for r in records], dtype=float) for rung in (1, 2, 3, 4)}
        row: Dict[str, Any] = {"criterion": criterion, "plans": len(records)}
        for rung in (1, 2, 3, 4):
            row[f"rung{rung}_median_s"] = float(np.median(times[rung]))
            ratios = times[1] / times[rung]
            row[f"rung{rung}_paired_median"] = float(np.median(ratios))
            row[f"rung{rung}_paired_min"] = float(ratios.min())
            row[f"rung{rung}_paired_max"] = float(ratios.max())
        rows.append(row)
    order = {c: i for i, c in enumerate(CRITERIA)}
    return sorted(rows, key=lambda r: order.get(r["criterion"], 99))


def plan_timing_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    body = [
        " & ".join([latex.criterion_tex(r["criterion"]), str(r["plans"]), f"{r['rung1_median_s']:.1f}",
                    f"{r['rung3_median_s']:.2f}", f"{r['rung4_median_s']:.2f}",
                    f"{r['rung3_paired_median']:.1f} ({r['rung3_paired_min']:.1f}--{r['rung3_paired_max']:.1f})",
                    f"{r['rung4_paired_median']:.1f} ({r['rung4_paired_min']:.1f}--{r['rung4_paired_max']:.1f})"])
        + r" \\"
        for r in rows
    ]
    header = [r"Criterion & plans & PyMedPhys [s] & GPU float64 [s] & GPU float32 [s]"
              r" & paired speed-up float64 & paired speed-up float32 \\",
              r" & & median & median & median & median (range) & median (range) \\"]
    return latex._tabular("lrrrrrr", header, body)


def accuracy_table(plan_json: Dict, sweep_rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Pass-rate agreement with PyMedPhys at both scales, matched cases only."""
    plan_by = {r["comparison"]: r for r in plan_json["acceptance"]}
    beamlet = summaries.summarise_sweep_pass_rates(sweep_rows) if sweep_rows else []
    body = []
    for rung in (2, 3, 4):
        plan_record = plan_by.get(f"rung {rung} vs rung 1") or plan_by.get(f"rung {rung} vs rung 2")
        plan_max = plan_record["max_abs_delta_pp"] if plan_record else float("nan")
        mine = [r for r in beamlet if r["rung"] == rung]
        n_cases = sum(r["n_cases"] for r in mine)
        beam_max = max((r["max_abs_delta_pp"] for r in mine), default=float("nan"))
        beam_mean = float(np.mean([r["mean_abs_delta_pp"] for r in mine])) if mine else float("nan")
        body.append(f"{latex.LABEL_TEX[summaries.RUNG_NAMES[rung]]} & 40 & {plan_max:.6f} & {n_cases} & "
                    f"{beam_max:.6f} & {beam_mean:.6f} \\\\")
    header = [r" & \multicolumn{2}{c}{Plan scale} & \multicolumn{3}{c}{Beamlet scale} \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
              r"Tested backend & pairs & max $|\Delta|$ [pp] & pairs & max $|\Delta|$ [pp] & mean $|\Delta|$ [pp] \\"]
    return latex._tabular("lrrrrr", header, body)


def interp_table(exp7: Optional[Dict]) -> Optional[List[str]]:
    """The EXP-0007 search-resolution sweep, unchanged."""
    if not exp7:
        return None
    rows = [r for r in exp7["rows"] if r["criterion"] == "3%/3mm/10%"]
    by: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    rates: Dict[int, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    ref = {(r["interp_fraction"], r["sample_id"]): r["pass_rate_pct"] for r in rows if r["rung"] == 1}
    deltas: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        by[r["interp_fraction"]][r["rung"]].append(r["seconds_best"])
        rates[r["interp_fraction"]][r["rung"]].append(r["pass_rate_pct"])
        if r["rung"] == 4 and (r["interp_fraction"], r["sample_id"]) in ref:
            deltas[r["interp_fraction"]].append(abs(r["pass_rate_pct"] - ref[(r["interp_fraction"], r["sample_id"])]))
    body = []
    for interp in sorted(by):
        r1, r4 = float(np.median(by[interp][1])), float(np.median(by[interp][4]))
        body.append(f"{interp} & {len(by[interp][1])} & {np.mean(rates[interp][1]):.4f} & {r1:.3f} & {r4:.4f} & "
                    f"{r1 / r4:.1f} & {max(deltas[interp]):.6f} \\\\")
    header = [r"$N_{\mathrm{int}}$ & $n$ & mean pass rate [\%] & PyMedPhys [s] & GPU float32 [s]"
              r" & speed-up & max $|\Delta|$ [pp] \\"]
    return latex._tabular("rrrrrrr", header, body)


# ── Figures ─────────────────────────────────────────────────────────────────


def performance_figure(plan_rows: Sequence[Dict], matched: Sequence[Dict], out_dir: Path) -> None:
    """Beamlet panels from the matched sweep, plan panels from the recorded ladder."""
    array = [r for r in matched if r["path"] == "array"]
    criteria = [c for c in CRITERIA if any(r["criterion"] == c for r in array)]
    if not criteria:
        return

    def by(rung: int, key: str) -> List[float]:
        return [next(r[key] for r in array if r["criterion"] == c and r["tested_rung"] == rung) for c in criteria]

    times = {RUNG_LABELS[1]: by(4, "reference_median_s")}
    time_spread = {RUNG_LABELS[1]: (by(4, "reference_q1_s"), by(4, "reference_q3_s"))}
    speedups, speed_spread = {}, {}
    for rung in (2, 3, 4):
        if not any(r["tested_rung"] == rung for r in array):
            continue
        times[RUNG_LABELS[rung]] = by(rung, "tested_median_s")
        time_spread[RUNG_LABELS[rung]] = (by(rung, "tested_q1_s"), by(rung, "tested_q3_s"))
        speedups[RUNG_LABELS[rung]] = by(rung, "median_paired")
        speed_spread[RUNG_LABELS[rung]] = (by(rung, "q1_paired"), by(rung, "q3_paired"))
    plan = {r["criterion"]: r for r in plan_rows}
    plan_times = {RUNG_LABELS[k]: [plan[c][f"rung{k}_median_s"] for c in criteria] for k in (1, 2, 3, 4)}
    plan_speed = {RUNG_LABELS[k]: [plan[c][f"rung{k}_paired_median"] for c in criteria] for k in (2, 3, 4)}
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    gamma_backend_performance_figure(
        [c.replace("/10%", "") for c in criteria], times, plan_times, speedups, plan_speed,
        str(out_dir / "figures" / "gamma_backend_performance.svg"),
        beamlet_time_spread=time_spread, beamlet_speedup_spread=speed_spread,
        beamlet_title="(a) One beamlet, 144,000 voxels",
        plan_title="(b) One plan, 67.5-100.5 million voxels",
    )


# ── Entry point ─────────────────────────────────────────────────────────────


def _load(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    return summaries.optional(path)


@app.command()
def main(
    plan_json: Path = typer.Option(..., help="CHG-0005 plan ladder JSON."),
    out_dir: Path = typer.Option(..., help="Report directory: tables/, figures/ and data/ are written into it."),
    exp7_json: Optional[Path] = typer.Option(None, help="EXP-0007 gamma_beamlet_results.json."),
    evidence_dir: Optional[Path] = typer.Option(None, help="EXP-0008 run directory."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build every table, figure and data file the report includes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )
    tables, data = out_dir / "tables", out_dir / "data"
    plan = json.loads(plan_json.read_text())
    plan_rows = plan_summary(plan)
    latex.write_tex(tables / "plan_timing.tex", plan_timing_table(plan_rows))
    latex.write_csv(data / "plan_timing.csv", plan_rows)
    exp7 = _load(exp7_json)
    interp = interp_table(exp7)
    if interp:
        latex.write_tex(tables / "interp.tex", interp)

    evidence = evidence_dir or Path("/nonexistent")
    matched_array = _load(evidence / "A" / "beamlet_matched_array.json")
    matched_tensor = _load(evidence / "A" / "beamlet_matched_tensor.json")
    matched_rows: List[Dict] = []
    sweep_rows: List[Dict] = []
    for payload in (matched_array, matched_tensor):
        if payload:
            sweep_rows.extend(payload["rows"])
    if matched_array and matched_tensor:
        summaries.validate_same_environment([matched_array, matched_tensor])
        summaries.validate_same_settings([matched_array, matched_tensor])
    if sweep_rows:
        matched_rows = summaries.summarise_matched(sweep_rows)
        latex.write_csv(data / "matched.csv", matched_rows)
        latex.write_tex(tables / "matched_array.tex", latex.matched_table(matched_rows, "array"))
        latex.write_tex(tables / "matched_tensor.tex", latex.matched_table(matched_rows, "tensor"))
        pass_rows = summaries.summarise_sweep_pass_rates(sweep_rows)
        latex.write_csv(data / "pass_rate_agreement.csv", pass_rows)
        performance_figure(plan_rows, matched_rows, out_dir)
    latex.write_tex(tables / "accuracy.tex", accuracy_table(plan, [r for r in sweep_rows if r["path"] == "array"]))

    beamlet_maps = _load(evidence / "B_beamlet" / "beamlet_maps.json")
    beamlet_map_rows = summaries.summarise_beamlet_maps(beamlet_maps["rows"]) if beamlet_maps else []
    if beamlet_map_rows:
        latex.write_csv(data / "beamlet_maps.csv", beamlet_map_rows)
        latex.write_tex(tables / "beamlet_maps.tex", latex.beamlet_maps_table(beamlet_map_rows))
    plan_maps = _load(evidence / "B_plan" / "plan_maps.json")
    plan_map_rows = summaries.summarise_plan_maps(plan_maps["rows"]) if plan_maps else []
    if plan_map_rows:
        latex.write_csv(data / "plan_maps.csv", plan_map_rows)
        latex.write_tex(tables / "plan_maps.tex", latex.plan_maps_table(plan_map_rows))

    cached = {p.stem.replace("time_", ""): summaries.load_payload(p)
              for p in sorted((evidence / "C").glob("time_*.json"))}
    integrated = {p.stem.replace("integrated_", ""): summaries.load_payload(p)
                  for p in sorted((evidence / "C").glob("integrated_*.json"))}
    pool_rows = summaries.summarise_pools(cached, integrated) if cached or integrated else []
    if pool_rows:
        latex.write_csv(data / "pools.csv", pool_rows)
        latex.write_tex(tables / "pools_cached.tex", latex.pools_table(pool_rows, "cached", POOL_NAMES))
        latex.write_tex(tables / "pools_integrated.tex", latex.pools_table(pool_rows, "integrated", POOL_NAMES))
        if exp7:
            medians = {(r["rung"], r["path"]): r["median_s"] for r in exp7["timing"] if r["criterion"] == "2%/2mm/10%"}
            projections = {(r["pool"], r["rung"], r["path"]): r["n_beamlets"] * medians[(r["rung"], r["path"])]
                           for r in pool_rows if (r["rung"], r["path"]) in medians and r["kind"] == "cached"}
            projection_rows = summaries.projection_errors(pool_rows, projections)
            latex.write_csv(data / "projection.csv", projection_rows)
            latex.write_tex(tables / "projection.tex", latex.projection_table(projection_rows, POOL_NAMES))

    scaling = _load(evidence / "D" / "scaling.json")
    scaling_rows = summaries.summarise_scaling(scaling["rows"]) if scaling else []
    if scaling_rows:
        latex.write_csv(data / "scaling.csv", scaling_rows)
        latex.write_csv(data / "scaling_crops.csv", scaling["crops"])
        latex.write_tex(tables / "scaling.tex", latex.scaling_table(scaling_rows))
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        gamma_scaling_figure(scaling_rows, str(out_dir / "figures" / "gamma_scaling.svg"))

    profile = _load(evidence / "E" / "profile.json")
    profile_rows = summaries.summarise_profile(profile["results"]) if profile else []
    if profile_rows:
        latex.write_csv(data / "profile.csv", profile_rows)
        latex.write_tex(tables / "profile.tex", latex.profile_table(profile_rows))

    values = latex.headline_values(matched_rows, beamlet_map_rows, plan_map_rows, pool_rows, scaling_rows, profile_rows)
    for row in plan_rows:
        suffix = {"1%/1mm/10%": "One", "2%/2mm/10%": "Two", "3%/3mm/10%": "Three"}.get(row["criterion"])
        if suffix:
            values[f"planPairedSingle{suffix}"] = f"{row['rung4_paired_median']:.1f}"
            values[f"planPairedDouble{suffix}"] = f"{row['rung3_paired_median']:.1f}"
    latex.write_tex(tables / "numbers.tex", latex.numbers_macros(values))
    (data / "numbers.json").write_text(json.dumps(values, indent=2) + "\n")
    typer.echo(f"Report assets written under {out_dir} ({len(values)} headline macros)")


if __name__ == "__main__":
    app()
