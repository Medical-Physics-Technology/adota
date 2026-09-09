"""Assemble the gamma-acceleration technical report's tables and figure.

Reads the two benchmarks that already exist -- the plan-level ladder written by
``scripts/gamma_benchmark.py report`` and the beamlet-level sweeps written by
``scripts/gamma_beamlet_benchmark.py sweep`` -- and emits, into one directory:

* ``tables/*.tex``   LaTeX ``tabular`` fragments, one per table in the report;
* ``figures/gamma_backend_performance.{svg,pdf,png}`` through
  :func:`src.figures.gamma_backend_performance.gamma_backend_performance_figure`;
* ``data/*.csv``     the numbers behind each table and the figure, so a panel can
  be traced back to the run that produced it.

Nothing is computed here that the benchmarks did not already measure: this
script selects, formats and cross-references. Keeping it separate from the
report source means the report can be rebuilt from the recorded JSONs without
re-running a single gamma evaluation.

Example::

    uv run python scripts/analysis/report_gamma_acceleration.py \\
        --plan-json docs/gamma_gpu/gamma_gpu_results.json \\
        --beamlet-sweep /scratch/mstryja/gamma_beamlet/sweep_main.json \\
        --beamlet-sweep /scratch/mstryja/gamma_beamlet/sweep_cpu.json \\
        --interp-sweep /scratch/mstryja/gamma_beamlet/sweep_interp_2.json \\
        --out-dir reports/technical-reports/gamma-pass-rate
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.figures.gamma_backend_performance import (  # noqa: E402
    gamma_backend_performance_figure,
)
from src.metrics.gamma_beamlet_report import (  # noqa: E402
    REFERENCE_RUNG,
    parity_table,
    timing_table,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Build the tables, figure and data files of the gamma-acceleration report.",
    add_completion=False,
)

# The three criteria the report compares across both scales. The plan corpus
# carries five; the two extra ones (1%/2mm/3% and 1%/3mm/0.1%) are reported in
# the plan table only, because the beamlet sweep does not include them.
SHARED_CRITERIA: Tuple[str, ...] = ("1%/1mm/10%", "2%/2mm/10%", "3%/3mm/10%")

RUNG_LABELS: Dict[int, str] = {
    1: "pymedphys cpu",
    2: "torch cpu float64",
    3: "torch cuda float64",
    4: "torch cuda float32",
}

# LaTeX-safe rung names for the table headers.
RUNG_TEX: Dict[int, str] = {
    1: r"\texttt{pymedphys}, CPU",
    2: r"\texttt{gamma\_torch}, CPU, float64",
    3: r"\texttt{gamma\_torch}, GPU, float64",
    4: r"\texttt{gamma\_torch}, GPU, float32",
}


def _tex_criterion(label: str) -> str:
    """``"3%/3mm/10%"`` in a form LaTeX will typeset."""
    return label.replace("%", r"\%")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write the rows behind a table, so every number has a machine-readable twin."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


def _write_tex(path: Path, lines: Sequence[str]) -> None:
    """Write one LaTeX fragment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", path)


# ── Plan-scale selection ────────────────────────────────────────────────────


def plan_summary(plan_json: Dict) -> List[Dict[str, object]]:
    """Median plan time and speed-up per criterion, from the recorded ladder."""
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in plan_json["performance"]:
        grouped[record["criterion"]].append(record)

    summary: List[Dict[str, object]] = []
    for criterion, records in grouped.items():
        times = {
            rung: np.array([record[f"rung{rung}_s"] for record in records], dtype=float)
            for rung in (1, 2, 3, 4)
        }
        row: Dict[str, object] = {
            "criterion": criterion,
            "plans": len(records),
            "median_voxels_m": float(
                np.median([record["n_voxels"] for record in records]) / 1e6
            ),
        }
        for rung in (1, 2, 3, 4):
            row[f"rung{rung}_median_s"] = float(np.median(times[rung]))
            row[f"rung{rung}_speedup"] = float(np.median(times[1] / times[rung]))
        summary.append(row)
    return sorted(summary, key=lambda item: SHARED_CRITERIA.index(item["criterion"])
                  if item["criterion"] in SHARED_CRITERIA else 99)


def beamlet_summary(rows: Sequence[Dict]) -> List[Dict[str, object]]:
    """Median beamlet time and speed-up per criterion, for the array path."""
    timing = [record for record in timing_table(rows) if record["path"] == "array"]
    grouped: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for record in timing:
        grouped[record["criterion"]][record["rung"]] = record

    summary: List[Dict[str, object]] = []
    for criterion, by_rung in grouped.items():
        row: Dict[str, object] = {"criterion": criterion}
        for rung, record in sorted(by_rung.items()):
            row[f"rung{rung}_median_s"] = record["median_s"]
            row[f"rung{rung}_max_s"] = record["max_s"]
            row[f"rung{rung}_speedup"] = record["speedup"]
            row[f"rung{rung}_total_speedup"] = record["total_speedup"]
            row[f"rung{rung}_beamlets"] = record["beamlets"]
        summary.append(row)
    return sorted(
        summary,
        key=lambda item: SHARED_CRITERIA.index(item["criterion"])
        if item["criterion"] in SHARED_CRITERIA
        else 99,
    )


# ── Table writers ───────────────────────────────────────────────────────────


def write_accuracy_table(
    plan_json: Dict, beamlet_rows: Sequence[Dict], out_dir: Path
) -> None:
    """Deviation of each rung from pymedphys, at both scales, in one table."""
    plan_by_comparison = {
        record["comparison"]: record for record in plan_json["acceptance"]
    }
    beamlet_parity = {
        (record["rung"], record["criterion"]): record
        for record in parity_table(beamlet_rows)
        if record["path"] == "array"
    }

    csv_rows: List[Dict[str, object]] = []
    body: List[str] = []
    for rung in (2, 3, 4):
        plan_record = plan_by_comparison.get(f"rung {rung} vs rung 1") or plan_by_comparison.get(
            f"rung {rung} vs rung 2"
        )
        plan_max = plan_record["max_abs_delta_pp"] if plan_record else float("nan")
        beamlet_max = max(
            (
                beamlet_parity[(rung, criterion)]["max_abs_delta_pp"]
                for criterion in SHARED_CRITERIA
                if (rung, criterion) in beamlet_parity
            ),
            default=float("nan"),
        )
        beamlet_mean = np.mean(
            [
                beamlet_parity[(rung, criterion)]["mean_abs_delta_pp"]
                for criterion in SHARED_CRITERIA
                if (rung, criterion) in beamlet_parity
            ]
            or [float("nan")]
        )
        body.append(
            f"{rung} & {RUNG_TEX[rung]} & {plan_max:.6f} & "
            f"{beamlet_max:.6f} & {beamlet_mean:.6f} \\\\"
        )
        csv_rows.append(
            {
                "rung": rung,
                "backend": RUNG_LABELS[rung],
                "plan_max_abs_delta_pp": plan_max,
                "beamlet_max_abs_delta_pp": beamlet_max,
                "beamlet_mean_abs_delta_pp": float(beamlet_mean),
            }
        )

    _write_tex(
        out_dir / "tables" / "accuracy.tex",
        [
            r"\begin{tabular}{clrrr}",
            r"\toprule",
            r" & & \multicolumn{1}{c}{Plan scale} & \multicolumn{2}{c}{Beamlet scale} \\",
            r"\cmidrule(lr){3-3}\cmidrule(lr){4-5}",
            r"Rung & Backend & max $|\Delta|$ [pp] & max $|\Delta|$ [pp] & mean $|\Delta|$ [pp] \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ],
    )
    _write_csv(out_dir / "data" / "accuracy.csv", csv_rows)


def write_timing_table(
    plan_json: Dict, beamlet_rows: Sequence[Dict], out_dir: Path
) -> None:
    """Median time and speed-up per criterion, at both scales."""
    plan_rows = {record["criterion"]: record for record in plan_summary(plan_json)}
    beamlet_rows_by_criterion = {
        record["criterion"]: record for record in beamlet_summary(beamlet_rows)
    }

    csv_rows: List[Dict[str, object]] = []
    body: List[str] = []
    for criterion in SHARED_CRITERIA:
        plan = plan_rows.get(criterion, {})
        beam = beamlet_rows_by_criterion.get(criterion, {})
        body.append(
            f"{_tex_criterion(criterion)} & "
            f"{beam.get('rung1_median_s', float('nan')):.3f} & "
            f"{beam.get('rung4_median_s', float('nan')):.3f} & "
            f"{beam.get('rung4_speedup', float('nan')):.1f} & "
            f"{beam.get('rung4_total_speedup', float('nan')):.1f} & "
            f"{plan.get('rung1_median_s', float('nan')):.1f} & "
            f"{plan.get('rung4_median_s', float('nan')):.2f} & "
            f"{plan.get('rung4_speedup', float('nan')):.1f} \\\\"
        )
        csv_rows.append({"criterion": criterion, **{f"beamlet_{k}": v for k, v in beam.items() if k != "criterion"},
                         **{f"plan_{k}": v for k, v in plan.items() if k != "criterion"}})

    _write_tex(
        out_dir / "tables" / "timing.tex",
        [
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r" & \multicolumn{4}{c}{One beamlet ($\num{144000}$ voxels)}"
            r" & \multicolumn{3}{c}{One plan ($67$--$100$ M voxels)} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
            r"Criterion & rung 1 & rung 4 & speed-up & pool"
            r" & rung 1 & rung 4 & speed-up \\",
            r" & [s] & [s] & (median) & speed-up & [s] & [s] & (median) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ],
    )
    _write_csv(out_dir / "data" / "timing.csv", csv_rows)


def write_entry_point_table(beamlet_rows: Sequence[Dict], out_dir: Path) -> None:
    """Array versus device-resident entry point, at beamlet scale."""
    timing = timing_table(beamlet_rows)
    lookup = {
        (record["path"], record["criterion"], record["rung"]): record for record in timing
    }
    csv_rows: List[Dict[str, object]] = []
    body: List[str] = []
    for criterion in SHARED_CRITERIA:
        cells: List[str] = [_tex_criterion(criterion)]
        record: Dict[str, object] = {"criterion": criterion}
        for path in ("array", "tensor"):
            for rung in (1, 4):
                entry = lookup.get((path, criterion, rung))
                value = entry["median_s"] if entry else float("nan")
                cells.append(f"{value:.3f}")
                record[f"{path}_rung{rung}_median_s"] = value
        speedup = (
            record["tensor_rung1_median_s"] / record["tensor_rung4_median_s"]
            if record.get("tensor_rung4_median_s")
            else float("nan")
        )
        cells.append(f"{speedup:.1f}")
        record["tensor_speedup"] = speedup
        body.append(" & ".join(cells) + r" \\")
        csv_rows.append(record)

    _write_tex(
        out_dir / "tables" / "entry_points.tex",
        [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r" & \multicolumn{2}{c}{Array path} & \multicolumn{2}{c}{Tensor path} & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Criterion & rung 1 [s] & rung 4 [s] & rung 1 [s] & rung 4 [s]"
            r" & speed-up (tensor) \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ],
    )
    _write_csv(out_dir / "data" / "entry_points.csv", csv_rows)


def write_interp_table(interp_rows: Sequence[Dict], out_dir: Path) -> None:
    """Cost against the search resolution $N_i$, at beamlet scale."""
    if not interp_rows:
        logger.warning("No interpolation-fraction sweep given; skipping that table")
        return
    timing = [record for record in timing_table(interp_rows) if record["path"] == "array"]
    parity = {
        (record["interp_fraction"], record["rung"]): record
        for record in parity_table(interp_rows)
        if record["path"] == "array"
    }
    by_interp: Dict[int, Dict[int, Dict]] = defaultdict(dict)
    for record in timing:
        by_interp[record["interp_fraction"]][record["rung"]] = record

    csv_rows: List[Dict[str, object]] = []
    body: List[str] = []
    for interp in sorted(by_interp):
        rungs = by_interp[interp]
        rung1 = rungs.get(1, {}).get("median_s", float("nan"))
        rung4 = rungs.get(4, {}).get("median_s", float("nan"))
        pass_rate = rungs.get(1, {}).get("mean_pass_rate_pct", float("nan"))
        deviation = parity.get((interp, 4), {}).get("max_abs_delta_pp", float("nan"))
        body.append(
            f"{interp} & {pass_rate:.4f} & {rung1:.3f} & {rung4:.4f} & "
            f"{rung1 / rung4:.1f} & {deviation:.6f} \\\\"
        )
        csv_rows.append(
            {
                "interp_fraction": interp,
                "rung1_median_s": rung1,
                "rung4_median_s": rung4,
                "speedup": rung1 / rung4 if rung4 else float("nan"),
                "rung4_max_abs_delta_pp": deviation,
                "mean_pass_rate_pct": pass_rate,
            }
        )

    _write_tex(
        out_dir / "tables" / "interp.tex",
        [
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"$N_i$ & mean pass rate [\%] & rung 1 [s] & rung 4 [s]"
            r" & speed-up & max $|\Delta|$ [pp] \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
        ],
    )
    _write_csv(out_dir / "data" / "interp.csv", csv_rows)


def write_figure(plan_json: Dict, beamlet_rows: Sequence[Dict], out_dir: Path) -> None:
    """Render the two-scale timing and speed-up figure."""
    plan = {record["criterion"]: record for record in plan_summary(plan_json)}
    beam = {record["criterion"]: record for record in beamlet_summary(beamlet_rows)}
    criteria = [criterion for criterion in SHARED_CRITERIA if criterion in beam]
    # save_figure_as_publication_formats writes three files but creates no
    # directory, so the destination has to exist before it is called.
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    def series(source: Dict[str, Dict], field: str) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for rung in (1, 2, 3, 4):
            values = [source.get(c, {}).get(f"rung{rung}_{field}") for c in criteria]
            if all(value is not None and np.isfinite(value) for value in values):
                out[RUNG_LABELS[rung]] = [float(value) for value in values]
        return out

    paths = gamma_backend_performance_figure(
        criteria=[criterion.replace("/10%", "") for criterion in criteria],
        beamlet_times_s=series(beam, "median_s"),
        plan_times_s=series(plan, "median_s"),
        beamlet_speedups={
            label: values
            for label, values in series(beam, "speedup").items()
            if label != RUNG_LABELS[REFERENCE_RUNG]
        },
        plan_speedups={
            label: values
            for label, values in series(plan, "speedup").items()
            if label != RUNG_LABELS[REFERENCE_RUNG]
        },
        figure_path=str(out_dir / "figures" / "gamma_backend_performance.svg"),
    )
    for path in paths:
        logger.info("Wrote %s", path)


def _load_rows(paths: Sequence[Path]) -> List[Dict]:
    """Concatenate the ``rows`` of several sweep JSONs."""
    rows: List[Dict] = []
    for path in paths:
        rows.extend(json.loads(path.read_text())["rows"])
    return rows


@app.command()
def main(
    plan_json: Path = typer.Option(..., help="Plan-level ladder JSON from gamma_benchmark report."),
    beamlet_sweep: List[Path] = typer.Option(..., help="Beamlet sweep JSON; repeat for several."),
    out_dir: Path = typer.Option(..., help="Report directory to write tables/, figures/ and data/ into."),
    interp_sweep: List[Path] = typer.Option([], help="Interpolation-fraction sweep JSON; repeat."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build every table, figure and data file the report includes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    plan = json.loads(plan_json.read_text())
    beamlet_rows = _load_rows(beamlet_sweep)
    interp_rows = _load_rows(interp_sweep)

    write_accuracy_table(plan, beamlet_rows, out_dir)
    write_timing_table(plan, beamlet_rows, out_dir)
    write_entry_point_table(beamlet_rows, out_dir)
    write_interp_table(interp_rows, out_dir)
    write_figure(plan, beamlet_rows, out_dir)
    typer.echo(f"Report assets written under {out_dir}")


if __name__ == "__main__":
    app()
