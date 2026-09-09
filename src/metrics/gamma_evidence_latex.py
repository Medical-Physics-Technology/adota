# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Render the EXP-0008 summary rows as LaTeX tabulars, CSVs and number macros.

Every table the report includes is written here from the rows
:mod:`src.metrics.gamma_evidence_summaries` produces, with a CSV twin of the
same numbers beside it. The headline numbers the prose quotes are written as
``\\newcommand`` macros into ``numbers.tex``, so the abstract, results and
conclusion cannot drift from the tables: they read the same values.

Conventions the review asked for and this module enforces: criterion labels
carry spaces between number and unit (``1\\% / 1 mm / 10\\%``), backends are
named descriptively rather than by rung, sample and repetition counts appear in
every table, and where a distribution exists its median is shown with the
interquartile range.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "criterion_tex",
    "write_csv",
    "write_tex",
    "matched_table",
    "beamlet_maps_table",
    "plan_maps_table",
    "pools_table",
    "projection_table",
    "scaling_table",
    "profile_table",
    "numbers_macros",
]

# LaTeX-safe backend names.
LABEL_TEX = {
    "PyMedPhys, CPU": r"PyMedPhys, CPU",
    "PyTorch, CPU, float64": r"PyTorch, CPU, float64",
    "PyTorch, GPU, float64": r"PyTorch, GPU, float64",
    "PyTorch, GPU, float32": r"PyTorch, GPU, float32",
}


def criterion_tex(label: str) -> str:
    """``"1%/1mm/10%"`` as ``1\\% / 1 mm / 10\\%``."""
    dose, _, rest = label.partition("%/")
    distance, _, cutoff = rest.partition("mm/")
    return rf"{dose}\% / {distance} mm / {cutoff.rstrip('%')}\%"


def _iqr(median: float, q1: float, q3: float, digits: int = 3) -> str:
    return rf"{median:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)
    return path


def write_tex(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", path)
    return path


def _tabular(spec: str, header: Sequence[str], body: Sequence[str]) -> List[str]:
    return [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", *header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]


# ── Experiment A ────────────────────────────────────────────────────────────


def matched_table(rows: Sequence[Dict[str, Any]], path: str) -> List[str]:
    """Paired speed-ups on matched cases, three conventions side by side."""
    body = []
    selected = [r for r in rows if r["path"] == path]
    for row in selected:
        body.append(
            " & ".join(
                [
                    criterion_tex(row["criterion"]),
                    LABEL_TEX[row["tested_label"]],
                    str(row["n_cases"]),
                    str(row["repeats"]),
                    _iqr(row["reference_median_s"], row["reference_q1_s"], row["reference_q3_s"]),
                    _iqr(row["tested_median_s"], row["tested_q1_s"], row["tested_q3_s"]),
                    rf"{row['median_paired']:.1f} [{row['q1_paired']:.1f}, {row['q3_paired']:.1f}]",
                    rf"{row['min_paired']:.1f}--{row['max_paired']:.1f}",
                    rf"{row['ratio_of_medians']:.1f}",
                    rf"{row['ratio_of_sums']:.1f}",
                ]
            )
            + r" \\"
        )
    header = [
        r"Criterion & Tested backend & $n$ & reps & PyMedPhys [s] & tested [s]"
        r" & paired & range & medians & sums \\",
        r" & & & & median [IQR] & median [IQR] & median [IQR] & & ratio & ratio \\",
    ]
    return _tabular("llrrrrrrrr", header, body)


def pass_rate_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    body = [
        " & ".join([criterion_tex(r["criterion"]), LABEL_TEX[r["label"]], str(r["n_cases"]),
                    f"{r['max_abs_delta_pp']:.6f}", f"{r['mean_abs_delta_pp']:.6f}"]) + r" \\"
        for r in rows
    ]
    header = [r"Criterion & Tested backend & $n$ & max $|\Delta|$ [pp] & mean $|\Delta|$ [pp] \\"]
    return _tabular("llrrr", header, body)


# ── Experiment B ────────────────────────────────────────────────────────────


def _map_row(row: Dict[str, Any], leading: Sequence[str]) -> str:
    return " & ".join(
        [
            *leading,
            LABEL_TEX[row["tested_label"]],
            LABEL_TEX[row["baseline_label"]].replace("PyMedPhys, CPU", "PyMedPhys"),
            row["other_dtype"],
            str(row["n_comparisons"]),
            rf"\num{{{row['n_evaluated_total']}}}",
            rf"\num{{{row['max_abs_delta']:.2e}}}",
            rf"\num{{{row['max_p99.99_abs_delta']:.2e}}}",
            str(row["mask_disagreements"]),
            str(row["boundary_crossings"]),
            f"{row['max_abs_pass_rate_delta_pp']:.6f}",
            f"{row['n_bitwise_identical']}/{row['n_comparisons']}",
        ]
    ) + r" \\"


def beamlet_maps_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    header = [
        r"Criterion & Tested & Baseline & dtype & $n$ & points & max $|\Delta\gamma|$"
        r" & p99.99 $|\Delta\gamma|$ & mask & cross. & max $|\Delta$GPR$|$ [pp] & bitwise \\"
    ]
    body = [_map_row(r, [criterion_tex(r["criterion"])]) for r in rows]
    return _tabular("lllrrrrrrrrr", header, body)


def plan_maps_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    header = [
        r"Plan & Criterion & Tested & Baseline & dtype & $n$ & points & max $|\Delta\gamma|$"
        r" & p99.99 $|\Delta\gamma|$ & mask & cross. & max $|\Delta$GPR$|$ [pp] & bitwise \\"
    ]
    body = [_map_row(r, [r["plan"].split("_")[0], criterion_tex(r["criterion"])]) for r in rows]
    return _tabular("llllrrrrrrrrr", header, body)


# ── Experiment C ────────────────────────────────────────────────────────────


def pools_table(rows: Sequence[Dict[str, Any]], kind: str, pool_names: Dict[str, str]) -> List[str]:
    body = []
    for row in [r for r in rows if r["kind"] == kind]:
        cells = [
            pool_names.get(row["pool"], row["pool"]),
            str(row["n_beamlets"]),
            LABEL_TEX[row["label"]],
            row["path"],
            str(row["passes"]),
            rf"{row['wall_median_s']:.1f} [{row['wall_min_s']:.1f}, {row['wall_max_s']:.1f}]",
            rf"{row['beamlets_per_s']:.1f}",
            rf"{row['per_beamlet_median_s']:.3f}",
            rf"{row['per_beamlet_p95_s']:.3f}",
            rf"{row['per_beamlet_p99_s']:.3f}",
            rf"{row['per_beamlet_max_s']:.2f}",
            "--" if row["peak_gpu_bytes"] is None else rf"{row['peak_gpu_bytes'] / 2**30:.2f}",
        ]
        if kind == "integrated":
            cells.insert(6, rf"{row['inference_sum_median_s']:.1f} / {row['gamma_sum_median_s']:.1f}")
        body.append(" & ".join(cells) + r" \\")
    if kind == "integrated":
        header = [
            r"Pool & $n$ & Backend & entry & passes & pass wall [s] & inference / gamma [s]"
            r" & beamlets/s & median [s] & p95 [s] & p99 [s] & max [s] & GPU [GiB] \\"
        ]
        return _tabular("lrllrrrrrrrrr", header, body)
    header = [
        r"Pool & $n$ & Backend & entry & passes & pass wall [s] & beamlets/s"
        r" & median [s] & p95 [s] & p99 [s] & max [s] & GPU [GiB] \\"
    ]
    return _tabular("lrllrrrrrrrr", header, body)


def projection_table(rows: Sequence[Dict[str, Any]], pool_names: Dict[str, str]) -> List[str]:
    body = [
        " & ".join(
            [pool_names.get(r["pool"], r["pool"]), str(r["n_beamlets"]), LABEL_TEX[r["label"]], r["path"],
             f"{r['projected_s']:.1f}", f"{r['wall_median_s']:.1f}", rf"{100 * r['relative_error']:+.1f}\%"]
        ) + r" \\"
        for r in rows
    ]
    return _tabular("lrllrrr", [r"Pool & $n$ & Backend & entry & projected [s] & measured [s] & error \\"], body)


# ── Experiment D ────────────────────────────────────────────────────────────


def scaling_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    body = []
    for r in rows:
        body.append(
            " & ".join(
                [
                    r["plan"].split("_")[0],
                    rf"\num{{{r['voxels']}}}",
                    rf"\num{{{r['n_evaluated']}}}",
                    str(r["repeats"]),
                    _iqr(r["rung1_median_s"], r["rung1_q1_s"], r["rung1_q3_s"], 2),
                    _iqr(r["rung3_median_s"], r["rung3_q1_s"], r["rung3_q3_s"], 3),
                    _iqr(r["rung4_median_s"], r["rung4_q1_s"], r["rung4_q3_s"], 3),
                    rf"{r['rung3_speedup']:.1f}",
                    rf"{r['rung4_speedup']:.1f}",
                    rf"\num{{{r['rung4_interp_samples']:.2e}}}",
                    str(r["rung4_iterations"]),
                    rf"{r['rung4_samples_per_s'] / 1e6:.1f}",
                    ("--" if r.get("rung4_peak_gpu_bytes") is None
                     else f"{r['rung4_peak_gpu_bytes'] / 2**30:.2f}"),
                ]
            )
            + r" \\"
        )
    header = [
        r"Plan & voxels & evaluated & reps & PyMedPhys [s] & GPU float64 [s] & GPU float32 [s]"
        r" & speed-up & speed-up & samples & iter. & Msamples/s & GPU [GiB] \\",
        r" & & points & & median [IQR] & median [IQR] & median [IQR] & float64 & float32"
        r" & (float32) & & (float32) & \\",
    ]
    return _tabular("lrrrrrrrrrrrr", header, body)


# ── Experiment E ────────────────────────────────────────────────────────────


def profile_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    body = [
        " & ".join(
            [
                r["role"],
                r["dtype"],
                f"{r['wall_unprofiled_s']:.4f}",
                f"{r['wall_profiled_s']:.4f}",
                f"{r['device_kernel_s']:.4f}",
                f"{r['device_memcpy_s']:.4f}",
                f"{r['host_launch_s']:.4f}",
                f"{r['host_synchronise_s']:.4f}",
                f"{r['host_allocate_s']:.4f}",
                f"{r['host_other_cpu_s']:.4f}",
                f"{r['kernel_launches_per_call']:.0f}",
                str(r["iterations_in_loop"]),
                rf"{100 * r['device_busy_fraction']:.0f}\%",
            ]
        )
        + r" \\"
        for r in rows
    ]
    header = [
        r"Beamlet & dtype & wall [s] & profiled [s] & kernels [s] & memcpy [s] & launch [s]"
        r" & sync [s] & alloc [s] & other host [s] & launches & loop iter. & device busy \\"
    ]
    return _tabular("llrrrrrrrrrrr", header, body)


# ── Headline numbers ────────────────────────────────────────────────────────


def numbers_macros(values: Dict[str, str]) -> List[str]:
    """``\\newcommand`` lines for every headline number the prose quotes.

    Macro names are letters only, as LaTeX requires; the caller supplies the
    formatted string so the table and the prose format a value the same way.
    """
    lines = ["% Generated by scripts/analysis/report_gamma_acceleration.py. Do not edit."]
    for name, value in values.items():
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name!r}")
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")
    return lines


def headline_values(
    matched: Sequence[Dict[str, Any]],
    beamlet_maps: Sequence[Dict[str, Any]],
    plan_maps: Sequence[Dict[str, Any]],
    pools: Sequence[Dict[str, Any]],
    scaling: Sequence[Dict[str, Any]],
    profile: Optional[Sequence[Dict[str, Any]]],
) -> Dict[str, str]:
    """Pick the numbers the abstract and conclusion quote, formatted once."""
    values: Dict[str, str] = {}

    def matched_row(rung: int, criterion: str, path: str = "array") -> Optional[Dict[str, Any]]:
        return next(
            (r for r in matched if r["tested_rung"] == rung and r["criterion"] == criterion and r["path"] == path),
            None,
        )

    letters = {"1%/1mm/10%": "One", "2%/2mm/10%": "Two", "3%/3mm/10%": "Three"}
    rung_letters = {2: "Cpu", 3: "GpuDouble", 4: "GpuSingle"}
    for criterion, suffix in letters.items():
        for rung, prefix in rung_letters.items():
            row = matched_row(rung, criterion)
            if row is None:
                continue
            values[f"paired{prefix}{suffix}"] = f"{row['median_paired']:.1f}"
            values[f"pairedQone{prefix}{suffix}"] = f"{row['q1_paired']:.1f}"
            values[f"pairedQthree{prefix}{suffix}"] = f"{row['q3_paired']:.1f}"
            values[f"sums{prefix}{suffix}"] = f"{row['ratio_of_sums']:.1f}"
            values[f"medians{prefix}{suffix}"] = f"{row['ratio_of_medians']:.1f}"
            values[f"nCases{prefix}{suffix}"] = str(row["n_cases"])
        reference = matched_row(4, criterion)
        if reference is not None:
            values[f"refMedian{suffix}"] = f"{reference['reference_median_s']:.3f}"
            values[f"refMax{suffix}"] = f"{reference['reference_max_s']:.2f}"
            values[f"gpuSingleMedian{suffix}"] = f"{reference['tested_median_s']:.3f}"
            values[f"gpuSingleMax{suffix}"] = f"{reference['tested_max_s']:.3f}"
        double = matched_row(3, criterion)
        if double is not None:
            values[f"gpuDoubleMedian{suffix}"] = f"{double['tested_median_s']:.3f}"

    if beamlet_maps:
        f64 = [r for r in beamlet_maps if r["other_dtype"] == "float64" and r["baseline_rung"] == 1]
        f32 = [r for r in beamlet_maps if r["other_dtype"] == "float32"]
        values["beamletMapsDoubleMaxDelta"] = f"{max(r['max_abs_delta'] for r in f64):.1e}"
        values["beamletMapsDoubleCrossings"] = str(sum(r["boundary_crossings"] for r in f64))
        values["beamletMapsDoubleMask"] = str(sum(r["mask_disagreements"] for r in f64))
        values["beamletMapsDoublePassRate"] = f"{max(r['max_abs_pass_rate_delta_pp'] for r in f64):.6f}"
        values["beamletMapsSingleMaxDelta"] = f"{max(r['max_abs_delta'] for r in f32):.1e}"
        values["beamletMapsSinglePassRate"] = f"{max(r['max_abs_pass_rate_delta_pp'] for r in f32):.3f}"
        values["beamletMapsSingleCrossings"] = str(sum(r["boundary_crossings"] for r in f32))
        values["beamletMapsPoints"] = f"{f64[0]['n_evaluated_total']}"
        values["beamletMapsBitwise"] = str(sum(r["n_bitwise_identical"] for r in f64))
        values["beamletMapsComparisons"] = str(sum(r["n_comparisons"] for r in f64))
    if plan_maps:
        f64 = [r for r in plan_maps if r["other_dtype"] == "float64" and r["baseline_rung"] == 1]
        f32 = [r for r in plan_maps if r["other_dtype"] == "float32"]
        values["planMapsDoubleMaxDelta"] = f"{max(r['max_abs_delta'] for r in f64):.1e}"
        values["planMapsDoubleCrossings"] = str(sum(r["boundary_crossings"] for r in f64))
        values["planMapsDoubleMask"] = str(sum(r["mask_disagreements"] for r in f64))
        values["planMapsDoublePassRate"] = f"{max(r['max_abs_pass_rate_delta_pp'] for r in f64):.6f}"
        values["planMapsSingleMaxDelta"] = f"{max(r['max_abs_delta'] for r in f32):.1e}"
        values["planMapsSinglePassRate"] = f"{max(r['max_abs_pass_rate_delta_pp'] for r in f32):.4f}"
        values["planMapsSingleCrossings"] = str(sum(r["boundary_crossings"] for r in f32))
        values["planMapsPoints"] = f"{sum(r['n_evaluated_total'] for r in f64)}"
    for row in pools:
        if row["kind"] != "cached":
            continue
        tag = row["pool"].replace("valsplit", "split").replace("test", "test")
        size_words = {200: "TwoHundred", 948: "All", 2000: "TwoThousand"}
        tag = "".join(ch for ch in tag if ch.isalpha()) + size_words.get(row["n_beamlets"], "")
        rung_tag = {1: "Ref", 3: "GpuDouble", 4: "GpuSingle"}[row["rung"]]
        values[f"pool{tag}{rung_tag}{row['path'].capitalize()}Wall"] = f"{row['wall_median_s']:.1f}"
        values[f"pool{tag}{rung_tag}{row['path'].capitalize()}Rate"] = f"{row['beamlets_per_s']:.1f}"
        values[f"pool{tag}N"] = str(row["n_beamlets"])
    for row in scaling:
        if row["voxels"] == min(r["voxels"] for r in scaling if r["plan"] == row["plan"]):
            plan_tag = "Lung" if "LUNG" in row["plan"] else "Prostate"
            values[f"scaling{plan_tag}SmallSpeedupSingle"] = f"{row['rung4_speedup']:.1f}"
        if row["voxels"] == max(r["voxels"] for r in scaling if r["plan"] == row["plan"]):
            plan_tag = "Lung" if "LUNG" in row["plan"] else "Prostate"
            values[f"scaling{plan_tag}FullSpeedupSingle"] = f"{row['rung4_speedup']:.1f}"
            values[f"scaling{plan_tag}FullSpeedupDouble"] = f"{row['rung3_speedup']:.1f}"
            values[f"scaling{plan_tag}FullVoxels"] = f"{row['voxels'] / 1e6:.1f}"
    if profile:
        for row in profile:
            tag = f"{row['role'].capitalize()}{'Double' if row['dtype'] == 'float64' else 'Single'}"
            values[f"profileBusy{tag}"] = f"{100 * row['device_busy_fraction']:.0f}"
            values[f"profileLaunches{tag}"] = f"{row['kernel_launches_per_call']:.0f}"
            values[f"profileHostOther{tag}"] = f"{100 * row['host_other_fraction']:.0f}"
    return values
