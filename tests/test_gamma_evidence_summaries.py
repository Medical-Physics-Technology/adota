"""Unit tests for the EXP-0008 summary and rendering modules.

Synthetic rows shaped like the harness output, so every reduction the report
tables depend on is exercised without a GPU or a dataset.
"""

from __future__ import annotations

import pytest

from src.metrics import gamma_evidence_latex as latex
from src.metrics import gamma_evidence_summaries as summaries
from src.metrics.gamma_beamlet_report import MatchedSetError


def _sweep_row(rung, sample_id, seconds, criterion="2%/2mm/10%", path="array", pass_rate=99.0):
    return {
        "sample_id": sample_id, "rung": rung, "criterion": criterion, "path": path, "interp_fraction": 10,
        "pass_rate_pct": pass_rate, "seconds_median": seconds, "seconds_best": seconds, "repeats": 5,
    }


def _sweep():
    rows = []
    for criterion in summaries.SHARED_CRITERIA:
        for sample_id, base in (("a", 1.0), ("b", 2.0), ("c", 8.0)):
            rows.append(_sweep_row(1, sample_id, base, criterion))
            rows.append(_sweep_row(3, sample_id, 0.25, criterion))
            rows.append(_sweep_row(4, sample_id, 0.2, criterion, pass_rate=99.01))
    return rows


def test_summarise_matched_reports_every_convention_with_counts():
    rows = summaries.summarise_matched(_sweep(), tested_rungs=(2, 3, 4))
    assert {r["tested_rung"] for r in rows} == {3, 4}  # rung 2 absent: skipped, not invented
    row = next(r for r in rows if r["tested_rung"] == 4 and r["criterion"] == "2%/2mm/10%")
    assert row["n_cases"] == 3 and row["repeats"] == 5
    assert row["median_paired"] == pytest.approx(10.0)
    assert row["ratio_of_medians"] == pytest.approx(2.0 / 0.2)
    assert row["ratio_of_sums"] == pytest.approx(11.0 / 0.6)
    assert row["tested_label"] == "PyTorch, GPU, float32"


def test_summarise_matched_raises_on_unmatched_rung():
    rows = _sweep() + [_sweep_row(2, "a", 1.5, "2%/2mm/10%")]
    with pytest.raises(MatchedSetError):
        summaries.summarise_matched(rows, tested_rungs=(2, 3, 4))


def test_sweep_pass_rate_deviation_is_against_rung_one_by_case():
    rows = summaries.summarise_sweep_pass_rates(_sweep())
    rung4 = [r for r in rows if r["rung"] == 4]
    assert all(r["max_abs_delta_pp"] == pytest.approx(0.01) for r in rung4)
    assert all(r["n_cases"] == 3 for r in rung4)


def _map_row(sample_id, tested, baseline, dtype, delta, crossings=0, mask=0, pr_delta=0.0, bitwise=False):
    return {
        "sample_id": sample_id, "criterion": "2%/2mm/10%", "tested_rung": tested, "baseline_rung": baseline,
        "other_dtype": dtype, "n_evaluated_both": 1000, "max_abs_delta": delta, "mean_abs_delta": delta / 2,
        "p99.99_abs_delta": delta * 0.9, "mask_disagreements": mask, "boundary_crossings": crossings,
        "pass_rate_delta_pp": pr_delta, "passed": crossings == 0, "bitwise_identical": bitwise,
    }


def test_beamlet_map_summary_aggregates_over_cases():
    rows = [
        _map_row("a", 3, 1, "float64", 1e-15, bitwise=False),
        _map_row("b", 3, 1, "float64", 5e-15, bitwise=True),
        _map_row("a", 4, 1, "float32", 3e-6, crossings=2, pr_delta=-0.02),
    ]
    summary = {(r["tested_rung"], r["baseline_rung"]): r for r in summaries.summarise_beamlet_maps(rows)}
    f64 = summary[(3, 1)]
    assert f64["n_comparisons"] == 2 and f64["n_evaluated_total"] == 2000
    assert f64["max_abs_delta"] == 5e-15 and f64["n_bitwise_identical"] == 1 and f64["all_gates_passed"]
    f32 = summary[(4, 1)]
    assert f32["boundary_crossings"] == 2 and f32["max_abs_pass_rate_delta_pp"] == pytest.approx(0.02)
    assert f32["all_gates_passed"] is False


def _pool_payload(rung, path, per_beamlet, passes=2):
    return {
        "results": [
            {
                "rung": rung, "path": path, "criterion": "2%/2mm/10%", "n_beamlets": len(per_beamlet),
                "sample_ids": [str(i) for i in range(len(per_beamlet))], "pass_rates_pct": [99.0] * len(per_beamlet),
                "passes": [
                    {"pass_index": k, "wall_s": sum(per_beamlet) * 1.05, "per_beamlet_s": list(per_beamlet),
                     "peak_gpu_bytes": 1 << 20 if rung != 1 else None}
                    for k in range(passes)
                ],
            }
        ]
    }


def test_pool_summary_and_projection_error():
    cached = {"test200": _pool_payload(1, "array", [0.2, 0.3, 1.0])}
    rows = summaries.summarise_pools(cached, {})
    (row,) = rows
    assert row["n_beamlets"] == 3 and row["passes"] == 2
    assert row["wall_median_s"] == pytest.approx(1.5 * 1.05)
    assert row["per_beamlet_max_s"] == 1.0 and row["beamlets_per_s"] == pytest.approx(3 / (1.5 * 1.05))
    errors = summaries.projection_errors(rows, {("test200", 1, "array"): 0.9})
    assert errors[0]["relative_error"] == pytest.approx((0.9 - 1.575) / 1.575)


def _scaling_row(plan, rung, voxels, seconds, n_eval=100):
    return {
        "plan": plan, "criterion": "2%/2mm/10%", "voxels": voxels, "rung": rung, "shape_zyx": [1, 1, voxels],
        "n_evaluated": n_eval, "n_above_cutoff": n_eval, "repeats": 3,
        "seconds_all": [seconds, seconds * 1.1, seconds * 0.9], "seconds_median": seconds,
        "seconds_q1": seconds * 0.95, "seconds_q3": seconds * 1.05, "iterations": 12, "shell_points": 500,
        "interp_samples": 10 ** 6, "samples_per_s": 10 ** 6 / seconds, "peak_gpu_bytes": 1 << 30 if rung != 1 else None,
    }


def test_scaling_summary_pairs_rungs_per_crop():
    rows = [
        _scaling_row("P", 1, 1000, 1.0), _scaling_row("P", 3, 1000, 0.5), _scaling_row("P", 4, 1000, 0.25),
        _scaling_row("P", 1, 8000, 8.0), _scaling_row("P", 3, 8000, 1.0), _scaling_row("P", 4, 8000, 0.5),
    ]
    summary = summaries.summarise_scaling(rows)
    assert [r["voxels"] for r in summary] == [1000, 8000]
    assert summary[1]["rung4_speedup"] == pytest.approx(16.0)
    assert summary[1]["rung4_speedup_min"] < 16.0 < summary[1]["rung4_speedup_max"]
    with pytest.raises(MatchedSetError):
        summaries.summarise_scaling([_scaling_row("P", 3, 1000, 0.5)])


def test_criterion_label_spacing_and_macro_names():
    assert latex.criterion_tex("1%/1mm/10%") == r"1\% / 1 mm / 10\%"
    lines = latex.numbers_macros({"pairedGpuSingleOne": "7.0"})
    assert lines[-1] == r"\newcommand{\pairedGpuSingleOne}{7.0}"
    with pytest.raises(ValueError):
        latex.numbers_macros({"bad_name": "1"})


def test_headline_values_come_from_the_summary_rows():
    matched = summaries.summarise_matched(_sweep(), tested_rungs=(3, 4))
    values = latex.headline_values(matched, [], [], [], [], None)
    assert values["pairedGpuSingleTwo"] == "10.0"
    assert values["nCasesGpuSingleTwo"] == "3"
    assert values["refMedianTwo"] == "2.000"


def test_environment_validation_refuses_mixed_runs():
    same = [{"environment": {"gpu": "A40", "torch": "2.8", "pymedphys": "0.41"}}] * 2
    summaries.validate_same_environment(same)
    with pytest.raises(ValueError):
        other = {"environment": {"gpu": "A100", "torch": "2.8", "pymedphys": "0.41"}}
        summaries.validate_same_environment(same + [other])
