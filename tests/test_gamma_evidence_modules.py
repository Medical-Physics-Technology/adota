"""Unit tests for the EXP-0008 evidence modules.

Everything that runs a GPU, a dataset or a checkpoint belongs to the
integration suite. What is testable here is the logic the evidence depends on:
that a paired speed-up refuses unmatched case sets, that map agreement
statistics count what they claim to count, that nested crops really nest and
keep their centre, and that the provenance manifest records what a re-run
needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.metrics.benchmark_provenance import environment_manifest, git_state, sha256_file, timestamped_run_dir
from src.metrics.gamma_beamlet_report import MatchedSetError, paired_speedups, timing_table
from src.metrics.gamma_map_agreement import GATES, check_gates, compare_maps, map_digest, pass_rate_pct
from src.metrics.gamma_scaling_benchmark import high_dose_centre, nested_crops


def _row(rung, sample_id, seconds, criterion="2%/2mm/10%", path="array", repeats=5):
    return {
        "sample_id": sample_id,
        "rung": rung,
        "criterion": criterion,
        "path": path,
        "interp_fraction": 10,
        "backend": "pymedphys" if rung == 1 else "torch",
        "device": "cpu" if rung in (1, 2) else "cuda:0",
        "dtype": None if rung == 1 else ("float64" if rung in (2, 3) else "float32"),
        "voxels": 144000,
        "energy_mev": 100.0,
        "pass_rate_pct": 99.0,
        "seconds_median": seconds,
        "seconds_best": seconds,
        "seconds_mean": seconds,
        "seconds_all": [seconds] * repeats,
        "repeats": repeats,
    }


# ── Paired speed-ups ────────────────────────────────────────────────────────


def test_paired_speedup_distinguishes_the_three_conventions():
    rows = [
        _row(1, "a", 1.0), _row(1, "b", 4.0), _row(1, "c", 10.0),
        _row(4, "a", 0.5), _row(4, "b", 0.5), _row(4, "c", 0.5),
    ]
    stats = paired_speedups(rows, 4, criterion="2%/2mm/10%")
    assert stats["ratios"] == pytest.approx([2.0, 8.0, 20.0])
    assert stats["median_paired"] == pytest.approx(8.0)
    assert stats["ratio_of_medians"] == pytest.approx(4.0 / 0.5)
    assert stats["ratio_of_sums"] == pytest.approx(15.0 / 1.5)
    assert stats["n_cases"] == 3 and stats["repeats"] == 5


def test_paired_speedup_refuses_unmatched_case_sets():
    rows = [_row(1, "a", 1.0), _row(1, "b", 2.0), _row(2, "a", 1.5)]
    with pytest.raises(MatchedSetError, match="different cases"):
        paired_speedups(rows, 2, criterion="2%/2mm/10%")


def test_paired_speedup_refuses_an_empty_side():
    rows = [_row(1, "a", 1.0)]
    with pytest.raises(MatchedSetError, match="no rows"):
        paired_speedups(rows, 3, criterion="2%/2mm/10%")


def test_ratio_of_medians_is_withheld_on_unmatched_sets():
    rows = [_row(1, "a", 1.0), _row(1, "b", 3.0), _row(2, "a", 1.0)]
    by_rung = {record["rung"]: record for record in timing_table(rows)}
    assert by_rung[2]["speedup"] is None
    assert by_rung[2]["total_speedup"] is None


# ── Map agreement ───────────────────────────────────────────────────────────


def _maps():
    rng = np.random.default_rng(3)
    reference = rng.random((4, 5, 6)).astype(np.float64) * 1.5
    reference[0, 0, :] = np.nan  # not evaluated
    return reference


def test_compare_maps_reports_zero_difference_for_identical_maps():
    reference = _maps()
    result = compare_maps(reference, reference.copy())
    assert result["max_abs_delta"] == 0.0
    assert result["mask_disagreements"] == 0
    assert result["boundary_crossings"] == 0
    assert result["pass_rate_delta_pp"] == 0.0
    assert result["bitwise_identical"] is True
    assert result["reference_digest"] == result["other_digest"] == map_digest(reference)


def test_compare_maps_counts_mask_disagreements_and_crossings():
    reference = _maps()
    other = reference.copy()
    other[1, 1, 1] = np.nan  # evaluated in the reference only
    # Push one passing point across the threshold and one failing point back.
    passing = np.argwhere(reference < 0.9)[0]
    failing = np.argwhere(reference > 1.1)[0]
    other[tuple(passing)] = 1.05
    other[tuple(failing)] = 0.95
    result = compare_maps(reference, other)
    assert result["mask_disagreements"] == 1
    assert result["boundary_crossings"] == 2
    assert result["bitwise_identical"] is False
    assert result["max_abs_delta"] > 0
    assert result["p99.99_abs_delta"] >= result["p50_abs_delta"]


def test_compare_maps_float32_against_float64_reports_native_dtypes():
    reference = _maps()
    other = reference.astype(np.float32)
    result = compare_maps(reference, other)
    assert result["reference_dtype"] == "float64" and result["other_dtype"] == "float32"
    assert 0 < result["max_abs_delta"] < 1e-6
    assert result["bitwise_identical"] is False


def test_gates_are_stricter_for_float64():
    comparison = {"pass_rate_delta_pp": 0.0, "mask_disagreements": 1, "boundary_crossings": 0}
    assert check_gates(comparison, "float64")["passed"] is False
    assert check_gates(comparison, "float32")["passed"] is True
    assert GATES["float64"]["max_pass_rate_delta_pp"] < GATES["float32"]["max_pass_rate_delta_pp"]


def test_pass_rate_matches_the_project_definition():
    gamma_map = np.array([np.nan, 0.5, 1.5, 0.0, 0.9])
    # Three positive values, one above 1: 1 - 1/3.
    assert pass_rate_pct(gamma_map) == pytest.approx(100 * (1 - 1 / 3))


# ── Nested crops ────────────────────────────────────────────────────────────


def test_nested_crops_nest_and_keep_the_centre():
    reference = np.zeros((40, 50, 60))
    reference[18:23, 23:28, 28:33] = 8.0
    reference[20, 25, 30] = 10.0  # after the block, so it is not overwritten
    evaluation = reference * 0.99
    crops = nested_crops(reference, evaluation, [1000, 20000])
    assert [c.target_voxels for c in crops] == [1000, 20000, 40 * 50 * 60]
    for smaller, larger in zip(crops, crops[1:]):
        for (a0, a1), (b0, b1) in zip(smaller.bounds_zyx, larger.bounds_zyx):
            assert b0 <= a0 and a1 <= b1
    centre = high_dose_centre(reference)
    for crop in crops:
        for (lo, hi), c in zip(crop.bounds_zyx, centre):
            assert lo <= c < hi
    assert crops[-1].shape == reference.shape
    assert crops[0].reference.max() == 10.0


def test_crop_description_carries_hashes_and_cutoff_population():
    reference = np.linspace(0, 1, 2 * 3 * 4).reshape(2, 3, 4)
    crop = nested_crops(reference, reference, [6])[0]
    description = crop.describe(cutoff_dose=0.5)
    assert description["n_above_cutoff"] == int((crop.reference >= 0.5).sum())
    assert len(description["reference_sha256"]) == 64


# ── Provenance ──────────────────────────────────────────────────────────────


def test_environment_manifest_records_threads_and_versions():
    manifest = environment_manifest(None)
    for key in ("python", "numpy", "torch", "torch_num_threads", "thread_env", "cpu_logical", "gpus"):
        assert key in manifest
    assert set(manifest["thread_env"]) >= {"OMP_NUM_THREADS", "NUMBA_NUM_THREADS"}


def test_git_state_reports_this_checkout(tmp_path):
    state = git_state(Path(__file__).resolve().parents[1], tmp_path, "adota")
    assert len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)
    if state["dirty"]:
        assert Path(state["diff_path"]).is_file()


def test_sha256_and_run_dir(tmp_path):
    payload = tmp_path / "x.json"
    payload.write_text(json.dumps({"a": 1}))
    assert len(sha256_file(payload)) == 64
    run_dir = timestamped_run_dir(tmp_path, "exp")
    assert run_dir.is_dir() and run_dir.name.startswith("exp_")
