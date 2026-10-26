"""Unit tests for the beamlet-scale gamma benchmark and its report reduction.

The benchmark's expensive parts -- building dose pairs from a checkpoint and
timing a real gamma evaluation -- need a GPU, a dataset and a trained model, so
they belong to the integration suite. What is testable here without any of that
is the logic around them: which device each rung resolves to, whether the cache
round-trips, and whether the report reduction attributes deviations and
speed-ups to the right rung. All three have been wrong at least once.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.gamma_beamlet_benchmark import (
    RUNGS,
    BeamletPair,
    GammaCase,
    criterion_label,
    load_pairs,
    save_pairs,
)
from src.metrics.gamma_beamlet_report import (
    build_report,
    parity_table,
    throughput_table,
    timing_table,
)

SCALE = {"min_ds": 0.0, "max_ds": 100.0}


def _pairs(count=2, shape=(4, 3, 3)):
    rng = np.random.default_rng(7)
    return [
        BeamletPair(
            sample_id=f"beamlet-{index}",
            energy_mev=100.0 + index,
            reference=rng.random(shape, dtype=np.float32),
            evaluation=rng.random(shape, dtype=np.float32),
        )
        for index in range(count)
    ]


def _row(rung, criterion="3%/3mm/10%", sample_id="a", pass_rate=99.0, seconds=1.0, path="array"):
    backend = "pymedphys" if rung == 1 else "torch"
    device = {1: "cpu", 2: "cpu", 3: "cuda:0", 4: "cuda:0"}[rung]
    dtype = {1: None, 2: "float64", 3: "float64", 4: "float32"}[rung]
    return {
        "sample_id": sample_id,
        "energy_mev": 100.0,
        "voxels": 144000,
        "criterion": criterion,
        "interp_fraction": 10,
        "max_gamma": 2.0,
        "rung": rung,
        "backend": backend,
        "device": device,
        "dtype": dtype,
        "path": path,
        "pass_rate_pct": pass_rate,
        "seconds_best": seconds,
        "seconds_mean": seconds,
        "repeats": 3,
    }


# ── Rung device resolution ──────────────────────────────────────────────────


def test_cli_device_moves_only_the_gpu_rungs():
    """``--device cuda:1`` must not drag the torch-CPU rung onto the GPU.

    Rung 2 exists to separate the implementation from the device. If the CLI
    override applied to it, rungs 2 and 3 would be the same measurement and the
    ladder would prove nothing.
    """
    assert RUNGS["rung2"].resolve_device("cuda:1") == "cpu"
    assert RUNGS["rung3"].resolve_device("cuda:1") == "cuda:1"
    assert RUNGS["rung4"].resolve_device("cuda:1") == "cuda:1"
    assert RUNGS["rung1"].resolve_device("cuda:1") is None


def test_backend_options_follow_the_resolved_device():
    assert RUNGS["rung1"].backend_options("cuda:1") is None
    assert RUNGS["rung2"].backend_options("cuda:1") == {"device": "cpu", "dtype": "float64"}
    assert RUNGS["rung4"].backend_options("cuda:2") == {"device": "cuda:2", "dtype": "float32"}


def test_criterion_label_matches_the_plan_level_form():
    assert criterion_label(GammaCase(3.0, 3.0, 10.0)) == "3%/3mm/10%"
    assert criterion_label(GammaCase(1.0, 2.0, 0.1)) == "1%/2mm/0.1%"


def test_gamma_case_carries_the_search_parameters_through():
    params = GammaCase(2.0, 2.0, 10.0, interp_fraction=5, max_gamma=3.0).as_params()
    assert params["interp_fraction"] == 5
    assert params["max_gamma"] == 3.0
    assert params["local_gamma"] is False


# ── Cache round trip ────────────────────────────────────────────────────────


def test_pair_cache_round_trips(tmp_path):
    original = _pairs(3)
    path = tmp_path / "pairs.npz"
    save_pairs(path, original, SCALE)
    restored, scale = load_pairs(path)

    assert scale == SCALE
    assert [pair.sample_id for pair in restored] == [pair.sample_id for pair in original]
    for before, after in zip(original, restored):
        assert after.energy_mev == pytest.approx(before.energy_mev)
        np.testing.assert_array_equal(after.reference, before.reference)
        np.testing.assert_array_equal(after.evaluation, before.evaluation)
        assert after.voxels == before.voxels


# ── Report reduction ────────────────────────────────────────────────────────


def test_parity_is_measured_against_pymedphys_not_against_the_mean():
    rows = [
        _row(1, sample_id="a", pass_rate=99.0),
        _row(1, sample_id="b", pass_rate=90.0),
        _row(4, sample_id="a", pass_rate=99.5),
        _row(4, sample_id="b", pass_rate=90.0),
    ]
    (record,) = parity_table(rows)
    assert record["rung"] == 4
    assert record["max_abs_delta_pp"] == pytest.approx(0.5)
    assert record["mean_abs_delta_pp"] == pytest.approx(0.25)
    assert record["worst_sample_id"] == "a"


def test_parity_pairs_rows_by_beamlet_and_criterion():
    """A rung measured on a different criterion must not be compared across it."""
    rows = [
        _row(1, criterion="1%/1mm/10%", sample_id="a", pass_rate=80.0),
        _row(1, criterion="3%/3mm/10%", sample_id="a", pass_rate=99.0),
        _row(4, criterion="3%/3mm/10%", sample_id="a", pass_rate=99.0),
    ]
    (record,) = parity_table(rows)
    assert record["criterion"] == "3%/3mm/10%"
    assert record["max_abs_delta_pp"] == pytest.approx(0.0)


def test_timing_reports_the_median_and_the_speedup_over_pymedphys():
    rows = [
        _row(1, sample_id="a", seconds=1.0),
        _row(1, sample_id="b", seconds=3.0),
        _row(1, sample_id="c", seconds=5.0),
        _row(4, sample_id="a", seconds=0.1),
        _row(4, sample_id="b", seconds=0.3),
        _row(4, sample_id="c", seconds=0.5),
    ]
    by_rung = {record["rung"]: record for record in timing_table(rows)}
    assert by_rung[1]["median_s"] == pytest.approx(3.0)
    assert by_rung[4]["median_s"] == pytest.approx(0.3)
    assert by_rung[4]["speedup"] == pytest.approx(10.0)
    assert by_rung[1]["speedup"] == pytest.approx(1.0)
    assert by_rung[4]["beamlets_per_s"] == pytest.approx(1 / 0.3)


def test_a_slower_rung_reports_a_speedup_below_one():
    """The torch-CPU rung is slower than pymedphys on a beamlet; say so."""
    rows = [_row(1, seconds=1.0), _row(2, seconds=15.0)]
    by_rung = {record["rung"]: record for record in timing_table(rows)}
    assert by_rung[2]["speedup"] == pytest.approx(1 / 15)


def test_throughput_scales_the_median_by_the_pool_size():
    rows = [_row(1, seconds=2.0), _row(4, seconds=0.5)]
    table = {(record["rung"], record["pool_size"]): record for record in
             throughput_table(rows, pool_sizes=(1, 200))}
    assert table[(1, 200)]["pass_seconds"] == pytest.approx(400.0)
    assert table[(4, 200)]["pass_seconds"] == pytest.approx(100.0)


def test_build_report_carries_every_table_and_the_raw_rows():
    rows = [_row(1), _row(3), _row(4)]
    report = build_report(rows, {"gpu": "NVIDIA A40"}, pool_sizes=(20,))
    assert set(report) == {"environment", "beamlets", "parity", "timing", "throughput", "rows"}
    assert report["rows"] == rows
    assert {record["rung"] for record in report["parity"]} == {3, 4}
    assert {record["rung"] for record in report["timing"]} == {1, 3, 4}
