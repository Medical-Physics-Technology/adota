"""Tests for the pipeline timing report (build / format / JSON)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_plan_opentps", Path(__file__).resolve().parent.parent / "scripts" / "run_plan_opentps.py"
)
rpo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rpo)


def _extraction():
    return {
        "n_spots": 100,
        "elapsed_s": 20.0,
        "timing_per_field": {
            0: {"rotation_s": 0.1, "crop_s_total": 0.5, "flux_s_total": 8.0, "save_s_total": 2.0},
            1: {"rotation_s": 0.1, "crop_s_total": 0.5, "flux_s_total": 8.0, "save_s_total": 2.0},
        },
    }


def _inference():
    return {
        "n_spots": 100, "elapsed_s": 6.0, "n_batches": 2, "batch_size": 56, "device": "cpu",
        "load_s": 4.0, "read_s": 1.5, "downsample_s": 2.5,
        "forward_s": 1.0,
        "save_s": 1.0, "upsample_s": 0.3, "save_write_s": 0.7,
        "ms_per_spot_load": 40.0, "ms_per_spot_read": 15.0, "ms_per_spot_downsample": 25.0,
        "ms_per_spot_forward": 10.0,
        "ms_per_spot_save": 10.0, "ms_per_spot_upsample": 3.0, "ms_per_spot_save_write": 7.0,
    }


def _accumulation():
    return {"n_spots": 100, "n_fields": 2, "elapsed_s": 3.0, "deposit_s": 2.0, "derotate_s": 0.8, "write_s": 0.2}


def test_build_report_aggregates_and_is_json_serializable() -> None:
    rep = rpo._build_timing_report(
        total_s=30.0, model_load_s=0.3, plan_load_s=0.5,
        extraction=_extraction(), inference=_inference(), accumulation=_accumulation(), figure_s=2.0,
    )
    assert rep["total_s"] == 30.0
    assert rep["n_spots"] == 100
    ex = rep["stages"]["extraction"]
    assert ex["steps"]["flux_projection"]["total_s"] == 16.0  # 8 + 8
    assert ex["ms_per_beamlet"] == 200.0  # 20 s / 100 * 1000
    inf_steps = rep["stages"]["inference"]["steps"]
    assert inf_steps["forward"]["ms_per_beamlet"] == 10.0
    # load is split into file read vs downsample-to-grid; save into upsample vs write.
    assert inf_steps["record_load"]["ms_per_beamlet"] == 15.0
    assert inf_steps["downsample"]["ms_per_beamlet"] == 25.0
    assert inf_steps["upsample"]["total_s"] == 0.3
    assert inf_steps["save_write"]["total_s"] == 0.7
    assert "write" in rep["stages"]["accumulation"]["steps"]
    assert rep["stages"]["comparison_figures"]["total_s"] == 2.0
    json.dumps(rep)  # must not raise


def test_format_report_table() -> None:
    rep = rpo._build_timing_report(
        total_s=30.0, model_load_s=0.3, plan_load_s=0.5,
        extraction=_extraction(), inference=_inference(), accumulation=_accumulation(), figure_s=2.0,
    )
    table = rpo._format_timing_report(rep)
    assert "TIMING SUMMARY" in table
    assert "flux projection" in table
    assert "record load (file read)" in table
    assert "downsample (CT -> ADoTA grid)" in table
    assert "ADoTA forward" in table
    assert "upsample (ADoTA -> ROI grid)" in table
    assert "save predictions to disk" in table
    assert "TOTAL" in table


def test_partial_stages_omitted() -> None:
    rep = rpo._build_timing_report(
        total_s=10.0, model_load_s=0.0, plan_load_s=0.5,
        extraction=_extraction(), inference=None, accumulation=None, figure_s=0.0,
    )
    stages = rep["stages"]
    assert "model_load" not in stages  # model_load_s == 0
    assert "inference" not in stages
    assert "accumulation" not in stages
    assert "comparison_figures" not in stages
    assert "extraction" in stages
