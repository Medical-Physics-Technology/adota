"""Per-beamlet metadata and site summary of the engine-driver dataset overview (synthetic CSVs only)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

from src.engine_drivers.dataset_overview import (
    METADATA_COLUMNS,
    SUMMARY_COLUMNS,
    beamlet_metadata,
    check_join,
    site_from_patient,
    site_summary,
)

ROWS: List[Dict[str, object]] = [
    {"sample_id": "s3", "pool_patient_id": "LUNG1-001", "pool_anatomy": "thoracic", "gantry_angle": 370.0,
     "energy_mev": 100.0, "bixel_shift_xy_mm_0": -3.0, "bixel_shift_xy_mm_1": 4.0, "field_dir": "f_lung_a"},
    {"sample_id": "s1", "pool_patient_id": "LUNG1-001", "pool_anatomy": "thoracic", "gantry_angle": 10.0,
     "energy_mev": 120.0, "bixel_shift_xy_mm_0": 3.0, "bixel_shift_xy_mm_1": -4.0, "field_dir": "f_lung_a"},
    {"sample_id": "s2", "pool_patient_id": "Lung_Dx-A0080", "pool_anatomy": "thoracic", "gantry_angle": 200.0,
     "energy_mev": 140.0, "bixel_shift_xy_mm_0": 9.0, "bixel_shift_xy_mm_1": 12.0, "field_dir": "f_dx_a"},
    {"sample_id": "s4", "pool_patient_id": "Prostate-AEC-074", "pool_anatomy": "pelvic", "gantry_angle": 90.0,
     "energy_mev": 180.0, "bixel_shift_xy_mm_0": 0.5, "bixel_shift_xy_mm_1": 0.5, "field_dir": "f_pro_a"},
]
PEAK = {"s1": "True", "s2": "False", "s3": "True", "s4": "True"}


def _write(tmp_path: Path, rows: Optional[List[Dict[str, object]]] = None, peak: Optional[Dict[str, str]] = None):
    """Write the three input CSVs; returns their paths."""
    frame = pd.DataFrame(rows if rows is not None else ROWS)
    index = frame.drop(columns=["field_dir"]).assign(extra="x")
    plan = pd.DataFrame({"candidate_id": frame["sample_id"], "patient_id": frame["pool_patient_id"],
                         "theta_x_deg": 0.0, "field_dir": frame["field_dir"]})
    # plan rows of other splits must not leak into the metadata
    plan = pd.concat([plan, pd.DataFrame([{"candidate_id": "train1", "patient_id": "LUNG1-999",
                                           "theta_x_deg": 0.0, "field_dir": "f_train"}])])
    peak = peak if peak is not None else PEAK
    features = pd.DataFrame({"sample_id": list(peak), "peak_inside_crop": list(peak.values()), "other": 1.0})
    paths = tmp_path / "index.csv", tmp_path / "plan.csv", tmp_path / "features.csv"
    for path, table in zip(paths, (index, plan, features)):
        table.to_csv(path, index=False)
    return paths


@pytest.mark.parametrize("patient, site", [("LUNG1-001", "thoracic"), ("Lung_Dx-A0080", "thoracic"),
                                           ("StageII-Colorectal-CT-012", "abdominal"),
                                           ("Prostate-AEC-074", "pelvic")])
def test_prefix_mapping(patient, site):
    assert site_from_patient(patient) == site


@pytest.mark.parametrize("patient", ["HNSCC-001", "LUNG1001", "Prostate-074", ""])
def test_unknown_prefix_raises(patient):
    with pytest.raises(ValueError, match="unknown patient id prefix"):
        site_from_patient(patient)


def test_metadata_columns_order_and_values(tmp_path):
    meta = beamlet_metadata(*_write(tmp_path))
    assert tuple(meta.columns) == METADATA_COLUMNS
    assert meta["sample_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert meta["site"].tolist() == ["thoracic", "thoracic", "thoracic", "pelvic"]
    assert meta["gantry_deg"].tolist() == [10.0, 200.0, 10.0, 90.0]
    assert (meta["couch_deg"] == 0.0).all()
    assert meta["spot_x_mm"].tolist() == [3.0, 9.0, -3.0, 0.5]
    assert meta["spot_y_mm"].tolist() == [-4.0, 12.0, 4.0, 0.5]
    assert meta["in_analysis"].dtype == bool
    assert meta["in_analysis"].tolist() == [True, False, True, True]
    assert "train1" not in set(meta["sample_id"])


def test_anatomy_disagreement_raises(tmp_path):
    rows = [dict(r) for r in ROWS]
    rows[3]["pool_anatomy"] = "abdominal"
    with pytest.raises(ValueError, match="disagrees with pool_anatomy"):
        beamlet_metadata(*_write(tmp_path, rows))


def test_field_with_two_gantries_raises(tmp_path):
    rows = [dict(r) for r in ROWS]
    rows[1]["gantry_angle"] = 11.0
    with pytest.raises(ValueError, match="field_id maps to more than one"):
        beamlet_metadata(*_write(tmp_path, rows))


def test_one_gantry_two_fields_raises(tmp_path):
    rows = [dict(r) for r in ROWS]
    rows[1]["field_dir"] = "f_lung_b"
    with pytest.raises(ValueError, match="maps to more than one field_id"):
        beamlet_metadata(*_write(tmp_path, rows))


def test_missing_feature_row_raises(tmp_path):
    peak = {k: v for k, v in PEAK.items() if k != "s2"}
    with pytest.raises(ValueError, match="no feature row"):
        beamlet_metadata(*_write(tmp_path, peak=peak))


def _errors(tmp_path: Path, ids: List[str]) -> Path:
    path = tmp_path / "errors.csv"
    pd.DataFrame({"sample_id": ids, "engine": "adota", "mape_pct": 1.0}).to_csv(path, index=False)
    return path


def test_check_join_counts_rows(tmp_path):
    meta = beamlet_metadata(*_write(tmp_path))
    assert check_join(meta, _errors(tmp_path, ["s4", "s3", "s2", "s1"])) == 4


@pytest.mark.parametrize("ids", [["s1", "s2", "s3"], ["s1", "s2", "s3", "s4", "s5"], ["s1", "s2", "s3", "s4", "s4"]])
def test_check_join_lost_rows(tmp_path, ids):
    meta = beamlet_metadata(*_write(tmp_path))
    with pytest.raises(ValueError, match="join lost rows"):
        check_join(meta, _errors(tmp_path, ids))


def test_check_join_duplicate_metadata(tmp_path):
    meta = beamlet_metadata(*_write(tmp_path))
    meta = pd.concat([meta, meta.iloc[[0]]])
    with pytest.raises(ValueError, match="join lost rows"):
        check_join(meta, _errors(tmp_path, ["s1", "s2", "s3", "s4"]))


def test_site_summary_tiny_frame(tmp_path):
    summary = site_summary(beamlet_metadata(*_write(tmp_path)))
    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert summary["site"].tolist() == ["thoracic", "pelvic"]
    thor = summary.iloc[0]
    assert (thor["patients"], thor["fields"], thor["beamlets"]) == (2, 2, 3)
    assert (thor["beamlets_per_patient_min"], thor["beamlets_per_patient_median"],
            thor["beamlets_per_patient_max"]) == (1, 1.5, 2)
    assert (thor["energy_min_mev"], thor["energy_median_mev"], thor["energy_max_mev"]) == (100.0, 120.0, 140.0)
    assert (thor["spot_x_min_mm"], thor["spot_x_max_mm"]) == (-3.0, 9.0)
    assert (thor["spot_y_min_mm"], thor["spot_y_max_mm"]) == (-4.0, 12.0)
    assert thor["in_analysis"] == 2
    pelvic = summary.iloc[1]
    assert (pelvic["patients"], pelvic["fields"], pelvic["beamlets"], pelvic["in_analysis"]) == (1, 1, 1, 1)


def test_site_summary_order_follows_sites():
    meta = pd.DataFrame({"sample_id": ["a", "b", "c"], "patient_id": ["P-1", "P-2", "P-3"],
                         "site": ["pelvic", "abdominal", "thoracic"], "field_id": ["f1", "f2", "f3"],
                         "gantry_deg": 0.0, "couch_deg": 0.0, "energy_mev": 100.0, "spot_x_mm": 0.0,
                         "spot_y_mm": 0.0, "in_analysis": [True, False, True]})
    assert site_summary(meta)["site"].tolist() == ["thoracic", "abdominal", "pelvic"]
