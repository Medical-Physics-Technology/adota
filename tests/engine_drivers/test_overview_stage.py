"""Tables of the dataset_overview stage (synthetic, CPU)."""
from __future__ import annotations

import pandas as pd

from src.engine_drivers.overview_stage import gantry_sectors, overview_summary, spot_cells


def _meta() -> pd.DataFrame:
    return pd.DataFrame({"sample_id": list("abcde"), "site": ["thoracic", "thoracic", "abdominal", "pelvic", "pelvic"],
                         "gantry_deg": [0.0, 14.9, 15.0, 359.9, 180.0],
                         "spot_x_mm": [1.0, 1.0, -2.0, 3.0, 3.0], "spot_y_mm": [5.0, 5.0, 0.0, 1.0, 2.0]})


def test_gantry_sectors_count_beamlets_with_left_closed_bins_and_keep_empty_sectors():
    out = gantry_sectors(_meta(), 15.0)
    assert len(out) == 3 * 24 and out["beamlets"].sum() == 5
    th = out[out["site"] == "thoracic"].set_index("gantry_bin_low_deg")["beamlets"]
    assert th[0.0] == 2 and th[15.0] == 0
    assert out[(out["site"] == "abdominal") & (out["gantry_bin_low_deg"] == 15.0)]["beamlets"].item() == 1
    assert out[(out["site"] == "pelvic") & (out["gantry_bin_low_deg"] == 345.0)]["beamlets"].item() == 1


def test_spot_cells_group_per_site_and_position():
    out = spot_cells(_meta())
    assert out["beamlets"].sum() == 5
    assert out[(out["site"] == "thoracic")]["beamlets"].tolist() == [2]
    assert list(out["site"].astype(str).unique()) == ["thoracic", "abdominal", "pelvic"]


def test_summary_has_one_table_per_site_and_states_the_missing_fields():
    summary = pd.DataFrame([{"site": s, "patients": 2, "fields": 4, "beamlets": 10, "in_analysis": 9,
                             "beamlets_per_patient_min": 4, "beamlets_per_patient_median": 5.0,
                             "beamlets_per_patient_max": 6, "energy_min_mev": 70.0, "energy_median_mev": 100.0,
                             "energy_max_mev": 150.0, "spot_x_min_mm": -1.0, "spot_x_max_mm": 1.0,
                             "spot_y_min_mm": -2.0, "spot_y_max_mm": 2.0} for s in ("thoracic", "pelvic")])
    classes = pd.DataFrame([{"site": s, "hu_class": c, "n_voxels": 1, "share": 1 / 3}
                            for s in ("thoracic", "pelvic") for c in ("below_-500", "-500_to_300", "above_300")])
    clipping = pd.DataFrame({"site": ["thoracic", "pelvic"], "n_crops": 1, "n_voxels": 3,
                             "n_clipped_low": 0, "n_clipped_high": 0})
    text = "\n".join(overview_summary(summary, classes, clipping, {"x": 18, "y": 18}))
    assert "## thoracic" in text and "## pelvic" in text
    assert "| patients / fields / beamlets | 2 / 4 / 10 |" in text
    assert "33.3%" in text and "Couch" in text and "Spot weight" in text and "18 x 18" in text
    assert "\u2014" not in text


def test_summary_reports_a_high_hu_pile_up_from_the_histogram():
    summary = pd.DataFrame([{"site": "pelvic", "patients": 1, "fields": 1, "beamlets": 1, "in_analysis": 1,
                             "beamlets_per_patient_min": 1, "beamlets_per_patient_median": 1.0,
                             "beamlets_per_patient_max": 1, "energy_min_mev": 70.0, "energy_median_mev": 70.0,
                             "energy_max_mev": 70.0, "spot_x_min_mm": 0.0, "spot_x_max_mm": 0.0,
                             "spot_y_min_mm": 0.0, "spot_y_max_mm": 0.0}])
    classes = pd.DataFrame([{"site": "pelvic", "hu_class": c, "n_voxels": 1, "share": 0.5}
                            for c in ("below_-500", "-500_to_300", "above_300")])
    clipping = pd.DataFrame({"site": ["pelvic"], "n_crops": 1, "n_voxels": 100, "n_clipped_low": 0,
                             "n_clipped_high": 0})
    hist = pd.DataFrame({"site": "pelvic", "bin_low_hu": [0.0, 2976.0, 3066.0], "count": [90, 8, 2]})
    text = "\n".join(overview_summary(summary, classes, clipping, {"x": 1, "y": 1}, hist))
    assert "pelvic 10 voxels (10.000%), most in the bin from 2976 HU (8)" in text
