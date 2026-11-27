"""Per-beamlet score and error table (synthetic, CPU)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.engine_drivers.score_table import beamlet_score_table, contribution_name


def _inputs():
    meta = pd.DataFrame({"sample_id": ["a", "b", "c"], "patient_id": ["p1", "p1", "p2"],
                         "site": ["thoracic", "thoracic", "pelvic"], "field_id": ["f1", "f1", "f2"],
                         "gantry_deg": [10.0, 10.0, 200.0], "energy_mev": [100.0, 120.0, 150.0],
                         "spot_x_mm": [0.0, 3.1, -3.1], "spot_y_mm": [0.0, 4.0, -4.0],
                         "in_analysis": [True, True, False]})
    contrib = pd.DataFrame({"sample_id": ["a", "b"], "depth and energy [max_grad_depth_mm]": [0.2, -0.1],
                            "WEPL spread across the beamlet [pflugfelder_hi]": [0.1, 0.05],
                            "total": [0.3, -0.05], "baseline": [0.5, 0.5], "decile": [9, 3],
                            "patient_id": ["p1", "p1"], "anatomy": ["thoracic", "thoracic"]})
    err = {e: pd.DataFrame({"sample_id": ["a", "b", "c"], "engine": e, "gpr_2pct_2mm_10pct": [99.0, 95.0, 90.0],
                            "mape_pct": [4.0, 6.0, 8.0]}) for e in ("adota", "pba")}
    return meta, contrib, err


def test_contribution_names_use_the_lead_feature_so_shared_labels_stay_distinct():
    assert contribution_name("WEPL spread across the beamlet [pflugfelder_hi]") == "contrib_pflugfelder_hi"
    assert (contribution_name("tissue changes in the peak zone [total_hu_change]")
            != contribution_name("tissue changes in the peak zone [interface_bp_distance]"))


def test_table_keeps_every_beamlet_and_adds_score_and_prefixed_errors():
    meta, contrib, err = _inputs()
    t = beamlet_score_table(meta, contrib, err).set_index("sample_id")
    assert list(t.index) == ["a", "b", "c"]
    assert t.loc["a", "difficulty_score"] == pytest.approx(0.8) and t.loc["b", "score_decile"] == 3
    assert pd.isna(t.loc["c", "difficulty_score"])
    assert t.loc["c", "pba_mape_pct"] == 8.0 and "adota_gpr_2pct_2mm_10pct" in t.columns
    sums = t.loc[["a", "b"], ["contrib_max_grad_depth_mm", "contrib_pflugfelder_hi"]].sum(axis=1)
    assert (sums - (t.loc[["a", "b"], "difficulty_score"] - t.loc[["a", "b"], "score_baseline"])).abs().max() < 1e-12


def test_errors_must_cover_every_beamlet():
    meta, contrib, err = _inputs()
    err["pba"] = err["pba"].iloc[:2]
    with pytest.raises(ValueError, match="pba errors"):
        beamlet_score_table(meta, contrib, err)


def test_a_score_outside_the_analysis_population_is_refused():
    meta, contrib, err = _inputs()
    meta.loc[1, "in_analysis"] = False
    with pytest.raises(ValueError, match="outside the analysis population"):
        beamlet_score_table(meta, contrib, err)


def test_site_disagreement_is_refused():
    meta, contrib, err = _inputs()
    contrib.loc[0, "anatomy"] = "pelvic"
    with pytest.raises(ValueError, match="anatomy disagrees"):
        beamlet_score_table(meta, contrib, err)
