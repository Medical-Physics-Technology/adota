"""Directory selection, never-overwrite saving and ranking for the saved-beamlet comparison."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.saved_beamlets import (
    INPUT_SUFFIXES,
    PREDICTION_SUFFIX,
    check_or_write_provenance,
    rank_beamlets,
    save_new_npy,
    survey_dirs,
)


def _block(root, name, cells, anatomy="thoracic"):
    d = root / name
    d.mkdir()
    for ix, iy in cells:
        stem = f"a{ix:02d}_{iy:02d}"
        for suffix in INPUT_SUFFIXES + (PREDICTION_SUFFIX,):
            if suffix != "_sim_res.json":
                (d / f"{stem}{suffix}").write_bytes(b"x")
        (d / f"{stem}_sim_res.json").write_text(json.dumps({
            "anatomy": anatomy, "patient_id": "P1", "provenance_uid": "ds/P1/uid", "initial_energy": 102.6,
            "gantry_angle": 137.9}))
    return d


def test_survey_keeps_full_lattices_and_says_why_it_drops_the_rest(tmp_path):
    full = _block(tmp_path, "full", [(i, j) for i in range(18) for j in range(18)][:300])
    small = _block(tmp_path, "smoke", [(0, 0), (0, 17), (17, 0), (17, 17), (9, 9)])
    other = _block(tmp_path, "grid3", [(i, j) for i in range(3) for j in range(3)])
    selected, excluded = survey_dirs([full, small, other], grid_n=18, min_beamlets=200)
    assert [b.name for b in selected] == ["full"]
    assert selected[0].energy == 102.6 and len(selected[0].stems) == 300
    reasons = {p.name: why for p, why in excluded}
    assert "5 beamlets" in reasons["smoke"] and "3 x 3" in reasons["grid3"]


def test_missing_predictions_are_counted_but_missing_inputs_exclude(tmp_path):
    d = _block(tmp_path, "nopred", [(i, j) for i in range(2) for j in range(2)])
    (d / "a01_01_ds_pred.npy").unlink()
    (selected,), _ = survey_dirs([d], grid_n=2, min_beamlets=1)
    assert selected.n_missing_predictions == 1
    (d / "a01_01_ds.npy").unlink()                             # a generator file missing: excluded
    selected, excluded = survey_dirs([d], grid_n=2, min_beamlets=1)
    assert not selected and "missing" in excluded[0][1]


def test_provenance_refuses_to_mix_two_models_in_one_directory(tmp_path):
    path = tmp_path / "pba_provenance.json"
    first = check_or_write_provenance(path, {"model_fingerprint": "aaa", "note": 1}, ("model_fingerprint",))
    again = check_or_write_provenance(path, {"model_fingerprint": "aaa", "note": 2}, ("model_fingerprint",))
    assert first == again == json.loads(path.read_text())    # the original record is kept
    with pytest.raises(RuntimeError, match="different setup"):
        check_or_write_provenance(path, {"model_fingerprint": "bbb"}, ("model_fingerprint",))


def test_save_new_npy_never_touches_an_existing_file(tmp_path):
    target = tmp_path / "a00_00_ds_pba.npy"
    assert save_new_npy(target, np.arange(3.0))
    before = target.stat()
    assert not save_new_npy(target, np.zeros(3))              # already there: left alone
    after = target.stat()
    assert (before.st_mtime_ns, before.st_ino, before.st_size) == (after.st_mtime_ns, after.st_ino, after.st_size)
    assert np.array_equal(np.load(target), np.arange(3.0))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a00_00_ds_pba.npy"]   # no temp files left


def test_ranking_breaks_ties_on_the_stricter_criteria():
    crit = criteria_from_config(None)
    rows = []
    for i, (g3, g2, g1, mad) in enumerate([(100, 99, 95, 1.0), (100, 99, 97, 1.0), (100, 100, 90, 2.0),
                                           (80, 70, 60, 5.0), (90, 80, 70, 3.0), (float("nan"), 0, 0, 0)]):
        rows.append({"stem": f"s{i}", "pba_g3_3_10": g3, "pba_g2_2_10": g2, "pba_g1_1_10": g1,
                     "pba_mean_abs_diff_pct": mad})
    best, worst = rank_beamlets(rows, "PBA", crit, n=2)
    assert [r["stem"] for r in best] == ["s2", "s1"]
    assert [r["stem"] for r in worst] == ["s3", "s4"]


def test_legacy_config_reads_as_the_two_default_engines():
    from src.pencil_beam.saved_beamlets import engines_from_config

    pba, adota = engines_from_config({"pencil_beam": {"model_dir": "m"}, "adota": {"model_dir": "a"}})
    assert (pba.name, pba.kind, pba.suffix, pba.tag, pba.provenance_file) == (
        "PBA", "pencil_beam", "_ds_pba.npy", "pba", "pba_provenance.json")
    assert (adota.name, adota.suffix, adota.tag, adota.provenance_file) == (
        "ADoTA", "_ds_pred.npy", "adota", "adota_provenance.json")
    assert adota.settings["model_dir"] == "a"


def test_engine_list_gives_each_model_its_own_file_and_provenance():
    from src.pencil_beam.saved_beamlets import engines_from_config

    engines = engines_from_config({"engines": [
        {"name": "PBA", "kind": "pencil_beam"},
        {"name": "v11", "kind": "adota", "model_dir": "models/v11"},
        {"name": "final attndrop0", "kind": "adota", "suffix": "_ds_pred_final_attndrop0.npy"}]})
    assert [e.suffix for e in engines] == ["_ds_pba.npy", "_ds_pred.npy", "_ds_pred_final_attndrop0.npy"]
    assert engines[2].tag == "final_attndrop0"
    assert engines[2].provenance_file == "adota_provenance_ds_pred_final_attndrop0.json"
    assert engines[1].settings == {"model_dir": "models/v11"}


@pytest.mark.parametrize("engines, match", [
    ([{"name": "a", "kind": "adota"}, {"name": "b", "kind": "adota"}], "share a suffix"),
    ([{"name": "a b", "kind": "adota"}, {"name": "a_b", "kind": "adota", "suffix": "_ds_x.npy"}], "share a tag"),
    ([{"name": "a", "kind": "adota", "suffix": "_ds.npy"}], "must look like"),
    ([{"name": "a", "kind": "adota", "suffix": "_pred.npy"}], "must look like"),
    ([{"name": "MC", "kind": "adota"}], "reserved"),
    ([{"name": "a", "kind": "fno"}], "kind"),
])
def test_engine_list_refuses_ambiguous_or_clobbering_engines(engines, match):
    from src.pencil_beam.saved_beamlets import engines_from_config

    with pytest.raises(ValueError, match=match):
        engines_from_config({"engines": engines})


def test_scores_are_reused_per_engine_and_only_new_engines_are_scored():
    from src.pencil_beam.saved_beamlets import engines_from_config, engines_to_score

    engines = engines_from_config({"engines": [{"name": "PBA", "kind": "pencil_beam"},
                                               {"name": "v11", "kind": "adota"},
                                               {"name": "new", "kind": "adota", "suffix": "_ds_pred_new.npy"}]})
    keys = ["g3_3_10", "r80_delta_mm"]
    cached = [{"stem": s, "pba_g3_3_10": 99.0, "pba_r80_delta_mm": 0.1, "v11_g3_3_10": 98.0,
               "v11_r80_delta_mm": 0.2} for s in ("a00_00", "a00_01")]
    assert [e.name for e in engines_to_score(cached, engines, keys)] == ["new"]
    del cached[1]["v11_r80_delta_mm"]                          # one beamlet incomplete: v11 again
    assert [e.name for e in engines_to_score(cached, engines, keys)] == ["v11", "new"]
    assert engines_to_score([], engines, keys) == engines


def _dose_block(tmp_path):
    """A one-beamlet block: MC, a PBA within 0.1% of it, and two ADoTA files in ADoTA's (D, H, W) layout.
    (Identical doses would give gamma 0 everywhere, which gamma_pass_rate divides by.)"""
    z, y, x = np.meshgrid(np.arange(12), np.arange(12), np.arange(40), indexing="ij")
    mc = np.exp(-((z - 6) ** 2 + (y - 6) ** 2) / 8.0) * np.exp(-((x - 25) ** 2) / 30.0) * 100
    np.save(tmp_path / "a00_00_ct.npy", np.zeros_like(mc))
    np.save(tmp_path / "a00_00_ds.npy", mc)
    np.save(tmp_path / "a00_00_ds_pba.npy", mc * 1.001)                                 # 0.1% high
    np.save(tmp_path / "a00_00_ds_pred.npy", np.moveaxis(mc * 0.999, -1, 0)[None])      # 0.1% low
    np.save(tmp_path / "a00_00_ds_pred_new.npy", np.moveaxis(mc * 1.08, -1, 0)[None])  # 8% high
    return mc


def test_score_beamlet_scores_every_engine_of_the_task(tmp_path):
    from src.pencil_beam.saved_beamlets import engines_from_config, load_doses, score_beamlet

    mc = _dose_block(tmp_path)
    engines = engines_from_config({"engines": [{"name": "PBA", "kind": "pencil_beam"},
                                               {"name": "v11", "kind": "adota"},
                                               {"name": "new", "kind": "adota", "suffix": "_ds_pred_new.npy"}]})
    doses = load_doses(tmp_path, "a00_00", engines)
    assert np.allclose(doses["v11"], mc * 0.999) and doses["v11"].shape == mc.shape   # ADoTA layout restored
    crit = criteria_from_config([{"dose": 3, "dist": 3, "cutoff": 10}, {"dose": 1, "dist": 1, "cutoff": 10}])
    row = score_beamlet((str(tmp_path), "a00_00", tuple(crit), "blk", tuple(engines)))
    assert row["pba_g3_3_10"] == row["v11_g3_3_10"] == 100.0
    assert row["pba_max_abs_diff_pct"] == pytest.approx(0.1, rel=1e-3)          # each engine its own file
    assert row["v11_mean_diff_pct"] < 0 < row["pba_mean_diff_pct"]
    assert row["new_max_abs_diff_pct"] == pytest.approx(8.0, rel=1e-3)
    only_new = score_beamlet((str(tmp_path), "a00_00", tuple(crit), "blk", tuple(engines[2:])))
    assert "pba_g3_3_10" not in only_new and only_new["new_mean_diff_pct"] == row["new_mean_diff_pct"]


def test_missing_predictions_go_to_the_engine_suffix_and_leave_the_others_alone(tmp_path, monkeypatch):
    import src.beamlets.inference as inference
    from src.pencil_beam.saved_beamlets import write_missing_predictions

    d = _block(tmp_path, "blk", [(0, 0), (0, 1)])
    (d / "a00_00_ds_pred.npy").write_bytes(b"v11 prediction")             # an existing model's file

    def fake_inference(directory, model, device, config):
        for f in directory.glob("*_ct.npy"):
            np.save(directory / f.name.replace("_ct.npy", "_ds_pred.npy"), np.full(3, 7.0))
    monkeypatch.setattr(inference, "run_inference", fake_inference)
    (selected,), _ = survey_dirs([d], grid_n=2, min_beamlets=1)
    n = write_missing_predictions(selected, None, None, {"checkpoint_sha256": "new"}, suffix="_ds_pred_new.npy",
                                  provenance_file="adota_provenance_ds_pred_new.json")
    assert n == 2
    assert np.array_equal(np.load(d / "a00_01_ds_pred_new.npy"), np.full(3, 7.0))
    assert (d / "a00_00_ds_pred.npy").read_bytes() == b"v11 prediction"    # untouched
    assert json.loads((d / "adota_provenance_ds_pred_new.json").read_text()) == {"checkpoint_sha256": "new"}
    assert not (d / "adota_provenance.json").exists()
    assert write_missing_predictions(selected, None, None, {"checkpoint_sha256": "new"},
                                     suffix="_ds_pred_new.npy") == 0      # rerun: nothing to add
