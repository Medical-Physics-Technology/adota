"""Directory selection, never-overwrite saving and ranking for the saved-beamlet comparison."""
from __future__ import annotations

import json

import numpy as np

from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.saved_beamlets import INPUT_SUFFIXES, rank_beamlets, save_new_npy, survey_dirs


def _block(root, name, cells, anatomy="thoracic"):
    d = root / name
    d.mkdir()
    for ix, iy in cells:
        stem = f"a{ix:02d}_{iy:02d}"
        for suffix in INPUT_SUFFIXES:
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


def test_blocks_without_saved_adota_predictions_are_excluded(tmp_path):
    d = _block(tmp_path, "nopred", [(i, j) for i in range(2) for j in range(2)])
    (d / "a01_01_ds_pred.npy").unlink()
    selected, excluded = survey_dirs([d], grid_n=2, min_beamlets=1)
    assert not selected and "missing" in excluded[0][1]


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
