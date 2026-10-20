"""Directory selection, read-only mirroring and ranking for the saved-beamlet comparison."""
from __future__ import annotations

import json

import pytest

from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.saved_beamlets import INPUT_SUFFIXES, mirror_inputs, rank_beamlets, survey_dirs


def _block(root, name, cells, anatomy="thoracic"):
    d = root / name
    d.mkdir()
    for ix, iy in cells:
        stem = f"a{ix:02d}_{iy:02d}"
        for suffix in INPUT_SUFFIXES[:-1]:
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


def test_mirror_links_inputs_and_never_writes_the_source(tmp_path):
    src_dir = _block(tmp_path, "src", [(0, 0), (1, 1)])
    before = sorted(p.name for p in src_dir.iterdir())
    (block,), _ = survey_dirs([src_dir], grid_n=2, min_beamlets=1)
    out = mirror_inputs(block, tmp_path / "out" / "src")
    assert all((out / f"a00_00{s}").is_symlink() for s in INPUT_SUFFIXES)
    (out / "a00_00_ds_pba.npy").write_bytes(b"new")            # new files land in the mirror
    assert sorted(p.name for p in src_dir.iterdir()) == before
    mirror_inputs(block, out)                                   # idempotent
    with pytest.raises(ValueError):
        mirror_inputs(block, src_dir / "inside")


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
