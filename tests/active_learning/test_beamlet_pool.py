"""The beamlet pool on synthetic data: the multi-energy plan, patient splits and
training matches, quotas and batches, the overshoot checks, packing with the flux
recomputed, and the checksummed move. No DICOM, no Monte Carlo."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from src.active_learning.beamlet_pool.batches import BatchSettings, my_batches, space_check
from src.active_learning.beamlet_pool.planning import PlanSettings, assign_batches, patient_quotas
from src.active_learning.beamlet_pool.registry import Claim, assign_splits, match_training
from src.active_learning.beamlet_pool.store import (
    StoreChecks,
    move_file,
    pack_batch,
    post_mc_reason,
    read_record,
    recompute_flux,
)
from src.mc_generation.config_writer import build_beamlet_field_plan_text, build_multi_energy_plan_text


def test_multi_energy_plan_groups_spots_into_one_control_point_per_energy():
    energies = [100.0, 150.5, 100.0, 210.0, 150.5]
    spots = [(float(i), -float(i)) for i in range(5)]
    text, where = build_multi_energy_plan_text(energies, spots, 90.0, (0.0, 0.0, 0.0))
    assert where == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]
    assert "###NumberOfControlPoints\n3\n" in text
    assert text.count("####Energy (MeV)") == 3 and "150.5" in text
    assert [int(line) for line in text.split("####NbOfScannedSpots\n")[1:] for line in [line.split("\n")[0]]] \
        == [2, 2, 1]


def test_single_energy_plan_is_unchanged_by_the_multi_energy_builder():
    single = build_beamlet_field_plan_text(120.0, [(1.0, 2.0), (3.0, 4.0)], 90.0, (0.0, 0.0, 0.0))
    multi, _ = build_multi_energy_plan_text([120.0, 120.0], [(1.0, 2.0), (3.0, 4.0)], 90.0, (0.0, 0.0, 0.0))
    # Same spots, energy and weights; only the cumulative meterset bookkeeping differs.
    strip = lambda t: [line for line in t.splitlines() if "Cumulative" not in line and not line[:1].isdigit()]  # noqa: E731
    assert strip(single) == strip(multi)


def screened() -> pd.DataFrame:
    rows = []
    for c, n in (("A", 20), ("B", 10)):
        for i in range(n):
            rows.append({"collection": c, "patient_id": f"{c}{i:02d}", "qc_pass": True, "n_training_matches": 0,
                         "training_matches": "", "z_extent_mm": 300.0})
    frame = pd.DataFrame(rows)
    frame.loc[0, ["n_training_matches", "training_matches"]] = [1, "abc"]
    frame.loc[1, "qc_pass"] = False
    return frame


def test_splits_respect_training_matches_qc_claims_and_fractions():
    claims = [Claim(role="test", patient_ids=["A00", "A02"]), Claim(role="excluded", patient_ids=["B03"])]
    out = assign_splits(screened(), claims, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=1)
    by = out.set_index("patient_id")["split"]
    assert by["A00"] == "excluded"                      # the training match wins over the claim
    assert by["A01"] == "excluded" and by["A02"] == "test" and by["B03"] == "excluded"
    drawn = out[out["split_reason"].str.startswith("drawn")]
    # 17 free patients in A: 11.9 / 2.55 / 2.55 -> 12 / 3 / 2 (largest remainder, ties in config order);
    # 9 in B (B03 excluded): 6.3 / 1.35 / 1.35 -> 6 / 2 / 1.
    assert drawn.groupby(["collection", "split"]).size().to_dict() == {
        ("A", "test"): 2, ("A", "train"): 12, ("A", "validation"): 3,
        ("B", "test"): 1, ("B", "train"): 6, ("B", "validation"): 2}
    again = assign_splits(screened(), claims, {"train": 0.7, "validation": 0.15, "test": 0.15}, seed=1)
    assert again["split"].tolist() == out["split"].tolist()


def test_training_match_needs_the_same_grid_and_accepts_either_z_end():
    # Pelvic training scan of StageII-Colorectal-CT-059: z origin recorded at the top slice.
    scans = pd.DataFrame({"patient_key": ["k1", "k2"], "image_origin_mm_0": [-192.63, -180.0],
                          "image_origin_mm_1": [-318.63, -300.0], "image_origin_mm_2": [1110.5, 31.5],
                          "image_size_vox_0": [382, 402], "image_size_vox_2": [255, 465],
                          "image_spacing_mm_2": [1.0, 1.0]})
    grid = {"grid_x": 382, "z_extent_mm": 255.0, "origin_x_mm": -192.63, "origin_y_mm": -318.63,
            "origin_z_mm": 860.5, "origin_z_top_mm": 1110.5}
    assert match_training(grid, scans) == ["k1"]
    assert match_training({**grid, "origin_z_top_mm": 900.0}, scans) == []          # neither end
    assert match_training({**grid, "origin_y_mm": -15.0}, scans) == []              # other patient, same size


def test_quotas_and_batches_keep_patients_whole_and_splits_interleaved():
    registry = pd.DataFrame({"patient_id": [f"p{i}" for i in range(20)],
                             "split": ["train"] * 14 + ["validation"] * 3 + ["test"] * 3})
    settings = PlanSettings(total_beamlets=2000, oversample=1.0)
    quotas = patient_quotas(registry, settings)
    assert quotas["p0"] == 100 and quotas["p15"] == 100
    plan = pd.DataFrame([{"patient_id": p, "split": s, "candidate_id": f"{p}_{k}"}
                         for p, s in zip(registry["patient_id"], registry["split"]) for k in range(quotas[p])])
    batch = assign_batches(plan, batch_size=500, seed=3)
    assert plan.assign(batch=batch).groupby("patient_id")["batch"].nunique().max() == 1
    first = plan.assign(batch=batch)[batch == 0]
    assert set(first["split"]) == {"train", "validation", "test"}
    assert my_batches(plan.assign(batch=batch), 1, 2) == [b for b in sorted(batch.unique()) if b % 2 == 1]


def sim_res(r80=150.0, peak_in_body=True, ratio=0.99, energy=150.0, share=None):
    out = {"r80_mm": r80, "peak_in_body": peak_in_body, "dose_deposition_ratio": ratio, "roi_size": [20, 20, 64],
           "initial_energy": energy, "beamlet_angles": [0.01, -0.02], "rays_entrence_point_proj": [10.0, 9.0, 0.0],
           "mc_seconds": 1.0}
    return out if share is None else {**out, "body_fraction_at_peak": share}


def test_post_mc_checks_catch_overshoot_and_a_peak_outside_the_patient():
    checks = StoreChecks(range_margin_mm=10.0)
    assert post_mc_reason(sim_res(r80=40.0), checks) == ""
    assert post_mc_reason(sim_res(r80=float("nan")), checks) == "mc_overshoot_crop"
    assert post_mc_reason(sim_res(r80=58.0), checks) == "mc_overshoot_crop"
    assert post_mc_reason(sim_res(r80=40.0, peak_in_body=False), checks) == "mc_peak_outside_patient"
    assert post_mc_reason(sim_res(r80=40.0, ratio=0.9), checks) == "mc_low_deposition"


def test_post_mc_check_judges_the_peak_slice_not_the_hottest_voxel():
    checks = StoreChecks(range_margin_mm=10.0)
    # the hottest voxel in air (noise) no longer rejects a beam that stops in the patient
    assert post_mc_reason(sim_res(r80=40.0, peak_in_body=False, share=0.9), checks) == ""
    assert post_mc_reason(sim_res(r80=40.0, peak_in_body=True, share=0.2), checks) == "mc_peak_outside_patient"
    assert post_mc_reason(sim_res(r80=40.0, share=0.5), checks) == ""
    assert post_mc_reason(sim_res(r80=40.0, share=0.4), StoreChecks(min_body_fraction_at_peak=0.3)) == ""


def write_raw(root, rows, sigmas, flux_noise=0.0):
    for row in rows.itertuples():
        d = root / row.field_dir
        d.mkdir(parents=True, exist_ok=True)
        meta = sim_res(r80=float(row.r80), energy=float(row.energy_mev))
        np.save(d / f"{row.candidate_id}_ct.npy", np.random.default_rng(1).integers(-1000, 1500, (20, 20, 64))
                .astype(np.int16))
        np.save(d / f"{row.candidate_id}_ds.npy", np.random.default_rng(2).random((20, 20, 64)).astype(np.float32))
        np.save(d / f"{row.candidate_id}_flux.npy", recompute_flux(meta, sigmas) * (1 + flux_noise))
        (d / f"{row.candidate_id}_sim_res.json").write_text(json.dumps(meta))


def plan_rows() -> pd.DataFrame:
    return pd.DataFrame({"candidate_id": ["c1", "c2", "c3", "c4"], "split": ["train", "train", "test", "train"],
                         "patient_id": "p", "anatomy": "thoracic", "energy_mev": [100.0, 120.0, 140.0, 160.0],
                         "gantry_deg": 30.0, "theta_x_deg": 0.0, "theta_y_deg": 0.0, "field_dir": "f",
                         "r80": [40.0, 70.0, 40.0, 40.0]})


def test_pack_stores_rejects_and_round_trips_with_the_flux_recomputed(tmp_path):
    rows, sigmas = plan_rows(), (3.0, 2.5)
    write_raw(tmp_path / "raw", rows[rows["candidate_id"] != "c4"], sigmas)     # c4: skipped by the generator
    index, shards = pack_batch(tmp_path / "raw", rows, tmp_path / "shards", 7, lambda e: sigmas, StoreChecks())
    assert index.set_index("candidate_id")["status"].to_dict() == {
        "c1": "stored", "c2": "rejected:mc_overshoot_crop", "c3": "stored", "c4": "missing"}
    assert set(shards) == {"train", "test"}
    with h5py.File(shards["train"]) as h5:
        record = read_record(h5["c1"])
        assert np.array_equal(record["flux"], np.load(tmp_path / "raw" / "f" / "c1_flux.npy"))
        assert np.array_equal(record["ct"], np.load(tmp_path / "raw" / "f" / "c1_ct.npy"))
        assert record["dose"].dtype == np.float32 and list(h5.keys()) == ["c1"]


def test_pack_refuses_a_flux_that_does_not_recompute(tmp_path):
    rows, sigmas = plan_rows().iloc[:1], (3.0, 2.5)
    write_raw(tmp_path / "raw", rows, sigmas, flux_noise=1e-3)
    with pytest.raises(RuntimeError, match="recomputed flux"):
        pack_batch(tmp_path / "raw", rows, tmp_path / "shards", 0, lambda e: sigmas, StoreChecks())


def test_move_checks_the_copy_and_removes_the_source(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x" * 1000)
    digest = move_file(src, tmp_path / "store" / "a.bin")
    assert not src.exists() and (tmp_path / "store" / "a.bin").read_bytes() == b"x" * 1000 and len(digest) == 64
    assert not list((tmp_path / "store").glob("*.partial"))


def test_space_check_refuses_a_batch_that_does_not_fit(tmp_path):
    settings = BatchSettings(work_root=str(tmp_path), store_root=str(tmp_path), reserve_work_gb=1e9)
    assert "GB needed" in space_check(settings, 10)
    assert space_check(BatchSettings(work_root=str(tmp_path), store_root=str(tmp_path), transient_gb=0.0,
                                     reserve_work_gb=0.0, reserve_store_gb=0.0), 1) is None


FAKE_MCSQUARE = '''#!{python}
"""Stands in for MCsquare: one dose file per spot of every control point, filled
with that control point's energy, at the beamlet-mode output names."""
import pathlib, SimpleITK as sitk, numpy as np
lines = pathlib.Path("PlanPencil.txt").read_text().splitlines()
out = pathlib.Path("Outputs"); out.mkdir(exist_ok=True)
c = -1
for i, line in enumerate(lines):
    if line == "####Energy (MeV)":
        c += 1
        energy = float(lines[i + 1])
    if line == "####NbOfScannedSpots":
        for d in range(int(lines[i + 1])):
            image = sitk.GetImageFromArray(np.full((2, 2, 2), energy, dtype=np.float32))
            sitk.WriteImage(image, str(out / f"Dose_Beamlet_0_{{c}}_{{d}}.mhd"))
'''


def test_beamlet_field_with_per_spot_energies_maps_every_output_back_to_its_spot(tmp_path):
    import stat
    import sys

    import SimpleITK as sitk

    from src.mc_generation.mcsquare_runner import MCSquareRunner

    install = tmp_path / "mcsquare"
    for sub in ("Materials", "Scanners", "BDL"):
        (install / sub).mkdir(parents=True)
    (install / "BDL" / "bdl.txt").write_text("")
    binary = install / "MCsquare_linux"
    binary.write_text(FAKE_MCSQUARE.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    runner = MCSquareRunner(install_dir=str(install), work_root=str(tmp_path / "work"), bdl_file="bdl.txt")
    ct = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    energies = [150.0, 90.5, 150.0, 210.0]
    seen = {i: float(sitk.GetArrayFromImage(d).max()) for i, d, meta in
            runner.run_beamlet_field(ct, energy=energies, gantry_angle=90.0, spots_xy=[(0, 0)] * 4,
                                     isocenter=(0, 0, 0), num_primaries=1e3)}
    assert seen == dict(enumerate(energies))


def test_space_check_adds_work_and_store_needs_on_one_file_system(tmp_path, monkeypatch):
    import src.active_learning.beamlet_pool.batches as batches

    monkeypatch.setattr(batches, "free_gb", lambda path: 100.0)
    settings = BatchSettings(work_root=str(tmp_path / "w"), store_root=str(tmp_path / "s"), transient_gb=20.0,
                             reserve_work_gb=30.0, reserve_store_gb=50.0)
    # 2000 raw beamlets (32 GB) + 20 GB transient + 6.2 GB packed + 50 GB reserve > 100 GB
    assert "work and store together" in space_check(settings, 2000)
    assert space_check(settings, 100) is None


def test_sync_copies_finished_batches_checked_and_resumes(tmp_path):
    from src.active_learning.beamlet_pool.batches import mirror
    from src.datasets.beamlet_h5 import sha256_of

    store, target = tmp_path / "store", tmp_path / "mirror"
    (store / "train").mkdir(parents=True)
    (store / "plan").mkdir()
    (store / "state").mkdir()
    (store / "plan" / "plan.csv").write_text("a,b\n")
    shard = store / "train" / "batch_0000.h5"
    shard.write_bytes(b"shard" * 100)
    marker = {"batch": 0, "files": {"train/batch_0000.h5": sha256_of(shard)}}
    (store / "state" / "batch_0000.json").write_text(json.dumps(marker))
    first = mirror(store, target, reserve_gb=0.0)
    assert first["batches_copied"] == 1 and (target / "train" / "batch_0000.h5").read_bytes() == b"shard" * 100
    assert (target / "state" / "batch_0000.json").exists() and shard.exists()        # the source stays
    assert mirror(store, target, reserve_gb=0.0)["batches_present"] == 1             # repeatable
    marker["files"]["train/batch_0000.h5"] = "0" * 64                                # a corrupted source
    (store / "state" / "batch_0000.json").write_text(json.dumps(marker))
    (target / "train" / "batch_0000.h5").unlink()
    with pytest.raises(RuntimeError, match="checksum"):
        mirror(store, target, reserve_gb=0.0)
