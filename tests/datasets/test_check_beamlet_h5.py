"""Tests for :mod:`scripts.check_beamlet_h5` (spec ``docs/dev/h5_v3_spec.md`` 6.3).

Builds tiny synthetic "old" (v2-shaped) and "new" (v3-shaped) HDF5 files directly
with h5py -- no host data, no `/scratch`, no `/RadiotherapyData`, no real
exclusion list. Arrays are ``(40, 40, 160)`` float32, well under the real
``(40, 40, D)``, so the tests run fast while exercising the same storage
parameters (``compression="gzip"``, ``chunks=True``) as the real files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Set

import h5py
import numpy as np
from typer.testing import CliRunner

from scripts.check_beamlet_h5 import (
    V3_ATTRS,
    app,
    check_generator_equivalence,
    compare_ids,
    run_checks,
    spot_key_histogram,
    ulp_distance,
)

SHAPE = (40, 40, 160)


def _v2_attrs(sample_id: str, i: int) -> dict:
    return {
        "dose_deposition_ratio": 0.99,
        "gantry_angle": float(340.0 + i),
        "id": sample_id,
        "initial_energy": float(0.3 + 0.01 * i),
        "beamlet_angles": np.array([-0.7, -0.7], dtype=np.float64),
        "stat_uncertainty": float("nan") if i % 2 else 0.01,
    }


def _dummy_v3_attrs(sample_id: str, i: int) -> dict:
    """One non-empty value per name in :data:`V3_ATTRS`, in the real dtypes."""
    return {
        "schema_version": np.int64(3),
        "source_dataset": "trainset_pelvis" if i % 2 == 0 else "initial_test_one_ct",
        "energy_mev": np.float64(100.0 + i),
        "gantry_angle_sim": np.float64(90.0),
        "isocenter_mm": np.array([200.0, 200.0, 123.0], dtype=np.float64),
        "image_origin_mm": np.array([-199.6, -372.6, -340.5], dtype=np.float64),
        "image_size_vox": np.array([400, 400, 248], dtype=np.int64),
        "image_spacing_mm": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "roi_size_vox": np.array([80, 80, 400], dtype=np.int64),
        "bixel_shift_xy_mm": np.array([-24.6, -31.6], dtype=np.float64),
        "ray_entrance_mm": np.array([0.0, 37.68, 42.71], dtype=np.float64),
        "ray_entrance_proj_mm": np.array([37.68, 42.71, 0.0], dtype=np.float64),
        "num_primaries": np.float64(1e7),
        "flux_model": "SingleGaussian",
        "flux_compute": "flux_projection_gpu_batched/float64/cpu",
        "bdl_file": "hptc_beam_model_rsnone.txt",
        "bdl_sha256": "0" * 64,
        "flux_sigma_xy_mm": np.array([3.0, 3.0], dtype=np.float64),
        "downsample_method": "average",
        "patient_key": f"patient-{i // 4:04d}",
        "spot_key": f"spot-{i:04d}",
        "metadata_json": json.dumps({"id": sample_id}),
    }


assert set(_dummy_v3_attrs("x", 0).keys()) == set(V3_ATTRS), "dummy attrs must cover every V3_ATTRS name"


def _build_pair(
    tmp_path: Path,
    *,
    n: int = 8,
    n_excluded: int = 2,
    keep_excluded_id: Optional[str] = None,
    omit_new_id: Optional[str] = None,
    perturb_ct_id: Optional[str] = None,
    perturb_flux_id: Optional[str] = None,
    perturb_dose_id: Optional[str] = None,
    drop_v3_attr: Optional[tuple] = None,
) -> Dict[str, object]:
    """A tiny "old" file, a derived "new" file, and their exclusion list.

    ``new`` normally holds exactly ``old - excluded`` with identical datasets and
    v2 attrs plus dummy v3 attrs; the ``keep_excluded_id`` / ``omit_new_id`` /
    ``perturb_*`` / ``drop_v3_attr`` knobs each introduce one specific fault, so
    a test can assert the checker names exactly that id.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = [f"id-{i:03d}" for i in range(n)]
    excluded_ids: Set[str] = set(ids[:n_excluded])
    exclusion_path = tmp_path / "exclusion.txt"
    exclusion_path.write_text("\n".join(sorted(excluded_ids)) + "\n")

    old_path = tmp_path / "old.h5"
    new_path = tmp_path / "new.h5"
    rng = np.random.default_rng(0)
    records: Dict[str, dict] = {}

    with h5py.File(old_path, "w") as old_h5:
        for i, sample_id in enumerate(ids):
            ct = rng.random(SHAPE, dtype=np.float32)
            dose = rng.random(SHAPE, dtype=np.float32)
            flux = rng.random(SHAPE, dtype=np.float32)
            v2 = _v2_attrs(sample_id, i)
            records[sample_id] = {"ct": ct, "dose": dose, "flux": flux, "v2": v2}
            group = old_h5.create_group(sample_id)
            for name, arr in (("ct", ct), ("dose", dose), ("flux", flux)):
                group.create_dataset(name, data=arr, compression="gzip", chunks=True)
            for key, value in v2.items():
                group.attrs[key] = value

    with h5py.File(new_path, "w") as new_h5:
        for i, sample_id in enumerate(ids):
            is_excluded = sample_id in excluded_ids
            if is_excluded and sample_id != keep_excluded_id:
                continue
            if sample_id == omit_new_id:
                continue
            rec = records[sample_id]
            ct, dose, flux = rec["ct"].copy(), rec["dose"].copy(), rec["flux"].copy()
            if sample_id == perturb_ct_id:
                ct[0, 0, 0] = ct[0, 0, 0] + 1.0
            if sample_id == perturb_dose_id:
                dose[0, 0, 0] = dose[0, 0, 0] + 1.0
            if sample_id == perturb_flux_id:
                flux[0, 0, 0] = np.nextafter(flux[0, 0, 0], np.float32(np.inf))
            group = new_h5.create_group(sample_id)
            for name, arr in (("ct", ct), ("dose", dose), ("flux", flux)):
                group.create_dataset(name, data=arr, compression="gzip", chunks=True)
            for key, value in rec["v2"].items():
                group.attrs[key] = value
            v3 = _dummy_v3_attrs(sample_id, i)
            if drop_v3_attr is not None and drop_v3_attr[0] == sample_id:
                v3.pop(drop_v3_attr[1])
            for key, value in v3.items():
                group.attrs[key] = value

    return {
        "old_path": old_path, "new_path": new_path, "exclusion_path": exclusion_path,
        "ids": ids, "excluded_ids": excluded_ids, "records": records,
    }


# --- Pure-function unit tests --------------------------------------------------


def test_ulp_distance_one_ulp() -> None:
    a = np.array([0.5, 1.0], dtype=np.float32)
    b = a.copy()
    b[0] = np.nextafter(a[0], np.float32(np.inf))
    assert ulp_distance(a, b) == 1
    assert ulp_distance(a, a) == 0


def test_compare_ids_pure() -> None:
    old_ids = {"a", "b", "c", "d"}
    excluded = {"b", "z"}  # "z" is listed but never existed in old
    report = compare_ids(old_ids, new_ids={"a", "c", "d"}, excluded=excluded)
    assert report["pass"] is True
    assert report["listed_not_in_old"] == ["z"]
    assert report["listed_but_present"] == []
    assert report["missing_from_new"] == []
    assert report["unexpected_in_new"] == []

    bad_report = compare_ids(old_ids, new_ids={"a", "b", "c", "e"}, excluded=excluded)
    assert bad_report["pass"] is False
    assert bad_report["listed_but_present"] == ["b"]
    assert bad_report["missing_from_new"] == ["d"]
    assert bad_report["unexpected_in_new"] == ["e"]

    allowed = compare_ids(old_ids, new_ids={"a", "c", "d", "e"}, excluded=excluded, allow_new=["e"])
    assert allowed["pass"] is True
    assert allowed["unexpected_in_new"] == []
    assert allowed["allowed_new"] == ["e"]


# --- Case 1: a clean rebuild ----------------------------------------------------


def test_clean_rebuild_passes(tmp_path: Path) -> None:
    built = _build_pair(tmp_path)
    report = run_checks(built["old_path"], built["new_path"], built["exclusion_path"], sample=0)
    assert report["verdict"] == "pass"
    assert report["flux_equivalence"] == "array_equal"
    assert report["a"]["pass"] is True
    assert report["b"]["pass"] is True
    assert report["c"]["pass"] is True
    assert report["d"]["pass"] is True
    assert report["e"]["pass"] is True
    assert report["f"]["pass"] is True


# --- Case 2: an excluded id still present in new --------------------------------


def test_excluded_id_still_present_fails(tmp_path: Path) -> None:
    built = _build_pair(tmp_path, keep_excluded_id="id-000")
    report = run_checks(built["old_path"], built["new_path"], built["exclusion_path"], sample=0)
    assert report["verdict"] == "fail"
    assert report["a"]["pass"] is False
    assert "id-000" in report["a"]["listed_but_present"]
    assert report["b"]["pass"] is False
    assert "id-000" in report["b"]["excluded_present_in_new"]


# --- Case 3: an unlisted retained id missing from new ---------------------------


def test_unlisted_id_missing_from_new_fails(tmp_path: Path) -> None:
    built = _build_pair(tmp_path, omit_new_id="id-005")
    report = run_checks(built["old_path"], built["new_path"], built["exclusion_path"], sample=0)
    assert report["verdict"] == "fail"
    assert report["a"]["pass"] is False
    assert "id-005" in report["a"]["missing_from_new"]


# --- Case 4: a perturbed ct value ------------------------------------------------


def test_perturbed_ct_fails_naming_id_and_dataset(tmp_path: Path) -> None:
    built = _build_pair(tmp_path, perturb_ct_id="id-003")
    report = run_checks(built["old_path"], built["new_path"], built["exclusion_path"], sample=0)
    assert report["verdict"] == "fail"
    assert report["c"]["pass"] is False
    assert any(f["id"] == "id-003" and f["dataset"] == "ct" for f in report["c"]["failures"])


# --- Case 5: a 1-ulp flux perturbation is a pass, downgraded to allclose --------


def test_one_ulp_flux_perturbation_downgrades_not_fails(tmp_path: Path) -> None:
    # Perturb the last retained id, which check (f) never samples (it takes the
    # first 5 of the 6 retained ids): (f) does its own strict torch.equal on the
    # raw flux with no allclose exception (CLAUDE.md: no relaxed tolerances beyond
    # the one the spec names), so a ulp difference on an (f)-sampled id would
    # correctly fail (f) even though (c) downgrades it. This test isolates (c).
    built = _build_pair(tmp_path, perturb_flux_id="id-007")
    report = run_checks(built["old_path"], built["new_path"], built["exclusion_path"], sample=0)
    assert report["verdict"] == "pass"
    assert report["flux_equivalence"] == "allclose"
    assert report["c"]["pass"] is True
    with h5py.File(built["old_path"], "r") as old_h5, h5py.File(built["new_path"], "r") as new_h5:
        from scripts.check_beamlet_h5 import compare_record

        flux_report = compare_record(old_h5["id-007"], new_h5["id-007"])["datasets"]["flux"]
    assert flux_report["verdict"] == "allclose"
    assert flux_report["max_ulp"] == 1


# --- Case 6: a missing v3 attr ---------------------------------------------------


def test_missing_v3_attr_fails_naming_id_and_attr(tmp_path: Path) -> None:
    built = _build_pair(tmp_path, drop_v3_attr=("id-002", "spot_key"))
    report = run_checks(built["old_path"], built["new_path"], built["exclusion_path"], sample=0)
    assert report["verdict"] == "fail"
    assert report["d"]["pass"] is False
    assert {"id": "id-002", "attr": "spot_key"} in report["d"]["missing"]


# --- Case 7: generator equivalence (f) ------------------------------------------


def test_generator_equivalence_pass_and_fail(tmp_path: Path) -> None:
    clean = _build_pair(tmp_path / "clean")
    depths = {sample_id: SHAPE[2] for sample_id in clean["ids"] if sample_id not in clean["excluded_ids"]}
    ok_ids = list(depths)[:2]
    ok_report = check_generator_equivalence(
        clean["old_path"], clean["new_path"], clean["exclusion_path"], ok_ids, depths)
    assert ok_report["pass"] is True
    assert ok_report["mismatched_ids"] == []

    dirty = _build_pair(tmp_path / "dirty", perturb_dose_id="id-005")
    dirty_depths = {sample_id: SHAPE[2] for sample_id in dirty["ids"] if sample_id not in dirty["excluded_ids"]}
    bad_report = check_generator_equivalence(
        dirty["old_path"], dirty["new_path"], dirty["exclusion_path"], ["id-005"], dirty_depths)
    assert bad_report["pass"] is False
    assert bad_report["mismatched_ids"] == ["id-005"]


def test_spot_key_histogram_from_attrs_and_index_csv(tmp_path: Path) -> None:
    built = _build_pair(tmp_path)
    from_attrs = spot_key_histogram(built["new_path"])
    assert from_attrs["pass"] is True
    assert from_attrs["source"] == "attrs"
    assert from_attrs["n_spot_keys"] == sum(from_attrs["histogram"].values())

    index_csv = built["new_path"].with_name(f"{built['new_path'].stem}_index.csv")
    with open(index_csv, "w", newline="") as fh:
        fh.write("spot_key\n")
        fh.write("s1\ns1\ns1\ns1\ns2\ns2\n")
    from_csv = spot_key_histogram(built["new_path"])
    assert from_csv["source"] == "index_csv"
    assert from_csv["histogram"] == {"2": 1, "4": 1}
    assert from_csv["mode"] in (2, 4)


# --- Case 8: the CLI ---------------------------------------------------------------


def test_cli_exit_codes(tmp_path: Path) -> None:
    runner = CliRunner()

    clean = _build_pair(tmp_path / "clean")
    out_ok = tmp_path / "clean" / "report.json"
    result_ok = runner.invoke(app, [
        "--old", str(clean["old_path"]), "--new", str(clean["new_path"]),
        "--exclusion-list", str(clean["exclusion_path"]), "--sample", "0", "--out", str(out_ok),
    ])
    assert result_ok.exit_code == 0, result_ok.output
    assert out_ok.exists()
    assert json.loads(out_ok.read_text())["verdict"] == "pass"

    broken = _build_pair(tmp_path / "broken", omit_new_id="id-005")
    out_bad = tmp_path / "broken" / "report.json"
    result_bad = runner.invoke(app, [
        "--old", str(broken["old_path"]), "--new", str(broken["new_path"]),
        "--exclusion-list", str(broken["exclusion_path"]), "--sample", "0", "--out", str(out_bad),
    ])
    assert result_bad.exit_code == 1
    assert json.loads(out_bad.read_text())["verdict"] == "fail"
