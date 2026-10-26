"""Tests for :mod:`src.datasets.beamlet_h5` and :mod:`scripts.build_beamlet_h5`
(spec ``docs/dev/h5_v3_spec.md`` section 6.2).

Synthetic data only, ``device="cpu"``, small ``(80, 80, 320)`` raw records (pooled
depth 160, exactly ``MIN_DEPTH``). No host BDL, no ``/RadiotherapyData``, no
``/scratch``; ``workers=0`` runs the build in-process so tests stay fast and easy
to debug (no spawn-pool overhead).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from scripts.build_beamlet_h5 import app, build
from src.adota.config import DEFAULT_SCALE
from src.datasets.beamlet_h5 import (
    DATASET_KWARGS,
    DATASET_NAMES,
    V2_ATTRS,
    V3_ATTRS,
    derive_keys,
    index_row,
    plan_candidates,
    sha256_of,
)
from src.datasets.beamlet_record import downsample_grid, normalise_dose
from src.loaders.generator import H5PYGenerator
from tests.utils.bdl import build_bdl_text
from tests.utils.beamlet_records import make_metadata, write_synthetic_record

RAW_SHAPE = (80, 80, 320)  # pools to (40, 40, 160): 160 == MIN_DEPTH exactly

# The section-5 file attrs the build must always write.
FILE_ATTR_NAMES = (
    "schema_version", "created_utc", "generator", "python", "numpy", "torch", "h5py", "cuda", "gpu_name",
    "scale_json", "bdl_file", "bdl_sha256", "flux_compute", "flux_batch", "downsample_method",
    "exclusion_list_path", "exclusion_list_sha256", "n_excluded", "n_excluded_not_found", "sources_json",
    "n_records", "n_skipped", "skip_log_path", "index_path", "source_v2_path", "flux_equivalence",
)


def _bdl_path(tmp_path_factory) -> Path:
    bdl_dir = tmp_path_factory.mktemp("bdl")
    path = bdl_dir / "bdl.txt"
    path.write_text(build_bdl_text())
    return path


@pytest.fixture
def bdl_path(tmp_path_factory) -> Path:
    return _bdl_path(tmp_path_factory)


def _write_ids_file(path: Path, ids: List[str]) -> Path:
    path.write_text("\n".join(ids) + "\n")
    return path


def _empty_exclusion_list(path: Path) -> Path:
    path.write_text("")
    return path


def _expected_pooled_dose(raw_dose: np.ndarray) -> np.ndarray:
    return downsample_grid(normalise_dose(raw_dose, DEFAULT_SCALE), "average")


# ── 1. Round trip through H5PYGenerator ─────────────────────────────────────


def test_round_trip_via_h5py_generator(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    ids = [f"rec-{i}" for i in range(3)]
    raw_doses = {}
    for i, sample_id in enumerate(ids):
        meta = write_synthetic_record(source_dir, sample_id, shape=RAW_SHAPE, seed=i)
        raw_doses[sample_id] = np.load(source_dir / f"{sample_id}_ds.npy")
        assert meta["id"] == sample_id

    out = tmp_path / "out.h5"
    excl = _empty_exclusion_list(tmp_path / "exclude.txt")
    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0)

    assert out.exists()
    exclude_for_generator = _empty_exclusion_list(tmp_path / "generator_exclude.txt")
    ds = H5PYGenerator(str(out), indexes=list(ids), augmentation=False, cropp=False, normalize=False,
                       indexes_to_exclude_list=str(exclude_for_generator), expected_shape=(160, 40, 40))
    assert len(ds) == 3
    for i, sample_id in enumerate(ids):
        x, e, y = ds[i]
        assert x.shape == (2, 160, 40, 40)
        assert e.shape == (1,)
        assert y.shape == (1, 160, 40, 40)
        expected_dose = _expected_pooled_dose(raw_doses[sample_id]).transpose(2, 0, 1)
        assert np.array_equal(y.squeeze(0).numpy(), expected_dose.astype(np.float32))
        assert y.dtype == torch.float32


# ── 2. Datasets and attrs ───────────────────────────────────────────────────


def test_datasets_and_attrs(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    write_synthetic_record(source_dir, "rec-0", shape=RAW_SHAPE, seed=0)
    out = tmp_path / "out.h5"
    excl = _empty_exclusion_list(tmp_path / "exclude.txt")
    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0)

    with h5py.File(out, "r") as h5:
        group = h5["rec-0"]
        for name in DATASET_NAMES:
            dset = group[name]
            assert dset.dtype == np.float32
            assert dset.compression == "gzip"
            assert dset.compression_opts == DATASET_KWARGS.get("compression_opts", 4) or dset.compression_opts == 4
            assert dset.chunks is not None

        for key in V2_ATTRS:
            assert key in group.attrs
        assert isinstance(group.attrs["beamlet_angles"], np.ndarray)
        assert group.attrs["beamlet_angles"].dtype == np.float64
        assert group.attrs["beamlet_angles"].shape == (2,)
        assert isinstance(group.attrs["id"], str) and group.attrs["id"] == "rec-0"
        assert isinstance(group.attrs["dose_deposition_ratio"], (float, np.floating))

        for key in V3_ATTRS:
            assert key in group.attrs
        assert group.attrs["schema_version"] == 3
        raw_text = (source_dir / "rec-0_metadata.json").read_text()
        assert group.attrs["metadata_json"] == raw_text


# ── 3. Exclusion list bookkeeping ───────────────────────────────────────────


def test_exclusion_bookkeeping(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    ids = [f"rec-{i}" for i in range(6)]
    for i, sample_id in enumerate(ids):
        write_synthetic_record(source_dir, sample_id, shape=RAW_SHAPE, seed=i)

    excl = tmp_path / "exclude.txt"
    excl.write_text("rec-1\nrec-3\nnot-a-real-id\n")
    out = tmp_path / "out.h5"
    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0)

    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"rec-0", "rec-2", "rec-4", "rec-5"}
        assert h5.attrs["n_excluded"] == 2
        assert h5.attrs["n_excluded_not_found"] == 1
        assert h5.attrs["exclusion_list_path"] == str(excl.resolve())
        assert h5.attrs["exclusion_list_sha256"] == sha256_of(excl)

    skip_rows = list(csv.DictReader((tmp_path / "out_skipped.csv").open()))
    excluded_rows = [r for r in skip_rows if r["reason"] == "excluded"]
    assert {r["sample_id"] for r in excluded_rows} == {"rec-1", "rec-3"}
    assert all(r["sample_id"] != "not-a-real-id" for r in skip_rows)


# ── 4. Missing exclusion list is a hard error ───────────────────────────────


def test_missing_exclusion_list_is_an_error(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "pelvis").mkdir(parents=True)
    out = tmp_path / "out.h5"
    with pytest.raises(FileNotFoundError):
        build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path,
              exclusion_list=tmp_path / "does_not_exist.txt", device="cpu", workers=0)

    runner = CliRunner()
    result = runner.invoke(app, [
        "--raw-root", str(raw_root), "--source", "pelvis", "--out", str(out), "--bdl", str(bdl_path),
        "--exclusion-list", str(tmp_path / "does_not_exist.txt"), "--device", "cpu", "--workers", "0",
    ])
    assert result.exit_code != 0

    result_missing_option = runner.invoke(app, [
        "--raw-root", str(raw_root), "--source", "pelvis", "--out", str(out), "--bdl", str(bdl_path),
        "--device", "cpu", "--workers", "0",
    ])
    assert result_missing_option.exit_code != 0


# ── 5. Skip reasons ──────────────────────────────────────────────────────────


def test_skip_reasons(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"

    write_synthetic_record(source_dir, "zero-dose", shape=RAW_SHAPE, seed=0, zero_dose=True)
    write_synthetic_record(source_dir, "bad-shape", shape=(78, 80, 320), seed=1)
    write_synthetic_record(source_dir, "short-depth", shape=(80, 80, 300), seed=2)
    write_synthetic_record(source_dir, "n-spots", shape=RAW_SHAPE, seed=3, n_spots=2)

    write_synthetic_record(source_dir, "load-error", shape=RAW_SHAPE, seed=4)
    (source_dir / "load-error_ds.npy").unlink()

    write_synthetic_record(source_dir, "json-error", shape=RAW_SHAPE, seed=5)
    (source_dir / "json-error_metadata.json").write_text("{not valid json")

    excl = _empty_exclusion_list(tmp_path / "exclude.txt")
    out = tmp_path / "out.h5"
    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0)

    with h5py.File(out, "r") as h5:
        assert list(h5.keys()) == []

    skip_rows = {r["sample_id"]: r["reason"] for r in csv.DictReader((tmp_path / "out_skipped.csv").open())}
    assert skip_rows["zero-dose"] == "zero_dose"
    assert skip_rows["bad-shape"] == "bad_shape"
    assert skip_rows["short-depth"] == "short_depth"
    assert skip_rows["n-spots"] == "n_spots"
    assert skip_rows["load-error"] == "load_error"
    assert skip_rows["json-error"] == "json_error"


# ── 6. File attrs present ───────────────────────────────────────────────────


def test_file_attrs_present(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    write_synthetic_record(source_dir, "rec-0", shape=RAW_SHAPE, seed=0)
    excl = _empty_exclusion_list(tmp_path / "exclude.txt")
    out = tmp_path / "out.h5"
    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0)

    with h5py.File(out, "r") as h5:
        for name in FILE_ATTR_NAMES:
            assert name in h5.attrs, f"missing file attr {name!r}"
        sources_json = json.loads(h5.attrs["sources_json"])
        assert "note" in sources_json and "pelvis" in sources_json


# ── 7. derive_keys ───────────────────────────────────────────────────────────


def test_derive_keys_patient_and_spot() -> None:
    meta_a = make_metadata("a", gantry_angle=340.0, energy_mev=100.0, beamlet_angles=(-0.7, -0.7))
    meta_b = make_metadata("b", gantry_angle=340.0, energy_mev=200.0, beamlet_angles=(0.3, 0.1))
    patient_a, spot_a = derive_keys(meta_a)
    patient_b, spot_b = derive_keys(meta_b)
    # Same patient geometry (image_origin/size/spacing untouched) -> same patient_key,
    # same isocenter/gantry_angle/bixel shift -> same spot_key despite differing energy/angles.
    assert patient_a == patient_b
    assert spot_a == spot_b

    meta_c = make_metadata("c", gantry_angle=10.0)
    _, spot_c = derive_keys(meta_c)
    assert spot_c != spot_a


# ── 8. Index CSV ─────────────────────────────────────────────────────────────


def test_index_csv_matches_attrs(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    ids = [f"rec-{i}" for i in range(2)]
    for i, sample_id in enumerate(ids):
        write_synthetic_record(source_dir, sample_id, shape=RAW_SHAPE, seed=i)
    excl = _empty_exclusion_list(tmp_path / "exclude.txt")
    out = tmp_path / "out.h5"
    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0)

    with h5py.File(out, "r") as h5:
        expected_rows = {gid: index_row(gid, dict(h5[gid].attrs)) for gid in h5.keys()}

    csv_rows = {r["sample_id"]: r for r in csv.DictReader((tmp_path / "out_index.csv").open())}
    assert set(csv_rows) == set(expected_rows)
    for sample_id, expected in expected_rows.items():
        actual = csv_rows[sample_id]
        for key, value in expected.items():
            assert key in actual
            assert actual[key] == str(value)


# ── 9. flux-batch independence ──────────────────────────────────────────────


def test_flux_batch_size_does_not_change_output(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    ids = [f"rec-{i}" for i in range(3)]
    for i, sample_id in enumerate(ids):
        write_synthetic_record(source_dir, sample_id, shape=RAW_SHAPE, seed=i)
    excl = _empty_exclusion_list(tmp_path / "exclude.txt")

    out1 = tmp_path / "out_b1.h5"
    out8 = tmp_path / "out_b8.h5"
    build(raw_root=raw_root, sources=["pelvis"], out=out1, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0, flux_batch_size=1)
    build(raw_root=raw_root, sources=["pelvis"], out=out8, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0, flux_batch_size=8)

    ignore = {"created_utc", "flux_batch", "index_path", "skip_log_path"}
    with h5py.File(out1, "r") as h1, h5py.File(out8, "r") as h8:
        assert set(h1.keys()) == set(h8.keys()) == set(ids)
        for key in h1.attrs:
            if key in ignore:
                continue
            assert key in h8.attrs
            v1, v8 = h1.attrs[key], h8.attrs[key]
            if isinstance(v1, np.ndarray):
                assert np.array_equal(v1, v8)
            else:
                assert v1 == v8
        for gid in ids:
            g1, g8 = h1[gid], h8[gid]
            for name in DATASET_NAMES:
                assert np.array_equal(g1[name][:], g8[name][:])
            for key in V2_ATTRS + V3_ATTRS:
                a1, a8 = g1.attrs[key], g8.attrs[key]
                if isinstance(a1, np.ndarray):
                    assert np.array_equal(a1, a8)
                else:
                    assert a1 == a8 or (isinstance(a1, float) and np.isnan(a1) and np.isnan(a8))


# ── 10. Resume ───────────────────────────────────────────────────────────────


def test_resume(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    ids = [f"rec-{i}" for i in range(4)]
    for i, sample_id in enumerate(ids):
        write_synthetic_record(source_dir, sample_id, shape=RAW_SHAPE, seed=i)
    excl = _empty_exclusion_list(tmp_path / "exclude.txt")
    out = tmp_path / "out.h5"

    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0, limit=2)
    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"rec-0", "rec-1"}
        first_ct = {gid: h5[gid]["ct"][:].copy() for gid in h5.keys()}

    # Simulate an interrupted build: the first run finished and was renamed to
    # `out`; put it back as `<out>.partial` so --resume has something to reopen,
    # and stamp one extra group as artificially incomplete (only "ct" present).
    partial = out.with_suffix(out.suffix + ".partial")
    out.rename(partial)
    with h5py.File(partial, "a") as h5:
        incomplete = h5.create_group("rec-incomplete")
        incomplete.create_dataset("ct", data=np.zeros((1, 1, 1), dtype=np.float32))

    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0, resume=True)

    with h5py.File(out, "r") as h5:
        # "rec-incomplete" is not a real candidate: it is deleted as incomplete
        # and never rebuilt, since plan_candidates never produces it.
        assert set(h5.keys()) == set(ids)
        for gid in ["rec-0", "rec-1"]:
            assert np.array_equal(h5[gid]["ct"][:], first_ct[gid])


# ── 11. ids-file restriction ─────────────────────────────────────────────────


def test_ids_file_restricts_candidates(tmp_path, bdl_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    ids = [f"rec-{i}" for i in range(4)]
    for i, sample_id in enumerate(ids):
        write_synthetic_record(source_dir, sample_id, shape=RAW_SHAPE, seed=i)

    ids_file = _write_ids_file(tmp_path / "ids.txt", ["rec-0", "rec-1"])
    excl = tmp_path / "exclude.txt"
    excl.write_text("rec-1\n")
    out = tmp_path / "out.h5"

    build(raw_root=raw_root, sources=["pelvis"], out=out, bdl=bdl_path, exclusion_list=excl,
          device="cpu", workers=0, ids_file=ids_file)

    with h5py.File(out, "r") as h5:
        assert set(h5.keys()) == {"rec-0"}

    skip_rows = {r["sample_id"]: r["reason"] for r in csv.DictReader((tmp_path / "out_skipped.csv").open())}
    assert skip_rows.get("rec-1") == "excluded"
    assert "rec-2" not in skip_rows and "rec-3" not in skip_rows


# ── plan_candidates unit coverage ────────────────────────────────────────────


def test_plan_candidates_orders_and_splits(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    source_dir = raw_root / "pelvis"
    ids = [f"rec-{i}" for i in range(3)]
    for i, sample_id in enumerate(ids):
        write_synthetic_record(source_dir, sample_id, shape=RAW_SHAPE, seed=i)

    retained, skip_rows = plan_candidates(raw_root, ["pelvis"], {"rec-1"}, None, None)
    assert retained == [("pelvis", "rec-0"), ("pelvis", "rec-2")]
    assert skip_rows == [{"sample_id": "rec-1", "source": "pelvis", "reason": "excluded", "detail": ""}]
