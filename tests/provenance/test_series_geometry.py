"""Series assembly from DICOM headers: grouping, geometry checks, exact loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.active_learning.beamlet_pool.planning import entries_from_registry
from src.active_learning.beamlet_pool.registry import CollectionSpec, grid_from_geometry, screen_patient
from src.active_learning.pool import PoolEntry, RecordResolver
from src.datasets.tcia import TCIADataset, best_candidate, ct_series_candidates
from src.provenance.dicom_qc import QCGates
from src.provenance.series_geometry import CT_IMAGE_STORAGE, check_geometry, read_directory

AXIAL = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def write_slice(path: Path, series_uid: str, z: float, *, x0: float = -10.0, sop_class: str = CT_IMAGE_STORAGE,
                modality: str = "CT", image_type=("ORIGINAL", "PRIMARY", "AXIAL"), orientation=AXIAL) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = "1.2.3", series_uid
    ds.Modality, ds.ImageType = modality, list(image_type)
    ds.PatientID = "P"
    ds.ImagePositionPatient = [x0, -10.0, z]
    ds.ImageOrientationPatient = list(orientation)
    ds.PixelSpacing, ds.SliceThickness = [1.0, 1.0], 2.0
    ds.Rows = ds.Columns = 4
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleIntercept, ds.RescaleSlope = 0, 1
    ds.PixelData = np.full((4, 4), int(z), dtype=np.int16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def write_series(directory: Path, uid: str, zs, prefix: str = "s", **kw) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i, z in enumerate(zs):
        write_slice(directory / f"{prefix}{i:03d}.dcm", uid, float(z), **kw)


def headers(directory: Path, uid: str):
    return read_directory(str(directory)).series[uid]


def test_uniform_series_stacks_in_order(tmp_path):
    write_series(tmp_path, "1.1", [8, 2, 6, 0, 4])
    geom = check_geometry(headers(tmp_path, "1.1"))
    assert geom.ok and geom.n_slices == 5 and geom.step_mm == pytest.approx(2.0)
    assert [pydicom.dcmread(f).ImagePositionPatient[2] for f in geom.files] == [0, 2, 4, 6, 8]
    assert geom.z_ends_mm == (0.0, 8.0) and geom.z_extent_mm == pytest.approx(10.0)


@pytest.mark.parametrize("zs, kw, reason", [
    ([0, 2, 4, 8, 10], {}, "slice gap"),                       # one slice missing
    ([0, 2, 4, 300, 302, 304], {}, "slice gap"),              # two acquisitions under one UID
    ([0, 2, 2, 4, 6], {}, "duplicate slice positions"),
    ([0, 2, 4, 4.5, 6, 8, 10], {}, "uneven step"),            # a stray slice between two others
])
def test_broken_stacks_are_named(tmp_path, zs, kw, reason):
    write_series(tmp_path, "1.1", zs, **kw)
    geom = check_geometry(headers(tmp_path, "1.1"))
    assert not geom.ok and any(reason in r for r in geom.reasons)


def test_sheared_and_reoriented_stacks_fail(tmp_path):
    write_series(tmp_path / "a", "1.1", [0, 2, 4])
    write_slice(tmp_path / "a" / "x.dcm", "1.1", 6.0, x0=-7.0)
    assert any("shifted in plane" in r for r in check_geometry(headers(tmp_path / "a", "1.1")).reasons)
    write_series(tmp_path / "b", "1.2", [0, 2, 4])
    write_slice(tmp_path / "b" / "x.dcm", "1.2", 6.0, orientation=(1.0, 0.0, 0.0, 0.0, 0.9, 0.1))
    assert "orientation varies" in check_geometry(headers(tmp_path / "b", "1.2")).reasons


def test_directory_is_split_by_each_files_own_series_uid(tmp_path):
    write_series(tmp_path, "1.1", [0, 2, 4, 6], prefix="a")
    write_series(tmp_path, "1.2", [0, 3, 6], prefix="b")
    write_slice(tmp_path / "loc.dcm", "1.1", 50.0, image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"))
    write_slice(tmp_path / "rt.dcm", "1.1", 0.0, sop_class="1.2.840.10008.5.1.4.1.1.481.3", modality="RTSTRUCT")
    (tmp_path / "notes.txt").write_text("not dicom")
    contents = read_directory(str(tmp_path))
    assert {uid: len(s) for uid, s in contents.series.items()} == {"1.1": 4, "1.2": 3}
    assert contents.skipped == {"localizer": 1, "not a CT image": 1, "not readable as DICOM": 1}
    assert check_geometry(contents.series["1.1"]).ok    # the localizer would have been a 44 mm gap


def make_patient(root: Path, patient: str = "P001") -> Path:
    """A patient with a large series that has a gap and a smaller one that stacks."""
    pdir = root / "Coll" / patient / "study"
    write_series(pdir / "big", "9.1", list(range(0, 40, 2)) + list(range(300, 340, 2)))
    write_series(pdir / "small", "9.2", list(range(0, 60, 2)))
    (pdir / "seg").mkdir()
    write_slice(pdir / "seg" / "seg.dcm", "9.3", 0.0, sop_class="1.2.840.10008.5.1.4.1.1.66.4", modality="SEG")
    return root


def test_selection_skips_the_series_that_does_not_stack(tmp_path):
    make_patient(tmp_path)
    gates = QCGates(min_slices=10, max_slice_gap_ratio=1.5)
    candidates = ct_series_candidates(str(tmp_path / "Coll" / "P001"), gates)
    assert sorted(c.geometry.series_uid for c in candidates) == ["9.1", "9.2"]
    big = next(c for c in candidates if c.geometry.series_uid == "9.1")
    assert not big.qc_pass and any("slice gap" in r for r in big.qc_reasons)
    assert best_candidate(candidates).geometry.series_uid == "9.2"
    ds = TCIADataset(root=str(tmp_path), collection="Coll", min_slices=10, qc={"max_slice_gap_ratio": 1.5})
    rec = ds.record(0)
    assert rec.series_uid == "9.2" and rec.n_slices == 30 and len(rec.files) == 30
    assert rec.load_image().GetSize()[2] == 30
    legacy = TCIADataset(root=str(tmp_path), collection="Coll", min_slices=10)
    assert legacy.record(0).series_uid == "9.1" and legacy.record(0).files == ()   # unchanged without the gate


def test_screen_matches_training_scans_on_every_series(tmp_path):
    make_patient(tmp_path)
    spec = CollectionSpec(name="Coll", anatomy="thoracic", root=str(tmp_path), collection="Coll",
                          qc={"min_slices": 10, "max_slice_gap_ratio": 1.5})
    big = check_geometry(headers(tmp_path / "Coll" / "P001" / "study" / "big", "9.1"))
    grid = grid_from_geometry(big)
    # the training scan is the series the screen does not select
    scans = pd.DataFrame([{"patient_key": "k1", "image_origin_mm_0": grid["origin_x_mm"],
                           "image_origin_mm_1": grid["origin_y_mm"], "image_origin_mm_2": grid["origin_z_top_mm"],
                           "image_size_vox_0": grid["grid_x"], "image_size_vox_2": grid["grid_z"],
                           "image_spacing_mm_2": 1.0, "source_dataset": "t"}])
    row = screen_patient(spec, "P001", scans)
    assert row["training_matches"] == "k1" and row["series_uid"] == "9.2" and row["qc_pass"]
    assert row["n_ct_series"] == 2 and row["n_ct_series_not_stacking"] == 1 and row["max_slice_gap_ratio"] == 1.5


def test_registry_entries_load_exactly_their_series(tmp_path):
    make_patient(tmp_path)
    entry = PoolEntry("train", "thoracic", "Coll", str(tmp_path), "Coll", "P001", "9.2", 30, 1.5)
    rec = RecordResolver([entry]).record(entry)
    assert rec.series_uid == "9.2" and len(rec.files) == 30
    with pytest.raises(ValueError, match="slices on disk"):
        RecordResolver([entry]).record(PoolEntry(*list(entry.__dict__.values())[:7], 29, 1.5))
    with pytest.raises(ValueError, match="does not stack"):
        RecordResolver([entry]).record(PoolEntry(*list(entry.__dict__.values())[:6], "9.1", 40, 1.5))
    old = pd.DataFrame([{"split": "train", "anatomy": "thoracic", "collection": "Coll", "root": str(tmp_path),
                         "dataset_collection": "Coll", "patient_id": "P001", "series_uid": "9.2", "n_slices": 30}])
    with pytest.raises(ValueError, match="predates"):
        entries_from_registry(old)
    assert entries_from_registry(old.assign(max_slice_gap_ratio=1.5))[0].max_slice_gap_ratio == 1.5
