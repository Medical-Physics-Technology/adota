"""Meaningful tests for the CT dataset abstraction on the real TCIA manifests.

Skipped when the manifests are absent. These assert the loader (a) selects the CT
series and not the 1-slice RTSTRUCT, (b) is patient-indexed with correct
provenance, (c) loads a geometrically-valid CT volume with a plausible HU range,
and (d) composes across anatomies with correct index math and provenance.
"""
import os

import pytest
import SimpleITK as sitk

from src.datasets.multi import MultiDataset
from src.datasets.registry import build_dataset_from_config
from src.datasets.tcia import TCIADataset

THORACIC = "/scratch/mstryja/manifest-1603198545583"
ABDOMINAL = "/scratch/mstryja/manifest-1646429317311"
PELVIC = "/scratch/mstryja/manifest-1684259732535"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PELVIC), reason="TCIA manifests not present on this host"
)


def test_patient_indexed_selects_ct_with_provenance():
    ds = TCIADataset(root=PELVIC, anatomy="pelvic", n_patients=3, selection="first")
    assert len(ds) == 3  # exactly n_patients, i.e. one CT per patient
    for rec in ds:
        assert rec.anatomy == "pelvic"
        assert rec.patient_id.startswith("Prostate-AEC")
        assert rec.series_uid  # non-empty DICOM SeriesInstanceUID
        # CT volume, not the 1-slice RTSTRUCT that shares the patient dir
        assert 50 <= rec.n_slices <= 1000
        assert rec.uid.startswith(f"{ds.name}/{rec.patient_id}/")
    # patient-indexed => distinct patients
    assert len(set(ds.patient_ids())) == 3


def test_load_image_is_valid_ct_volume():
    ds = TCIADataset(root=PELVIC, anatomy="pelvic", n_patients=1)
    img, rec = ds.load(0)
    assert img.GetDimension() == 3
    size = img.GetSize()  # (x, y, z)
    assert size[2] == rec.n_slices and size[0] > 0 and size[1] > 0
    assert all(s > 0 for s in img.GetSpacing())
    arr = sitk.GetArrayFromImage(img)
    # a real CT spans from air (~ -1000 HU) to dense bone/contrast (> 500 HU)
    assert arr.min() < -500 and arr.max() > 500


def test_explicit_patient_ids_and_deterministic_random():
    one = TCIADataset(root=PELVIC, anatomy="pelvic", patient_ids=["Prostate-AEC-058"])
    assert len(one) == 1 and one.record(0).patient_id == "Prostate-AEC-058"
    a = TCIADataset(root=PELVIC, anatomy="pelvic", n_patients=4, selection="random", seed=42)
    b = TCIADataset(root=PELVIC, anatomy="pelvic", n_patients=4, selection="random", seed=42)
    assert a.patient_ids() == b.patient_ids()  # reproducible sampling


@pytest.mark.skipif(not (os.path.isdir(THORACIC) and os.path.isdir(ABDOMINAL)),
                    reason="need >=3 manifests for the multi-dataset test")
def test_multidataset_index_math_and_provenance():
    cfg = {
        "datasets": [
            {"name": "NSCLC-Radiomics", "anatomy": "thoracic", "root": THORACIC, "n_patients": 2},
            {"name": "Colorectal", "anatomy": "abdominal", "root": ABDOMINAL, "n_patients": 3},
            {"name": "Prostate-AEC", "anatomy": "pelvic", "root": PELVIC, "n_patients": 2},
        ]
    }
    ds = build_dataset_from_config(cfg)
    assert isinstance(ds, MultiDataset)
    assert len(ds) == 7
    assert ds.counts_by_anatomy() == {"thoracic": 2, "abdominal": 3, "pelvic": 2}
    # boundary indices map to the right sub-dataset's anatomy
    assert ds.record(0).anatomy == "thoracic"
    assert ds.record(2).anatomy == "abdominal"   # first of the 3 colorectal
    assert ds.record(5).anatomy == "pelvic"      # first of the 2 prostate
    assert ds.record(-1).anatomy == "pelvic"
    # global provenance uids are unique across the whole composite
    assert len({ds.record(i).uid for i in range(len(ds))}) == 7
