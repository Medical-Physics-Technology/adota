"""Tests for the synthetic phantom CT source (src/datasets/phantom.py)."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from src.datasets.base import CTDataset
from src.datasets.phantom import (
    PhantomDataset,
    PhantomRecord,
    PhantomSpec,
    build_phantom_dataset,
    build_phantom_image,
)


def test_plain_water_box_is_uniform_and_correctly_oriented():
    spec = PhantomSpec(kind="water", size=(40, 50, 60), spacing=(1.0, 1.0, 1.0), water_hu=0)
    img = build_phantom_image(spec)
    # SimpleITK size is (x, y, z); numpy array is (z, y, x).
    assert img.GetSize() == (40, 50, 60)
    arr = sitk.GetArrayFromImage(img)
    assert arr.shape == (60, 50, 40)
    assert np.all(arr == 0)
    assert img.GetSpacing() == (1.0, 1.0, 1.0)
    assert img.GetDirection() == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_air_shell_geometry_and_hu_values():
    d = 5
    spec = PhantomSpec(kind="water", size=(60, 60, 60), water_hu=0, air_hu=-1024,
                       air_layer_depth=d)
    arr = sitk.GetArrayFromImage(build_phantom_image(spec))
    # Border of thickness d on every face is air; the core is water.
    assert np.all(arr[:d] == -1024) and np.all(arr[-d:] == -1024)
    assert np.all(arr[:, :d] == -1024) and np.all(arr[:, -d:] == -1024)
    assert np.all(arr[:, :, :d] == -1024) and np.all(arr[:, :, -d:] == -1024)
    core = arr[d:-d, d:-d, d:-d]
    assert np.all(core == 0)
    # Exact water/air voxel counts.
    assert core.size == (60 - 2 * d) ** 3
    assert int((arr == 0).sum()) == core.size


def test_content_hash_is_deterministic_and_sensitive():
    a = PhantomSpec(size=(10, 10, 10), air_layer_depth=0)
    b = PhantomSpec(size=(10, 10, 10), air_layer_depth=0)
    c = PhantomSpec(size=(10, 10, 10), air_layer_depth=5)
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash
    assert len(a.content_hash) == 16


def test_phantom_record_is_ctrecord_compatible():
    spec = PhantomSpec(name="plain", size=(20, 20, 20))
    rec = PhantomRecord(dataset_name="water_phantom", anatomy="phantom",
                        patient_id="plain", spec=spec)
    # Duck-typed interface the generation spine relies on.
    assert rec.uid == f"water_phantom/plain/{spec.content_hash}"
    assert rec.series_uid == spec.content_hash
    assert rec.n_slices == 20
    img = rec.load_image()
    assert isinstance(img, sitk.Image)
    assert img.GetSize() == (20, 20, 20)


def test_build_phantom_dataset_from_config():
    cfg = {
        "name": "water_phantom",
        "phantoms": [
            {"kind": "water", "name": "plain", "size": [30, 30, 30], "air_layer_depth": 0},
            {"kind": "water", "name": "air5mm", "size": [30, 30, 30], "air_layer_depth": 5},
        ],
    }
    ds = build_phantom_dataset(cfg)
    assert isinstance(ds, (PhantomDataset, CTDataset))
    assert len(ds) == 2
    assert ds.patient_ids() == ["plain", "air5mm"]
    assert ds.anatomy == "phantom"
    # Distinct geometry -> distinct provenance.
    assert ds.record(0).series_uid != ds.record(1).series_uid
    img, rec = ds.load(0)
    assert img.GetSize() == (30, 30, 30)
    assert rec.dataset_name == "water_phantom"


def test_unknown_kind_raises():
    import pytest
    with pytest.raises(ValueError):
        build_phantom_image(PhantomSpec(kind="tungsten", size=(10, 10, 10)))
