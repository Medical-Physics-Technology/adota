"""The exclusion list of H5PYGenerator is optional, and a named one must exist."""

import h5py
import pytest

from src.loaders.generator import H5PYGenerator


@pytest.fixture
def h5_file(tmp_path):
    path = tmp_path / "records.h5"
    with h5py.File(path, "w") as f:
        for rid in ("a", "b", "c"):
            f.create_group(rid)
    return path


@pytest.mark.parametrize("kwargs", [{}, {"indexes_to_exclude_list": None}, {"indexes_to_exclude_list": ""}])
def test_no_list_keeps_every_record(h5_file, kwargs):
    assert H5PYGenerator(str(h5_file), **kwargs).record_ids == ["a", "b", "c"]


def test_a_named_list_drops_its_records(h5_file, tmp_path):
    listing = tmp_path / "exclude.txt"
    listing.write_text("b\n")
    assert H5PYGenerator(str(h5_file), indexes_to_exclude_list=str(listing)).record_ids == ["a", "c"]


def test_a_named_list_that_is_missing_raises(h5_file, tmp_path):
    with pytest.raises(FileNotFoundError):
        H5PYGenerator(str(h5_file), indexes_to_exclude_list=str(tmp_path / "missing.txt"))
