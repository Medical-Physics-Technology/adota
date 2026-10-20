"""The 250 MeV cut on beamlet HDF5 record lists."""
from __future__ import annotations

import h5py
import pytest

from src.datasets.energy_cut import MAX_VALID_MC_ENERGY_MEV, drop_above_energy
from src.training.data import load_record_ids


def _h5(path, energies, with_attr=True):
    with h5py.File(path, "w") as f:
        for i, e in enumerate(energies):
            g = f.create_group(f"r{i}")
            if with_attr:
                g.attrs["energy_mev"] = e
            g.attrs["initial_energy"] = (e - 70.0) / 200.0
    return path


def test_cut_keeps_order_and_the_boundary(tmp_path):
    path = _h5(tmp_path / "d.h5", [80.0, 250.0, 250.01, 120.0, 269.0])
    kept, dropped = drop_above_energy(path, ["r0", "r1", "r2", "r3", "r4"], MAX_VALID_MC_ENERGY_MEV)
    assert kept == ["r0", "r1", "r3"] and dropped == ["r2", "r4"]
    assert drop_above_energy(path, ["r4", "r0"], None) == (["r4", "r0"], [])     # None: no cut


def test_old_files_use_the_scale_and_refuse_without_it(tmp_path):
    path = _h5(tmp_path / "v2.h5", [100.0, 260.0], with_attr=False)
    scale = {"min_energy": 70.0, "max_energy": 270.0}
    assert drop_above_energy(path, ["r0", "r1"], 250.0, scale) == (["r0"], ["r1"])
    with pytest.raises(KeyError):
        drop_above_energy(path, ["r0"], 250.0)


def test_training_record_list_applies_exclusions_then_the_cut(tmp_path):
    path = _h5(tmp_path / "d.h5", [80.0, 255.0, 120.0, 90.0])
    excl = tmp_path / "exclude.txt"
    excl.write_text("r3\n")
    assert load_record_ids(path, excl) == ["r0", "r1", "r2"]                    # no cut by default
    assert load_record_ids(path, excl, max_energy_mev=250.0) == ["r0", "r2"]
