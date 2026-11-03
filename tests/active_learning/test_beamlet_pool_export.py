"""Exporting a pool split to the v3 training layout, on a synthetic shard: the
same recipe as the v3 build (normalise, then average-pool to 2 mm), the flux
recomputed from the metadata, the v2 + v3 attributes and the pool's identity."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from src.active_learning.beamlet_pool.export import POOL_ATTRS, export_split, pool_metadata
from src.active_learning.beamlet_pool.store import recompute_flux
from src.adota.config import DEFAULT_SCALE
from src.datasets.beamlet_h5 import V2_ATTRS, V3_ATTRS
from src.datasets.beamlet_record import downsample_grid, normalise_dose

ROI = [8, 8, 32]
SIGMAS = [3.5, 3.1]


def pool_sim_res(energy: float, k: int) -> dict:
    return {"simulation_log": {"energy": [energy], "isocenter": [100.0, 90.0, 50.0 + k],
                               "bixelgrid_shifts_xy": [[1.0 * k, -2.0]], "gantry_angle": 90.0},
            "initial_energy": energy, "gantry_angle": 30.0 + k, "num_primaries": 1e6, "stat_uncertainty": float("nan"),
            "id": f"uuid-{k}", "dose_deposition_ratio": 0.99, "beamlet_angles": [0.5, -0.25],
            "rays_entrence_point_proj": [4.0, 4.0, 0.0], "rays_entrence_point": [0.0, 4.0, 4.0], "roi_size": ROI,
            "image_origin": [-100.0, -100.0, -50.0], "image_spacing": [1.0, 1.0, 1.0], "image_size": [200, 200, 100],
            "patient_id": "P-001", "anatomy": "pelvic", "dataset_name": "Prostate-AEC"}


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "test").mkdir()
    with h5py.File(tmp_path / "test" / "batch_0000.h5", "w") as h5:
        for k, cid in enumerate(["cb", "ca"]):
            group = h5.create_group(cid)
            group.create_dataset("ct", data=rng.integers(-1000, 1500, ROI).astype(np.int16))
            group.create_dataset("dose", data=rng.random(ROI, dtype=np.float32) * 1e7)
            group.attrs["sim_res"] = json.dumps(pool_sim_res(100.0 + 20 * k, k))
            group.attrs["flux_sigma_xy_mm"] = np.asarray(SIGMAS)
    return tmp_path


def test_pool_metadata_moves_angles_and_primaries_where_the_writer_reads_them():
    meta = pool_metadata(pool_sim_res(120.0, 0))
    assert meta["simulation_log"]["beamlet_angles"] == [0.5, -0.25]
    assert meta["simulation_log"]["sim_params"]["Num_Primaries"] == 1e6
    assert "sim_params" not in pool_sim_res(120.0, 0)["simulation_log"]


def test_export_follows_the_v3_recipe_and_keeps_the_pool_identity(store, tmp_path):
    out = tmp_path / "out" / "pool_test.h5"
    rows = export_split(store, "test", out, scale=DEFAULT_SCALE, bdl_file="bdl.txt", bdl_sha256="abc")
    assert [r["sample_id"] for r in rows] == ["ca", "cb"]
    with h5py.File(store / "test" / "batch_0000.h5", "r") as src, h5py.File(out, "r") as h5:
        assert h5.attrs["n_records"] == 2 and h5.attrs["split"] == "test"
        group, raw = h5["cb"], src["cb"]
        for name in ("ct", "dose", "flux"):
            assert group[name].shape == (4, 4, 16) and group[name].dtype == np.float32
        expected = downsample_grid(normalise_dose(raw["dose"][...], DEFAULT_SCALE))
        np.testing.assert_allclose(group["dose"][...], expected, rtol=1e-6)
        flux = recompute_flux(json.loads(raw.attrs["sim_res"]), SIGMAS)
        np.testing.assert_allclose(group["flux"][...], downsample_grid(flux), rtol=1e-6)
        assert set(V2_ATTRS) | set(V3_ATTRS) | set(POOL_ATTRS) <= set(group.attrs)
        assert group.attrs["energy_mev"] == 100.0
        assert group.attrs["initial_energy"] == pytest.approx((100.0 - 70.0) / 200.0)
        assert group.attrs["num_primaries"] == 1e6 and group.attrs["source_dataset"] == "beamlet_pool"
        assert group.attrs["pool_patient_id"] == "P-001" and group.attrs["pool_shard"] == "test/batch_0000.h5"
    index = pd.read_csv(tmp_path / "out" / "pool_test_index.csv")
    assert index["sample_id"].tolist() == ["ca", "cb"] and index["pool_anatomy"].eq("pelvic").all()
    assert index["patient_key"].nunique() == 1


def test_export_limits_and_refuses_to_overwrite(store, tmp_path):
    out = tmp_path / "pool_test.h5"
    assert len(export_split(store, "test", out, scale=DEFAULT_SCALE, bdl_file="b", bdl_sha256="c", limit=1)) == 1
    with pytest.raises(FileExistsError):
        export_split(store, "test", out, scale=DEFAULT_SCALE, bdl_file="b", bdl_sha256="c")
    with pytest.raises(FileNotFoundError):
        export_split(store, "validation", tmp_path / "other.h5", scale=DEFAULT_SCALE, bdl_file="b", bdl_sha256="c")
