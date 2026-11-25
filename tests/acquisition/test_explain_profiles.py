"""Beam-axis profiles and loaders of the difficulty explanation module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.acquisition.explain.profiles import BeamletInputs, beam_axis_profile, load_grid_beamlet
from src.acquisition.features import FEATURE_NAMES, compute_features
from src.processing.range_energy import energy_to_range_mm

D, H, W, DZ = 160, 30, 30, 2.0


def _phantom(energy=100.0, slab=None):
    ct = np.zeros((D, H, W), dtype=np.float32)
    if slab is not None:
        ct[slab[0]:slab[1]] = 1000.0
    yy, xx = np.mgrid[0:H, 0:W]
    lateral = np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / (2 * 3.0**2)).astype(np.float32)
    flux = np.repeat(lateral[None], D, axis=0)
    return BeamletInputs("synthetic", "grid", ct, flux, energy)


def test_water_phantom_profiles():
    prof = beam_axis_profile(_phantom(100.0))
    for name in ("depth_mm", "hu_axis", "hu_flux_weighted", "rsp_axis", "wepl_axis_mm", "idd_analytic"):
        assert getattr(prof, name).shape == (D,), name
    assert prof.idd_mc is None and prof.mc_bp_slice is None
    assert prof.idd_analytic.max() == pytest.approx(1.0)
    np.testing.assert_allclose(prof.wepl_axis_mm, (np.arange(D) + 1) * DZ, rtol=0.05)
    assert abs(prof.bp_slice * DZ - energy_to_range_mm(100.0)) < 10.0
    assert prof.bp_zone_mm[0] <= prof.bp_slice * DZ <= prof.bp_zone_mm[1]


def test_bone_slab_shifts_peak_and_profile():
    water = beam_axis_profile(_phantom(100.0))
    bone = beam_axis_profile(_phantom(100.0, slab=(30, 40)))
    assert bone.bp_slice < water.bp_slice
    assert bone.hu_flux_weighted[35] == pytest.approx(1000.0)
    assert abs(bone.hu_flux_weighted[20]) < 1e-3 and abs(bone.hu_flux_weighted[45]) < 1e-3


_BLOCK = Path("/scratch/mstryja/DoTA_dataset_v2/energy_gantry_full_abdominal_StageII-Colorectal-CT-222_e135_g197p8_v1")
_CSV = Path("/scratch/mstryja/adota_runs/exp0014/difficulty_engine_grids/energy_gantry_full/features_analytic.csv")


def test_grid_beamlet_reproduces_saved_features():
    for path in (_BLOCK / "a00_12_ct.npy", _CSV):
        if not path.exists():
            pytest.skip(f"missing {path}: needs the DoTA_dataset_v2 grid block and the engine-grids feature CSV")
    import pandas as pd

    table = pd.read_csv(_CSV)
    row = table[(table.block == _BLOCK.name) & (table.stem == "a00_12")].iloc[0]
    inp = load_grid_beamlet(_BLOCK, "a00_12", 135.0)
    prof = beam_axis_profile(inp)
    feats = compute_features(inp.ct_hu, inp.flux, inp.energy_mev, prof.analytic_dose)
    for name in FEATURE_NAMES:
        assert feats[name] == pytest.approx(row[name], rel=1e-6, abs=1e-12), name
    assert prof.bp_slice == int(row.bp_slice)
    assert prof.bp_zone_mm == pytest.approx((row.bp_range_min_mm, row.bp_range_max_mm))
    assert inp.mc_dose.shape == (D, H, W) and prof.mc_bp_slice is not None
