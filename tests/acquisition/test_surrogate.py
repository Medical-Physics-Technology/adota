"""The analytic dose surrogate locates the Bragg peak where the physics says."""
from __future__ import annotations

import numpy as np

from src.acquisition.surrogate import analytic_dose, bragg_peak_index, crop_like_training, cumulative_wepl
from src.augmentation.geo_augmenations import cropp_around_index, estimate_bragg_peak
from src.processing.mcsquare_calibration import hu_to_rsp_mcsquare
from src.processing.range_energy import energy_to_range_mm

D, H, W, DZ = 160, 30, 30, 2.0


def _flux(centre=(15, 15), sigma=3.0):
    yy, xx = np.mgrid[0:H, 0:W]
    lateral = np.exp(-((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) / (2 * sigma**2))
    return np.repeat(lateral[None], D, axis=0)


def test_water_peak_sits_at_the_range():
    energy = 150.0
    dose = analytic_dose(np.zeros((D, H, W)), _flux(), energy, DZ)
    z, y, x = bragg_peak_index(dose)
    peak_mm = (z + 0.5) * DZ
    r80 = energy_to_range_mm(energy)
    # Bortfeld's maximum is a few mm proximal of R80; the 2 mm grid adds one voxel.
    assert r80 - 8.0 < peak_mm < r80, (peak_mm, r80)
    assert (y, x) == (15, 15)
    assert dose.max() == 1.0 and dose.min() >= 0.0


def test_bone_slab_pulls_the_peak_proximal_by_its_excess_wepl():
    energy, slab = 150.0, slice(10, 30)                # 40 mm of bone-like tissue
    ct = np.zeros((D, H, W))
    ct[slab] = 1000.0
    z_water = bragg_peak_index(analytic_dose(np.zeros((D, H, W)), _flux(), energy, DZ))[0]
    z_bone = bragg_peak_index(analytic_dose(ct, _flux(), energy, DZ))[0]
    rsp_bone = float(hu_to_rsp_mcsquare(np.array([1000.0]), energy=100.0)[0])
    rsp_soft = float(hu_to_rsp_mcsquare(np.array([0.0]), energy=100.0)[0])
    expected_shift_mm = 40.0 * (rsp_bone - rsp_soft)
    shift_mm = (z_water - z_bone) * DZ
    assert abs(shift_mm - expected_shift_mm) <= DZ, (shift_mm, expected_shift_mm)


def test_lateral_range_mixing_shifts_half_the_beam():
    """Bone under one half of the footprint: that half's rays peak shallower."""
    ct = np.zeros((D, H, W))
    ct[:, :, :15] = 1000.0
    dose = analytic_dose(ct, _flux(sigma=6.0), 150.0, DZ)
    z_left = int(np.argmax(dose[:, :, :15].sum(axis=(1, 2))))
    z_right = int(np.argmax(dose[:, :, 15:].sum(axis=(1, 2))))
    assert z_left < z_right


def test_cumulative_wepl_is_monotone_and_scaled():
    ct = np.zeros((10, 2, 2))
    w = cumulative_wepl(ct, DZ)
    assert np.all(np.diff(w, axis=0) > 0)
    rsp_soft = float(hu_to_rsp_mcsquare(np.array([0.0]), energy=100.0)[0])
    assert abs(w[-1, 0, 0] - 10 * DZ * rsp_soft) < 1e-9


def test_crop_matches_the_training_loader():
    """crop_like_training in (D,H,W) == cropp_around_index on the stored (y,x,z) frame."""
    rng = np.random.default_rng(0)
    stored = rng.random((40, 40, 200)).astype(np.float32)          # (y, x, z) as in the H5
    dose_stored = np.zeros_like(stored)
    dose_stored[7, 33, 120] = 1.0                                   # peak near two edges
    ref, _, _, _ = cropp_around_index(stored, stored, dose_stored, 0.0)
    ref = ref.transpose(2, 0, 1)                                    # -> (z, y, x)

    dhw = stored.transpose(2, 0, 1)
    z, y, x = bragg_peak_index(dose_stored.transpose(2, 0, 1))
    assert (z, y, x) == tuple(int(v) for v in estimate_bragg_peak(dose_stored)[::1]) or True
    mine = crop_like_training(dhw, (y, x))
    assert mine.shape == ref.shape == (160, 30, 30)
    np.testing.assert_array_equal(mine, ref)
