"""Tests for the body-mask / centre-of-mass isocenter helpers (COM approach)."""
import numpy as np
import SimpleITK as sitk

from src.mc_generation.geometry import (
    body_center_of_mass,
    body_fraction_at_peak,
    dose_in_body_fraction,
    extraction_isocenter_physical,
    isocenters_from_world,
    mc_isocenter,
)


def _phantom_with_offcenter_body(size=(200, 200, 200), body=(-1024.0, 0.0)):
    """Air volume with a water box shifted toward one corner (off the grid centre)."""
    air, water = body
    arr = np.full(size[::-1], air, dtype=np.float32)  # (z,y,x)
    # water box occupying x[20:120], y[30:110], z[40:160] -> centre far from grid centre
    arr[40:160, 30:110, 20:120] = water
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img, arr


def test_body_com_lands_in_body_not_grid_centre():
    img, arr = _phantom_with_offcenter_body()
    com = body_center_of_mass(img, body_hu_threshold=-500.0)   # world (x,y,z)
    # expected COM of the water box (index centres): x~(20+120)/2=70, y~70, z~100
    assert com[0] == np.isclose(com[0], 70, atol=2) or abs(com[0] - 70) <= 2
    assert abs(com[1] - 70) <= 2
    assert abs(com[2] - 100) <= 2
    # and it is clearly NOT the grid centre (100,100,100)
    grid_ext = extraction_isocenter_physical(img)   # ~ (99.5,99.5,99.5)
    assert np.linalg.norm(com - grid_ext) > 20


def test_isocenters_from_world_matches_grid_convention():
    img, _ = _phantom_with_offcenter_body()
    # feed the grid-centre world point -> should reproduce mc_isocenter / extraction
    grid_ext = extraction_isocenter_physical(img)
    world = grid_ext + np.asarray(img.GetSpacing())/2.0   # invert the -spacing/2
    iso_mc, iso_ext = isocenters_from_world(img, world)
    assert np.allclose(iso_ext, grid_ext, atol=1e-6)
    assert np.allclose(iso_mc, np.asarray(mc_isocenter(img)), atol=1.0)


def test_dose_in_body_fraction():
    mask = np.zeros((10, 10, 10), dtype=np.float32)
    mask[:, :, 5:] = 1.0                      # right half in body
    dose = np.zeros((10, 10, 10), dtype=np.float32)
    dose[:, :, 6] = 1.0                       # all dose in body
    assert dose_in_body_fraction(dose, mask) == 1.0
    dose2 = np.zeros_like(dose)
    dose2[:, :, 1] = 1.0                      # all dose in air
    assert dose_in_body_fraction(dose2, mask) == 0.0
    dose3 = np.zeros_like(dose)
    dose3[:, :, 1] = 1.0
    dose3[:, :, 6] = 3.0                      # 3/4 in body
    assert abs(dose_in_body_fraction(dose3, mask) - 0.75) < 1e-6


def _beamlet(body_end=None):
    """A 20 x 20 x 300 crop: body from depth 20 (to ``body_end``), a pencil beam on
    the axis peaking at depth 200, and one noisy air voxel in front of the patient."""
    mask = np.zeros((20, 20, 300), dtype=np.float32)
    mask[:, :, 20:body_end] = 1.0
    depth = np.arange(300)
    curve = 0.3 + np.exp(-0.5 * ((depth - 200) / 6.0) ** 2)      # entrance plateau + flat peak
    curve[205:] *= np.exp(-(depth[205:] - 205) / 3.0)            # distal fall-off
    lateral = np.exp(-0.5 * ((np.arange(20) - 9.5) / 3.0) ** 2)
    dose = curve[None, None, :] * lateral[:, None, None] * lateral[None, :, None]
    dose[3, 15, 5] = 2.0 * dose.max()                            # air spike, hotter than the peak
    return dose.astype(np.float32), mask


def test_body_fraction_at_peak_ignores_a_spike_in_air():
    dose, mask = _beamlet()
    peak = np.unravel_index(int(np.argmax(dose)), dose.shape)
    assert mask[peak] == 0.0                                     # the hottest voxel is in air
    assert body_fraction_at_peak(dose, mask) == 1.0


def test_body_fraction_at_peak_catches_a_peak_beyond_the_patient():
    dose, mask = _beamlet(body_end=150)
    assert body_fraction_at_peak(dose, mask) == 0.0
    assert body_fraction_at_peak(np.zeros_like(dose), mask) == 0.0
