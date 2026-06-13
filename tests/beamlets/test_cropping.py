"""Unit + A/B parity tests for :mod:`src.beamlets.cropping`."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from src.beamlets.cropping import crop_around_spatial_point, extract_beamlet_roi

ORIGIN = (-12.0, -34.0, 5.0)
SPACING = (1.0, 1.0, 2.0)
DISTANCES = (420.0, 2014.9, 2584.1)  # d_nozzle, d_smx, d_smy


def _image_from_array(arr: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(arr.astype(np.float32))  # (z, y, x)
    image.SetOrigin(ORIGIN)
    image.SetSpacing(SPACING)
    return image


def test_crop_centres_on_spatial_point() -> None:
    """The lateral window is centred on the spatial point's voxel."""
    nz, ny, nx = 12, 12, 20
    arr = np.full((nz, ny, nx), -1024.0, dtype=np.float32)
    marker_xyz = (4, 5, 6)  # (ix, iy, iz)
    arr[marker_xyz[2], marker_xyz[1], marker_xyz[0]] = 500.0
    image = _image_from_array(arr)
    point = image.TransformContinuousIndexToPhysicalPoint(
        [float(c) for c in marker_xyz]
    )

    roi = (4, 4, 8)  # (H, W, D)
    crop, corner, np_indexes, oob = crop_around_spatial_point(image, point, roi)

    assert crop.shape == (4, 4, 8)
    assert not oob
    # Centre voxel -> crop (z=H//2, y=W//2, x=ix).
    z, y, x = np.unravel_index(int(np.argmax(crop)), crop.shape)
    assert (z, y, x) == (2, 2, 4)
    # crp is np_indexes (iz, iy, ix); corner is the lower index corner.
    assert np_indexes == (6, 5, 4)
    assert corner == [0, 5 - 2, 6 - 2]


def test_oob_air_padded_and_inbounds_bit_identical() -> None:
    """Edge crops pad with air; the in-bounds part equals the raw slice."""
    rng = np.random.default_rng(1)
    nz, ny, nx = 10, 10, 12
    arr = rng.uniform(-500, 500, size=(nz, ny, nx)).astype(np.float32)
    image = _image_from_array(arr)

    # Point near the y=0 / z=0 corner so the window overhangs the grid.
    point = image.TransformContinuousIndexToPhysicalPoint([3.0, 1.0, 1.0])
    roi = (6, 6, 8)  # (H, W, D)
    crop, corner, np_indexes, oob = crop_around_spatial_point(image, point, roi)

    assert crop.shape == (6, 6, 8)
    assert oob
    # In-bounds overlap: the window starts at index -2 in y and z, so the
    # first two rows/cols are padding and the data lands at dst offset 2.
    np.testing.assert_array_equal(crop[2:6, 2:6, 0:8], arr[0:4, 0:4, 0:8])
    # Padded rows are exactly air.
    assert np.all(crop[0:2, :, :] == -1024.0)
    assert np.all(crop[:, 0:2, :] == -1024.0)


def test_ct_array_path_is_bit_identical() -> None:
    """Supplying ct_array yields byte-identical crop/entrance/crp vs fetching it.

    This guards the per-field GetArrayFromImage optimization: the fast path
    (pre-copied array) must equal the original path exactly.
    """
    rng = np.random.default_rng(9)
    arr = rng.uniform(-1000, 1000, size=(30, 40, 50)).astype(np.float32)
    image = _image_from_array(arr)
    iso = image.TransformContinuousIndexToPhysicalPoint([20.0, 18.0, 15.0])
    roi = (8, 8, 20)
    precomputed = sitk.GetArrayFromImage(image)

    base = crop_around_spatial_point(image, iso, roi)
    fast = crop_around_spatial_point(image, iso, roi, ct_array=precomputed)
    np.testing.assert_array_equal(base[0], fast[0])
    assert base[1:] == fast[1:]

    e_base = extract_beamlet_roi(image, *DISTANCES, (2.0, -3.0), iso, roi)
    e_fast = extract_beamlet_roi(image, *DISTANCES, (2.0, -3.0), iso, roi, ct_array=precomputed)
    np.testing.assert_array_equal(e_base[0], e_fast[0])
    np.testing.assert_array_equal(e_base[1], e_fast[1])
    assert e_base[2:] == e_fast[2:]


def test_reinsertion_round_trip() -> None:
    """Re-inserting an in-bounds crop at np_indexes reproduces the source region."""
    rng = np.random.default_rng(2)
    nz, ny, nx = 16, 16, 20
    arr = rng.uniform(-500, 500, size=(nz, ny, nx)).astype(np.float32)
    image = _image_from_array(arr)
    point = image.TransformContinuousIndexToPhysicalPoint([5.0, 8.0, 9.0])

    roi = (6, 6, 10)
    crop, corner, np_indexes, oob = crop_around_spatial_point(image, point, roi)
    assert not oob
    iz, iy, ix = np_indexes
    region = arr[iz - 3 : iz + 3, iy - 3 : iy + 3, 0:10]
    np.testing.assert_array_equal(crop, region)


# --- A/B parity vs datagenerator -------------------------------------------


def test_parity_extract_beamlet_roi(datagenerator_utils) -> None:
    """In-bounds crop + entrance match datagenerator's get_roi_with_indexes_ct_only."""
    rng = np.random.default_rng(5)
    nz, ny, nx = 60, 60, 60
    arr = rng.uniform(-1000, 1000, size=(nz, ny, nx)).astype(np.float32)
    image = _image_from_array(arr)

    iso = image.TransformContinuousIndexToPhysicalPoint([30.0, 30.0, 30.0])
    roi = (20, 20, 40)
    spot = (3.0, -4.0)
    d_nozzle, d_smx, d_smy = 420.0, 2014.9, 2584.1

    ours_crop, ours_entrance, ours_crp, oob = extract_beamlet_roi(
        image, d_nozzle, d_smx, d_smy, spot, iso, roi
    )
    theirs_crop, theirs_entrance, theirs_crp = (
        datagenerator_utils.get_roi_with_indexes_ct_only(
            image, d_nozzle, d_smx, d_smy, spot, iso, roi
        )
    )

    assert not oob
    np.testing.assert_array_equal(ours_crop, theirs_crop)
    np.testing.assert_allclose(ours_entrance, theirs_entrance, atol=1e-6)
    assert tuple(ours_crp) == tuple(theirs_crp)
