"""Ground-truth tests for :mod:`src.beamlets.rotation`.

These pin the rotation contract on a synthetic phantom with non-trivial
origin/spacing: the isocenter is an exact fixed point, 0 deg is the identity,
and a positive angle rotates content counter-clockwise in the ``(x, y)`` plane.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from src.beamlets.rotation import rotate_ct_around_isocenter

# Phantom geometry: non-zero origin, non-unit z spacing.
SIZE_XYZ = (21, 21, 3)
ORIGIN = (-10.0, -20.0, 5.0)
SPACING = (1.0, 1.0, 2.0)
ISO_INDEX = (10.0, 10.0, 1.0)  # continuous voxel index of the isocenter


def _phantom_with_marker(marker_index_xyz: tuple[int, int, int]) -> sitk.Image:
    """Build an air phantom with a single bright marker voxel."""
    nx, ny, nz = SIZE_XYZ
    arr = np.full((nz, ny, nx), -1024.0, dtype=np.float32)  # (z, y, x)
    mx, my, mz = marker_index_xyz
    arr[mz, my, mx] = 1000.0
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(ORIGIN)
    image.SetSpacing(SPACING)
    return image


def _isocenter_physical(image: sitk.Image) -> tuple[float, float, float]:
    return image.TransformContinuousIndexToPhysicalPoint(ISO_INDEX)


def _argmax_index_xyz(image: sitk.Image) -> tuple[int, int, int]:
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    z, y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    return (int(x), int(y), int(z))


def test_identity_at_zero_degrees() -> None:
    """A 0 deg rotation returns the input content unchanged."""
    image = _phantom_with_marker((15, 10, 1))
    rotated = rotate_ct_around_isocenter(image, 0.0, _isocenter_physical(image))
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(rotated), sitk.GetArrayFromImage(image), atol=1e-3
    )


def test_isocenter_voxel_is_fixed() -> None:
    """A marker placed at the isocenter stays at the isocenter under rotation."""
    image = _phantom_with_marker((10, 10, 1))  # at ISO_INDEX
    rotated = rotate_ct_around_isocenter(image, 37.0, _isocenter_physical(image))
    assert _argmax_index_xyz(rotated) == (10, 10, 1)


def test_forward_then_backward_is_identity() -> None:
    """rotate(A) then rotate(-A) around the same isocenter returns the input.

    This is the round-trip guarantee the extract -> accumulate pipeline relies on.
    """
    # Use a smooth (band-limited) field; random noise is pathological for
    # double linear interpolation and not representative of CT/dose data.
    zz, yy, xx = np.meshgrid(
        np.arange(SIZE_XYZ[2]), np.arange(SIZE_XYZ[1]), np.arange(SIZE_XYZ[0]), indexing="ij"
    )
    arr = (yy * 3.0 + xx * 2.0).astype(np.float32)
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(ORIGIN)
    image.SetSpacing(SPACING)
    iso = _isocenter_physical(image)

    rotated = rotate_ct_around_isocenter(image, 30.0, iso, default_value=0.0)
    restored = rotate_ct_around_isocenter(rotated, -30.0, iso, default_value=0.0)

    # The border ring leaves the grid under rotation; compare the interior.
    a = sitk.GetArrayFromImage(restored)[:, 4:-4, 4:-4]
    b = arr[:, 4:-4, 4:-4]
    assert float(np.abs(a - b).max()) < 1.0


def test_positive_angle_is_counter_clockwise() -> None:
    """+90 deg sends a +x marker to +y about the isocenter (CCW in x-y)."""
    # Marker 5 voxels along +x from the isocenter -> physical offset (+5, 0).
    image = _phantom_with_marker((15, 10, 1))
    rotated = rotate_ct_around_isocenter(image, 90.0, _isocenter_physical(image))
    # CCW 90: (+5, 0) -> (0, +5) -> index (10, 15).
    assert _argmax_index_xyz(rotated) == (10, 15, 1)


def test_180_degrees_flips_through_isocenter() -> None:
    """180 deg maps a +x marker to the symmetric -x position about the isocenter."""
    image = _phantom_with_marker((15, 10, 1))
    rotated = rotate_ct_around_isocenter(image, 180.0, _isocenter_physical(image))
    # (+5, 0) -> (-5, 0) -> index (5, 10).
    assert _argmax_index_xyz(rotated) == (5, 10, 1)


def test_oob_filled_with_air() -> None:
    """Rotation introduces no values below the air fill."""
    image = _phantom_with_marker((15, 10, 1))
    rotated = rotate_ct_around_isocenter(image, 45.0, _isocenter_physical(image))
    assert float(sitk.GetArrayFromImage(rotated).min()) >= -1024.0 - 1e-3
