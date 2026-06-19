"""Ground-truth tests for :mod:`src.beamlets.rotation`.

These pin the rotation contract on a synthetic phantom with non-trivial
origin/spacing: the isocenter is an exact fixed point, 0 deg is the identity,
and a positive angle rotates content counter-clockwise in the ``(x, y)`` plane.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from src.beamlets.rotation import (
    expanded_reference_grid,
    rotate_ct_around_isocenter,
)

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


# --- Phase 1: expanded-grid rotation (no clipping) -------------------------

# Off-centre isocenter on a 21x21 grid, so a rotation clips the fixed grid.
_OFF_ORIGIN = (0.0, 0.0, 0.0)
_OFF_SPACING = (1.0, 1.0, 1.0)
_OFF_ISO_INDEX = (5.0, 5.0, 1.0)


def _corner_phantom() -> sitk.Image:
    """21x21x3 air phantom with a bright marker at each axial corner (z=1)."""
    arr = np.full((3, 21, 21), -1024.0, dtype=np.float32)  # (z, y, x)
    for ix, iy in ((0, 0), (20, 0), (0, 20), (20, 20)):
        arr[1, iy, ix] = 1000.0
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(_OFF_ORIGIN)
    image.SetSpacing(_OFF_SPACING)
    return image


def _off_iso(image: sitk.Image) -> tuple[float, float, float]:
    return image.TransformContinuousIndexToPhysicalPoint(_OFF_ISO_INDEX)


def test_expanded_rotation_preserves_all_corners() -> None:
    """Expansion keeps every corner marker; the fixed grid clips some."""
    image = _corner_phantom()
    iso = _off_iso(image)
    nn = sitk.sitkNearestNeighbor

    fixed = rotate_ct_around_isocenter(image, 40.0, iso, interpolator=nn)
    expanded = rotate_ct_around_isocenter(image, 40.0, iso, expand=True, interpolator=nn)

    n_fixed = int((sitk.GetArrayFromImage(fixed) > 500).sum())
    n_expanded = int((sitk.GetArrayFromImage(expanded) > 500).sum())
    assert n_expanded == 4  # nothing lost
    assert n_fixed < 4  # the fixed grid clips at least one corner


def test_expanded_identity_at_zero() -> None:
    """At 0 deg the expanded grid matches the input grid and content."""
    image = _corner_phantom()
    expanded = rotate_ct_around_isocenter(image, 0.0, _off_iso(image), expand=True)
    assert expanded.GetSize() == image.GetSize()
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(expanded), sitk.GetArrayFromImage(image), atol=1e-3
    )


def test_expanded_grid_grows_for_oblique() -> None:
    """The expanded grid is larger for an oblique angle, unchanged at 0 deg."""
    image = _corner_phantom()
    iso = _off_iso(image)
    same = expanded_reference_grid(image, 0.0, iso)
    oblique = expanded_reference_grid(image, 45.0, iso)
    assert same.GetSize()[:2] == image.GetSize()[:2]
    assert oblique.GetSize()[0] > image.GetSize()[0]
    assert oblique.GetSize()[1] > image.GetSize()[1]
    assert oblique.GetSize()[2] == image.GetSize()[2]  # z unchanged


def test_expanded_isocenter_is_fixed_point() -> None:
    """A marker at the isocenter stays at the isocenter physical point."""
    arr = np.full((3, 21, 21), -1024.0, dtype=np.float32)
    arr[1, 5, 5] = 1000.0  # at the off-centre isocenter index (x=5, y=5, z=1)
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(_OFF_ORIGIN)
    image.SetSpacing(_OFF_SPACING)
    iso = _off_iso(image)

    expanded = rotate_ct_around_isocenter(
        image, 37.0, iso, expand=True, interpolator=sitk.sitkNearestNeighbor
    )
    arr_e = sitk.GetArrayFromImage(expanded)
    z, y, x = np.unravel_index(int(arr_e.argmax()), arr_e.shape)
    phys = expanded.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
    np.testing.assert_allclose(phys, iso, atol=1.0)


def test_derotate_into_original_grid_recovers_input() -> None:
    """rotate(expand) then de-rotate into the original grid recovers the input."""
    zz, yy, xx = np.meshgrid(np.arange(3), np.arange(21), np.arange(21), indexing="ij")
    smooth = (yy * 2.0 + xx * 3.0).astype(np.float32)  # band-limited
    image = sitk.GetImageFromArray(smooth)
    image.SetOrigin(_OFF_ORIGIN)
    image.SetSpacing(_OFF_SPACING)
    iso = _off_iso(image)

    expanded = rotate_ct_around_isocenter(image, 40.0, iso, expand=True, default_value=0.0)
    # De-rotate into the *original* grid (the accumulation pattern).
    restored = rotate_ct_around_isocenter(
        expanded, -40.0, iso, reference=image, default_value=0.0
    )
    a = sitk.GetArrayFromImage(restored)[:, 4:-4, 4:-4]
    b = smooth[:, 4:-4, 4:-4]
    assert float(np.abs(a - b).max()) < 1.0


# --- P1: field-level coarse (2mm) rotation grid (out_spacing_factor=2) -------


def test_coarse_grid_doubles_spacing_and_halves_size() -> None:
    """out_spacing_factor=2 builds a 2x-coarser grid on all three axes."""
    image = _corner_phantom()  # 21x21x3, 1mm spacing
    iso = _off_iso(image)
    fine = expanded_reference_grid(image, 30.0, iso)  # factor 1 (default)
    coarse = expanded_reference_grid(image, 30.0, iso, out_spacing_factor=2)

    assert coarse.GetSpacing() == tuple(2.0 * s for s in image.GetSpacing())
    # Half the voxels per axis (within +/-1 from the ceil), z halved too.
    for c, f in zip(coarse.GetSize(), fine.GetSize()):
        assert c <= (f + 1) // 2 + 1
    assert coarse.GetSize()[2] == (image.GetSize()[2] + 1) // 2
    # Cell-center alignment: origin offset by +(f-1)/2 * fine_spacing (= +0.5mm
    # at f=2) from the factor-1 grid, so the coarse voxels land on the 1mm
    # cell-centres (matching the align_corners=False down-sample convention).
    sx, sy, _ = image.GetSpacing()
    np.testing.assert_allclose(
        coarse.GetOrigin()[:2],
        (fine.GetOrigin()[0] + 0.5 * sx, fine.GetOrigin()[1] + 0.5 * sy),
        atol=1e-9,
    )


def test_coarse_isocenter_is_fixed_point() -> None:
    """Rotating to the 2mm grid keeps content at the isocenter at the isocenter."""
    arr = np.full((3, 21, 21), -1024.0, dtype=np.float32)
    arr[:, 4:7, 4:7] = 1000.0  # 3x3 (all-z) blob centred on the isocenter index
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin(_OFF_ORIGIN)
    image.SetSpacing(_OFF_SPACING)
    iso = _off_iso(image)

    coarse = rotate_ct_around_isocenter(
        image, 37.0, iso, expand=True, out_spacing_factor=2, default_value=-1024.0
    )
    arr_c = sitk.GetArrayFromImage(coarse)
    z, y, x = np.unravel_index(int(arr_c.argmax()), arr_c.shape)
    phys = coarse.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
    # Peak stays within one 2mm voxel of the isocenter physical point.
    np.testing.assert_allclose(phys[:2], iso[:2], atol=2.0)


def test_coarse_rotation_no_clipping() -> None:
    """The 2mm expanded grid's physical bbox contains every rotated corner."""
    image = _corner_phantom()
    iso = _off_iso(image)
    coarse = expanded_reference_grid(image, 40.0, iso, out_spacing_factor=2)
    nx, ny, _nz = image.GetSize()
    ix0, iy0, iz0 = float(iso[0]), float(iso[1]), float(iso[2])
    theta = np.radians(40.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    sx, sy, _ = coarse.GetSize()
    for cx, cy in ((0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)):
        px, py, _pz = image.TransformIndexToPhysicalPoint((cx, cy, 0))
        rx = ix0 + (px - ix0) * cos_t - (py - iy0) * sin_t
        ry = iy0 + (px - ix0) * sin_t + (py - iy0) * cos_t
        idx = coarse.TransformPhysicalPointToContinuousIndex((rx, ry, iz0))
        assert -0.5 <= idx[0] <= sx - 0.5  # corner lands inside the coarse grid
        assert -0.5 <= idx[1] <= sy - 0.5


def test_coarse_derotate_upsample_recovers_smooth_field() -> None:
    """rotate->2mm then de-rotate->1mm (reference=ct) recovers a smooth field.

    Uses a depth-thick phantom: the +0.5mm cell-center offset is negligible for a
    realistic CT but would clip a razor-thin (few-slice) phantom at the z edges.
    """
    zz, yy, xx = np.meshgrid(np.arange(16), np.arange(21), np.arange(21), indexing="ij")
    smooth = (yy * 2.0 + xx * 3.0 + zz * 1.0).astype(np.float32)  # band-limited
    image = sitk.GetImageFromArray(smooth)
    image.SetOrigin(_OFF_ORIGIN)
    image.SetSpacing(_OFF_SPACING)
    iso = image.TransformContinuousIndexToPhysicalPoint((5.0, 5.0, 8.0))

    coarse = rotate_ct_around_isocenter(
        image, 40.0, iso, expand=True, out_spacing_factor=2, default_value=0.0
    )
    # De-rotate + up-sample back to the original 1mm grid in one resample.
    restored = rotate_ct_around_isocenter(
        coarse, -40.0, iso, reference=image, default_value=0.0
    )
    a = sitk.GetArrayFromImage(restored)[3:-3, 5:-5, 5:-5]
    b = smooth[3:-3, 5:-5, 5:-5]
    # Larger tolerance than the 1mm round-trip: 2mm sampling loses resolution.
    assert float(np.abs(a - b).max()) < 4.0
