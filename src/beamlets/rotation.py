"""Per-field CT rotation around the isocenter (SimpleITK).

The gantry rotation is a rotation in the axial (``x``-``y``) plane around the
isocenter. We delegate it to SimpleITK's resampling with an
:class:`SimpleITK.Euler3DTransform` centred on the physical isocenter: SimpleITK
does the physical-to-index bookkeeping internally and correctly (handling origin,
spacing and direction), so the isocenter is an exact fixed point and the
out-of-bounds regions are filled with air by construction. This removes the
hand-rolled pixel-pivot maths that was the source of the rotation-pivot bug
(suspect S2).

Sign convention (pinned by the unit tests): a positive ``angle_deg`` rotates the
image **content counter-clockwise** in the ``(x, y)`` physical plane (x right,
y up) about the isocenter. The extraction rotates each field by
``A = (-1) * (gantry_angle - 90)`` degrees.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import SimpleITK as sitk

from src.beamlets import AIR_HU

logger = logging.getLogger(__name__)

__all__ = [
    "derotation_subgrid",
    "expanded_reference_grid",
    "rotate_ct_around_isocenter",
]


def _euler_z_transform(
    angle_deg: float, isocenter_physical: Sequence[float]
) -> sitk.Euler3DTransform:
    """The axial-plane resample transform used by :func:`rotate_ct_around_isocenter`.

    The resample transform maps *output* points to *input* points, so the image
    content rotates by the opposite sign of the transform's rotation; negating here
    makes a positive ``angle_deg`` a counter-clockwise content rotation.
    """
    transform = sitk.Euler3DTransform()
    transform.SetCenter([float(c) for c in isocenter_physical])
    transform.SetRotation(0.0, 0.0, math.radians(-angle_deg))
    return transform


def derotation_subgrid(
    source: sitk.Image,
    source_bounds_zyx: tuple,
    angle_deg: float,
    isocenter_physical: Sequence[float],
    reference: sitk.Image,
    margin: int = 1,
):
    """The part of ``reference`` that can sample a given region of ``source``.

    Resampling ``source`` onto ``reference`` is wasted work wherever the result is
    known to be zero. When the source is a dose deposit grid its non-zero support
    is a small fraction of the grid (roughly a tenth for a typical plan), and every
    output voxel whose sample point lies outside that support has all eight of its
    trilinear neighbours equal to zero, so it resamples to **exactly** zero. This
    returns the output sub-grid that covers the support, so the caller can resample
    only there and leave the rest of the output at zero.

    Nothing is dropped: the skipped region is exactly zero, not merely negligible.
    The values inside the sub-grid can still differ from a full-grid resample in the
    last bit, because the sub-grid carries its own origin and ITK then evaluates
    ``(origin + spacing * lo) + spacing * i`` where the full grid evaluates
    ``origin + spacing * (lo + i)``; the reassociation moves a sample coordinate by
    an ulp and with it an interpolation weight. On a clinical plan this reached 2
    ulp on 0.003% of voxels (1e-5% of Dmax), with the dose integral unchanged to
    twelve digits.

    The ``margin`` of one source voxel is what makes that exact: a sample point can
    only pick up a non-zero voxel if it lies within one voxel of the support, so
    growing the source region by one voxel before mapping it covers every output
    voxel that can be non-zero. One further voxel is added in output index space to
    absorb rounding.

    Args:
        source: The image being resampled from (the rotated-frame deposit grid).
        source_bounds_zyx: ``((z_lo, y_lo, x_lo), (z_hi, y_hi, x_hi))`` inclusive
            NumPy index bounds of the region of interest within ``source``.
        angle_deg: The rotation passed to :func:`rotate_ct_around_isocenter`.
        isocenter_physical: The physical rotation centre ``(x, y, z)``.
        reference: The full output grid (the CT).
        margin: Source voxels of interpolation support to include (1 is exact for
            trilinear; a higher-order interpolator needs a wider margin).

    Returns:
        ``(sub_reference, numpy_slices)`` where ``sub_reference`` is an empty image
        on ``reference``'s grid covering the mapped region and ``numpy_slices`` is
        the matching ``(z, y, x)`` slice tuple into a ``reference``-shaped array.
        ``(None, None)`` if the mapped region misses ``reference`` entirely.
    """
    lo_zyx, hi_zyx = source_bounds_zyx
    src_nx, src_ny, src_nz = source.GetSize()
    src_size_zyx = (src_nz, src_ny, src_nx)
    lo = [max(0, int(lo_zyx[k]) - margin) for k in range(3)]
    hi = [min(src_size_zyx[k] - 1, int(hi_zyx[k]) + margin) for k in range(3)]

    # source physical -> reference physical is the inverse of the resample map.
    inverse = _euler_z_transform(angle_deg, isocenter_physical).GetInverse()
    corners = []
    for iz in (lo[0], hi[0]):
        for iy in (lo[1], hi[1]):
            for ix in (lo[2], hi[2]):
                point = source.TransformIndexToPhysicalPoint((ix, iy, iz))
                corners.append(
                    reference.TransformPhysicalPointToContinuousIndex(
                        inverse.TransformPoint(point)
                    )
                )

    ref_size = reference.GetSize()  # (x, y, z)
    out_lo, out_hi = [], []
    for k in range(3):
        values = [c[k] for c in corners]
        low = max(0, int(math.floor(min(values))) - 1)
        high = min(ref_size[k] - 1, int(math.ceil(max(values))) + 1)
        if low > high:
            return None, None
        out_lo.append(low)
        out_hi.append(high)

    sub = sitk.Image(
        out_hi[0] - out_lo[0] + 1,
        out_hi[1] - out_lo[1] + 1,
        out_hi[2] - out_lo[2] + 1,
        reference.GetPixelID(),
    )
    sub.SetSpacing(reference.GetSpacing())
    sub.SetDirection(reference.GetDirection())
    sub.SetOrigin(reference.TransformIndexToPhysicalPoint(tuple(out_lo)))
    slices = (
        slice(out_lo[2], out_hi[2] + 1),  # z
        slice(out_lo[1], out_hi[1] + 1),  # y
        slice(out_lo[0], out_hi[0] + 1),  # x
    )
    return sub, slices


def expanded_reference_grid(
    image: sitk.Image,
    angle_deg: float,
    isocenter_physical: Sequence[float],
    out_spacing_factor: int = 1,
) -> sitk.Image:
    """Build an empty grid that fully contains ``image`` rotated about the isocenter.

    Rotating around an off-centre isocenter with a fixed grid clips the content
    that leaves the grid. This returns a reference grid (same direction,
    larger/shifted in the axial ``x``-``y`` plane) sized to the bounding box of the
    rotated input corners, so a subsequent resample loses no information.

    Args:
        image: The image to be rotated.
        angle_deg: The (content) rotation angle in degrees.
        isocenter_physical: The physical rotation centre ``(x, y, z)``.
        out_spacing_factor: Output voxel size multiplier (default ``1`` = the input
            spacing, the original behaviour). ``2`` builds a 2x-coarser grid on all
            three axes (the field-level down-sample); the rotate resample then
            rotates **and** down-samples in one pass.

    Returns:
        An empty :class:`SimpleITK.Image` to use as the resample reference.
    """
    nx, ny, nz = image.GetSize()
    sx, sy, _sz = image.GetSpacing()
    iso_x, iso_y = float(isocenter_physical[0]), float(isocenter_physical[1])
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Forward-map (input -> output) the four axial corners: rotate by +angle
    # about the isocenter (the inverse of the output->input resample transform).
    xs, ys = [], []
    for ix, iy in ((0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)):
        px, py, _pz = image.TransformIndexToPhysicalPoint((ix, iy, 0))
        dx, dy = px - iso_x, py - iso_y
        xs.append(iso_x + dx * cos_t - dy * sin_t)
        ys.append(iso_y + dx * sin_t + dy * cos_t)

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if out_spacing_factor == 1:
        # Original behaviour, kept byte-identical (z size/spacing unchanged).
        out_nx = int(math.ceil((max_x - min_x) / sx)) + 1
        out_ny = int(math.ceil((max_y - min_y) / sy)) + 1
        reference = sitk.Image(out_nx, out_ny, nz, image.GetPixelID())
        reference.SetSpacing(image.GetSpacing())
        reference.SetDirection(image.GetDirection())
        reference.SetOrigin((min_x, min_y, image.GetOrigin()[2]))
        return reference

    f = int(out_spacing_factor)
    in_spacing = image.GetSpacing()
    out_spacing = tuple(s * f for s in in_spacing)
    out_nx = int(math.ceil((max_x - min_x) / out_spacing[0])) + 1
    out_ny = int(math.ceil((max_y - min_y) / out_spacing[1])) + 1
    out_nz = (nz + f - 1) // f
    # Cell-center alignment: a factor-f trilinear down-sample with
    # ``align_corners=False`` samples coarse voxel j at fine position
    # ``f*j + (f-1)/2``, i.e. the fine cell-centres. Offsetting the origin by
    # ``+(f-1)/2 * fine_spacing`` on every axis makes the coarse voxel centres
    # land on those cell-centres, so the model -- trained on the
    # ``align_corners=False`` down-sample -- sees no sub-voxel shift. (The half-
    # voxel air sliver this trims at the grid corners is outside the patient.)
    half = tuple((f - 1) / 2.0 * s for s in in_spacing)
    reference = sitk.Image(out_nx, out_ny, out_nz, image.GetPixelID())
    reference.SetSpacing(out_spacing)
    reference.SetDirection(image.GetDirection())
    reference.SetOrigin(
        (min_x + half[0], min_y + half[1], image.GetOrigin()[2] + half[2])
    )
    return reference


def rotate_ct_around_isocenter(
    image: sitk.Image,
    angle_deg: float,
    isocenter_physical: Sequence[float],
    reference: Optional[sitk.Image] = None,
    expand: bool = False,
    interpolator: int = sitk.sitkLinear,
    default_value: float = float(AIR_HU),
    out_spacing_factor: int = 1,
) -> sitk.Image:
    """Rotate a CT image in the axial plane around the physical isocenter.

    Args:
        image: The CT grid as a SimpleITK image.
        angle_deg: Rotation angle in degrees. Positive rotates the content
            counter-clockwise in the ``(x, y)`` plane about the isocenter.
        isocenter_physical: Physical point ``(x, y, z)`` to rotate around (the
            field isocenter), in the image's physical frame.
        reference: Output grid to resample into. When given, the result lies on
            this grid (used to de-rotate back into the original CT grid). When
            ``None``, see ``expand``.
        expand: When ``True`` (and ``reference`` is ``None``), resample into a
            grid sized to contain the full rotated content (no clipping). When
            ``False`` (default), use the input grid -- the original fixed-size
            behaviour.
        interpolator: SimpleITK interpolator (default linear).
        default_value: Fill value for out-of-bounds voxels (default air HU).
        out_spacing_factor: Output voxel-size multiplier for the expanded grid
            (default ``1`` = input spacing, unchanged). ``2`` rotates into a
            2x-coarser (2mm) grid -- the field-level down-sample. Only used when
            ``reference is None and expand``.

    Returns:
        A new SimpleITK image with the content rotated, on ``reference`` (if
        given), else the expanded grid (if ``expand``), else the input grid.
    """
    # Rotation about the z-axis (axial plane); see :func:`_euler_z_transform` for
    # the sign convention (pinned by the tests).
    transform = _euler_z_transform(angle_deg, isocenter_physical)

    if reference is None:
        reference = (
            expanded_reference_grid(
                image, angle_deg, isocenter_physical, out_spacing_factor
            )
            if expand
            else image
        )

    logger.debug(
        "Rotating CT by %.3f deg around isocenter %s (out size %s)",
        angle_deg,
        tuple(round(float(c), 3) for c in isocenter_physical),
        reference.GetSize(),
    )
    return sitk.Resample(
        image,
        reference,
        transform,
        interpolator,
        float(default_value),
        image.GetPixelID(),
    )
