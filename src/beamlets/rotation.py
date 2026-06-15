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

__all__ = ["expanded_reference_grid", "rotate_ct_around_isocenter"]


def expanded_reference_grid(
    image: sitk.Image,
    angle_deg: float,
    isocenter_physical: Sequence[float],
) -> sitk.Image:
    """Build an empty grid that fully contains ``image`` rotated about the isocenter.

    Rotating around an off-centre isocenter with a fixed grid clips the content
    that leaves the grid. This returns a reference grid (same spacing/direction,
    larger/shifted in the axial ``x``-``y`` plane, ``z`` unchanged) sized to the
    bounding box of the rotated input corners, so a subsequent resample loses no
    information.

    Args:
        image: The image to be rotated.
        angle_deg: The (content) rotation angle in degrees.
        isocenter_physical: The physical rotation centre ``(x, y, z)``.

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
    out_nx = int(math.ceil((max_x - min_x) / sx)) + 1
    out_ny = int(math.ceil((max_y - min_y) / sy)) + 1

    reference = sitk.Image(out_nx, out_ny, nz, image.GetPixelID())
    reference.SetSpacing(image.GetSpacing())
    reference.SetDirection(image.GetDirection())
    reference.SetOrigin((min_x, min_y, image.GetOrigin()[2]))
    return reference


def rotate_ct_around_isocenter(
    image: sitk.Image,
    angle_deg: float,
    isocenter_physical: Sequence[float],
    reference: Optional[sitk.Image] = None,
    expand: bool = False,
    interpolator: int = sitk.sitkLinear,
    default_value: float = float(AIR_HU),
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

    Returns:
        A new SimpleITK image with the content rotated, on ``reference`` (if
        given), else the expanded grid (if ``expand``), else the input grid.
    """
    transform = sitk.Euler3DTransform()
    transform.SetCenter([float(c) for c in isocenter_physical])
    # Rotation about the z-axis (axial plane). The resampling transform maps
    # output points to input points, so the image content rotates by the
    # opposite sign of the transform's rotation; negating here makes a positive
    # angle_deg a counter-clockwise content rotation (pinned by the tests).
    transform.SetRotation(0.0, 0.0, math.radians(-angle_deg))

    if reference is None:
        reference = (
            expanded_reference_grid(image, angle_deg, isocenter_physical)
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
