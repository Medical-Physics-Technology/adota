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
from typing import Sequence

import SimpleITK as sitk

from src.beamlets import AIR_HU

logger = logging.getLogger(__name__)

__all__ = ["rotate_ct_around_isocenter"]


def rotate_ct_around_isocenter(
    image: sitk.Image,
    angle_deg: float,
    isocenter_physical: Sequence[float],
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
        interpolator: SimpleITK interpolator (default linear).
        default_value: Fill value for out-of-bounds voxels (default air HU).

    Returns:
        A new SimpleITK image on the same grid (origin/spacing/size/direction)
        as ``image``, with the content rotated.
    """
    transform = sitk.Euler3DTransform()
    transform.SetCenter([float(c) for c in isocenter_physical])
    # Rotation about the z-axis (axial plane). The resampling transform maps
    # output points to input points, so the image content rotates by the
    # opposite sign of the transform's rotation; negating here makes a positive
    # angle_deg a counter-clockwise content rotation (pinned by the tests).
    transform.SetRotation(0.0, 0.0, math.radians(-angle_deg))

    logger.debug(
        "Rotating CT by %.3f deg around isocenter %s",
        angle_deg,
        tuple(round(float(c), 3) for c in isocenter_physical),
    )
    return sitk.Resample(
        image,
        image,
        transform,
        interpolator,
        float(default_value),
        image.GetPixelID(),
    )
