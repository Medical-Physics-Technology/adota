"""Isocenter coordinate handling: plan frame -> CT frame (x-flip convention).

The OpenTPS plan (``PlanPencil.txt`` isocenter, spot geometry) and the structure
masks use an **x-coordinate that is flipped relative to the CT / dose array**: a
plan isocenter at index ``x`` corresponds to CT index ``(Nx - 1) - x``. Verified
on LUNG1-193: the plan isocenter sits at x=189 while the GTV (and the MCsquare
high-dose region) is at x=310 = ``Nx-1-189``.

This was the source of the rotation pivot landing on the *mirror* of the true
target (suspect S3) -- the original datagenerator code applied the same flip via
``pivot_x = max_range_x - iso_x``. Every place that turns the plan isocenter into
a physical point (the rotation pivot / crop reference) or a CT index (mask
orientation) must apply this flip, so the rotation pivots on the true target and
the extracted sub-volume is the correct model input.

Only ``x`` is flipped here. The structure masks additionally need a y-flip, which
:mod:`src.beamlets.structures` detects automatically against the corrected
isocenter.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import SimpleITK as sitk

from src.loaders.plan_parser import Plan

__all__ = [
    "plan_isocenter_index_ct",
    "isocenter_physical",
    "isocenter_index_zyx",
]


def plan_isocenter_index_ct(
    plan_iso_xyz: Sequence[float], nx: int
) -> Tuple[float, float, float]:
    """Convert a plan isocenter ``(x, y, z)`` index to the CT frame (x flipped).

    Args:
        plan_iso_xyz: Plan isocenter as a continuous voxel index ``(x, y, z)``.
        nx: CT grid size along x.

    Returns:
        CT-frame ``(x, y, z)`` index with ``x -> (nx - 1) - x``.
    """
    return (
        (nx - 1) - float(plan_iso_xyz[0]),
        float(plan_iso_xyz[1]),
        float(plan_iso_xyz[2]),
    )


def isocenter_physical(
    plan_iso_xyz: Sequence[float], ct: sitk.Image
) -> Tuple[float, float, float]:
    """Physical point of the plan isocenter in the CT frame (x flipped).

    Args:
        plan_iso_xyz: Plan isocenter index ``(x, y, z)``.
        ct: The CT image (provides the grid for the index->physical transform).

    Returns:
        The isocenter physical point ``(x, y, z)`` -- the rotation pivot / crop
        reference.
    """
    idx = plan_isocenter_index_ct(plan_iso_xyz, ct.GetSize()[0])
    return ct.TransformContinuousIndexToPhysicalPoint(list(idx))


def isocenter_index_zyx(plan: Plan, ct: sitk.Image) -> Tuple[float, float, float]:
    """Plan isocenter as a CT-frame ``(z, y, x)`` index (x flipped).

    Used as the reference for structure-mask orientation.

    Args:
        plan: The parsed plan.
        ct: The CT image.

    Returns:
        ``(z, y, x)`` continuous voxel index in the CT frame.
    """
    iso = plan.fractions[0].fields[0].isocenter
    x, y, z = plan_isocenter_index_ct(iso, ct.GetSize()[0])
    return (z, y, x)
