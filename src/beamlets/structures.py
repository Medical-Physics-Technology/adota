"""Load structure masks correctly oriented onto the dose/CT voxel grid.

The OpenTPS-exported structure masks (``target.mhd`` / ``OAR_*.mhd``) share the
CT voxel grid (same size) but carry an unusable physical origin, so SimpleITK
physical resampling misplaces them. They are also stored with one or more axes
flipped relative to the CT. Rather than hardcode a flip, we **auto-detect** it
per plan: among the eight axis-flip orientations (which preserve the shape), pick
the one that places the *target* centroid nearest the plan isocenter, then apply
that same flip to every structure. The target is, by construction, centred on
the isocenter, so this is a robust, plan-independent signal.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from src.loaders.plan_directory import PlanDirectory
from src.loaders.plan_parser import Plan

logger = logging.getLogger(__name__)

__all__ = [
    "isocenter_index_zyx",
    "apply_flip",
    "detect_target_flip",
    "load_oriented_structures",
]

# Above this centroid-to-isocenter distance (voxels) the best orientation is
# still used but flagged as unreliable.
_ISO_DIST_WARN = 40.0

Flip = Tuple[bool, bool, bool]


def isocenter_index_zyx(plan: Plan) -> Tuple[float, float, float]:
    """Return the plan isocenter as a continuous voxel index in ``(z, y, x)``.

    The plan isocenter is stored as a continuous voxel index in ``(x, y, z)``.
    """
    iso = plan.fractions[0].fields[0].isocenter
    return (iso[2], iso[1], iso[0])


def apply_flip(mask: np.ndarray, flips: Flip) -> np.ndarray:
    """Flip a ``(z, y, x)`` mask along the axes selected in ``flips``."""
    slicer = tuple(slice(None, None, -1) if f else slice(None) for f in flips)
    return mask[slicer]


def detect_target_flip(
    target_mask: np.ndarray,
    iso_zyx: Tuple[float, float, float],
) -> Tuple[Flip, float]:
    """Pick the axis-flip placing the target centroid nearest the isocenter.

    Args:
        target_mask: Boolean target mask ``(z, y, x)``.
        iso_zyx: Isocenter voxel index ``(z, y, x)``.

    Returns:
        Tuple ``(flips, distance)`` -- the best ``(flip_z, flip_y, flip_x)`` and
        the centroid-to-isocenter distance in voxels.
    """
    iso = np.asarray(iso_zyx, dtype=float)
    best: Optional[Flip] = None
    best_dist = float("inf")
    for flips in itertools.product((False, True), repeat=3):
        idx = np.argwhere(apply_flip(target_mask, flips))
        if idx.size == 0:
            continue
        dist = float(np.linalg.norm(idx.mean(axis=0) - iso))
        if dist < best_dist:
            best_dist = dist
            best = flips
    if best is None:
        raise ValueError("Target mask is empty; cannot detect orientation")
    return best, best_dist


def load_oriented_structures(
    plan_directory: PlanDirectory,
    target_keyword: str = "target",
) -> Tuple[Dict[str, np.ndarray], Flip]:
    """Load the plan's structure masks, oriented onto the CT/dose grid.

    Args:
        plan_directory: The loaded plan directory (contours + plan).
        target_keyword: Substring identifying the target contour (case-insensitive).

    Returns:
        Tuple ``(masks, flips)`` -- a mapping ``name -> boolean (z, y, x) mask``
        (all oriented) and the detected ``(flip_z, flip_y, flip_x)``.

    Raises:
        ValueError: If there are no contours or no target contour is found.
    """
    if not plan_directory.contours:
        raise ValueError("Plan directory has no structure contours")

    raw = {
        name: sitk.GetArrayFromImage(image).astype(bool)
        for name, image in plan_directory.contours.items()
    }
    target_name = next(
        (n for n in raw if target_keyword.lower() in n.lower()), None
    )
    if target_name is None:
        raise ValueError(
            f"No target contour (containing {target_keyword!r}) among {list(raw)}"
        )

    iso = isocenter_index_zyx(plan_directory.plan)
    flips, dist = detect_target_flip(raw[target_name], iso)
    if dist > _ISO_DIST_WARN:
        logger.warning(
            "Detected structure flip %s places the target %.1f voxels from the "
            "isocenter (> %.0f); orientation may be unreliable.",
            flips,
            dist,
            _ISO_DIST_WARN,
        )
    else:
        logger.info(
            "Structure orientation: flip(z,y,x)=%s (target %.1f voxels from isocenter)",
            flips,
            dist,
        )

    oriented = {name: apply_flip(mask, flips) for name, mask in raw.items()}
    return oriented, flips
