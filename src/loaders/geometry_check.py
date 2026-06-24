"""Fast plan-geometry preflight: verify CT / structures / MC dose share a grid.

Reads only the image **headers** (size / spacing / origin / direction) via
``sitk.ImageFileReader.ReadImageInformation`` -- no pixel data -- so the whole
check is sub-millisecond. It catches an overwritten or foreign CT (a different
patient, a wrong export) *before* any ADoTA work runs, instead of surfacing later
as a confusing shape mismatch in the comparison / gamma stage.

Compatibility model
-------------------
The CT is the reference grid (ADoTA reconstructs the dose on it). The structure
masks are segmentations of that CT, and the MC dose is normally scored on it, so
all three must occupy the **same physical region**:

* **direction** matrices must match (same orientation);
* the physical bounding boxes must overlap almost entirely -- the smaller grid
  must sit (>= ``overlap_min``) inside the larger. A legitimately coarser or
  z-cropped dose/scoring grid is a sub-region of the CT and passes; a foreign CT
  is shifted/disjoint and fails (an ``error``).

A grid that co-registers but is **not identical** (different size/spacing/origin
within the same region) is a ``warn``: the figure/gamma compare arrays voxel-for-
voxel and would need a resample first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Tuple

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

__all__ = ["GridGeometry", "PlanGeometryError", "check_plan_geometry"]


class PlanGeometryError(ValueError):
    """Raised when the plan's CT / structures / dose grids are incompatible."""


@dataclass(frozen=True)
class GridGeometry:
    """A grid's physical metadata (sitk ``(x, y, z)`` order), header-only."""

    name: str
    size: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: Tuple[float, ...]  # row-major 3x3

    @classmethod
    def from_image(cls, name: str, img: sitk.Image) -> "GridGeometry":
        return cls(name, tuple(img.GetSize()), tuple(img.GetSpacing()),
                   tuple(img.GetOrigin()), tuple(img.GetDirection()))

    @classmethod
    def from_header(cls, name: str, path: Path) -> "GridGeometry":
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()  # header only -- no pixels loaded
        return cls(name, tuple(reader.GetSize()), tuple(reader.GetSpacing()),
                   tuple(reader.GetOrigin()), tuple(reader.GetDirection()))

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        """Axis-aligned physical bounding box from the eight grid corners."""
        d = np.array(self.direction, dtype=float).reshape(3, 3)
        origin = np.array(self.origin, dtype=float)
        spacing = np.array(self.spacing, dtype=float)
        corners = [
            origin + d @ (np.array(idx, dtype=float) * spacing)
            for idx in product(*[(0, s - 1) for s in self.size])
        ]
        corners = np.array(corners)
        return corners.min(axis=0), corners.max(axis=0)

    def fmt(self) -> str:
        sx, sy, sz = self.size
        sp = self.spacing
        og = self.origin
        return (
            f"{self.name:10s} {sx}x{sy}x{sz}  "
            f"sp({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})  "
            f"origin({og[0]:.1f},{og[1]:.1f},{og[2]:.1f})"
        )


def _overlap_fraction(a: GridGeometry, b: GridGeometry) -> float:
    """Fraction of the *smaller* grid's physical bbox volume inside the overlap."""
    amin, amax = a.bbox()
    bmin, bmax = b.bbox()
    inter = np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0.0, None).prod()
    smaller = min((amax - amin).prod(), (bmax - bmin).prod())
    return float(inter / smaller) if smaller > 0 else 0.0


def _compare(ct: GridGeometry, g: GridGeometry, *, overlap_min: float,
             spacing_tol: float, origin_tol: float) -> List[Tuple[str, str]]:
    """Classify ``g`` against the reference ``ct`` -> list of (severity, message)."""
    if not np.allclose(np.array(ct.direction), np.array(g.direction), atol=1e-3):
        return [("error", f"CT vs {g.name}: direction/orientation matrices differ")]
    overlap = _overlap_fraction(ct, g)
    if overlap < overlap_min:
        d = tuple(round(g.origin[i] - ct.origin[i], 1) for i in range(3))
        return [("error",
                 f"CT vs {g.name}: physical regions overlap only {overlap * 100:.0f}% "
                 f"(origin delta = {d} mm) -- different physical region; "
                 f"wrong or overwritten CT?")]
    identical = (
        ct.size == g.size
        and np.allclose(np.array(ct.spacing), np.array(g.spacing), atol=spacing_tol)
        and np.allclose(np.array(ct.origin), np.array(g.origin), atol=origin_tol)
    )
    if not identical:
        return [("warn",
                 f"CT vs {g.name}: co-registered but the grids are not identical "
                 f"({ct.size} vs {g.size}); the array comparison / gamma need the "
                 f"same grid (a resample would be required)")]
    return []


def check_plan_geometry(
    plan_directory,
    mode: str = "error",
    *,
    overlap_min: float = 0.9,
    spacing_tol: float = 1e-3,
    origin_tol: float = 1.0,
) -> List[Tuple[str, str]]:
    """Verify the CT, structure masks and MC dose share one physical grid.

    Logs a small geometry table (always) and the detected issues. Behaviour on a
    hard (``error``-severity) mismatch is controlled by ``mode``:

    * ``"error"`` (default): raise :class:`PlanGeometryError` -- aborts before any
      stage runs;
    * ``"warn"``: log it as a warning and continue;
    * ``"off"``: only log the geometry table, run no checks.

    Returns the list of ``(severity, message)`` issues found.
    """
    if mode not in ("error", "warn", "off"):
        raise ValueError(f"geometry_check mode must be error/warn/off, got {mode!r}")

    ct = GridGeometry.from_image("CT", plan_directory.ct)
    # De-duplicate structure grids (the masks usually all share one grid) so the
    # report shows the distinct grids, not one row per mask.
    seen, structs = set(), []
    for name, img in plan_directory.contours.items():
        g = GridGeometry.from_image(name, img)
        key = (g.size, g.spacing, tuple(round(o, 3) for o in g.origin), g.direction)
        if key not in seen:
            seen.add(key)
            label = "structures" if len(seen) == 1 else f"structures({name})"
            structs.append(GridGeometry(label, g.size, g.spacing, g.origin, g.direction))
    dose = (
        GridGeometry.from_header("Dose", plan_directory.mc_dose_path)
        if plan_directory.mc_dose_path is not None
        else None
    )

    table = [ct.fmt()] + [g.fmt() for g in structs] + ([dose.fmt()] if dose else [])
    logger.info("Plan geometry (header-only preflight):\n  %s", "\n  ".join(table))

    if mode == "off":
        return []

    issues: List[Tuple[str, str]] = []
    for g in structs + ([dose] if dose else []):
        issues += _compare(ct, g, overlap_min=overlap_min,
                            spacing_tol=spacing_tol, origin_tol=origin_tol)

    for sev, msg in issues:
        (logger.warning if sev == "warn" else logger.error)("Geometry %s: %s", sev, msg)

    errors = [m for sev, m in issues if sev == "error"]
    if errors:
        summary = "; ".join(errors)
        if mode == "error":
            raise PlanGeometryError(
                f"Plan geometry check failed for {Path(plan_directory.plan_dir).name}: "
                f"{summary}"
            )
        logger.warning("Plan geometry check found errors (mode=warn, continuing): %s",
                       summary)
    else:
        logger.info("Plan geometry check passed (CT / structures / dose co-registered).")
    return issues
