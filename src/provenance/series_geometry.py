"""Which files make up a CT series, and whether they stack into a valid volume.

A series directory on disk is not a series. It can hold files of another series
(a second reconstruction, a localizer), objects that are not CT images, or
unreadable files; and the files of one series can still fail to form a volume
(missing slices, two acquisitions under one UID, a tilted or shifted stack).
The SimpleITK reader trusts the directory: it stacks what it gets at the spacing
of the first two slices, so any of these gives a geometrically wrong CT without
an error.

This module reads every header of a directory (no pixel data), groups the files
by the ``SeriesInstanceUID`` each file carries itself, keeps CT image objects
only, and checks the geometry of each group:

- one orientation and one in-plane grid (rows, columns, pixel spacing);
- every slice at the same in-plane position (the slice origins projected on the
  row and column directions agree), so the stack is not sheared;
- distinct positions along the slice normal (no duplicates);
- an even step: no step larger than ``max_gap_ratio`` times the median step
  (a missing slice), none smaller than the median over that ratio.

The result lists the files in stacking order, which is what a reader is given.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
_TAGS = ["SOPClassUID", "SeriesInstanceUID", "Modality", "ImageType", "ImagePositionPatient",
         "ImageOrientationPatient", "PixelSpacing", "Rows", "Columns"]


@dataclass(frozen=True)
class SliceHeader:
    """The geometry of one CT image file."""

    path: str
    series_uid: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, ...]
    pixel_spacing: Tuple[float, float]
    rows: int
    columns: int


@dataclass
class DirectoryContents:
    """The CT slices of one directory per series, and what was set aside."""

    series: Dict[str, List[SliceHeader]] = field(default_factory=dict)
    skipped: Dict[str, int] = field(default_factory=dict)   # reason -> number of files


@dataclass
class SeriesGeometry:
    """The checked geometry of one series; ``reasons`` is empty when it stacks."""

    series_uid: str
    files: Tuple[str, ...]
    n_slices: int
    step_mm: float
    max_step_mm: float
    min_step_mm: float
    origin_mm: Tuple[float, float, float]       # position of the first slice in stacking order
    z_ends_mm: Tuple[float, float]               # lowest and highest slice z
    pixel_spacing_mm: Tuple[float, float]        # DICOM order: between rows (y), between columns (x)
    rows: int
    columns: int
    reasons: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def z_extent_mm(self) -> float:
        return self.n_slices * self.step_mm


def _floats(value, n: int) -> Optional[Tuple[float, ...]]:
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return out if len(out) == n else None


def read_directory(series_dir: str) -> DirectoryContents:
    """Every file of ``series_dir`` (not recursive), grouped by its own series UID."""
    out = DirectoryContents()
    groups: Dict[str, List[SliceHeader]] = defaultdict(list)

    def skip(reason: str) -> None:
        out.skipped[reason] = out.skipped.get(reason, 0) + 1

    for name in sorted(os.listdir(series_dir)):
        path = os.path.join(series_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=_TAGS)
        except (InvalidDicomError, OSError, ValueError):
            skip("not readable as DICOM")
            continue
        if str(getattr(ds, "SOPClassUID", "")) != CT_IMAGE_STORAGE or getattr(ds, "Modality", None) != "CT":
            skip("not a CT image")
            continue
        image_type = [str(t).upper() for t in (getattr(ds, "ImageType", None) or [])]
        if "LOCALIZER" in image_type:
            skip("localizer")
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        position = _floats(getattr(ds, "ImagePositionPatient", None), 3)
        orientation = _floats(getattr(ds, "ImageOrientationPatient", None), 6)
        spacing = _floats(getattr(ds, "PixelSpacing", None), 2)
        if not uid or position is None or orientation is None or spacing is None:
            skip("geometry tags missing")
            continue
        groups[uid].append(SliceHeader(path, uid, position, orientation, spacing,
                                       int(getattr(ds, "Rows", 0) or 0), int(getattr(ds, "Columns", 0) or 0)))
    out.series = dict(groups)
    return out


def check_geometry(slices: Sequence[SliceHeader], max_gap_ratio: float = 1.5,
                   orientation_tol: float = 1e-3, duplicate_tol_mm: float = 0.01) -> SeriesGeometry:
    """Stack ``slices`` along their normal and check that they form a volume."""
    if not slices:
        raise ValueError("no slices")
    reasons: List[str] = []
    orient = np.array([s.orientation for s in slices])
    if np.abs(orient - orient[0]).max() > orientation_tol:
        reasons.append("orientation varies")
    if len({(s.rows, s.columns, s.pixel_spacing) for s in slices}) > 1:
        reasons.append("in-plane grid varies")
    row_dir, col_dir = orient[0, :3], orient[0, 3:]
    normal = np.cross(row_dir, col_dir)
    pos = np.array([s.position for s in slices])
    order = np.argsort(pos @ normal, kind="stable")
    pos = pos[order]
    depth = pos @ normal
    in_plane = np.stack([pos @ row_dir, pos @ col_dir], axis=1)
    pixel = float(min(slices[0].pixel_spacing))
    if np.abs(in_plane - in_plane[0]).max() > 0.5 * pixel:
        reasons.append("slices shifted in plane (tilt or table move)")
    steps = np.diff(depth)
    if steps.size:
        n_dup = int((steps <= duplicate_tol_mm).sum())
        if n_dup:
            reasons.append(f"{n_dup} duplicate slice positions")
        real = steps[steps > duplicate_tol_mm]
        step = float(np.median(real)) if real.size else 0.0
        if real.size and real.max() > max_gap_ratio * step:
            reasons.append(f"slice gap {real.max():.2f} mm > {max_gap_ratio} x step {step:.2f} mm")
        if real.size and real.min() < step / max_gap_ratio:
            reasons.append(f"uneven step {real.min():.2f} mm < step {step:.2f} mm / {max_gap_ratio}")
        max_step, min_step = float(steps.max()), float(steps.min())
    else:
        step = max_step = min_step = 0.0
        reasons.append("single slice")
    z = pos[:, 2]
    return SeriesGeometry(
        series_uid=slices[0].series_uid, files=tuple(slices[i].path for i in order), n_slices=len(slices),
        step_mm=step, max_step_mm=max_step, min_step_mm=min_step,
        origin_mm=tuple(float(v) for v in pos[0]), z_ends_mm=(float(z.min()), float(z.max())),
        pixel_spacing_mm=slices[0].pixel_spacing, rows=slices[0].rows, columns=slices[0].columns, reasons=reasons)
