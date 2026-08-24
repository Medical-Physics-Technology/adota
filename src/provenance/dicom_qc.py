"""DICOM acquisition provenance + quality gating (shared by datasets and registry).

Two concerns, one module:

* **Provenance** -- pull the acquisition parameters that matter for data quality
  and traceability (pixel spacing, slice thickness, kVp, tube current, kernel,
  manufacturer, ...) from a series' DICOM header. Recorded on every CT record and
  carried into every generated beamlet, regardless of gating.
* **Quality gating** -- decide whether a series is good enough to use. The original
  datagenerator gated on ``x,y spacing < 1.5 mm``, ``z spacing < 2 mm`` and
  ``z-slices > 70`` (``get_qualified_indexes``); adota currently gates only on slice
  count + modality. :class:`QCGates` reproduces the datagenerator gates (opt-in) and
  extends them to kVp / tube current / kernel.

Gating is **opt-in**: the spacing/acquisition gates default to ``None`` (not
enforced), so existing behaviour is unchanged; a dataset turns them on via its
``qc:`` config. Provenance is always extracted.

Note: Phase 1 uses ``SliceThickness`` as the z-spacing proxy (cheap, header-only).
The Phase-2 registry builder can verify true inter-slice spacing via SimpleITK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from os.path import join
from typing import Dict, List, Optional, Sequence, Tuple

import pydicom


@dataclass(frozen=True)
class QCGates:
    """Quality thresholds. Spacing/acquisition gates are opt-in (``None`` = off).

    Datagenerator equivalents: ``max_spacing_xy=1.5``, ``max_spacing_z=2.0``,
    ``min_slices=70`` (it used ``z-slices > 70``).
    """

    min_slices: int = 50
    max_slices: int = 1000
    require_ct: bool = True
    require_monochrome2: bool = True
    max_spacing_xy: Optional[float] = None      # mm; fail if x or y spacing >= this
    max_spacing_z: Optional[float] = None        # mm; fail if slice thickness >= this
    min_z_extent_mm: Optional[float] = None      # mm; fail if n_slices*thickness < this
    min_xy_extent_mm: Optional[float] = None     # mm; fail if either in-plane extent < this
    kvp_range: Optional[Tuple[float, float]] = None
    min_tube_current: Optional[float] = None     # mA
    exclude_kernels: Tuple[str, ...] = ()        # case-insensitive substrings


def _num(v) -> Optional[float]:
    """Coerce a DICOM value (possibly a multi-value or string) to float, or None."""
    if v is None:
        return None
    try:
        if isinstance(v, (list, tuple)) or hasattr(v, "__iter__") and not isinstance(v, str):
            v = list(v)[0]
        return float(v)
    except (TypeError, ValueError, IndexError):
        return None


def params_from_header(ds: "pydicom.dataset.Dataset", n_slices: int) -> Dict:
    """Extract acquisition provenance from an already-read DICOM header."""
    ps = getattr(ds, "PixelSpacing", None)
    sx = _num(ps[0]) if ps is not None and len(ps) > 0 else None
    sy = _num(ps[1]) if ps is not None and len(ps) > 1 else None
    kernel = getattr(ds, "ConvolutionKernel", None)
    if isinstance(kernel, (list, tuple)):
        kernel = "\\".join(str(k) for k in kernel)
    return {
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
        "modality": getattr(ds, "Modality", None),
        "photometric": getattr(ds, "PhotometricInterpretation", None),
        "n_slices": int(n_slices),
        "pixel_spacing_x": sx,
        "pixel_spacing_y": sy,
        "slice_thickness": _num(getattr(ds, "SliceThickness", None)),
        "kvp": _num(getattr(ds, "KVP", None)),
        "tube_current": _num(getattr(ds, "XRayTubeCurrent", None)),
        "exposure": _num(getattr(ds, "Exposure", None)),
        "convolution_kernel": str(kernel) if kernel else None,
        "manufacturer": (getattr(ds, "Manufacturer", None) or None),
        "model": (getattr(ds, "ManufacturerModelName", None) or None),
        "study_date": (getattr(ds, "StudyDate", None) or None),
        "rows": _num(getattr(ds, "Rows", None)),
        "columns": _num(getattr(ds, "Columns", None)),
    }


def read_acquisition_params(series_dir: str, dcm_files: Optional[Sequence[str]] = None) -> Dict:
    """Read acquisition provenance from a series directory (first DICOM header)."""
    files = list(dcm_files) if dcm_files is not None else sorted(glob(join(series_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"no .dcm files in {series_dir}")
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)
    return params_from_header(ds, len(files))


def check_quality(params: Dict, gates: QCGates) -> Tuple[bool, List[str]]:
    """Return ``(qc_pass, reasons)``; empty reasons == pass. Missing gated data fails."""
    reasons: List[str] = []
    n = int(params.get("n_slices") or 0)
    if not (gates.min_slices <= n <= gates.max_slices):
        reasons.append(f"n_slices={n}")
    if gates.require_ct and params.get("modality") != "CT":
        reasons.append(f"modality={params.get('modality')}")
    if gates.require_monochrome2 and params.get("photometric") != "MONOCHROME2":
        reasons.append(f"photometric={params.get('photometric')}")

    if gates.max_spacing_xy is not None:
        sx, sy = params.get("pixel_spacing_x"), params.get("pixel_spacing_y")
        if sx is None or sy is None:
            reasons.append("pixel_spacing_missing")
        elif sx >= gates.max_spacing_xy or sy >= gates.max_spacing_xy:
            reasons.append(f"pixel_spacing=({sx},{sy})>=({gates.max_spacing_xy})")
    if gates.max_spacing_z is not None:
        st = params.get("slice_thickness")
        if st is None:
            reasons.append("slice_thickness_missing")
        elif st >= gates.max_spacing_z:
            reasons.append(f"slice_thickness={st}>={gates.max_spacing_z}")
    if gates.min_z_extent_mm is not None:
        st = params.get("slice_thickness")
        if st is None:
            reasons.append("slice_thickness_missing")
        else:
            z_extent = n * st
            if z_extent < gates.min_z_extent_mm:
                reasons.append(f"z_extent={z_extent:.0f}<{gates.min_z_extent_mm}")
    if gates.min_xy_extent_mm is not None:
        sx, sy = params.get("pixel_spacing_x"), params.get("pixel_spacing_y")
        cols, rows = params.get("columns"), params.get("rows")
        if None in (sx, sy, cols, rows):
            reasons.append("in_plane_extent_missing")
        else:
            ex, ey = cols * sx, rows * sy
            if min(ex, ey) < gates.min_xy_extent_mm:
                reasons.append(f"xy_extent=({ex:.0f},{ey:.0f})<{gates.min_xy_extent_mm}")
    if gates.kvp_range is not None:
        kv = params.get("kvp")
        if kv is None:
            reasons.append("kvp_missing")
        elif not (gates.kvp_range[0] <= kv <= gates.kvp_range[1]):
            reasons.append(f"kvp={kv}")
    if gates.min_tube_current is not None:
        tc = params.get("tube_current")
        if tc is None:
            reasons.append("tube_current_missing")
        elif tc < gates.min_tube_current:
            reasons.append(f"tube_current={tc}")
    if gates.exclude_kernels and params.get("convolution_kernel"):
        k = params["convolution_kernel"].lower()
        hit = next((x for x in gates.exclude_kernels if x.lower() in k), None)
        if hit:
            reasons.append(f"kernel~{hit}")
    return (len(reasons) == 0, reasons)


def gates_from_dict(d: Optional[Dict], **base) -> QCGates:
    """Build :class:`QCGates` from a config dict merged over ``base`` defaults."""
    merged = dict(base)
    if d:
        merged.update(d)
    kr = merged.get("kvp_range")
    if kr is not None:
        merged["kvp_range"] = (float(kr[0]), float(kr[1]))
    ek = merged.get("exclude_kernels")
    if ek is not None:
        merged["exclude_kernels"] = tuple(ek)
    allowed = QCGates.__dataclass_fields__.keys()
    return QCGates(**{k: v for k, v in merged.items() if k in allowed})
