"""Range-fidelity metrics for proton beamlet depth-dose (IDD) curves.

These functions quantify the clinically relevant distal-edge quantities of a
1-D integrated depth-dose (IDD) profile ``I(z)``: the Bragg-peak depth (R100),
the distal fall-off depths (R90 / R80 / R50 / R20) and the distal fall-off
width (R20 - R80). They are used to compare a Monte-Carlo ground-truth dose
against an ADoTA prediction at the beamlet level, addressing the distal-range
fidelity that is especially important in proton therapy.

Design notes
------------
* Working on the *laterally integrated* IDD (``dose.sum(axis=(1, 2))`` along the
  beam axis) rather than a single central ray suppresses the Monte-Carlo
  statistical noise at the distal edge, so the extracted range is well-defined.
* Beamlet voxels are ~2 mm while the clinically relevant range tolerance is
  ~1 mm, so all depths are extracted with sub-voxel resolution via cubic
  interpolation onto a fine grid.
* All depths are in millimetres, measured from the entrance face (z = 0) along
  the beam axis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.interpolate import interp1d

# Distal fall-off levels (fraction of the Bragg-peak dose) extracted for every
# IDD curve. R80 is the conventional clinical range definition.
DISTAL_LEVELS: tuple[float, ...] = (0.9, 0.8, 0.5, 0.2)

# Default fine-grid oversampling factor used for sub-voxel extraction.
DEFAULT_OVERSAMPLE: int = 20


@dataclass
class RangeMetrics:
    """Range-related quantities extracted from a single IDD curve.

    All depths are in millimetres from the entrance face. ``dfw_mm`` is the
    distal fall-off width R20 - R80, a measure of distal-edge steepness; a
    larger value means a less sharp (smoother) fall-off.
    """

    r100_mm: float  # Bragg-peak depth
    r90_mm: float  # distal 90% fall-off depth
    r80_mm: float  # distal 80% fall-off depth (clinical range)
    r50_mm: float  # distal 50% fall-off depth
    r20_mm: float  # distal 20% fall-off depth
    dfw_mm: float  # distal fall-off width = R20 - R80
    peak_dose: float  # IDD value at the Bragg peak (physical units)

    def as_dict(self, prefix: str = "") -> dict:
        """Flat dict of the metrics, optionally with a key prefix."""
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def _nan_metrics() -> RangeMetrics:
    nan = float("nan")
    return RangeMetrics(nan, nan, nan, nan, nan, nan, 0.0)


def _fine_grid(
    idd: np.ndarray, dz_mm: float, oversample: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic-interpolate the IDD onto a ``dz_mm / oversample`` grid."""
    n = len(idd)
    z = np.arange(n, dtype=float) * dz_mm
    if n < 4:
        # Too few points for cubic; fall back to linear on the native grid.
        return z, idd.astype(float)
    z_fine = np.linspace(z[0], z[-1], (n - 1) * oversample + 1)
    interp = interp1d(z, idd, kind="cubic")
    return z_fine, interp(z_fine)


def _distal_crossing(
    z: np.ndarray, y: np.ndarray, peak_idx: int, level_value: float
) -> float:
    """Deepest depth on the distal side where ``y`` crosses ``level_value``.

    Scans the distal region (``z >= z[peak_idx]``) for the last sample still at
    or above ``level_value`` and linearly interpolates the crossing to the next
    (sub-threshold) sample, giving a sub-voxel distal depth.
    """
    above = np.where(y[peak_idx:] >= level_value)[0]
    if len(above) == 0:
        return z[peak_idx]
    k = peak_idx + int(above[-1])
    if k >= len(y) - 1:
        return z[k]
    y0, y1 = y[k], y[k + 1]
    if y0 == y1:
        return z[k]
    frac = (y0 - level_value) / (y0 - y1)
    return float(z[k] + frac * (z[k + 1] - z[k]))


def compute_range_metrics(
    idd: np.ndarray,
    dz_mm: float,
    *,
    oversample: int = DEFAULT_OVERSAMPLE,
    min_peak_dose: float = 0.0,
) -> RangeMetrics:
    """Extract range metrics from a 1-D integrated depth-dose curve.

    Args:
        idd: Integrated depth dose ``I(z)`` along the beam axis, shape ``(D,)``.
        dz_mm: Voxel spacing along the beam axis in millimetres.
        oversample: Fine-grid oversampling factor for sub-voxel extraction.
        min_peak_dose: If the Bragg-peak dose is at or below this value the
            curve is considered empty and all metrics are returned as NaN.

    Returns:
        A :class:`RangeMetrics` instance. Depths are NaN for an empty curve.
    """
    idd = np.asarray(idd, dtype=float)
    if idd.ndim != 1:
        raise ValueError(f"idd must be 1-D, got shape {idd.shape}")
    if idd.size == 0 or float(np.max(idd)) <= min_peak_dose:
        return _nan_metrics()

    z, y = _fine_grid(idd, dz_mm, oversample)
    peak_idx = int(np.argmax(y))
    peak_dose = float(y[peak_idx])
    r100 = float(z[peak_idx])

    depths = {
        level: _distal_crossing(z, y, peak_idx, level * peak_dose)
        for level in DISTAL_LEVELS
    }
    r80 = depths[0.8]
    r20 = depths[0.2]
    return RangeMetrics(
        r100_mm=r100,
        r90_mm=depths[0.9],
        r80_mm=r80,
        r50_mm=depths[0.5],
        r20_mm=r20,
        dfw_mm=r20 - r80,
        peak_dose=peak_dose,
    )


def integrated_depth_dose(dose: np.ndarray) -> np.ndarray:
    """Sum a beamlet dose ``(D, H, W)`` laterally into an IDD ``(D,)``."""
    dose = np.asarray(dose)
    if dose.ndim != 3:
        raise ValueError(f"dose must be 3-D (D, H, W), got shape {dose.shape}")
    return dose.sum(axis=(1, 2))


# Signed range-metric deltas reported as (prediction - reference) in mm.
_DELTA_FIELDS: tuple[str, ...] = (
    "r100_mm",
    "r90_mm",
    "r80_mm",
    "r50_mm",
    "r20_mm",
    "dfw_mm",
)


def range_metric_deltas(pred: RangeMetrics, ref: RangeMetrics) -> dict:
    """Signed differences (prediction - reference) for each range metric.

    Keys are suffixed ``_delta_mm`` (e.g. ``r80_delta_mm``). ``r80_delta_mm`` is
    the clinically critical range error; ``dfw_delta_mm`` quantifies distal
    fall-off width mismatch (positive => prediction is smoother than the MC GT).
    """
    return {
        f"{field.replace('_mm', '')}_delta_mm": getattr(pred, field)
        - getattr(ref, field)
        for field in _DELTA_FIELDS
    }
