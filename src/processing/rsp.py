"""
Shared HU → density / RSP conversion utilities.

References
----------
- Schneider U, Pedroni E, Lomax A (1996). "The calibration of CT
  Hounsfield units for radiotherapy treatment planning."
  Phys Med Biol 41(1):111-124.
- Schneider W, Bortfeld T, Schlegel W (2000). "Correlation between CT
  numbers and tissue parameters needed for Monte Carlo simulations of
  clinical dose distributions." Phys Med Biol 45(2):459-478.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ── Schneider 1996 Table 3: piecewise-linear HU → mass density [g/cm³] ──
# Control points for linear interpolation (same approach as OpenTPS
# PiecewiseHU2Density).
_HU_DENSITY_TABLE = np.array(
    [
        # HU,       density [g/cm³]
        [-1024.0, 0.0012],  # vacuum / air
        [-950.0, 0.044],  # inflated lung
        [-700.0, 0.302],  # lung tissue
        [-100.0, 0.924],  # adipose / fat
        [0.0, 1.000],  # water
        [15.0, 1.020],  # soft tissue
        [100.0, 1.076],  # muscle
        [300.0, 1.145],  # soft bone / cartilage
        [500.0, 1.331],  # spongy bone
        [1000.0, 1.824],  # dense bone
        [1500.0, 2.196],  # cortical bone
        [2000.0, 2.568],
        [3071.0, 2.862],  # extrapolation guard
    ]
)

_HU_KNOTS = _HU_DENSITY_TABLE[:, 0]
_DENS_KNOTS = _HU_DENSITY_TABLE[:, 1]

DENSITY_WATER = 1.0  # g/cm³


def hu_to_density(ct_hu: np.ndarray) -> np.ndarray:
    """Convert HU → mass density [g/cm³] via piecewise-linear table."""
    return np.interp(ct_hu, _HU_KNOTS, _DENS_KNOTS)


def hu_to_rsp_density(ct_hu: np.ndarray) -> np.ndarray:
    """Approximate RSP ≈ ρ(HU) / ρ_water  (energy-independent)."""
    return hu_to_density(ct_hu) / DENSITY_WATER


def hu_to_rsp(
    ct_hu: np.ndarray,
    calibration: Optional[dict] = None,
    calibration_path: Optional[Path] = None,
) -> np.ndarray:
    """Convert HU volume → relative stopping power (RSP).

    Uses a Schneider-style piecewise-linear calibration.  Supply either
    a pre-loaded ``calibration`` dict or a ``calibration_path`` YAML.
    If neither is given, falls back to :func:`hu_to_rsp_density`.
    """
    if calibration is None and calibration_path is not None:
        with open(calibration_path) as f:
            calibration = yaml.safe_load(f)

    if calibration is None:
        return hu_to_rsp_density(ct_hu)

    rsp = np.zeros_like(ct_hu, dtype=np.float64)
    for seg in calibration["segments"]:
        mask = (ct_hu >= seg["hu_min"]) & (ct_hu <= seg["hu_max"])
        rsp[mask] = seg["slope"] * ct_hu[mask] + seg["intercept"]
    rsp_min = calibration.get("rsp_min", 0.001)
    rsp_max = calibration.get("rsp_max", 2.5)
    np.clip(rsp, rsp_min, rsp_max, out=rsp)
    return rsp
