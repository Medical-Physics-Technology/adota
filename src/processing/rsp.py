"""Shared HU -> density / RSP conversion utilities.

Thin compatibility layer over the single canonical physical model,
:mod:`src.processing.mcsquare_calibration` (the MCsquare ``default``-scanner
calibration that matches how the ground-truth dose was generated). Every ADoTA
metric (WEPL / Pflugfelder HI, Interface Severity Index, Bragg-peak estimation,
...) resolves HU -> density / RSP through that one model, so results are
physically consistent and defensible. New code should import
:mod:`src.processing.mcsquare_calibration` directly; these wrappers are kept for
backward compatibility.
"""

from pathlib import Path
from typing import Optional

import numpy as np

from src.processing.mcsquare_calibration import get_default_calibration

DENSITY_WATER = 1.0  # g/cm^3


def hu_to_density(ct_hu: np.ndarray) -> np.ndarray:
    """HU -> mass density [g/cm^3] (canonical MCsquare default-scanner table)."""
    return get_default_calibration().convert_hu_to_density(
        np.asarray(ct_hu, dtype=float)
    )


def hu_to_rsp_density(ct_hu: np.ndarray) -> np.ndarray:
    """Deprecated alias: returns the canonical MCsquare RSP.

    Historically this returned the crude density ratio ``rho/rho_water``; it now
    delegates to the proper stopping-power-based RSP so no caller silently uses
    the old approximation. Prefer :func:`hu_to_rsp`.
    """
    return get_default_calibration().convert_hu_to_rsp(np.asarray(ct_hu, dtype=float))


def hu_to_rsp(
    ct_hu: np.ndarray,
    calibration: Optional[dict] = None,
    calibration_path: Optional[Path] = None,
) -> np.ndarray:
    """Convert HU volume -> relative stopping power (RSP).

    Uses the canonical MCsquare ``default``-scanner calibration unless an
    explicit piecewise ``calibration`` dict (or ``calibration_path`` YAML with a
    ``segments`` list) is supplied, in which case that override is honoured for
    backward compatibility.
    """
    if calibration is None and calibration_path is not None:
        import yaml

        with open(calibration_path) as f:
            calibration = yaml.safe_load(f)

    if calibration is None:
        return get_default_calibration().convert_hu_to_rsp(
            np.asarray(ct_hu, dtype=float)
        )

    # Explicit piecewise-linear segment override (legacy calibration YAML).
    ct_hu = np.asarray(ct_hu, dtype=np.float64)
    rsp = np.zeros_like(ct_hu, dtype=np.float64)
    for seg in calibration["segments"]:
        mask = (ct_hu >= seg["hu_min"]) & (ct_hu <= seg["hu_max"])
        rsp[mask] = seg["slope"] * ct_hu[mask] + seg["intercept"]
    np.clip(
        rsp,
        calibration.get("rsp_min", 0.001),
        calibration.get("rsp_max", 2.5),
        out=rsp,
    )
    return rsp
