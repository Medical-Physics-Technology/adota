"""
Pflugfelder (2007) lateral tissue heterogeneity index.

Implements the Water-Equivalent Path Length (WEPL) based heterogeneity
index described in:

    Pflugfelder D, Wilkens JJ, Oelfke U (2007).
    "Worst case optimization: a method to account for uncertainties in
    the optimization of intensity modulated proton therapy."
    Phys Med Biol 53(6):1689-1700.

The HI is defined as the coefficient of variation of the WEPL map over
the lateral beam cross-section (within the flux footprint):

    HI = σ(WEPL) / μ(WEPL)

A homogeneous beam path (e.g. pure water) yields HI ≈ 0.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from src.processing.rsp import hu_to_rsp


def compute_wepl_map(
    ct_hu: np.ndarray,
    resolution_mm: Tuple[float, float, float],
    bp_depth_mm: float,
    calibration: Optional[dict] = None,
) -> np.ndarray:
    """Compute the per-ray WEPL map up to the Bragg-peak depth.

    Parameters
    ----------
    ct_hu : ndarray, shape (D, H, W)
        CT volume in Hounsfield Units.
    resolution_mm : tuple of 3 floats
        Voxel spacing (depth, height, width) in mm.
    bp_depth_mm : float
        Bragg-peak depth along axis 0 in mm.
    calibration : dict, optional
        Schneider HU→RSP calibration.  If *None*, density-ratio RSP is used.

    Returns
    -------
    wepl_map : ndarray, shape (H, W)
        Water-equivalent path length [mm] for each lateral ray.
    """
    dz_mm = resolution_mm[0]
    n_slices = int(np.clip(np.round(bp_depth_mm / dz_mm), 1, ct_hu.shape[0]))

    rsp = hu_to_rsp(ct_hu[:n_slices], calibration=calibration)
    wepl_map = rsp.sum(axis=0) * dz_mm  # (H, W)
    return wepl_map


def compute_pflugfelder_hi(
    wepl_map: np.ndarray,
    flux_2d: np.ndarray,
    flux_threshold_frac: float = 0.10,
) -> Dict[str, float]:
    """Pflugfelder heterogeneity index from a WEPL map.

    Parameters
    ----------
    wepl_map : ndarray, shape (H, W)
        Water-equivalent path length map [mm].
    flux_2d : ndarray, shape (H, W)
        Lateral flux footprint (e.g. ``flux.sum(axis=0)``).
    flux_threshold_frac : float
        Fraction of max flux below which rays are excluded.

    Returns
    -------
    dict with keys ``"hi"``, ``"wepl_mean"``, ``"wepl_std"``.
    """
    mask = flux_2d >= flux_threshold_frac * flux_2d.max()
    if mask.sum() == 0:
        return {"hi": 0.0, "wepl_mean": 0.0, "wepl_std": 0.0}

    wepl_vals = wepl_map[mask]
    mu = float(np.mean(wepl_vals))
    sigma = float(np.std(wepl_vals))
    hi = sigma / mu if mu > 0 else 0.0

    return {"hi": hi, "wepl_mean": mu, "wepl_std": sigma}


def pflugfelder_hi(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    gt_dose: np.ndarray,
    resolution_mm: Tuple[float, float, float],
    flux_threshold_frac: float = 0.10,
    calibration: Optional[dict] = None,
) -> Dict[str, float]:
    """Convenience wrapper: CT + dose → Pflugfelder HI.

    Uses the GT IDD argmax as the Bragg-peak depth.

    Parameters
    ----------
    ct_hu : ndarray, shape (D, H, W)
        CT volume in Hounsfield Units.
    flux : ndarray, shape (D, H, W)
        Flux volume.
    gt_dose : ndarray, shape (D, H, W)
        Ground-truth dose volume.
    resolution_mm : tuple of 3 floats
        Voxel spacing (depth, height, width) in mm.
    flux_threshold_frac : float
        Fraction of max flux below which lateral rays are excluded.
    calibration : dict, optional
        Schneider HU→RSP calibration.

    Returns
    -------
    dict with keys ``"hi"``, ``"wepl_mean"``, ``"wepl_std"``.
    """
    # BP depth from GT IDD
    idd = gt_dose.sum(axis=(1, 2))
    bp_slice = int(np.argmax(idd))
    bp_depth_mm = float(bp_slice) * resolution_mm[0]

    if bp_depth_mm <= 0:
        return {"hi": 0.0, "wepl_mean": 0.0, "wepl_std": 0.0}

    wepl_map = compute_wepl_map(
        ct_hu, resolution_mm, bp_depth_mm, calibration=calibration
    )
    flux_2d = flux.sum(axis=0)

    return compute_pflugfelder_hi(wepl_map, flux_2d, flux_threshold_frac)
