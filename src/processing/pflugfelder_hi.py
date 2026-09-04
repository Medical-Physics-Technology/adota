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

from typing import Dict, Tuple

import numpy as np

from src.processing.mcsquare_calibration import (
    DEFAULT_ENERGY_MEV,
    hu_to_rsp_mcsquare,
)


def compute_wepl_map(
    ct_hu: np.ndarray,
    resolution_mm: Tuple[float, float, float],
    bp_depth_mm: float,
    rsp_energy_mev: float = DEFAULT_ENERGY_MEV,
) -> np.ndarray:
    """Compute the per-ray WEPL map up to the Bragg-peak depth.

    The WEPL of each lateral ray is the depth-integrated relative stopping power
    (RSP) up to the Bragg-peak depth. RSP is computed with the self-contained
    MCsquare ``default``-scanner calibration
    (:func:`src.processing.mcsquare_calibration.hu_to_rsp_mcsquare`), i.e.
    ``rho(HU) * SP_material(HU, E) / SP_water(E)`` -- the same conversion
    MCsquare used to generate the ground-truth dose. This replaces the earlier
    density-ratio approximation, which overestimated bone RSP by ~15-25%.

    Parameters
    ----------
    ct_hu : ndarray, shape (D, H, W)
        CT volume in Hounsfield Units. Axis 0 is the beam (depth) direction.
    resolution_mm : tuple of 3 floats
        Voxel spacing (depth, height, width) in mm.
    bp_depth_mm : float
        Bragg-peak depth along axis 0 in mm.
    rsp_energy_mev : float
        Reference energy for the RSP conversion (OpenTPS convention: 100 MeV;
        RSP is only weakly energy dependent).

    Returns
    -------
    wepl_map : ndarray, shape (H, W)
        Water-equivalent path length [mm] for each lateral ray.
    """
    dz_mm = resolution_mm[0]
    n_slices = int(np.clip(np.round(bp_depth_mm / dz_mm), 1, ct_hu.shape[0]))

    rsp = hu_to_rsp_mcsquare(ct_hu[:n_slices], energy=rsp_energy_mev)
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


def compute_parallel_beam_wepl_diff(
    wepl_map: np.ndarray,
    flux_2d: np.ndarray,
    resolution_mm: Tuple[float, float, float],
    flux_threshold_frac: float = 0.10,
) -> Dict[str, float]:
    """Transverse WEPL heterogeneity: density difference *across* the aperture.

    Detects the "half the beamlet through bone, half through air" situation that
    the scalar Pflugfelder CV (``wepl_std``) cannot distinguish from random
    lateral scatter, because it discards the spatial arrangement of the WEPL
    values. Here we keep it:

    * ``wepl_halfsplit_diff`` - split the flux footprint into two halves about
      the flux centroid (along the row and the column axis) and take the larger
      of the two flux-weighted mean-WEPL differences between halves. A bimodal
      bone/air aperture gives a large value; uniform scatter gives ~0.
    * ``wepl_lateral_grad_p95`` - the 95th percentile of the transverse WEPL
      gradient magnitude inside the footprint (peak sharpness of the split).

    Both are input-only (CT + flux + a Bragg-peak depth for the WEPL map).
    """
    mask = flux_2d >= flux_threshold_frac * flux_2d.max()
    if mask.sum() < 4:
        return {"wepl_halfsplit_diff": 0.0, "wepl_lateral_grad_p95": 0.0}

    H, W = wepl_map.shape
    yy, xx = np.indices((H, W))
    wm = flux_2d[mask]
    cy = float(np.average(yy[mask], weights=wm))
    cx = float(np.average(xx[mask], weights=wm))

    def flux_wmean(sel: np.ndarray) -> float:
        w = flux_2d[sel]
        return float(np.average(wepl_map[sel], weights=w)) if w.sum() > 0 else np.nan

    diffs = []
    for coord, c in ((yy, cy), (xx, cx)):
        a = mask & (coord < c)
        b = mask & (coord >= c)
        if a.sum() and b.sum():
            ma, mb = flux_wmean(a), flux_wmean(b)
            if np.isfinite(ma) and np.isfinite(mb):
                diffs.append(abs(ma - mb))
    halfsplit = max(diffs) if diffs else 0.0

    dy, dx = resolution_mm[1], resolution_mm[2]
    gy, gx = np.gradient(wepl_map, dy, dx)
    grad = np.sqrt(gy * gy + gx * gx)
    p95 = float(np.percentile(grad[mask], 95))

    return {"wepl_halfsplit_diff": halfsplit, "wepl_lateral_grad_p95": p95}


def pflugfelder_hi(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    gt_dose: np.ndarray,
    resolution_mm: Tuple[float, float, float],
    flux_threshold_frac: float = 0.10,
    rsp_energy_mev: float = DEFAULT_ENERGY_MEV,
) -> Dict[str, float]:
    """Convenience wrapper: CT + dose → Pflugfelder HI.

    Uses the GT IDD argmax as the Bragg-peak depth and the MCsquare
    ``default``-scanner RSP calibration for the WEPL map.

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
    rsp_energy_mev : float
        Reference energy for the RSP conversion (default 100 MeV).

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
        ct_hu, resolution_mm, bp_depth_mm, rsp_energy_mev=rsp_energy_mev
    )
    flux_2d = flux.sum(axis=0)

    return compute_pflugfelder_hi(wepl_map, flux_2d, flux_threshold_frac)
