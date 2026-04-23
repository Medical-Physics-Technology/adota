"""
Interface Severity Index (ISI).

Quantifies how many, and how severe, the tissue-class interfaces are
along the beam path inside a region of interest (typically a sphere
around the Bragg peak).

Uses the 24-class Schneider decomposition implemented in
:mod:`scripts.analysis.hu_to_sp` — the same segmentation used elsewhere
in the project — so that interface severity is consistent with the
TITUS / TRACer material model.

Physics motivation
------------------
Multiple Coulomb scattering (MCS) at density interfaces is the dominant
driver of Bragg-peak distal-edge degradation (Sawakuchi et al., *PMB*
53:4605, 2008; Chang et al., *Radiat. Phys. Chem.* 137:121, 2017).
The severity of a single interface scales approximately with the
squared relative-stopping-power (RSP) jump,
:math:`(\\Delta\\mathrm{RSP})^2`.

Method
------
1. Segment the CT volume into 24 tissue classes via
   :func:`segment_tissue` (Schneider piecewise-constant LUT).
2. Build a symmetric 24×24 severity matrix
   :math:`W_{ab} = f(\\overline{\\mathrm{RSP}}_a, \\overline{\\mathrm{RSP}}_b)`
   where *f* is :math:`(\\Delta\\mathrm{RSP})^2` by default.
3. For every pair of neighbouring voxels with different classes, add
   :math:`W_{c(v), c(v')}` to a per-voxel severity map (split evenly
   between the two voxels so each interface is counted once in sums).
4. Restrict to the BP region (sphere of radius *r* around the
   Bragg-peak voxel) and to voxels above a flux threshold.
5. Aggregate to four scalars: ``isi_sum``, ``isi_max``, ``isi_mean``,
   ``isi_axial_sum``.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np

from src.processing.tissue_decomposition import (
    N_TISSUE_CLASSES,
    TISSUE_LUT,
    hu_to_density,
    hu_to_rsp_schneider,
    segment_tissue,
)

# ── Representative HU / RSP / density per class ───────────────────────────
# Midpoint of each TISSUE_LUT interval.  Classes already cap at 1600 HU
# in the Schneider LUT, so no manual capping is required.
_CLASS_HU_REPR = np.array(
    [0.5 * (lo + hi) for lo, hi, _ in TISSUE_LUT], dtype=np.float64
)
_CLASS_RSP = hu_to_rsp_schneider(_CLASS_HU_REPR)
_CLASS_DENSITY = hu_to_density(_CLASS_HU_REPR)

SeverityMode = Literal["rsp_sq", "rsp_abs", "density_sq", "custom"]


def build_severity_matrix(
    mode: SeverityMode = "rsp_sq",
    custom: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a symmetric ``(N, N)`` interface-severity matrix.

    Parameters
    ----------
    mode
        ``"rsp_sq"`` — :math:`W_{ab} = (\\mathrm{RSP}_a - \\mathrm{RSP}_b)^2`
        (default; MCS-motivated).
        ``"rsp_abs"`` — :math:`W_{ab} = |\\mathrm{RSP}_a - \\mathrm{RSP}_b|`.
        ``"density_sq"`` — :math:`W_{ab} = (\\rho_a - \\rho_b)^2`.
        ``"custom"`` — use the matrix supplied via *custom*.
    custom
        User-supplied ``(N, N)`` severity matrix (required when
        ``mode == "custom"``).

    Returns
    -------
    W : ndarray, shape ``(N, N)``, float64
        Symmetric with ``W[a, a] = 0`` (``N = N_TISSUE_CLASSES = 24``).
    """
    if mode == "custom":
        if custom is None:
            raise ValueError("mode='custom' requires a custom matrix")
        W = np.asarray(custom, dtype=np.float64)
        if W.shape != (N_TISSUE_CLASSES, N_TISSUE_CLASSES):
            raise ValueError(
                f"custom matrix must be ({N_TISSUE_CLASSES}, {N_TISSUE_CLASSES})"
            )
        return W

    if mode == "rsp_sq":
        diff = _CLASS_RSP[:, None] - _CLASS_RSP[None, :]
        W = diff * diff
    elif mode == "rsp_abs":
        W = np.abs(_CLASS_RSP[:, None] - _CLASS_RSP[None, :])
    elif mode == "density_sq":
        diff = _CLASS_DENSITY[:, None] - _CLASS_DENSITY[None, :]
        W = diff * diff
    else:
        raise ValueError(f"Unknown severity mode: {mode!r}")

    np.fill_diagonal(W, 0.0)
    return W


def compute_interface_severity_map(
    class_volume: np.ndarray,
    severity_matrix: np.ndarray,
    axial_only: bool = False,
) -> np.ndarray:
    """Per-voxel interface severity.

    For each pair of neighbouring voxels ``v, v'`` with different
    class labels, ``severity_matrix[c(v), c(v')]`` is added, split
    evenly between ``v`` and ``v'`` so each interface is counted
    exactly once in aggregate sums.

    Parameters
    ----------
    class_volume : ndarray, shape ``(D, H, W)``, int
        Per-voxel tissue-class labels.
    severity_matrix : ndarray, shape ``(N, N)``
        Class-pair severity weights (see :func:`build_severity_matrix`).
    axial_only
        If *True*, only count interfaces along axis 0 (beam direction).
    """
    c_vol = class_volume.astype(np.int64, copy=False)
    severity = np.zeros(c_vol.shape, dtype=np.float64)

    # Axial (beam direction)
    w = severity_matrix[c_vol[:-1], c_vol[1:]]
    severity[:-1] += 0.5 * w
    severity[1:] += 0.5 * w

    if not axial_only:
        w = severity_matrix[c_vol[:, :-1], c_vol[:, 1:]]
        severity[:, :-1] += 0.5 * w
        severity[:, 1:] += 0.5 * w

        w = severity_matrix[c_vol[:, :, :-1], c_vol[:, :, 1:]]
        severity[:, :, :-1] += 0.5 * w
        severity[:, :, 1:] += 0.5 * w

    return severity


def _build_sphere_mask(
    shape: Tuple[int, int, int],
    centre: Tuple[int, int, int],
    radius_mm: float,
    resolution_mm: Tuple[float, float, float],
) -> np.ndarray:
    """Boolean sphere mask with physical radius, accounting for voxel spacing."""
    D, H, W = shape
    kz, ky, kx = centre
    dz, dy, dx = resolution_mm
    zz, yy, xx = np.ogrid[:D, :H, :W]
    dist_sq = (
        ((zz - kz) * dz) ** 2 + ((yy - ky) * dy) ** 2 + ((xx - kx) * dx) ** 2
    )
    return dist_sq <= radius_mm**2


def compute_isi_metrics(
    severity_map: np.ndarray,
    region_mask: np.ndarray,
    flux: Optional[np.ndarray] = None,
    flux_threshold_frac: float = 0.10,
    axial_severity_map: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Aggregate a severity map into scalar ISI metrics.

    Returns a dict with keys ``"isi_sum"``, ``"isi_max"``, ``"isi_mean"``,
    ``"isi_axial_sum"``.  Returns zeros if the region × flux mask is empty.
    """
    mask = region_mask.copy()
    if flux is not None:
        f_max = float(np.abs(flux).max())
        if f_max > 0:
            mask &= np.abs(flux) >= flux_threshold_frac * f_max

    defaults = dict(isi_sum=0.0, isi_max=0.0, isi_mean=0.0, isi_axial_sum=0.0)
    if mask.sum() == 0:
        return defaults

    vals = severity_map[mask]
    isi_sum = float(vals.sum())
    isi_max = float(vals.max())
    active = vals > 0
    isi_mean = float(vals[active].mean()) if active.any() else 0.0

    if axial_severity_map is not None:
        isi_axial_sum = float(axial_severity_map[mask].sum())
    else:
        isi_axial_sum = 0.0

    return dict(
        isi_sum=isi_sum,
        isi_max=isi_max,
        isi_mean=isi_mean,
        isi_axial_sum=isi_axial_sum,
    )


def interface_severity(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    gt_dose: np.ndarray,
    resolution_mm: Tuple[float, float, float],
    severity_mode: SeverityMode = "rsp_sq",
    sphere_radius_mm: float = 15.0,
    flux_threshold_frac: float = 0.10,
    custom_matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Convenience wrapper: CT + flux + dose → ISI scalars.

    Builds a sphere of radius *sphere_radius_mm* around the Bragg-peak
    voxel (from GT IDD argmax + lateral dose peak), segments the CT
    into 24 Schneider tissue classes, and aggregates interface
    severities within the sphere × flux mask.

    Returns
    -------
    dict with keys ``"isi_sum"``, ``"isi_max"``, ``"isi_mean"``,
    ``"isi_axial_sum"``.
    """
    D, H, W = ct_hu.shape

    # -- Bragg-peak voxel from GT dose ------------------------------------
    idd = gt_dose.sum(axis=(1, 2))
    bp_k = int(np.argmax(idd))
    dose_bp_slice = gt_dose[bp_k]
    if dose_bp_slice.max() > 1e-9:
        bp_y = int(np.argmax(dose_bp_slice.max(axis=1)))
        bp_x = int(np.argmax(dose_bp_slice.max(axis=0)))
    else:
        bp_y, bp_x = H // 2, W // 2

    # -- Tissue segmentation + severity matrix ---------------------------
    class_vol = segment_tissue(ct_hu)
    W_mat = build_severity_matrix(mode=severity_mode, custom=custom_matrix)

    # -- Severity maps ----------------------------------------------------
    sev_iso = compute_interface_severity_map(class_vol, W_mat, axial_only=False)
    sev_axial = compute_interface_severity_map(class_vol, W_mat, axial_only=True)

    # -- Sphere region mask ----------------------------------------------
    sphere_mask = _build_sphere_mask(
        shape=(D, H, W),
        centre=(bp_k, bp_y, bp_x),
        radius_mm=sphere_radius_mm,
        resolution_mm=resolution_mm,
    )

    return compute_isi_metrics(
        severity_map=sev_iso,
        region_mask=sphere_mask,
        flux=flux,
        flux_threshold_frac=flux_threshold_frac,
        axial_severity_map=sev_axial,
    )
