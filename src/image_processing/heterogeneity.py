"""
Beam-aligned CT heterogeneity metrics.

Provides three complementary scores that quantify the structural complexity
of a CT volume as "seen" by a proton beamlet:

* **G_φ** – beam-weighted gradient magnitude
* **R**   – beam-axis roughness (lateral-mean HU jitter along depth)
* **H_φ** – combined heterogeneity = G_φ × R

These metrics are used to correlate CT complexity with dose-prediction
model performance (MAPE, GPR, etc.).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_weights(phi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize *phi* so that all voxel weights sum to 1.

    Parameters
    ----------
    phi : np.ndarray
        (D, H, W) non-negative beamlet-shape projection weights.
    eps : float
        Values below *eps* are treated as zero.

    Returns
    -------
    np.ndarray
        Weight array summing to 1 (float32), or all zeros if *phi* is
        effectively zero everywhere.
    """
    s = float(np.sum(phi))
    if s <= eps:
        return np.zeros_like(phi, dtype=np.float32)
    return (phi / s).astype(np.float32)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def gradient_magnitude_3d(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Spacing-aware 3-D gradient magnitude via central differences.

    Parameters
    ----------
    volume : np.ndarray
        (D, H, W) HU volume (converted to float32 internally).
    spacing_zyx : tuple[float, float, float]
        Voxel spacing in the same axis order as the array: (z, y, x).

    Returns
    -------
    np.ndarray
        |∇I| with shape (D, H, W), dtype float32.
    """
    vol = volume.astype(np.float32, copy=False)
    dz, dy, dx = map(float, spacing_zyx)

    # np.gradient uses second-order accurate central differences in the interior
    g_z, g_y, g_x = np.gradient(vol, dz, dy, dx, edge_order=1)
    return np.sqrt(g_z * g_z + g_y * g_y + g_x * g_x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def beam_weighted_gradient(
    ct_hu: np.ndarray,
    phi: np.ndarray,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Beam-weighted gradient magnitude **G_φ**.

    .. math::

        G_\\phi = \\sum_r w(r) \\, \\|\\nabla I(r)\\|_2

    where :math:`w = \\phi / \\sum \\phi` (optionally restricted to *mask*).

    Parameters
    ----------
    ct_hu : np.ndarray
        (D, H, W) CT volume in Hounsfield units.
    phi : np.ndarray
        (D, H, W) non-negative beamlet projection weights (e.g. fluence).
    spacing_zyx : tuple[float, float, float]
        Voxel spacing (z, y, x) in mm.
    mask : np.ndarray or None
        Optional boolean mask; only masked voxels contribute.
    eps : float
        Numerical guard for weight normalization.

    Returns
    -------
    float
        Scalar G_φ value.  Returns 0 if *phi* is all zeros.

    Raises
    ------
    ValueError
        If shapes of *ct_hu*, *phi*, or *mask* do not match.
    """
    if ct_hu.shape != phi.shape:
        raise ValueError(
            f"ct_hu and phi must have same shape, got {ct_hu.shape} vs {phi.shape}"
        )

    ct = ct_hu.astype(np.float32, copy=False)
    ph = phi.astype(np.float32, copy=False)

    if mask is not None:
        if mask.shape != ct.shape:
            raise ValueError(
                f"mask must match ct_hu shape, got {mask.shape} vs {ct.shape}"
            )
        m = mask.astype(bool, copy=False)
        ct = np.where(m, ct, 0.0)
        ph = np.where(m, ph, 0.0)

    grad_mag = gradient_magnitude_3d(ct, spacing_zyx=spacing_zyx)
    w = _normalize_weights(ph, eps=eps)

    return float(np.sum(w * grad_mag))


def beam_axis_roughness(
    ct_hu: np.ndarray,
    axis: int = 0,
    mask: np.ndarray | None = None,
) -> float:
    """Beam-axis roughness **R**.

    .. math::

        R = \\frac{1}{K-1} \\sum_{k} |\\mu_{k+1} - \\mu_k|

    where :math:`\\mu_k` is the lateral-mean HU at depth index *k* along the
    chosen *axis*.

    Parameters
    ----------
    ct_hu : np.ndarray
        (D, H, W) CT volume in HU.
    axis : int
        Which axis represents beam/depth (default 0 for the D dimension).
    mask : np.ndarray or None
        Optional boolean mask; per-slice means are computed only over masked
        voxels. Slices with no masked voxels are skipped.

    Returns
    -------
    float
        Scalar roughness value.  Returns 0 if fewer than two valid slices.
    """
    vol = ct_hu.astype(np.float32, copy=False)

    if mask is None:
        mu = vol.mean(axis=tuple(i for i in range(vol.ndim) if i != axis))
        diffs = np.abs(np.diff(mu))
        return float(diffs.mean()) if diffs.size > 0 else 0.0

    if mask.shape != vol.shape:
        raise ValueError(
            f"mask must match ct_hu shape, got {mask.shape} vs {vol.shape}"
        )
    m = mask.astype(bool, copy=False)

    # Move depth axis to front for straightforward looping
    vol_0 = np.moveaxis(vol, axis, 0)
    m_0 = np.moveaxis(m, axis, 0)

    mus = []
    for k in range(vol_0.shape[0]):
        mk = m_0[k]
        if mk.any():
            mus.append(float(vol_0[k][mk].mean()))
        else:
            mus.append(np.nan)

    mu_arr = np.array(mus, dtype=np.float32)
    valid = ~np.isnan(mu_arr)

    if valid.sum() < 2:
        return 0.0

    # Differences only where both neighbours are valid
    diffs = []
    for k in range(mu_arr.size - 1):
        if valid[k] and valid[k + 1]:
            diffs.append(abs(float(mu_arr[k + 1] - mu_arr[k])))

    return float(np.mean(diffs)) if len(diffs) > 0 else 0.0


def beam_aligned_global_heterogeneity(
    ct_hu: np.ndarray,
    phi: np.ndarray,
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axis: int = 0,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute the full beam-aligned heterogeneity metric set.

    Returns
    -------
    dict
        ``{"G_phi": ..., "R": ..., "H_phi": ...}``

        * **G_phi** – beam-weighted gradient magnitude
        * **R** – beam-axis roughness
        * **H_phi** – product G_phi × R
    """
    g_phi = beam_weighted_gradient(
        ct_hu,
        phi,
        spacing_zyx=spacing_zyx,
        mask=mask,
        eps=eps,
    )
    r = beam_axis_roughness(ct_hu, axis=axis, mask=mask)
    return {"G_phi": g_phi, "R": r, "H_phi": float(g_phi * r)}
