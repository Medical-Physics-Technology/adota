"""
Global intensity heterogeneity metrics for CT HU volumes.

Unlike beam-aligned heterogeneity (:mod:`src.image_processing.heterogeneity`),
these metrics describe the **intensity distribution** of the volume without
regard for spatial arrangement.  They are particularly useful for
characterising the overall tissue-composition complexity of a CT grid.

Computed metrics:
  * **mean**, **std**, **var** – basic descriptive statistics
  * **IQR**, **MAD** – robust dispersion measures
  * **entropy**, **uniformity** – histogram-based information content
  * **skewness**, **kurtosis_excess** – moment-based shape descriptors

Notes
-----
* Use a mask to exclude air / background (recommended).
* Entropy and uniformity depend on binning; use a fixed HU bin width for
  cross-case comparability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalIntensityHeterogeneity:
    """Immutable container for global intensity heterogeneity metrics.

    Attributes
    ----------
    mean, std, var : float
        Basic descriptive statistics of masked HU intensities.
    iqr : float
        Inter-quartile range (Q75 − Q25).
    mad : float
        Median absolute deviation.
    entropy : float
        Shannon entropy of the HU histogram (natural log, nats).
    uniformity : float
        Energy / uniformity of the HU histogram (sum of squared bin probs).
    skewness : float
        Fisher-Pearson standardised moment coefficient.
    kurtosis_excess : float
        Excess kurtosis (Gaussian baseline = 0).
    n_voxels : int
        Number of voxels contributing to the statistics.
    hu_min, hu_max : float
        Observed HU range after masking.
    """

    mean: float
    std: float
    var: float
    iqr: float
    mad: float
    entropy: float
    uniformity: float
    skewness: float
    kurtosis_excess: float
    n_voxels: int
    hu_min: float
    hu_max: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Element-wise natural log clamped away from zero."""
    return np.log(np.clip(x, eps, None))


def _quantile(x: np.ndarray, q: float) -> float:
    """Scalar quantile helper."""
    return float(np.quantile(x, q))


def _central_moments(
    x: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return (mean, m2, m3, m4) — central moments of order 1–4."""
    mu = float(np.mean(x))
    xc = x - mu
    m2 = float(np.mean(xc**2))
    m3 = float(np.mean(xc**3))
    m4 = float(np.mean(xc**4))
    return mu, m2, m3, m4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def global_intensity_heterogeneity(
    ct_hu: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    hu_range: tuple[float, float] | None = None,
    bin_width_hu: float = 25.0,
    eps: float = 1e-12,
) -> GlobalIntensityHeterogeneity:
    """Compute global intensity heterogeneity metrics from a 3-D HU array.

    Parameters
    ----------
    ct_hu : np.ndarray
        CT volume in HU with shape ``(D, H, W)``.
    mask : np.ndarray or None
        Boolean / integer ROI mask of the same shape as *ct_hu*.
        If ``None``, all voxels are used.
    hu_range : tuple[float, float] or None
        ``(hu_min, hu_max)`` for histogram binning.  If ``None`` the robust
        1st–99th percentile range of the data is used.  Fixing this value
        is recommended for cross-case comparability.
    bin_width_hu : float
        Histogram bin width in HU.  A fixed width is preferred over a fixed
        number of bins for consistent entropy / uniformity comparisons.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    GlobalIntensityHeterogeneity
        Frozen dataclass with all computed metrics.

    Raises
    ------
    ValueError
        If *mask* shape does not match *ct_hu* or no voxels are selected.
    """
    volume = ct_hu.astype(np.float32, copy=False)

    if mask is not None:
        if mask.shape != volume.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match ct_hu shape {volume.shape}"
            )
        x = volume[mask.astype(bool, copy=False)]
    else:
        x = volume.reshape(-1)

    n = int(x.size)
    if n == 0:
        raise ValueError("No voxels selected (empty mask).")

    # ---- Basic statistics ----
    mu, m2, m3, m4 = _central_moments(x)
    var = m2
    std = float(np.sqrt(max(var, 0.0)))

    # ---- Robust dispersion ----
    q25 = _quantile(x, 0.25)
    q75 = _quantile(x, 0.75)
    iqr = float(q75 - q25)

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    # ---- Skewness & excess kurtosis (moment-based) ----
    if std <= eps:
        skew = 0.0
        kurt_excess = -3.0  # degenerate; define as Gaussian baseline shift
    else:
        skew = float(m3 / (std**3 + eps))
        kurt_excess = float(m4 / (std**4 + eps) - 3.0)

    # ---- Histogram-based entropy & uniformity ----
    if hu_range is None:
        # Robust clipping range to reduce metal / streak outlier influence
        hmin = _quantile(x, 0.01)
        hmax = _quantile(x, 0.99)
    else:
        hmin, hmax = map(float, hu_range)

    if hmax <= hmin + eps:
        # All values approximately constant
        entropy = 0.0
        uniformity = 1.0
    else:
        nbins = max(int(np.ceil((hmax - hmin) / float(bin_width_hu))), 1)
        edges = np.linspace(hmin, hmax, nbins + 1, dtype=np.float32)

        counts, _ = np.histogram(x, bins=edges)
        p = counts.astype(np.float64)
        p = p / max(p.sum(), 1.0)

        entropy = float(-np.sum(p * _safe_log(p, eps=eps)))
        uniformity = float(np.sum(p * p))

    return GlobalIntensityHeterogeneity(
        mean=float(mu),
        std=std,
        var=float(var),
        iqr=iqr,
        mad=mad,
        entropy=entropy,
        uniformity=uniformity,
        skewness=skew,
        kurtosis_excess=kurt_excess,
        n_voxels=n,
        hu_min=float(np.min(x)),
        hu_max=float(np.max(x)),
    )
