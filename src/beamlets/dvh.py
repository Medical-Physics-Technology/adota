"""Dose-volume histogram (DVH) computation, adapted from OpenTPS.

Faithful port of OpenTPS' ``DVH.computeDVH`` / ``computeDx`` / ``computeVg`` for
our numpy-based pipeline: a DVH is built from a dose grid (in Gy) and a boolean
ROI mask on the **same voxel grid**. The cumulative histogram (volume receiving
at least a given dose) uses 4096 bins over ``[0, max_dvh]`` exactly as OpenTPS,
and the Dx metrics interpolate the cumulative curve the same way.

Mask/dose alignment (orientation) is handled upstream in
:mod:`src.beamlets.structures`; here both arrays are assumed already aligned.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DVH"]


class DVH:
    """Cumulative dose-volume histogram of a ROI for one dose grid.

    Attributes:
        name: ROI name.
        Dmean / Dstd / Dmin / Dmax: dose statistics over the ROI (Gy).
        D98 / D95 / D50 / D5 / D2: dose received by at least x% of the volume (Gy).
    """

    def __init__(
        self,
        mask: np.ndarray,
        dose: np.ndarray,
        spacing: Sequence[float],
        name: str = "",
        max_dvh: Optional[float] = None,
        n_bins: int = 4096,
    ) -> None:
        """Compute the DVH.

        Args:
            mask: Boolean ROI mask ``(z, y, x)``.
            dose: Dose grid ``(z, y, x)`` in Gy, same shape as ``mask``.
            spacing: Voxel spacing ``(sx, sy, sz)`` in mm (for absolute volume).
            name: ROI name.
            max_dvh: Upper dose (Gy) of the histogram range; defaults to
                ``1.05 * dose.max()``.
            n_bins: Number of histogram bins (OpenTPS uses 4096).
        """
        self.name = name
        mask = np.asarray(mask, dtype=bool)
        dose = np.asarray(dose, dtype=np.float64)
        if mask.shape != dose.shape:
            raise ValueError(f"mask {mask.shape} and dose {dose.shape} must match")

        d = dose[mask]
        self.n_voxels = int(d.size)
        if self.n_voxels == 0:
            raise ValueError(f"ROI {name!r} mask is empty")

        dose_max = float(dose.max())
        max_dvh = 1.05 * dose_max if max_dvh is None else float(max_dvh)
        bin_size = max_dvh / n_bins
        bin_edges = np.arange(0.0, max_dvh + 0.5 * bin_size, bin_size)
        # Extend the last edge so any voxel above max_dvh lands in the top bin.
        bin_edges[-1] = max_dvh + dose_max
        self._dose = bin_edges[:n_bins] + 0.5 * bin_size

        h, _ = np.histogram(d, bin_edges)
        h = np.flip(h, 0)
        h = np.cumsum(h)
        h = np.flip(h, 0)
        self._volume = h * 100.0 / len(d)  # cumulative volume in %
        self._volume_absolute = (
            h * spacing[0] * spacing[1] * spacing[2] / 1000.0  # cm^3
        )

        self.Dmean = float(d.mean())
        self.Dstd = float(d.std())
        self.Dmin = float(d.min())
        self.Dmax = float(d.max())
        self.D98 = self.compute_dx(98)
        self.D95 = self.compute_dx(95)
        self.D50 = self.compute_dx(50)
        self.D5 = self.compute_dx(5)
        self.D2 = self.compute_dx(2)

    @property
    def histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(dose_bins_gy, cumulative_volume_pct)`` arrays for plotting."""
        return self._dose, self._volume

    def compute_dx(self, percentile: float) -> float:
        """Dose received by at least ``percentile``% of the ROI volume (Gy).

        Linear interpolation of the cumulative DVH, matching OpenTPS.
        """
        volume = self._volume
        index = int(np.searchsorted(-volume, -percentile))
        if index > len(volume) - 2:
            index = len(volume) - 2
        v1, v2 = volume[index], volume[index + 1]
        if v1 == v2:
            dx = self._dose[index]
        else:
            w2 = (v1 - percentile) / (v1 - v2)
            w1 = (percentile - v2) / (v1 - v2)
            dx = w1 * self._dose[index] + w2 * self._dose[index + 1]
            if dx < 0:
                dx = 0.0
        return float(dx)

    def compute_vg(self, dose_gy: float, return_percentage: bool = True) -> float:
        """Volume receiving at least ``dose_gy`` Gy (in % or cm^3)."""
        index = int(np.searchsorted(self._dose, dose_gy))
        index = min(index, len(self._volume) - 1)
        return float(
            self._volume[index] if return_percentage else self._volume_absolute[index]
        )

    def metrics(self) -> dict:
        """Return the common DVH metrics as a dict (Gy)."""
        return {
            "n_voxels": self.n_voxels,
            "Dmin": self.Dmin,
            "Dmean": self.Dmean,
            "Dmax": self.Dmax,
            "Dstd": self.Dstd,
            "D98": self.D98,
            "D95": self.D95,
            "D50": self.D50,
            "D5": self.D5,
            "D2": self.D2,
        }
