"""An analytic dose surrogate for a beamlet that has not been simulated.

The difficulty metrics need a dose for two things only: to locate the Bragg peak
(the depth zone, the peak slice, the lateral peak voxel) and to weight the edge
metrics. Before Monte Carlo runs there is no dose, so this module builds one from
the model inputs alone: the CT, the flux and the beam energy.

The construction is a ray-cast pencil-beam model with no fitted parameters:

1. Per lateral ray ``(i, j)``, the cumulative water-equivalent depth
   ``W(s, i, j) = dz * sum_{s' <= s} RSP(CT[s', i, j])`` with the MCsquare-
   calibrated stopping-power conversion the ground truth itself used.
2. Bortfeld's analytic Bragg curve ``D(w; R0)`` in water, with the range
   ``R0 = R80(E)`` from the Grevillot fit (:mod:`src.acquisition.bragg_curve`).
3. ``dose(s, i, j) = flux(s, i, j) * D(W(s, i, j); R0)``.

Because the water-equivalent depth is per ray, a ray through bone peaks shallower
than its neighbour through lung, so the lateral range mixing that the WEPL-spread
metrics measure is present in the surrogate. What is *not* modelled: multiple
Coulomb scattering beyond what the flux already carries, and nuclear halo; both
are second order for locating the peak, which is all the metrics ask of it.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.acquisition.bragg_curve import beam_energy_spread_mev, bragg_curve
from src.processing.mcsquare_calibration import hu_to_rsp_mcsquare
from src.processing.pflugfelder_hi import DEFAULT_ENERGY_MEV
from src.processing.range_energy import energy_to_range_mm


def cumulative_wepl(ct_hu: np.ndarray, dz_mm: float,
                    rsp_energy_mev: float = DEFAULT_ENERGY_MEV) -> np.ndarray:
    """Water-equivalent depth reached at the *exit* of every voxel, ``(D, H, W)``, in mm.

    Voxel ``s`` contributes its full thickness, so ``W[s]`` is the WEPL at depth
    ``(s + 1) * dz``. The evaluation point for the Bragg curve is the voxel
    centre, which :func:`analytic_dose` handles.
    """
    rsp = hu_to_rsp_mcsquare(np.asarray(ct_hu, dtype=np.float64), energy=rsp_energy_mev)
    return np.cumsum(rsp, axis=0) * float(dz_mm)


def analytic_dose(ct_hu: np.ndarray, flux: np.ndarray, energy_mev: float,
                  dz_mm: float, rsp_energy_mev: float = DEFAULT_ENERGY_MEV) -> np.ndarray:
    """The surrogate dose on the beamlet grid, ``(D, H, W)``, normalised to a
    maximum of one.

    Args:
        ct_hu: CT in Hounsfield units, depth first.
        flux: Flux channel on the same grid; only its shape is used, so its
            normalisation does not matter.
        energy_mev: Beam energy, which sets the range.
        dz_mm: Depth voxel size.
        rsp_energy_mev: Reference energy of the stopping-power conversion.
    """
    wepl_exit = cumulative_wepl(ct_hu, dz_mm, rsp_energy_mev)
    # Water-equivalent depth at the voxel centre: half the voxel's own contribution back.
    rsp = hu_to_rsp_mcsquare(np.asarray(ct_hu, dtype=np.float64), energy=rsp_energy_mev)
    wepl_centre = wepl_exit - 0.5 * rsp * float(dz_mm)
    r0_mm = float(energy_to_range_mm(float(energy_mev)))
    spread = beam_energy_spread_mev(float(energy_mev))
    depth_dose = bragg_curve(wepl_centre.ravel(), r0_mm, spread).reshape(wepl_centre.shape)
    dose = np.abs(np.asarray(flux, dtype=np.float64)) * depth_dose
    peak = dose.max()
    return dose / peak if peak > 0 else dose


def bragg_peak_index(dose: np.ndarray) -> Tuple[int, int, int]:
    """``(z, y, x)`` of the Bragg peak the way the training loader finds it.

    Mirrors ``estimate_bragg_peak`` as applied to the *stored* record frame
    ``(y, x, z)``: the lateral ``y`` is the argmax of the y-marginal, then the
    ``(x, z)`` of the maximum within that ``y`` plane. Reproduced here in the
    ``(D, H, W)`` frame so that a surrogate dose centres the 30x30 crop exactly
    where the ground truth would have.
    """
    y_star = int(np.argmax(dose.sum(axis=(0, 2))))
    plane = dose[:, y_star, :]                       # (z, x)
    z_star, x_star = np.unravel_index(int(np.argmax(plane)), plane.shape)
    return int(z_star), y_star, int(x_star)


def crop_like_training(volume: np.ndarray, centre_yx: Tuple[int, int],
                       roi: Tuple[int, int, int] = (160, 30, 30)) -> np.ndarray:
    """The training crop in the ``(D, H, W)`` frame: depth ``[0, roi[0])``, and a
    lateral window centred on ``centre_yx`` with the loader's edge clamping.

    Equivalent to ``cropp_around_index`` on the stored ``(y, x, z)`` frame.
    """
    depth, height, width = roi

    def window(centre: int, size: int, extent: int) -> slice:
        start, end = centre - size // 2, centre + size // 2
        if start <= 0:
            start, end = 0, size
        if end > extent:
            start, end = extent - size, extent
        return slice(start, end)

    return volume[:depth, window(centre_yx[0], height, volume.shape[1]),
                  window(centre_yx[1], width, volume.shape[2])]


def peak_inside_crop(features: Dict[str, float], depth_voxels: int, dz_mm: float) -> bool:
    """Whether the dose that located this record's peak actually stops inside the
    model's crop: the peak slice is not the last one and the distal 10 percent
    crossing was found before the crop ends.

    The study could not evaluate this for a candidate; here it is computed from
    whichever dose each arm used, so the two arms can be compared on beamlets
    where both see a peak, and the surrogate's verdict can be scored against the
    ground truth's as a binary range prediction. A beamlet whose peak leaves the
    crop is not a valid model input at selection time either.
    """
    last_mm = (depth_voxels - 1) * dz_mm
    return bool(features["bp_slice"] < depth_voxels - 1 and features["bp_range_max_mm"] < last_mm)

