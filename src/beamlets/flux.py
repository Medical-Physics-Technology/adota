"""Proton-flux projection for the ADoTA input channel.

Faithful port of datagenerator's ``flux_spatial_spread`` and ``flux_projection``.
The numerics are **frozen**: the ADoTA model was trained on exactly this flux
channel, so the meshgrid/axis order and the nearest-energy spot-size lookup are
kept verbatim. Only typing and documentation are cleaned up.

``flux_spatial_spread`` uses the *nearest* ``MeanEnergy`` row of the BDL (this is
what the training data used); the interpolating :meth:`BeamDataLibrary.spot_sizes`
is a separate, OpenTPS-equivalent accessor and is deliberately not used here.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from src.beamlets.bdl import BeamDataLibrary

logger = logging.getLogger(__name__)

__all__ = ["flux_spatial_spread", "flux_projection", "flux_projection_gpu"]


def flux_spatial_spread(bdl: BeamDataLibrary, energy: float) -> Tuple[float, float]:
    """Spot sigmas ``(sigma_x, sigma_y)`` for the nearest tabulated mean energy.

    Args:
        bdl: The beam data library for this plan.
        energy: Beam energy in MeV.

    Returns:
        ``(SpotSize1x, SpotSize1y)`` of the row whose ``MeanEnergy`` is closest to
        ``energy``.
    """
    table = bdl.energy_table
    closest = table.iloc[(table["MeanEnergy"] - energy).abs().argsort()[:1]]
    sigma_x = float(closest["SpotSize1x"].values[0])
    sigma_y = float(closest["SpotSize1y"].values[0])
    return sigma_x, sigma_y


def flux_projection(
    beamlet_entrence: Sequence[float],
    beamlet_direction: Sequence[float],
    sigmas_xy: Sequence[float],
    shape: Sequence[int],
    initial_energy: Optional[float] = None,
    spacing: np.ndarray = np.asarray([1, 1, 1], dtype=np.float32),
) -> np.ndarray:
    """Generate a proton-flux projection along an angled beamlet.

    Faithful port; the meshgrid construction and rotation order are unchanged.

    Args:
        beamlet_entrence: Beamlet entrance coordinate in crop-local voxels,
            ordered as expected by the caller (``[y, z, x]`` projection).
        beamlet_direction: Beamlet direction angles ``(theta_x, theta_y)`` in
            degrees.
        sigmas_xy: Spatial spread ``(sigma_x, sigma_y)`` of the proton flux.
        shape: Output array shape (the crop's ``(z, y, x)`` numpy shape).
        initial_energy: Optional scalar the flux is multiplied by.
        spacing: Voxel spacing used to convert the sigmas to voxel units.

    Returns:
        The flux projection as a numpy array of the requested ``shape``.
    """
    # Rotation helpers for the angled beamlet.
    R_x = lambda theta: np.array(  # noqa: E731 - kept verbatim from source
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    R_y = lambda theta: np.array(  # noqa: E731 - kept verbatim from source
        [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    )

    x_0, y_0, z_0 = beamlet_entrence
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    z = np.arange(0, shape[2], 1)
    [xx, yy, zz] = np.meshgrid(x, y, z)

    theta_x_deg, theta_y_deg = beamlet_direction
    theta_x = theta_x_deg / 180 * np.pi
    theta_y = theta_y_deg / 180 * np.pi

    [x_t, y_t, z_t] = R_y(theta_x) @ R_x(theta_y) @ np.array(
        [xx.flatten() - x_0, yy.flatten() - y_0, zz.flatten() - z_0]
    )
    x_t = x_t.reshape(xx.shape)
    y_t = y_t.reshape(yy.shape)
    z_t = z_t.reshape(zz.shape)

    sigma_x = sigmas_xy[0] / spacing[0]
    sigma_y = sigmas_xy[1] / spacing[1]
    coef = 1 / (2 * np.pi * sigma_x * sigma_y)
    flux = coef * np.exp(-(x_t**2) / 2 / (sigma_x**2) - y_t**2 / 2 / (sigma_y**2))

    if initial_energy is not None:
        flux = flux * initial_energy

    return flux


def flux_projection_gpu(
    beamlet_entrence: Sequence[float],
    beamlet_direction: Sequence[float],
    sigmas_xy: Sequence[float],
    shape: Sequence[int],
    initial_energy: Optional[float] = None,
    spacing: np.ndarray = np.asarray([1, 1, 1], dtype=np.float32),
    device: str = "cuda",
) -> np.ndarray:
    """GPU/torch twin of :func:`flux_projection` (identical math, on ``device``).

    A drop-in replacement that performs the *exact same* operations as the NumPy
    :func:`flux_projection` -- same ``meshgrid("xy")`` layout, same rotation order
    (``R_y(theta_x) @ R_x(theta_y)``), same coefficient and Gaussian -- but in
    Torch on ``device`` (e.g. ``"cuda"``) so thousands of spots can be computed
    without the per-spot NumPy meshgrid/exp cost.

    The computation is done in **float64** to match the NumPy path; the result is
    returned as a host ``float64`` NumPy array of the requested ``shape``, so the
    saved beamlet is byte-compatible with the CPU path (the residual difference is
    at float64 round-off, far below the float32 the array is stored/consumed at --
    see ``tests/beamlets/test_flux_gpu.py`` for the equivalence proof).

    Args:
        beamlet_entrence: Beamlet entrance coordinate in crop-local voxels
            (``[y, z, x]`` projection), as for :func:`flux_projection`.
        beamlet_direction: Beamlet direction angles ``(theta_x, theta_y)`` [deg].
        sigmas_xy: Spatial spread ``(sigma_x, sigma_y)`` of the proton flux.
        shape: Output array shape (the crop's ``(z, y, x)`` numpy shape).
        initial_energy: Optional scalar the flux is multiplied by.
        spacing: Voxel spacing used to convert the sigmas to voxel units.
        device: Torch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"`` ...).

    Returns:
        The flux projection as a host ``float64`` NumPy array of ``shape``.
    """
    import torch

    dev = torch.device(device)
    dtype = torch.float64

    def R_x(theta: float) -> "torch.Tensor":
        c, s = math.cos(theta), math.sin(theta)
        return torch.tensor(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=dtype, device=dev
        )

    def R_y(theta: float) -> "torch.Tensor":
        c, s = math.cos(theta), math.sin(theta)
        return torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=dtype, device=dev
        )

    x_0, y_0, z_0 = beamlet_entrence
    x = torch.arange(0, shape[1], 1, dtype=dtype, device=dev)
    y = torch.arange(0, shape[0], 1, dtype=dtype, device=dev)
    z = torch.arange(0, shape[2], 1, dtype=dtype, device=dev)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="xy")

    theta_x_deg, theta_y_deg = beamlet_direction
    theta_x = theta_x_deg / 180 * math.pi
    theta_y = theta_y_deg / 180 * math.pi

    coords = torch.stack(
        [xx.flatten() - x_0, yy.flatten() - y_0, zz.flatten() - z_0]
    )
    rotated = R_y(theta_x) @ R_x(theta_y) @ coords
    x_t = rotated[0].reshape(xx.shape)
    y_t = rotated[1].reshape(yy.shape)

    sigma_x = sigmas_xy[0] / spacing[0]
    sigma_y = sigmas_xy[1] / spacing[1]
    coef = 1 / (2 * math.pi * sigma_x * sigma_y)
    flux = coef * torch.exp(
        -(x_t**2) / 2 / (sigma_x**2) - y_t**2 / 2 / (sigma_y**2)
    )

    if initial_energy is not None:
        flux = flux * initial_energy

    return flux.detach().cpu().numpy()
