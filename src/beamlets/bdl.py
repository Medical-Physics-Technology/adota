"""Beam data library (BDL) parsing and spot/angle conversions.

The BDL holds the gantry geometry (nozzle and scanning-magnet distances) and the
per-energy double-Gaussian spot model. The original datagenerator
``bdl_extract_geo`` fell back to a hardcoded HPTC file whenever the requested path
failed to open, which silently used the wrong geometry for non-HPTC plans
(suspect S1). Here the path is **explicit and required**:
:meth:`BeamDataLibrary.from_file` raises if the file is missing, so each plan uses
its own ``bdl.txt``.

The :class:`BeamDataLibrary` API mirrors OpenTPS' ``BDL`` so the downstream
dose-normalization chain matches OpenTPS exactly. OpenTPS converts an MCsquare
dose grid to Gy with::

    dose_gy = dose * delivered_protons * 1.602176e-19 * 1000 * n_fractions

where ``delivered_protons = Σ_layers meterset * computeMU2Protons(energy)`` and
``computeMU2Protons`` interpolates ``ProtonsMU`` against ``NominalEnergy``.
:meth:`BeamDataLibrary.compute_mu_to_protons` is that primitive; the full
plan-level normalization is assembled in the dose-comparison stage (suspect S7).

The spot-position/angle conversions take ``(d_smx, d_smy)`` directly so they do
not re-read the BDL per spot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import interpolate

logger = logging.getLogger(__name__)

__all__ = [
    "BeamDataLibrary",
    "spot_position_to_angles",
    "angles_to_spot_position",
]

_NOZZLE_PREFIX = "Nozzle"
_SMX_PREFIX = "SMX"
_SMY_PREFIX = "SMY"
_ENERGY_HEADER_PREFIX = "NominalEnergy"


@dataclass(frozen=True)
class BeamDataLibrary:
    """Parsed beam data library (OpenTPS-compatible).

    Attributes:
        nozzle_isocenter: Nozzle-exit-to-isocenter distance (``d_nozzle``).
        smx: Scanning-magnet-x-to-isocenter distance (``d_smx``).
        smy: Scanning-magnet-y-to-isocenter distance (``d_smy``).
        energy_table: Per-energy parameters with the BDL header columns
            (``NominalEnergy``, ``MeanEnergy``, ``ProtonsMU``, ``SpotSize1x``,
            ``SpotSize1y``, ``Divergence1x/1y``, ``Correlation1x/1y``, ...).
        source_path: The file this library was parsed from.
    """

    nozzle_isocenter: float
    smx: float
    smy: float
    energy_table: pd.DataFrame
    source_path: Path

    # -- Geometry convenience -------------------------------------------------

    @property
    def d_nozzle(self) -> float:
        """Alias for :attr:`nozzle_isocenter` (datagenerator naming)."""
        return self.nozzle_isocenter

    @property
    def d_smx(self) -> float:
        """Alias for :attr:`smx` (datagenerator naming)."""
        return self.smx

    @property
    def d_smy(self) -> float:
        """Alias for :attr:`smy` (datagenerator naming)."""
        return self.smy

    @property
    def distances(self) -> Tuple[float, float, float]:
        """Return ``(d_nozzle, d_smx, d_smy)``."""
        return (self.nozzle_isocenter, self.smx, self.smy)

    # -- Energy-table columns -------------------------------------------------

    @property
    def nominal_energy(self) -> np.ndarray:
        """Nominal energies (MeV), the interpolation x-axis."""
        return self.energy_table["NominalEnergy"].to_numpy()

    @property
    def protons_mu(self) -> np.ndarray:
        """Protons per monitor unit, per nominal energy."""
        return self.energy_table["ProtonsMU"].to_numpy()

    # -- OpenTPS-equivalent interpolators -------------------------------------

    def compute_mu_to_protons(self, energy: float) -> float:
        """Protons per MU at ``energy`` (OpenTPS ``computeMU2Protons``).

        Linear interpolation of ``ProtonsMU`` against ``NominalEnergy`` (clamped
        outside the table, matching ``np.interp``).

        Args:
            energy: Beam energy in MeV.

        Returns:
            Protons per MU at the requested energy.
        """
        return float(np.interp(energy, self.nominal_energy, self.protons_mu))

    def spot_sizes(self, energy: float) -> Tuple[float, float]:
        """Spot sigmas ``(sigma_x, sigma_y)`` at ``energy`` (OpenTPS ``spotSizes``).

        Linear interpolation with extrapolation, matching OpenTPS.

        Args:
            energy: Beam energy in MeV.

        Returns:
            ``(SpotSize1x, SpotSize1y)`` at the requested energy.
        """
        return (
            self._interp_column("SpotSize1x", energy),
            self._interp_column("SpotSize1y", energy),
        )

    def divergences(self, energy: float) -> Tuple[float, float]:
        """Beam divergences ``(div_x, div_y)`` at ``energy`` (OpenTPS ``divergences``)."""
        return (
            self._interp_column("Divergence1x", energy),
            self._interp_column("Divergence1y", energy),
        )

    def correlations(self, energy: float) -> Tuple[float, float]:
        """Beam correlations ``(corr_x, corr_y)`` at ``energy`` (OpenTPS ``correlations``)."""
        return (
            self._interp_column("Correlation1x", energy),
            self._interp_column("Correlation1y", energy),
        )

    def _interp_column(self, column: str, energy: float) -> float:
        """Linearly interpolate (with extrapolation) ``column`` at ``energy``."""
        interpolator = interpolate.interp1d(
            self.nominal_energy,
            self.energy_table[column].to_numpy(),
            kind="linear",
            fill_value="extrapolate",
        )
        return float(interpolator(energy))

    # -- Construction ---------------------------------------------------------

    @classmethod
    def from_file(cls, bdl_path: Path) -> "BeamDataLibrary":
        """Parse a beam data library text file.

        Args:
            bdl_path: Path to the BDL file (e.g. the plan-local ``bdl.txt``).

        Returns:
            The parsed :class:`BeamDataLibrary`.

        Raises:
            FileNotFoundError: If ``bdl_path`` does not exist.
            ValueError: If the geometry block or energy table is missing.
        """
        bdl_path = Path(bdl_path)
        if not bdl_path.is_file():
            raise FileNotFoundError(f"Beam data library not found: {bdl_path}")

        lines = bdl_path.read_text().splitlines()

        nozzle_isocenter = _value_after_prefix(lines, _NOZZLE_PREFIX, bdl_path)
        smx = _value_after_prefix(lines, _SMX_PREFIX, bdl_path)
        smy = _value_after_prefix(lines, _SMY_PREFIX, bdl_path)

        header_idx = _find_index(lines, _ENERGY_HEADER_PREFIX)
        if header_idx is None:
            raise ValueError(
                f"No '{_ENERGY_HEADER_PREFIX}' energy table header in {bdl_path}"
            )
        columns = lines[header_idx].split()
        rows = [
            [float(v) for v in line.split()]
            for line in lines[header_idx + 1 :]
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"Empty energy table in {bdl_path}")
        energy_table = pd.DataFrame(np.asarray(rows), columns=columns)

        logger.info(
            "BDL %s: d_nozzle=%.1f d_smx=%.1f d_smy=%.1f, %d energies",
            bdl_path.name,
            nozzle_isocenter,
            smx,
            smy,
            len(energy_table),
        )
        return cls(
            nozzle_isocenter=nozzle_isocenter,
            smx=smx,
            smy=smy,
            energy_table=energy_table,
            source_path=bdl_path,
        )


def _find_index(lines: list[str], prefix: str) -> int | None:
    """Return the index of the last line starting with ``prefix``, or None.

    The geometry labels (``Nozzle``/``SMX``/``SMY``) can appear earlier inside a
    descriptive header, so the *last* match is the authoritative value line; the
    energy header is unique so first/last coincide.
    """
    found = None
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            found = i
    return found


def _value_after_prefix(lines: list[str], prefix: str, bdl_path: Path) -> float:
    """Return the float on the line following the last line matching ``prefix``."""
    idx = _find_index(lines, prefix)
    if idx is None or idx + 1 >= len(lines):
        raise ValueError(f"Missing '{prefix}' geometry value in {bdl_path}")
    return float(lines[idx + 1])


def spot_position_to_angles(
    y_spot_position: float,
    z_spot_position: float,
    d_smx: float,
    d_smy: float,
) -> Tuple[float, float]:
    """Convert a bixelgrid shift to beamlet angles (degrees).

    Args:
        y_spot_position: Bixelgrid shift along the y-axis (DICOM frame).
        z_spot_position: Bixelgrid shift along the z-axis (DICOM frame).
        d_smx: Scanning-magnet-x-to-isocenter distance.
        d_smy: Scanning-magnet-y-to-isocenter distance.

    Returns:
        ``(theta_y, theta_z)`` in degrees.
    """
    theta_y = np.rad2deg(np.arctan(z_spot_position / d_smy))
    theta_z = np.rad2deg(np.arctan(y_spot_position / d_smx))
    return (theta_y, theta_z)


def angles_to_spot_position(
    theta_y: float,
    theta_z: float,
    d_smx: float,
    d_smy: float,
) -> Tuple[float, float]:
    """Inverse of :func:`spot_position_to_angles`.

    Args:
        theta_y: Beamlet angle for the y-shift, in degrees.
        theta_z: Beamlet angle for the z-shift, in degrees.
        d_smx: Scanning-magnet-x-to-isocenter distance.
        d_smy: Scanning-magnet-y-to-isocenter distance.

    Returns:
        ``(y_spot_position, z_spot_position)``.
    """
    z_spot_position = d_smy * np.tan(np.deg2rad(theta_y))
    y_spot_position = d_smx * np.tan(np.deg2rad(theta_z))
    return (y_spot_position, z_spot_position)
