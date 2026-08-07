"""Self-contained MCsquare CT calibration: HU -> RSP (relative stopping power).

Faithful port of OpenTPS's ``MCsquareCTCalibration.convertHU2RSP`` so the ADoTA
project computes water-equivalent path length exactly the way MCsquare did when
it generated the ground-truth dose, without importing OpenTPS at runtime.

The relative stopping power follows the MCsquare/Schneider definition::

    RSP(HU, E) = rho(HU) * SP_material(HU, E) / SP_water(E)

where

* ``rho(HU)``            piecewise-linear HU -> mass density (scanner table),
* ``material(HU)``       step assignment HU -> Schneider material (scanner table:
                         the material of the largest threshold <= HU),
* ``SP_material(., E)``  proton mass stopping power [MeV cm^2/g] at energy E,
                         linearly interpolated from that material's G4 table,
* ``SP_water(E)``        the same for water.

This corrects the previous density-ratio approximation (``RSP ~= rho/rho_water``)
which dropped the stopping-power ratio and overestimated bone RSP by ~15-25%.

The calibration data (the MCsquare ``default`` scanner, matching the DoTA data
generation) lives under ``src/processing/data/mcsquare_default/``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA_DIR = Path(__file__).parent / "data" / "mcsquare_default"

# MCsquare/OpenTPS evaluate the planning RSP at a fixed reference energy; RSP is
# only weakly energy dependent and 100 MeV is the OpenTPS default.
DEFAULT_ENERGY_MEV = 100.0


def _read_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a whitespace-separated 2-column numeric table, skipping comments."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def _read_hu_material(path: Path) -> tuple[np.ndarray, list[str]]:
    """Read HU_Material_Conversion.txt -> (sorted HU thresholds, material names)."""
    hus, names = [], []
    with open(path) as f:
        for raw in f:
            body = raw.split("#", 1)
            data = body[0].strip()
            if not data:
                continue
            parts = data.split()
            if len(parts) < 2:
                continue
            # Material name is taken from the trailing comment (matches the
            # material directory name), e.g. "-950  40  # Schneider_Lung".
            name = body[1].strip() if len(body) > 1 and body[1].strip() else parts[1]
            hus.append(float(parts[0]))
            names.append(name)
    hu_arr = np.asarray(hus, dtype=float)
    order = np.argsort(hu_arr)
    return hu_arr[order], [names[i] for i in order]


class MCsquareCTCalibration:
    """HU -> RSP calibration backed by the MCsquare ``default`` scanner tables."""

    def __init__(self, data_dir: Path = _DATA_DIR):
        self.data_dir = Path(data_dir)
        self._hu_dens, self._dens = _read_two_column(
            self.data_dir / "HU_Density_Conversion.txt"
        )
        self._mat_hu, self._mat_names = _read_hu_material(
            self.data_dir / "HU_Material_Conversion.txt"
        )
        # Per-material proton mass stopping-power tables (energy -> SP).
        self._mat_sp: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in set(self._mat_names) | {"Water"}:
            e, s = _read_two_column(self.data_dir / "materials" / f"{name}.dat")
            self._mat_sp[name] = (e, s)

    # ── density ─────────────────────────────────────────────────────────
    def convert_hu_to_density(self, hu: np.ndarray) -> np.ndarray:
        """Piecewise-linear HU -> mass density [g/cm^3] (extrapolated, >= 0)."""
        dens = np.interp(hu, self._hu_dens, self._dens)
        # np.interp clamps outside the table; OpenTPS extrapolates linearly, but
        # the table is flat at both ends here so clamping is numerically identical.
        return np.clip(dens, 0.0, None)

    # ── stopping power ──────────────────────────────────────────────────
    def _material_sp(self, name: str, energy: float) -> float:
        e, s = self._mat_sp[name]
        return float(np.interp(energy, e, s))

    def water_sp(self, energy: float = DEFAULT_ENERGY_MEV) -> float:
        """Proton mass stopping power of water [MeV cm^2/g] at ``energy``."""
        return self._material_sp("Water", energy)

    def _hu_material_indices(self, hu: np.ndarray) -> np.ndarray:
        """Index of the assigned material: largest threshold <= HU (step)."""
        idx = np.searchsorted(self._mat_hu, hu, side="right") - 1
        return np.clip(idx, 0, len(self._mat_names) - 1)

    def convert_hu_to_sp(
        self, hu: np.ndarray, energy: float = DEFAULT_ENERGY_MEV
    ) -> np.ndarray:
        """HU -> proton mass stopping power [MeV cm^2/g] at ``energy``."""
        sp_per_material = np.array(
            [self._material_sp(n, energy) for n in self._mat_names], dtype=float
        )
        return sp_per_material[self._hu_material_indices(hu)]

    # ── RSP ─────────────────────────────────────────────────────────────
    def convert_hu_to_rsp(
        self, hu: np.ndarray, energy: float = DEFAULT_ENERGY_MEV
    ) -> np.ndarray:
        """HU -> relative (to water) proton stopping power, RSP(HU, E)."""
        hu = np.asarray(hu, dtype=float)
        density = self.convert_hu_to_density(hu)
        sp = self.convert_hu_to_sp(hu, energy)
        return density * sp / self.water_sp(energy)


@lru_cache(maxsize=2)
def get_default_calibration(data_dir: Optional[str] = None) -> MCsquareCTCalibration:
    """Cached MCsquare ``default``-scanner calibration (data loaded once)."""
    return MCsquareCTCalibration(Path(data_dir) if data_dir else _DATA_DIR)


def hu_to_rsp_mcsquare(
    ct_hu: np.ndarray, energy: float = DEFAULT_ENERGY_MEV
) -> np.ndarray:
    """Convenience: HU volume -> RSP using the MCsquare default calibration."""
    return get_default_calibration().convert_hu_to_rsp(ct_hu, energy)
