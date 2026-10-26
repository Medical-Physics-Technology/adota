"""HU -> relative linear stopping power, derived from MCsquare's own physics data.

MCsquare (MHD input, ``data_ct_mhd.c``) gives every voxel a mass density by linear
interpolation of ``HU_Density_Conversion.txt`` and a material by a step lookup in
``HU_Material_Conversion.txt`` (``Binary_Search``: the last row whose HU is
strictly below the voxel's). Each material carries a Geant4 mass stopping-power
table (``G4_Stop_Pow.dat``; ``DB_STOP_POW = SP_GEANT4`` in ``define.h``). The
stopping power relative to water is therefore

    RLSP(HU) = rho(HU) * (S/rho)_material(HU)(E) / (S/rho)_water(E)

evaluated at a reference energy (the ratio varies by well under 1% over the
therapeutic range for tissue). The pencil beam needs exactly this curve so that
its water-equivalent depths match the MC's; DoseCUDA's shipped Hitachi curve would
not. Note that HU 0 is ``Schneider_AT_AG_SI4`` at 1.000 g/cm3 in the default
scanner, not MCsquare's ``Water``, so even a "water" phantom has RLSP != 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "hu_to_rlsp_table",
    "material_stopping_power",
    "read_hu_density",
    "read_hu_material",
    "read_material_list",
    "write_rlsp_csv",
]


def _numeric_rows(path: Path) -> list:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            rows.append(line.split())
    return rows


def read_hu_density(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """``(hu, density)`` from an MCsquare ``HU_Density_Conversion.txt``."""
    rows = _numeric_rows(path)
    return (np.array([float(r[0]) for r in rows]), np.array([float(r[1]) for r in rows]))


def read_hu_material(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """``(hu_thresholds, material_labels)`` from ``HU_Material_Conversion.txt``."""
    rows = _numeric_rows(path)
    return (np.array([float(r[0]) for r in rows]), np.array([int(r[1]) for r in rows]))


def read_material_list(path: Path) -> Dict[int, str]:
    """Material label -> directory name from ``Materials/list.dat``."""
    return {int(r[0]): r[1] for r in _numeric_rows(path) if len(r) >= 2}


def material_stopping_power(materials_dir: Path, name: str, energy_mev: float) -> float:
    """Geant4 mass stopping power (MeV cm2/g) of one material at ``energy_mev``."""
    table = np.loadtxt(Path(materials_dir) / name / "G4_Stop_Pow.dat")
    return float(np.interp(energy_mev, table[:, 0], table[:, 1]))


def hu_to_rlsp_table(
    install_dir: Path,
    scanner: str = "default",
    hu_range: Tuple[int, int] = (-1050, 3071),
    reference_energy_mev: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tabulate RLSP at every integer HU in ``hu_range`` (inclusive).

    One row per HU so that DoseCUDA's linear interpolation between rows
    reproduces MCsquare's material steps to within one HU.

    Args:
        install_dir: MCsquare install (holds ``Scanners/`` and ``Materials/``).
        scanner: Scanner sub-directory.
        hu_range: HU span to tabulate; outside it DoseCUDA clamps to the end rows.
        reference_energy_mev: Energy at which the stopping-power ratio is taken.

    Returns:
        ``(hu, rlsp)`` float arrays.
    """
    install_dir = Path(install_dir)
    scanner_dir = install_dir / "Scanners" / scanner
    materials_dir = install_dir / "Materials"
    hu_d, rho_d = read_hu_density(scanner_dir / "HU_Density_Conversion.txt")
    hu_m, labels = read_hu_material(scanner_dir / "HU_Material_Conversion.txt")
    names = read_material_list(materials_dir / "list.dat")

    s_water = material_stopping_power(materials_dir, "Water", reference_energy_mev)
    ratio = {lab: material_stopping_power(materials_dir, names[lab], reference_energy_mev) / s_water
             for lab in set(labels.tolist())}

    hu = np.arange(hu_range[0], hu_range[1] + 1, dtype=float)
    rho = np.interp(hu, hu_d, rho_d)
    # Binary_Search: largest index with threshold < HU, clamped to [0, n-1].
    index = np.clip(np.searchsorted(hu_m, hu, side="left") - 1, 0, len(hu_m) - 1)
    spr = np.array([ratio[lab] for lab in labels[index]])
    return hu, rho * spr


def write_rlsp_csv(path: Path, hu: np.ndarray, rlsp: np.ndarray) -> None:
    """Write ``HU,RLSP`` in DoseCUDA's ``HU_RLSP.csv`` layout."""
    lines = ["HU,RLSP"] + [f"{h:.2f},{r:.6f}" for h, r in zip(hu, rlsp)]
    Path(path).write_text("\n".join(lines) + "\n")
