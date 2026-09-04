"""Proton energy <-> water-equivalent range conversion.

Self-contained port of OpenTPS ``opentps.core.processing.rangeEnergy`` so the
ADoTA project does not depend on OpenTPS at runtime. The fits (Grevillot et al.)
map proton beam energy [MeV] to the water-equivalent range R80 (the depth of the
distal 80% dose fall-off) and back.

References
----------
L. Grevillot et al., "A Monte Carlo pencil beam scanning model for proton
treatment plan simulation using GATE/GEANT4." Phys Med Biol 56(16):5203-5219,
2011; and Nucl. Instrum. Methods B 268(20):3295-3305, 2010. The coefficients
below are copied verbatim from the OpenTPS implementation.
"""

from __future__ import annotations

from typing import Union

import numpy as np

Number = Union[float, np.ndarray]


def range_to_energy(r80_cm: Number) -> Number:
    """Water-equivalent range R80 [cm] -> proton energy [MeV]."""
    if isinstance(r80_cm, np.ndarray):
        r80 = np.asarray(r80_cm, dtype=float).copy()
        r80[r80 < 1.0] = 1.0
        ln = np.log(r80)
        return np.exp(
            3.464048
            + 0.561372013 * ln
            - 0.004900892 * ln * ln
            + 0.001684756748 * ln * ln * ln
        )
    if r80_cm <= 0.0:
        return 0.0
    ln = np.log(r80_cm)
    return float(
        np.exp(
            3.464048
            + 0.561372013 * ln
            - 0.004900892 * ln * ln
            + 0.001684756748 * ln * ln * ln
        )
    )


def energy_to_range(energy_mev: Number) -> Number:
    """Proton energy [MeV] -> water-equivalent range R80 [cm]."""
    if isinstance(energy_mev, np.ndarray):
        e = np.asarray(energy_mev, dtype=float).copy()
        e[e < 1.0] = 1.0
        ln = np.log(e)
        return np.exp(
            -5.5064 + 1.2193 * ln + 0.15248 * ln * ln - 0.013296 * ln * ln * ln
        )
    if energy_mev <= 0.0:
        return 0.0
    ln = np.log(energy_mev)
    return float(
        np.exp(-5.5064 + 1.2193 * ln + 0.15248 * ln * ln - 0.013296 * ln * ln * ln)
    )


def range_mm_to_energy(r80_mm: Number) -> Number:
    """Water-equivalent range R80 [mm] -> proton energy [MeV]."""
    return range_to_energy(np.asarray(r80_mm, dtype=float) / 10.0 if isinstance(r80_mm, np.ndarray) else r80_mm / 10.0)


def energy_to_range_mm(energy_mev: Number) -> Number:
    """Proton energy [MeV] -> water-equivalent range R80 [mm]."""
    r = energy_to_range(energy_mev)
    return r * 10.0
