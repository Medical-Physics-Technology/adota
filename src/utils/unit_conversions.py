"""Physical unit conversions for MCsquare (and ADoTA, PBA) beamlet doses.

**What MCsquare writes.** ``Dose.mhd`` is energy per mass per *simulated* proton,
in eV/g (``PostProcess_Scoring`` divides by the number of simulated primaries,
normalisation 1.0). The plan's spot weights do not scale it. MCsquare itself only
converts to Gy for its DVHs (``compute_simulation.c``):

    Gy = dose * 1.602176e-19 [J/eV] * 1000 [g/kg] * delivered_protons * fractions

with ``delivered_protons = sum(MU * protons_per_MU(E))``. For a UPenn-format BDL
(ours) ``protons_per_MU`` is the BDL's ``ProtonsMU`` column, interpolated linearly
in nominal energy and **extrapolated** linearly from the outermost two nodes
(``data_beam_model.c``). Plan-level doses use the same chain in
:mod:`src.beamlets.dose_scaling`.

ADoTA learns MCsquare's beamlet doses, and the pencil beam is commissioned onto
them, so all three share that unit.

**Reporting a single beamlet in Gy** needs a number of protons. The convention
here is **Gy per MU** (:func:`mcsquare_to_gy_per_mu`): the dose one monitor unit of
that beamlet delivers, with MCsquare's own protons-per-MU.

**Legacy.** The earlier ``to_gy`` multiplied by 1.60218e-10, treating the value as
MeV/g. That equals the dose of exactly 1e6 protons. It is kept, renamed, as
:func:`legacy_to_gy_per_1e6_protons`, so numbers already reported with it stay
reproducible; ``to_gy`` remains as a deprecated alias of it.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence, Union

import numpy as np

__all__ = [
    "EV_PER_G_TO_GY",
    "legacy_to_gy_per_1e6_protons",
    "mcsquare_protons_per_mu",
    "mcsquare_to_gy_per_mu",
    "mcsquare_to_gy_per_proton",
    "to_gy",
]

# 1 eV/g = 1.602176e-19 J / 1e-3 kg; MCsquare's own constant (compute_simulation.c).
EV_PER_G_TO_GY = 1.602176e-19 * 1000.0
_LEGACY_FACTOR = 1.60218e-10          # the old to_gy: eV/g per proton x 1e6 protons

ArrayLike = Union[float, np.ndarray]


def mcsquare_to_gy_per_proton(dose_ev_per_g: ArrayLike) -> ArrayLike:
    """MCsquare beamlet dose (eV/g per simulated proton) to Gy per proton."""
    return dose_ev_per_g * EV_PER_G_TO_GY


def mcsquare_protons_per_mu(energy_mev: float, nominal_energies: Sequence[float],
                            protons_per_mu: Sequence[float]) -> float:
    """Protons per MU at ``energy_mev`` exactly as MCsquare computes it (UPenn BDL).

    Linear interpolation between the bracketing BDL nodes, and linear
    *extrapolation* from the first or last two nodes outside the table (MCsquare
    clamps the node index, not the value). This differs from
    :meth:`src.beamlets.bdl.BeamDataLibrary.compute_mu_to_protons`, which clamps the
    value (the OpenTPS convention) and so disagrees above the last node.
    """
    e = np.asarray(nominal_energies, dtype=float)
    p = np.asarray(protons_per_mu, dtype=float)
    i = int(np.clip(np.searchsorted(e, energy_mev, side="right") - 1, 0, len(e) - 2))
    w = (energy_mev - e[i]) / (e[i + 1] - e[i])
    return float(p[i] + w * (p[i + 1] - p[i]))


def mcsquare_to_gy_per_mu(dose_ev_per_g: ArrayLike, energy_mev: float, bdl) -> ArrayLike:
    """MCsquare beamlet dose (eV/g per simulated proton) to Gy per MU of that beamlet.

    Args:
        dose_ev_per_g: MCsquare-scale dose (MC, ADoTA, or PBA in MCsquare units).
        energy_mev: The beamlet's nominal energy.
        bdl: A :class:`~src.beamlets.bdl.BeamDataLibrary` (UPenn format) or the path
            to its file; ``ProtonsMU`` is taken from it as MCsquare does.

    Raises:
        ValueError: If the BDL is not in the UPenn format MCsquare reads ProtonsMU from.
    """
    if isinstance(bdl, (str, Path)):
        from src.beamlets.bdl import BeamDataLibrary

        bdl_path = Path(bdl)
        bdl = BeamDataLibrary.from_file(bdl_path)
    else:
        bdl_path = getattr(bdl, "source_path", None)
    if bdl_path is not None and Path(bdl_path).exists():
        header = Path(bdl_path).read_text().splitlines()[0].strip()
        if header not in ("--UPenn beam model (double gaussian)--", "--Lookup table BDL format--"):
            raise ValueError(f"{bdl_path}: not a UPenn BDL, so MCsquare does not use its ProtonsMU")
    ppm = mcsquare_protons_per_mu(energy_mev, bdl.nominal_energy, bdl.protons_mu)
    return dose_ev_per_g * EV_PER_G_TO_GY * ppm


def legacy_to_gy_per_1e6_protons(dose: ArrayLike) -> ArrayLike:
    """The former ``to_gy``: ``dose * 1.60218e-10``, i.e. Gy for exactly 1e6 protons.

    Kept so that numbers reported with it reproduce exactly. New results should use
    :func:`mcsquare_to_gy_per_mu` (or :mod:`src.beamlets.dose_scaling` for plans).
    """
    return dose * _LEGACY_FACTOR


def to_gy(dose_in_mev_per_g: ArrayLike) -> ArrayLike:
    """Deprecated alias of :func:`legacy_to_gy_per_1e6_protons` (it was never MeV/g)."""
    warnings.warn("to_gy is Gy per 1e6 protons, not a MeV/g conversion; use mcsquare_to_gy_per_mu, or "
                  "legacy_to_gy_per_1e6_protons to reproduce old numbers", DeprecationWarning, stacklevel=2)
    return legacy_to_gy_per_1e6_protons(dose_in_mev_per_g)
