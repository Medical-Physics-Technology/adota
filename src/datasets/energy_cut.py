"""Drop beamlets above the energy MCsquare simulates correctly (250 MeV).

MCsquare's ICRU nuclear tables end at 250 MeV; above that its nuclear physics reads
past the tables (:data:`src.mc_generation.mcsquare_runner.MCSQUARE_NUCLEAR_DATA_MAX_MEV`).
The v3 training file holds 853 such beamlets out of 70,146, so their Monte Carlo
"ground truth" is undefined. Training, active-learning splits and evaluation remove
them when they *build* their record lists, by id, so splits that are already frozen
(and integer-indexed lists) are not shifted.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np

from src.mc_generation.mcsquare_runner import MCSQUARE_NUCLEAR_DATA_MAX_MEV

logger = logging.getLogger(__name__)

__all__ = ["MAX_VALID_MC_ENERGY_MEV", "drop_above_energy", "record_energies_mev"]

MAX_VALID_MC_ENERGY_MEV = MCSQUARE_NUCLEAR_DATA_MAX_MEV


def record_energies_mev(dataset_path: Path, record_ids: Sequence[str], scale: Optional[dict] = None) -> np.ndarray:
    """Beam energy (MeV) of each record of a beamlet HDF5.

    v3 files carry ``energy_mev``; older files only the normalised ``initial_energy``,
    which is de-normalised with ``scale`` (``min_energy`` / ``max_energy``).

    Raises:
        KeyError: For a record with no ``energy_mev`` when no ``scale`` is given.
    """
    out = np.empty(len(record_ids), dtype=float)
    with h5py.File(str(dataset_path), "r") as f:
        for i, rid in enumerate(record_ids):
            attrs = f[rid].attrs
            if "energy_mev" in attrs:
                out[i] = float(attrs["energy_mev"])
            elif scale is not None:
                lo, hi = float(scale["min_energy"]), float(scale["max_energy"])
                out[i] = lo + (hi - lo) * float(attrs["initial_energy"])
            else:
                raise KeyError(f"{rid}: no energy_mev attribute; pass the dataset's scale")
    return out


def drop_above_energy(dataset_path: Path, record_ids: Sequence[str], max_energy_mev: Optional[float],
                      scale: Optional[dict] = None) -> Tuple[List[str], List[str]]:
    """``(kept, dropped)``: records at or below ``max_energy_mev``, in input order.

    ``None`` keeps everything (the pre-cut behaviour, for reproducing old runs).
    """
    record_ids = list(record_ids)
    if max_energy_mev is None or not record_ids:
        return record_ids, []
    energies = record_energies_mev(dataset_path, record_ids, scale)
    keep = energies <= float(max_energy_mev)
    kept = [r for r, k in zip(record_ids, keep) if k]
    dropped = [r for r, k in zip(record_ids, keep) if not k]
    if dropped:
        logger.info("energy cut: %d of %d records above %.0f MeV removed (MCsquare's nuclear data ends there)",
                    len(dropped), len(record_ids), max_energy_mev)
    return kept, dropped
