"""Synthetic HDF5 records for the retrospective-benchmark tests.

A record is stored the way the training set stores it: ``ct``, ``flux`` and
``dose`` as ``(40, 40, 200)`` float32 grids in the ``(y, x, z)`` frame, min-max
normalised, with the beam energy normalised in ``attrs["initial_energy"]``. The
CT is water with a bone slab and a lateral bone half, so the difficulty metrics
have something to measure; the dose is a Bragg-shaped profile along ``z`` so the
validation crop and the range metrics find a peak.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from src.adota.config import DEFAULT_SCALE

STORED_SHAPE = (40, 40, 200)


def _normalise_ct(hu: np.ndarray) -> np.ndarray:
    return (hu - DEFAULT_SCALE["min_ct"]) / (DEFAULT_SCALE["max_ct"] - DEFAULT_SCALE["min_ct"])


def synthetic_record(rng: np.random.Generator, energy_mev: float = 120.0,
                     dose_kind: str = "bragg") -> dict:
    """One stored record. ``dose_kind`` is ``"bragg"`` (a peak inside the crop),
    ``"garbage"`` (random noise, to prove a path never reads it) or ``"nan"``."""
    y, x, z = STORED_SHAPE
    hu = np.zeros(STORED_SHAPE, dtype=np.float32)
    z0 = int(rng.integers(10, 40))
    hu[:, :, z0:z0 + 12] = 800.0                       # a bone slab across the beam
    hu[:, : x // 2, :] += float(rng.uniform(100, 400))  # a denser lateral half
    hu += rng.normal(0.0, 20.0, size=STORED_SHAPE).astype(np.float32)
    yy, xx = np.mgrid[0:y, 0:x]
    lateral = np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
    flux = np.repeat(lateral[:, :, None], z, axis=2)
    depth = np.arange(z, dtype=np.float32)
    peak = 40.0 + 0.6 * (energy_mev - 100.0)
    profile = 0.3 + 0.7 * np.exp(-((depth - peak) ** 2) / (2 * 4.0 ** 2))
    profile[depth > peak + 8] = 0.0
    dose = (flux * profile[None, None, :] * 0.8).astype(np.float32)
    if dose_kind == "garbage":     # a separate stream, so the CT and flux stay identical
        dose = np.random.default_rng(999).random(STORED_SHAPE, dtype=np.float32)
    elif dose_kind == "nan":
        dose = np.full(STORED_SHAPE, np.nan, dtype=np.float32)
    energy_norm = (energy_mev - DEFAULT_SCALE["min_energy"]) / (
        DEFAULT_SCALE["max_energy"] - DEFAULT_SCALE["min_energy"])
    return {"ct": _normalise_ct(hu).astype(np.float32), "flux": flux, "dose": dose,
            "attrs": {"initial_energy": energy_norm, "gantry_angle": float(rng.integers(0, 360)),
                      "beamlet_angles": np.array([0.3, -0.2]), "dose_deposition_ratio": 0.99,
                      "id": "", "stat_uncertainty": np.nan}}


def write_dataset(path: Path, ids: Sequence[str], seed: int = 0, dose_kind: str = "bragg",
                  energies: Sequence[float] | None = None) -> Path:
    rng = np.random.default_rng(seed)
    energies = list(energies) if energies is not None else [
        float(e) for e in rng.uniform(90.0, 150.0, size=len(ids))]
    with h5py.File(path, "w") as handle:
        for sample_id, energy in zip(ids, energies):
            record = synthetic_record(rng, energy, dose_kind)
            group = handle.create_group(sample_id)
            for key in ("ct", "flux", "dose"):
                group.create_dataset(key, data=record[key])
            for key, value in record["attrs"].items():
                group.attrs[key] = sample_id if key == "id" else value
    return path


def write_exclusion_file(path: Path, ids: Sequence[str] = ()) -> Path:
    path.write_text("\n".join(ids) + "\n")
    return path
