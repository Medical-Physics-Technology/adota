"""Synthetic raw beamlet records for the HDF5 builder tests.

A raw record is the triple ``<id>_ct.npy``, ``<id>_ds.npy``, ``<id>_metadata.json``
that the Monte Carlo generator writes. The metadata mirrors the real JSON key for
key so the loader, writer and checker tests exercise the same structure the host
data has, on arrays small enough to build in a test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


def make_metadata(
    sample_id: str,
    *,
    energy_mev: float = 112.3,
    beamlet_angles: Sequence[float] = (-0.7, -0.7),
    gantry_angle: float = 340.0,
    gantry_angle_sim: int = 90,
    isocenter: Sequence[float] = (200.0, 200.0, 123.0),
    image_origin: Sequence[float] = (-199.6, -372.6, -340.5),
    image_size: Sequence[int] = (400, 400, 248),
    image_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    roi_size: Sequence[int] = (80, 80, 400),
    bixel_shift: Sequence[float] = (-24.6, -31.6),
    entrance: Sequence[float] = (37.68, 42.71, 0.0),
    ddr: float = 0.99,
    stat: float = float("nan"),
    n_spots: int = 1,
    num_primaries: float = 1e7,
) -> dict:
    """The generator's metadata JSON with the given values (real key layout)."""
    entrance = [float(v) for v in entrance]
    return {
        "code": 200,
        "status": "success",
        "bdl_file_path": "../../MCsquare/BDL/hptc_beam_model_rsnone.txt",
        "stat_uncertainty": stat,
        "simulation_log": {
            "image_path": "../mcsquare_handler/input.mhd",
            "pencil_plan_path": "../../mcsquare_handler/PlanPencil.txt",
            "simulation_output_dir": "../../mcsquare_handler/",
            "sim_params": {
                "Energy_MHD_Output": True,
                "Num_Primaries": num_primaries,
                "E_Cut_Pro": 0.5,
                "Num_Threads": 0,
                "RNG_Seed": 0,
                "D_Max": 0.2,
                "Epsilon_Max": 0.25,
                "Te_Min": 0.05,
            },
            "bdl_file_path": "../../MCsquare/BDL/hptc_beam_model_rsnone.txt",
            "energy": [float(energy_mev)],
            "bixelgrid_shifts_xy": [[float(v) for v in bixel_shift]] * n_spots,
            "isocenter": [float(v) for v in isocenter],
            "gantry_angle": gantry_angle_sim,
            "handle_results_path": "../../mcsquare_handler/",
            "weights": [1] * n_spots,
            "n_spots": n_spots,
            "beamlet_angles": [float(v) for v in beamlet_angles],
        },
        "dose_deposition_ratio": ddr,
        "id": sample_id,
        "initial_energy": float(energy_mev),
        "gantry_angle": gantry_angle,
        "roi_size": [int(v) for v in roi_size],
        "image_origin": [float(v) for v in image_origin],
        "image_spacing": [float(v) for v in image_spacing],
        "image_size": [int(v) for v in image_size],
        "rays_entrence_point": [0.0, entrance[0], entrance[1]],
        "rays_entrence_point_proj": entrance,
    }


def write_synthetic_record(
    record_dir: Path,
    sample_id: str,
    *,
    shape: Tuple[int, int, int] = (80, 80, 400),
    seed: int = 0,
    dose_scale: float = 1e6,
    zero_dose: bool = False,
    **metadata_kwargs,
) -> dict:
    """Write one raw record into ``record_dir`` and return its metadata dict.

    The CT is int16 in the HU range the scale expects, the dose is non-negative
    float32 (all zeros when ``zero_dose``), and the JSON is ``json.dumps`` of
    :func:`make_metadata`.
    """
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ct = rng.integers(-1024, 3072, size=shape, dtype=np.int16)
    if zero_dose:
        dose = np.zeros(shape, dtype=np.float32)
    else:
        dose = (rng.random(shape, dtype=np.float32) * dose_scale).astype(np.float32)
    metadata = make_metadata(sample_id, roi_size=shape, **metadata_kwargs)
    np.save(record_dir / f"{sample_id}_ct.npy", ct)
    np.save(record_dir / f"{sample_id}_ds.npy", dose)
    (record_dir / f"{sample_id}_metadata.json").write_text(json.dumps(metadata, indent=1))
    return metadata
