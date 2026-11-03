"""Check that a mixed-energy beamlet-mode call gives the same doses as single-energy calls.

The pool gives every beamlet its own energy and runs a whole field in one MCsquare
beamlet-mode call, with one control point per energy
(:func:`src.mc_generation.config_writer.build_multi_energy_plan_text`). Before
that is used in production, the same beamlets are simulated both ways on one
field and compared per beamlet: integral dose ratio, R80 difference and the
gamma pass rate at 2%/2 mm (1 mm grid, 10% cutoff). The two runs differ only in
their random streams, so the differences are Monte Carlo noise at the chosen
primaries, and the thresholds are set for that.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.acquisition import prepare_ct
from src.active_learning.pool import PoolEntry, RecordResolver
from src.beamlets.bdl import BeamDataLibrary, angles_to_spot_position
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.robustness import field_geometry
from src.mc_generation.sweep import RobustnessConfig
from src.metrics.gamma_pass_rate import gamma_index
from src.metrics.range_metrics import compute_range_metrics

logger = logging.getLogger(__name__)

GAMMA = {"dose_percent_threshold": 2, "distance_mm_threshold": 2, "interp_fraction": 10, "max_gamma": 2,
         "lower_percent_dose_cutoff": 10, "random_subset": None, "local_gamma": False, "quiet": True}
PASS = {"min_gamma_pass": 0.99, "max_integral_ratio_error": 0.01, "max_abs_dr80_mm": 1.0}


def _idd(dose: np.ndarray) -> np.ndarray:
    return dose.sum(axis=(0, 1))      # sitk array is (z, y, x); the beam runs along x after rotation


def _beam_box(dose: np.ndarray, margin: int = 5) -> tuple:
    """Bounding box of the voxels above 0.1% of the maximum, with a margin."""
    idx = np.argwhere(dose > 1e-3 * float(dose.max()))
    lo, hi = idx.min(axis=0) - margin, idx.max(axis=0) + margin + 1
    return tuple(slice(max(0, int(a)), int(b)) for a, b in zip(lo, hi))


def compare_mixed_energy(rows: pd.DataFrame, entry: PoolEntry, runner: MCSquareRunner, bdl: BeamDataLibrary,
                         cfg: RobustnessConfig) -> pd.DataFrame:
    """Simulate ``rows`` (one patient, one gantry, distinct energies) both ways and compare."""
    if rows["gantry_deg"].nunique() != 1 or rows["patient_id"].nunique() != 1:
        raise ValueError("the check needs the beamlets of one patient and one field")
    rec = RecordResolver([entry]).record(entry)
    ct = prepare_ct(rec.load_image(), cfg.iso_spacing_mm)
    geom = field_geometry(ct, float(rows["gantry_deg"].iloc[0]), cfg, bdl)
    spots = [angles_to_spot_position(tx, ty, bdl.d_smx, bdl.d_smy)
             for tx, ty in zip(rows["theta_x_deg"], rows["theta_y_deg"])]
    energies = rows["energy_mev"].astype(float).tolist()
    common = dict(gantry_angle=geom.mc_gantry, isocenter=geom.iso_mc, num_primaries=cfg.num_primaries,
                  num_threads=cfg.num_threads, rng_seed=cfg.rng_seed)
    mixed = {i: sitk.GetArrayFromImage(d) for i, d, _ in
             runner.run_beamlet_field(geom.ct, energy=energies, spots_xy=spots, **common)}
    out: List[Dict] = []
    for i, (energy, spot) in enumerate(zip(energies, spots)):
        single = next(sitk.GetArrayFromImage(d) for _, d, _ in
                      runner.run_beamlet_field(geom.ct, energy=energy, spots_xy=[spot], **common))
        box = _beam_box(single)
        a, b = single[box], mixed[i][box]
        _, rates = gamma_index(a.astype(np.float64), b.astype(np.float64), {"y_max": float(a.max()), "y_min": 0.0},
                               dict(GAMMA), (1.0, 1.0, 1.0), cutoff=10.0, backend="torch")
        r_a = compute_range_metrics(_idd(a), 1.0).r80_mm
        r_b = compute_range_metrics(_idd(b), 1.0).r80_mm
        out.append({"candidate_id": rows["candidate_id"].iloc[i], "energy_mev": energy,
                    "integral_ratio": float(b.sum() / a.sum()), "dr80_mm": float(r_b - r_a),
                    "gamma_pass": float(rates[0])})
        logger.info("check %s %.1f MeV: integral %.4f, dR80 %+.2f mm, gamma %.4f", out[-1]["candidate_id"],
                    energy, out[-1]["integral_ratio"], out[-1]["dr80_mm"], out[-1]["gamma_pass"])
    frame = pd.DataFrame(out)
    frame["passes"] = ((frame["gamma_pass"] >= PASS["min_gamma_pass"])
                       & ((frame["integral_ratio"] - 1).abs() <= PASS["max_integral_ratio_error"])
                       & (frame["dr80_mm"].abs() <= PASS["max_abs_dr80_mm"]))
    return frame
