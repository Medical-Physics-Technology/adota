"""Labelling a selected batch: Monte Carlo through the existing generator.

The loop buys labels by handing its chosen candidates to the same MCsquare path that
produced the reference dataset (:func:`src.mc_generation.robustness.generate_beamlets`),
so a beamlet bought here is written exactly like a beamlet of the training set: same
crop, same flux construction, same QA gates, same ``sim_res`` fields.

Work is grouped by ``(patient, gantry, energy)``, because that grouping is what the
cost is made of. Each group pays one CT rotation into the beam's-eye frame and one
MCsquare setup (about 12 s) regardless of how many beamlets it holds, so a batch
spread thinly over many groups spends more on setup than on physics. The loop keeps
groups fat by sampling a subset of pool CTs per cycle rather than a few beamlets on
every CT; :func:`batch_cost_estimate` reports the split before anything is simulated.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.active_learning.pool import PoolEntry, RecordResolver
from src.beamlets.bdl import BeamDataLibrary
from src.mc_generation.geometry import reduce_vacuum_to_air, resample_to_isotropic
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.robustness import field_geometry, generate_beamlets
from src.mc_generation.sweep import RobustnessConfig

logger = logging.getLogger(__name__)

SETUP_SECONDS_PER_GROUP = 12.0
"""Measured MCsquare per-run CT/material/scoring setup, paid once per group."""

SECONDS_PER_BEAMLET = 3.3
"""Measured in beamlet mode at 1e6 primaries (``mc_seconds`` of the energy-gantry
run). Only used for the pre-flight estimate; the manifest records what was spent."""


def _lattice_index(value: float, lattice: np.ndarray) -> int:
    return int(np.argmin(np.abs(lattice - value)))


def batch_cost_estimate(chosen: pd.DataFrame) -> dict:
    """Groups, setup seconds and Monte Carlo seconds this batch will cost."""
    groups = chosen.groupby(["patient_id", "gantry_deg", "energy_mev"]).size()
    setup = SETUP_SECONDS_PER_GROUP * len(groups)
    mc = SECONDS_PER_BEAMLET * len(chosen)
    return {"n_beamlets": int(len(chosen)), "n_groups": int(len(groups)),
            "beamlets_per_group_mean": float(groups.mean()) if len(groups) else 0.0,
            "beamlets_per_group_min": int(groups.min()) if len(groups) else 0,
            "estimated_setup_seconds": float(setup),
            "estimated_mc_seconds": float(mc),
            "estimated_total_hours": float((setup + mc) / 3600.0)}


def label_batch(
    chosen: pd.DataFrame,
    entries: Sequence[PoolEntry],
    runner: MCSquareRunner,
    bdl: BeamDataLibrary,
    cfg: RobustnessConfig,
    output_root: Path,
    *,
    lattice: Optional[np.ndarray] = None,
    resolver: Optional[RecordResolver] = None,
) -> dict:
    """Simulate every selected candidate and write it to its field directory.

    Args:
        chosen: Selected rows, carrying ``candidate_id``, ``patient_id``,
            ``gantry_deg``, ``energy_mev``, ``theta_x_deg``, ``theta_y_deg`` and
            ``field_dir``.
        entries: The pool rows the candidates came from, for CT resolution.
        runner: The MCsquare boundary.
        bdl: The beam model.
        cfg: Generator settings; ``beamlet_mode`` and ``beamlet_block_size`` decide
            how the groups are sent to MCsquare.
        output_root: Where the field directories live.
        lattice: The steering lattice the candidates were drawn on, used to record a
            grid index in ``sim_res``. Defaults to the generator's own lattice.
        resolver: Reused CT resolver, so a multi-cycle loop scans DICOM once.

    Returns:
        Totals plus a per-group breakdown, including measured Monte Carlo seconds
        read back from the beamlets that were written.
    """
    if lattice is None:
        lattice = np.linspace(cfg.theta_x_range[0], cfg.theta_x_range[1], cfg.grid_n)
    resolver = resolver or RecordResolver(entries)
    by_patient = {e.patient_id: e for e in entries}
    started = time.perf_counter()
    totals = {"saved": 0, "qa_skipped": 0, "existing": 0}
    groups: List[dict] = []

    for patient_id, per_patient in chosen.groupby("patient_id", sort=False):
        entry = by_patient.get(str(patient_id))
        if entry is None:
            raise KeyError(f"{patient_id} is not in the pool selection")
        rec = resolver.record(entry)
        ct = reduce_vacuum_to_air(resample_to_isotropic(rec.load_image(), cfg.iso_spacing_mm))
        for gantry, per_field in per_patient.groupby("gantry_deg", sort=False):
            geom = field_geometry(ct, float(gantry), cfg, bdl)
            for energy, block in per_field.groupby("energy_mev", sort=False):
                cells: List[Tuple[int, int, float, float]] = []
                stems: List[str] = []
                for row in block.itertuples(index=False):
                    tx, ty = float(row.theta_x_deg), float(row.theta_y_deg)
                    cells.append((_lattice_index(tx, lattice), _lattice_index(ty, lattice),
                                  tx, ty))
                    stems.append(str(row.candidate_id))
                out_dir = Path(output_root) / str(block["field_dir"].iloc[0])
                stats = generate_beamlets(rec, runner, bdl, cfg, geom, float(energy),
                                          cells, out_dir, stems)
                for key in totals:
                    totals[key] += stats[key]
                groups.append({"patient_id": str(patient_id), "gantry_deg": float(gantry),
                               "energy_mev": float(energy), "n": int(len(cells)), **stats})
                logger.info("  %s g%.1f e%g: saved=%d qa=%d existing=%d (%d/%d groups)",
                            patient_id, float(gantry), float(energy), stats["saved"],
                            stats["qa_skipped"], stats["existing"], len(groups),
                            len(chosen.groupby(["patient_id", "gantry_deg", "energy_mev"])))

    wall = time.perf_counter() - started
    mc_seconds = read_mc_seconds(chosen, output_root)
    logger.info("labelled %d beamlets (%d simulated, %d already on disk) in %.2f h wall "
                "(%.0f s of MC), qa_skipped=%d",
                totals["saved"] + totals["existing"], totals["saved"], totals["existing"],
                wall / 3600.0, mc_seconds, totals["qa_skipped"])
    return {**totals, "n_groups": len(groups), "wall_seconds": float(wall),
            "mc_seconds": float(mc_seconds), "groups": groups}


def read_mc_seconds(chosen: pd.DataFrame, output_root: Path) -> float:
    """Sum the Monte Carlo seconds actually spent on a batch, from its ``sim_res``."""
    total = 0.0
    for row in chosen.itertuples(index=False):
        path = Path(output_root) / str(row.field_dir) / f"{row.candidate_id}_sim_res.json"
        if not path.exists():
            continue
        try:
            total += float(json.loads(path.read_text()).get("mc_seconds") or 0.0)
        except (ValueError, OSError):
            continue
    return total


def labelled_records(chosen: pd.DataFrame, output_root: Path) -> List[Tuple[str, str]]:
    """``(field_dir, stem)`` for every candidate that survived to disk.

    A selected candidate can be missing: the QA gates (deposition ratio, dose in the
    crop) run after the simulation and drop beamlets the input-only validity check
    could not see. Those are paid for and not trained on, and the count is the
    difference the cycle manifest records.
    """
    out: List[Tuple[str, str]] = []
    for row in chosen.itertuples(index=False):
        directory = Path(output_root) / str(row.field_dir)
        if (directory / f"{row.candidate_id}_sim_res.json").exists():
            out.append((str(directory), str(row.candidate_id)))
    return out


def group_sizes(chosen: pd.DataFrame) -> Dict[str, int]:
    """Group-size histogram, for the pre-flight report."""
    sizes = chosen.groupby(["patient_id", "gantry_deg", "energy_mev"]).size()
    return {"n_groups": int(len(sizes)), "min": int(sizes.min()) if len(sizes) else 0,
            "median": int(sizes.median()) if len(sizes) else 0,
            "max": int(sizes.max()) if len(sizes) else 0}
