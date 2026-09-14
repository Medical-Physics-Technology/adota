"""Candidate generation, and scoring a whole pool of CTs before any simulation.

A candidate is ``(CT, gantry, energy, theta_x, theta_y)``. Version 0 of the
generator, the one the design fixed for the first runs: gantry uniform at random
(the CT is rotated to the canonical beam's-eye frame, so gantry is metadata),
isocenter at the grid centre, energy from a discrete layer set spanning the training
range, steering uniform on the generator's lattice inside the screened window.

Candidates are **content-addressed**: the id is a hash of the five numbers that
define the beamlet, so the same candidate selected in two cycles maps to the same
files, is simulated once, and resume is free.

Scoring runs one CT per worker process. The cost per CT is dominated by
:func:`src.mc_generation.robustness.field_geometry` (one CT rotation per gantry), so
candidates are drawn a few gantries at a time rather than one gantry each.
"""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.acquisition import BeamletCandidate, DifficultyScorer, prepare_ct, score_candidates
from src.acquisition.features import FeatureConfig
from src.active_learning.pool import PoolEntry, RecordResolver
from src.beamlets.bdl import BeamDataLibrary
from src.mc_generation.sweep import RobustnessConfig, angle_tag, energy_tag

logger = logging.getLogger(__name__)


@dataclass
class CandidateConfig:
    """Version-0 candidate generation."""

    n_gantry_per_ct: int = 4
    n_per_field: int = 64
    """Candidates per (CT, gantry). The CT rotation is paid once per gantry, so this
    is the knob that trades pool size against scoring time."""
    energies: List[float] = field(default_factory=lambda: [90.0, 110.0, 130.0, 150.0, 170.0])
    theta_half_range_deg: float = 1.5
    """Steering window. 1.5 rather than 2.0: at 2 degrees the corner beamlets walk
    out of thin thoracic anatomy (the robustness screening result)."""
    lattice_n: int = 18
    gantry_min_deg: float = 0.0
    gantry_max_deg: float = 360.0
    seed: int = 20260908


def _stable_seed(*parts) -> int:
    """A reproducible seed from arbitrary parts.

    ``hash()`` on a string is salted per interpreter, so it cannot be used here: the
    gantries are drawn inside worker processes and must match a rerun's.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha1(key).digest()[:8], "big")


def candidate_id(patient_id: str, gantry: float, energy: float, tx: float, ty: float) -> str:
    """Stable, filename-safe id for one beamlet on one patient."""
    key = f"{patient_id}|{gantry:.4f}|{energy:.4f}|{tx:.5f}|{ty:.5f}"
    return "c" + hashlib.sha1(key.encode()).hexdigest()[:12]


def field_dir_name(anatomy: str, patient_id: str, energy: float, gantry: float,
                   prefix: str = "al", version: Optional[int] = 1) -> str:
    """Output directory for one (patient, gantry, energy), in generator style."""
    name = f"{prefix}_{anatomy}_{patient_id}_e{energy_tag(energy)}_g{angle_tag(gantry)}"
    return name if version is None else f"{name}_v{version}"


def draw_gantries(cfg: CandidateConfig, patient_uid: str) -> List[float]:
    """The field angles for one CT: seeded per patient, so a rerun redraws nothing."""
    rng = np.random.default_rng(_stable_seed(cfg.seed, patient_uid, "gantry"))
    return [float(g) for g in rng.uniform(cfg.gantry_min_deg, cfg.gantry_max_deg,
                                          size=cfg.n_gantry_per_ct)]


def generate_candidates(patient_id: str, patient_uid: str,
                        cfg: CandidateConfig) -> List[BeamletCandidate]:
    """Version-0 candidates for one CT, grouped by the gantries drawn for it."""
    lattice = np.linspace(-cfg.theta_half_range_deg, cfg.theta_half_range_deg, cfg.lattice_n)
    rng = np.random.default_rng(_stable_seed(cfg.seed, patient_uid, "candidates"))
    out: List[BeamletCandidate] = []
    for gantry in draw_gantries(cfg, patient_uid):
        for _ in range(cfg.n_per_field):
            energy = float(rng.choice(cfg.energies))
            tx = float(rng.choice(lattice))
            ty = float(rng.choice(lattice))
            out.append(BeamletCandidate(
                gantry_deg=gantry, energy_mev=energy, theta_x_deg=tx, theta_y_deg=ty,
                candidate_id=candidate_id(patient_id, gantry, energy, tx, ty)))
    return out


def score_one_ct(entry: PoolEntry, cand_cfg: CandidateConfig, rob_cfg: RobustnessConfig,
                 bdl_path: str, feature_config: Optional[FeatureConfig] = None,
                 prefix: str = "al", version: Optional[int] = 1) -> pd.DataFrame:
    """Generate and score every candidate on one CT. One row per candidate.

    Runs in a worker process: it loads its own CT and its own beam model, and returns
    a frame rather than any array, so nothing large crosses the process boundary.
    """
    resolver = RecordResolver([entry])
    rec = resolver.record(entry)
    bdl = BeamDataLibrary.from_file(bdl_path)
    candidates = generate_candidates(rec.patient_id, rec.uid, cand_cfg)
    ct = prepare_ct(rec.load_image(), rob_cfg.iso_spacing_mm)
    frame = score_candidates(ct, candidates, bdl, {"full": DifficultyScorer.load()},
                             rob_cfg, feature_config or FeatureConfig(), prepared=True)
    # score_candidates already carries candidate_id through from the dataclass, in
    # input order; check that rather than re-inserting it, so a future change to
    # either side shows up here instead of silently misaligning ids and scores.
    if list(frame["candidate_id"]) != [c.candidate_id for c in candidates]:
        raise RuntimeError("scored candidates came back out of order")
    frame.insert(1, "patient_id", rec.patient_id)
    frame.insert(2, "anatomy", rec.anatomy)
    frame.insert(3, "dataset_name", rec.dataset_name)
    frame.insert(4, "provenance_uid", rec.uid)
    frame["field_dir"] = [field_dir_name(rec.anatomy, rec.patient_id, e, g, prefix, version)
                          for e, g in zip(frame["energy_mev"], frame["gantry_deg"])]
    return frame


def _score_one_ct_star(args) -> pd.DataFrame:
    return score_one_ct(*args)


def pin_worker_threads() -> None:
    """One thread per worker, for the libraries that would otherwise take all of them.

    SimpleITK defaults its global thread pool to every core (48 here), and each worker
    resamples and rotates a CT, so an unpinned pool of ten workers asks for 480
    threads and the machine spends its time context-switching. Both settings are
    runtime calls rather than environment variables on purpose: a forked worker has
    already imported these libraries, so the usual ``OMP_NUM_THREADS`` trick would
    come too late to have any effect.
    """
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def score_pool(entries: Sequence[PoolEntry], cand_cfg: CandidateConfig,
               rob_cfg: RobustnessConfig, bdl_path: str, *, n_workers: int = 8,
               prefix: str = "al", version: Optional[int] = 1) -> pd.DataFrame:
    """Score every candidate on every CT, one worker process per CT.

    Each worker is pinned to a single thread by :func:`pin_worker_threads`; without
    that the pool oversubscribes the machine by an order of magnitude.
    """
    payload = [(e, cand_cfg, rob_cfg, bdl_path, None, prefix, version) for e in entries]
    frames: List[pd.DataFrame] = []
    if n_workers <= 1:
        for args in payload:
            frames.append(_score_one_ct_star(args))
            logger.info("scored %s (%d/%d)", args[0].patient_id, len(frames), len(payload))
    else:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=pin_worker_threads) as pool:
            for frame in pool.map(_score_one_ct_star, payload):
                frames.append(frame)
                logger.info("scored %s (%d/%d)", frame["patient_id"].iloc[0],
                            len(frames), len(payload))
    table = pd.concat(frames, ignore_index=True)
    n_valid = int(table["valid"].sum())
    logger.info("candidate pool: %d scored, %d valid (%.1f%%), %d CTs",
                len(table), n_valid, 100.0 * n_valid / max(len(table), 1), len(entries))
    return table
