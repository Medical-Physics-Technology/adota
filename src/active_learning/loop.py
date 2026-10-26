"""The active-learning cycle: sample, label, retrain, validate, record.

One cycle is::

    generate candidates on pool CTs -> score (input-only) -> strategy -> batch
    label the batch with Monte Carlo -> add it to the training sources
    retrain from the previous cycle's weights -> validate on the frozen set
    write the cycle manifest

The manifest is what makes the loop resumable and the run readable: every cycle
records what was selected, what it cost in Monte Carlo seconds, and what the model
scored afterwards. A crashed run restarts from the last complete cycle, and the
Monte Carlo itself is resumable per beamlet underneath, so an interrupted labelling
phase re-simulates only what is missing.

The budget axis of every plot is **Monte Carlo seconds**, not cycle count and not
sample count, because that is what a beamlet actually costs.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.active_learning.candidates import CandidateConfig, score_pool
from src.active_learning.dataset import read_training_sources, write_training_sources
from src.active_learning.oracle import batch_cost_estimate, label_batch, labelled_records
from src.active_learning.pool import PoolEntry, RecordResolver, read_pool
from src.active_learning.sampling import select, selection_fingerprint
from src.active_learning.training import cycle_checkpoint, cycle_training_config, run_training
from src.active_learning.validation import evaluate_frozen_set, summarise, summarise_by
from src.adota.utils import load_model
from src.beamlets.bdl import BeamDataLibrary
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.sweep import RobustnessConfig

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Everything one arm of the loop needs. One arm is one strategy."""

    name: str = "al"
    strategy: str = "score"
    arm: str = "warm"
    n_cycles: int = 2
    beamlets_per_cycle: int = 2500
    n_cts_per_cycle: int = 12
    """CTs drawn from the pool per cycle. Fewer CTs means fatter Monte Carlo groups
    and less setup overhead; more CTs means a more diverse batch. See
    :func:`src.active_learning.oracle.batch_cost_estimate`."""
    score_column: str = "score_full"
    score_alpha: float = 2.0
    max_per_patient_frac: float = 0.20
    seed: int = 20260908
    device_index: int = 0
    n_score_workers: int = 12
    output_root: str = "/scratch/mstryja/DoTA_dataset_v2"
    beamlet_prefix: str = "al"
    beamlet_version: Optional[int] = 1
    pool_csv: str = "registry/al_pool_selection.csv"
    validation_manifest: str = ""
    init_checkpoint: str = "models/DoTA_v3_grid_search_v11/best_model.pth"
    hyperparams: str = "models/DoTA_v3_grid_search_v11/hyperparams.json"
    train_config: str = "scripts/config_al_train.yaml"
    train_overrides: Dict = field(default_factory=dict)
    gamma_params: Dict = field(default_factory=dict)
    scale: Dict = field(default_factory=dict)
    evaluate_initial: bool = True
    """Measure the starting checkpoint on the frozen set before any cycle. This is
    the zero-budget point of every learning curve, so it is on by default."""
    checkpoint_selection: str = "last"
    """Which checkpoint a cycle hands to the next one; see
    :func:`src.active_learning.training.cycle_checkpoint` for why ``last``."""


def _cycle_dir(run_dir: Path, cycle: int) -> Path:
    return run_dir / f"cycle_{cycle:02d}"


def _load_validation_entries(manifest: str, output_root: Path) -> Tuple[List[Tuple[str, str]],
                                                                       pd.DataFrame]:
    """The frozen validation beamlets that made it to disk, plus their manifest."""
    table = pd.read_csv(manifest)
    entries = [(str(output_root / row.field_dir), str(row.candidate_id))
               for row in table.itertuples(index=False)
               if (output_root / row.field_dir / f"{row.candidate_id}_sim_res.json").exists()]
    if not entries:
        raise FileNotFoundError(
            f"no simulated beamlets found for the validation manifest {manifest}; "
            "build the validation set before running the loop")
    if len(entries) < len(table):
        logger.warning("validation set: %d of %d manifest rows are on disk "
                       "(the rest were dropped by Monte Carlo QA)", len(entries), len(table))
    return entries, table


def evaluate_checkpoint(checkpoint: Path, cfg: LoopConfig, entries, table,
                        out_prefix: Path) -> Dict:
    """Score one checkpoint on the frozen validation set; write the per-sample rows."""
    device = torch.device(f"cuda:{cfg.device_index}" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint), Path(cfg.hyperparams), device)
    frame = evaluate_frozen_set(model, entries, device=device, scale=cfg.scale,
                                gamma_params=cfg.gamma_params)
    frame.to_csv(f"{out_prefix}_samples.csv", index=False)
    by_decile = summarise_by(frame, table, "score_decile")
    by_decile.to_csv(f"{out_prefix}_by_decile.csv", index=False)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summarise(frame)


def run_cycle(
    cycle: int,
    cfg: LoopConfig,
    run_dir: Path,
    pool: Sequence[PoolEntry],
    resolver: RecordResolver,
    cand_cfg: CandidateConfig,
    rob_cfg: RobustnessConfig,
    runner: MCSquareRunner,
    bdl: BeamDataLibrary,
    bdl_path: str,
    init_checkpoint: Path,
    validation: Tuple[List[Tuple[str, str]], pd.DataFrame],
) -> dict:
    """One full cycle. Returns its manifest; writes it before returning."""
    cycle_dir = _cycle_dir(run_dir, cycle)
    cycle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cycle_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        logger.info("cycle %d already complete (%d beamlets, %.0f MC seconds); skipping",
                    cycle, manifest.get("n_labelled", 0), manifest.get("mc_seconds", 0.0))
        return manifest

    started = time.time()
    rng = np.random.default_rng([cfg.seed, cycle])
    output_root = Path(cfg.output_root)
    val_entries, val_table = validation

    # ── Candidates: a subset of the pool, scored before anything is simulated ──
    n_cts = min(cfg.n_cts_per_cycle, len(pool))
    picked = rng.choice(len(pool), size=n_cts, replace=False)
    entries = [pool[int(i)] for i in sorted(picked)]
    logger.info("cycle %d: scoring candidates on %d pool CTs", cycle, len(entries))
    table = score_pool(entries, cand_cfg, rob_cfg, bdl_path, n_workers=cfg.n_score_workers,
                       prefix=cfg.beamlet_prefix, version=cfg.beamlet_version)
    table.to_csv(cycle_dir / "candidates.csv", index=False)

    # ── Selection ─────────────────────────────────────────────────────────────
    chosen = select(table, cfg.strategy, cfg.beamlets_per_cycle, rng=rng,
                    score_column=cfg.score_column, alpha=cfg.score_alpha,
                    max_per_patient_frac=cfg.max_per_patient_frac)
    chosen.to_csv(cycle_dir / "selection.csv", index=False)
    cost = batch_cost_estimate(chosen)
    logger.info("cycle %d: %d beamlets in %d groups (%.1f per group), estimated %.1f h",
                cycle, cost["n_beamlets"], cost["n_groups"],
                cost["beamlets_per_group_mean"], cost["estimated_total_hours"])

    # ── Labelling ─────────────────────────────────────────────────────────────
    label_stats = label_batch(chosen, entries, runner, bdl, rob_cfg, output_root,
                              resolver=resolver)
    new_records = labelled_records(chosen, output_root)

    # ── Training sources: everything bought so far ────────────────────────────
    sources_csv = run_dir / "training_sources.csv"
    previous = read_training_sources(sources_csv) if sources_csv.exists() else []
    write_training_sources(previous + new_records, sources_csv)
    logger.info("cycle %d: training sources now %d beamlets (+%d)",
                cycle, len(previous) + len(new_records), len(new_records))

    # ── Retraining ────────────────────────────────────────────────────────────
    train_cfg = cycle_training_config(
        Path(cfg.train_config), run_name=f"{cfg.name}_c{cycle:02d}",
        sources_csv=sources_csv, overrides=cfg.train_overrides)
    train_run = run_training(
        train_cfg, config_path=cycle_dir / "train_config.yaml",
        runs_dir=cycle_dir / "training", init_checkpoint=init_checkpoint,
        device_index=cfg.device_index, log_path=cycle_dir / "training.log")
    checkpoint = cycle_checkpoint(train_run, cfg.checkpoint_selection)

    # ── Validation on the frozen set ──────────────────────────────────────────
    metrics = evaluate_checkpoint(checkpoint, cfg, val_entries, val_table,
                                  cycle_dir / "validation")

    manifest = {
        "cycle": cycle,
        "strategy": cfg.strategy,
        "arm": cfg.arm,
        "patients": [e.patient_id for e in entries],
        "n_candidates": int(len(table)),
        "n_valid_candidates": int(table["valid"].sum()),
        "n_selected": int(len(chosen)),
        "n_labelled": int(len(new_records)),
        "n_training_sources": int(len(previous) + len(new_records)),
        "selection": selection_fingerprint(chosen, cfg.score_column),
        "cost_estimate": cost,
        "labelling": {k: v for k, v in label_stats.items() if k != "groups"},
        "mc_seconds": float(label_stats["mc_seconds"]),
        "init_checkpoint": str(init_checkpoint),
        "checkpoint": str(checkpoint),
        "train_run": str(train_run),
        "metrics": metrics,
        "wall_seconds": float(time.time() - started),
    }
    (cycle_dir / "groups.json").write_text(json.dumps(label_stats["groups"], indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def run_loop(cfg: LoopConfig, run_dir: Path, cand_cfg: CandidateConfig,
             rob_cfg: RobustnessConfig, runner: MCSquareRunner, bdl: BeamDataLibrary,
             bdl_path: str) -> List[dict]:
    """Run every cycle of one arm, resuming whatever is already complete."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "loop_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    pool = read_pool(Path(cfg.pool_csv), role="pool")
    resolver = RecordResolver(pool)
    validation = _load_validation_entries(cfg.validation_manifest, Path(cfg.output_root))
    logger.info("arm %s/%s: %d pool CTs, %d frozen validation beamlets",
                cfg.arm, cfg.strategy, len(pool), len(validation[0]))

    metrics_log = run_dir / "metrics.jsonl"
    manifests: List[dict] = []
    checkpoint = Path(cfg.init_checkpoint)

    if cfg.evaluate_initial:
        baseline_path = run_dir / "baseline_metrics.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
        else:
            logger.info("measuring the starting checkpoint on the frozen validation set")
            baseline = evaluate_checkpoint(checkpoint, cfg, *validation,
                                           run_dir / "baseline")
            baseline_path.write_text(json.dumps(baseline, indent=2))
        _append(metrics_log, {"cycle": -1, "mc_seconds_cumulative": 0.0,
                              "beamlets_cumulative": 0, "strategy": cfg.strategy,
                              "arm": cfg.arm, "metrics": baseline})
        logger.info("baseline: GPR %.4f (p05 %.4f), MAPE %.2f%%, |dR80| median %.2f mm",
                    baseline["gpr_mean"], baseline["gpr_p05"],
                    baseline["mape_pct_mean"], baseline["abs_dr80_median_mm"])

    mc_seconds = 0.0
    beamlets = 0
    for cycle in range(cfg.n_cycles):
        manifest = run_cycle(cycle, cfg, run_dir, pool, resolver, cand_cfg, rob_cfg,
                             runner, bdl, bdl_path, checkpoint, validation)
        manifests.append(manifest)
        mc_seconds += float(manifest["mc_seconds"])
        beamlets += int(manifest["n_labelled"])
        checkpoint = Path(manifest["checkpoint"])
        _append(metrics_log, {"cycle": cycle, "mc_seconds_cumulative": mc_seconds,
                              "beamlets_cumulative": beamlets, "strategy": cfg.strategy,
                              "arm": cfg.arm, "metrics": manifest["metrics"]})
        m = manifest["metrics"]
        logger.info("cycle %d done: GPR %.4f (p05 %.4f), MAPE %.2f%%, "
                    "|dR80| median %.2f mm | %d beamlets, %.0f MC seconds cumulative",
                    cycle, m["gpr_mean"], m["gpr_p05"], m["mape_pct_mean"],
                    m["abs_dr80_median_mm"], beamlets, mc_seconds)
    return manifests


def _append(path: Path, row: dict) -> None:
    """Append one JSON line; the learning curve is read straight off this file."""
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
