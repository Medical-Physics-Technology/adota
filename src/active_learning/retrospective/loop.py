"""The retrospective cycle: score the pool, select, grow the training set, train,
validate, record; plus the shared cycle-0 baseline every strategy resumes from.

Three stages, each its own entry point in ``scripts/al_retro_loop.py``:

1. ``prepare_splits``: the exclusion list, ``V``, ``T`` and the cycle-0 set,
   written to CSV once so every run provably shares them.
2. ``run_cycle0``: train from random weights on the cycle-0 set for
   ``epochs_per_cycle`` epochs; save the checkpoint with full RNG state. No
   scoring happens here.
3. ``run_strategy``: resume from that checkpoint and, for each cycle, score the
   remaining pool through the scorer interface, select ``N`` records with the
   named strategy, add them, train ``epochs_per_cycle`` more epochs, validate.

Every run is a standard training run directory (manifest, ``metrics.jsonl``,
checkpoints with RNG state) and can be read on its own, without the other
strategies: the cycle-0 metrics are copied into each strategy's log, and the
manifest records the path and hash of the cycle-0 checkpoint it started from.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.active_learning.retrospective.dataset import (
    DEFAULT_EXCLUDE_PATH,
    Splits,
    SplitSpec,
    apply_exclusions,
    assert_schedule,
    batch_size_from_fraction,
    build_splits,
    cross_check_exclusions,
    read_exclusion_list,
    read_record_ids,
    read_splits,
    record_metadata,
    write_splits,
)
from src.active_learning.retrospective.sampling import select, selection_fingerprint
from src.active_learning.retrospective.scoring import build_scorer, score_distribution
from src.active_learning.retrospective.trainer import (
    CycleSpec,
    MetricSettings,
    TrainState,
    build_train_state,
    build_validation_bundle,
    restore_train_state,
    train_cycle,
    training_loader,
    verify_state_matches_checkpoint,
)
from src.active_learning.retrospective.validation import draw_eval_subsample
from src.adota.config import DEFAULT_GAMMA_PARAMS, DEFAULT_SCALE
from src.evaluation.cli import resolve_device
from src.schemas.configs import TrainingConfig
from src.training.logging_utils import log_phase
from src.training.run_dir import MetricsLog, save_resolved_config, write_manifest
from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)

PROVENANCE_CSV = ("/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/"
                  "uuid_provenance_map.csv")


@dataclass
class RetroConfig:
    """Everything one run of the retrospective benchmark needs."""

    experiment: str = "EXP-0009"
    dataset_path: str = ""
    exclude_indexes_path: str = DEFAULT_EXCLUDE_PATH
    record_provenance_csv: Optional[str] = PROVENANCE_CSV
    splits_dir: str = "/scratch/mstryja/adota_runs/al_retro/splits"
    runs_dir: str = "/scratch/mstryja/adota_runs/al_retro"
    max_records: Optional[int] = None
    """Subsample ``D`` before splitting; the smoke test's knob, never the real run's."""
    val_fraction: float = 0.15
    initial_fraction: float = 0.20
    split_seed: int = 42
    initial_seed: int = 20260910
    batch_fraction: float = 0.10
    n_cycles: int = 5
    epochs_per_cycle: int = 50
    eval_every_n_epochs: int = 5
    eval_subsample_size: int = 1000
    eval_subsample_seed: int = 20260910
    strategy: str = "random"
    seed: int = 1234
    device_index: Optional[int] = None
    checkpoint_every_n_epochs: int = 10
    scorer: Dict[str, Any] = field(default_factory=lambda: {"name": "difficulty"})
    gamma_params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_GAMMA_PARAMS))
    gamma_resolution_mm: List[float] = field(default_factory=lambda: [2.0, 2.0, 2.0])
    gamma_cutoff_percent: float = 10.0
    gamma_backend: str = "torch"
    training: Dict[str, Any] = field(default_factory=dict)
    """The :class:`TrainingConfig` block: model, optimizer, loader, scale."""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RetroConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown retrospective config keys {unknown}")
        return cls(**raw)

    def training_config(self) -> TrainingConfig:
        valid = {f.name for f in fields(TrainingConfig)}
        block = {k: v for k, v in self.training.items() if k in valid}
        block.update(dataset_path=self.dataset_path,
                     excluded_indexes_file=self.exclude_indexes_path,
                     seed=self.seed, runs_dir=self.runs_dir,
                     device_index=self.device_index if self.device_index is not None else 0,
                     gamma_params=dict(self.gamma_params),
                     checkpoint_every_n_epochs=self.checkpoint_every_n_epochs,
                     gpr_resolution_mm=tuple(self.gamma_resolution_mm))
        block.setdefault("scale", dict(DEFAULT_SCALE))
        if "input_shape" in block:
            block["input_shape"] = tuple(block["input_shape"])
        return TrainingConfig(**block)

    def metric_settings(self) -> MetricSettings:
        cfg = self.training_config()
        return MetricSettings(scale=dict(cfg.scale), gamma_params=dict(self.gamma_params),
                              resolution_mm=tuple(self.gamma_resolution_mm),
                              gamma_cutoff_percent=self.gamma_cutoff_percent,
                              gamma_backend=self.gamma_backend,
                              lps_dx_mm=cfg.lps_dx_mm, lps_dy_mm=cfg.lps_dy_mm)


# ── Stage 1: splits ─────────────────────────────────────────────────────────


def prepare_splits(cfg: RetroConfig) -> Dict[str, Any]:
    """Apply the exclusion list, draw ``V`` and the cycle-0 set, write them, and
    write the per-record metadata the fingerprints need."""
    excluded = read_exclusion_list(Path(cfg.exclude_indexes_path))
    check = cross_check_exclusions(Path(cfg.exclude_indexes_path))
    all_ids = read_record_ids(Path(cfg.dataset_path))
    kept = apply_exclusions(all_ids, excluded)
    if cfg.max_records is not None and cfg.max_records < len(kept):
        rng = np.random.RandomState(cfg.seed)
        picked = sorted(rng.choice(len(kept), size=cfg.max_records, replace=False).tolist())
        kept = [kept[i] for i in picked]
        logger.warning("max_records: subsampled D to %d records (smoke test only)", len(kept))
    spec = SplitSpec(val_fraction=cfg.val_fraction, initial_fraction=cfg.initial_fraction,
                     val_seed=cfg.split_seed, initial_seed=cfg.initial_seed)
    splits = build_splits(kept, spec)
    out = Path(cfg.splits_dir)
    summary_path = write_splits(splits, out, spec, extra={
        "dataset_path": cfg.dataset_path, "exclude_indexes_path": cfg.exclude_indexes_path,
        "n_records_file": len(all_ids), "n_excluded_listed": len(excluded),
        "n_after_exclusion": len(kept) if cfg.max_records is None else len(kept),
        "max_records": cfg.max_records, "exclusion_cross_check": check})
    meta = record_metadata(Path(cfg.dataset_path), splits.training + splits.validation,
                           Path(cfg.record_provenance_csv) if cfg.record_provenance_csv else None,
                           scale=cfg.training_config().scale)
    meta.to_csv(out / "record_metadata.csv", index=False)
    logger.info("splits written to %s (fingerprint %s)", summary_path, splits.fingerprint()[:12])
    return json.loads(summary_path.read_text())


@dataclass
class RunInputs:
    """What every run reads from the splits directory."""

    splits: Splits
    metadata: pd.DataFrame
    subsample_ids: List[str]
    batch_size: int
    splits_summary: Dict[str, Any]

    def fingerprint(self) -> Dict[str, Any]:
        digest = hashlib.sha256("\n".join(self.subsample_ids).encode()).hexdigest()
        return {"splits_dir": self.splits_summary.get("splits_dir"),
                "splits_fingerprint": self.splits.fingerprint(),
                "n_validation": len(self.splits.validation),
                "n_training": len(self.splits.training), "n_initial": len(self.splits.initial),
                "n_pool_start": len(self.splits.pool), "batch_size": self.batch_size,
                "eval_subsample_size": len(self.subsample_ids), "eval_subsample_sha256": digest}


def load_run_inputs(cfg: RetroConfig) -> RunInputs:
    directory = Path(cfg.splits_dir)
    splits = read_splits(directory)
    metadata = pd.read_csv(directory / "record_metadata.csv")
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    summary = json.loads((directory / "splits.json").read_text())
    summary["splits_dir"] = str(directory)
    subsample = draw_eval_subsample(splits.validation, cfg.eval_subsample_size,
                                    cfg.eval_subsample_seed)
    return RunInputs(splits=splits, metadata=metadata, subsample_ids=subsample,
                     batch_size=batch_size_from_fraction(len(splits.training), cfg.batch_fraction),
                     splits_summary=summary)


# ── Run directories and manifests ───────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_manifest(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(updates)
    path.write_text(json.dumps(manifest, indent=2, cls=NumpyEncoder, default=str))


def _write_run_manifest(run_dir: Path, cfg: RetroConfig, inputs: RunInputs, **extra: Any) -> None:
    write_manifest(run_dir, config=cfg, dataset_path=Path(cfg.dataset_path),
                   excluded_indexes_path=Path(cfg.exclude_indexes_path),
                   extra={"experiment": cfg.experiment, "kind": "al_retrospective",
                          "inputs": inputs.fingerprint(), "cycles": [], **extra})
    save_resolved_config(cfg, run_dir / "config.yaml")


def _build_state(cfg: RetroConfig) -> TrainState:
    device = resolve_device(cfg.device_index)
    train_cfg = cfg.training_config()
    state = build_train_state(train_cfg, device)
    n_params = sum(p.numel() for p in state.base_model.parameters())
    log_phase("INIT", f"Device {device} | DoTA3D_v3 {n_params / 1e6:.2f} M params | "
                      f"compile={'on' if train_cfg.compile else 'off'} "
                      f"tf32={'on' if train_cfg.allow_tf32 else 'off'}")
    return state


def _copy_metrics(source_log: Path, target: MetricsLog, source_run: str) -> int:
    """Copy the cycle-0 rows into a strategy's log, so it reads on its own."""
    n = 0
    for line in source_log.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            row["inherited_from"] = source_run
            target.log(row)
            n += 1
    return n


# ── Stage 2: cycle 0 ────────────────────────────────────────────────────────


def run_cycle0(cfg: RetroConfig, run_dir: Path) -> Path:
    """Train the shared baseline; returns the checkpoint every strategy resumes from."""
    inputs = load_run_inputs(cfg)
    _write_run_manifest(run_dir, cfg, inputs, strategy="cycle0", role="cycle0_baseline")
    train_cfg = cfg.training_config()
    state = _build_state(cfg)
    validation = build_validation_bundle(train_cfg, inputs.splits.validation,
                                         inputs.subsample_ids, cfg.metric_settings())
    loader = training_loader(train_cfg, inputs.splits.initial, inputs.splits.validation, cycle=0)
    log_phase("INIT", f"cycle 0: {len(inputs.splits.initial)} training records, "
                      f"{len(inputs.splits.validation)} validation, "
                      f"{len(inputs.subsample_ids)} in the evaluation subsample")
    metrics_log = MetricsLog(run_dir / "metrics.jsonl")
    checkpoint_dir = run_dir / "checkpoints" / "cycle_00"
    spec = CycleSpec(cycle=0, strategy="cycle0", n_train=len(inputs.splits.initial),
                     epochs=cfg.epochs_per_cycle, cumulative_epoch_start=0,
                     eval_every=cfg.eval_every_n_epochs)
    spec.start_epoch = _resume_epoch(checkpoint_dir, state, spec)
    result = train_cycle(state, loader, validation, spec, run_dir=run_dir,
                         metrics_log=metrics_log, checkpoint_dir=checkpoint_dir,
                         checkpoint_every=cfg.checkpoint_every_n_epochs)
    checkpoint = Path(result["checkpoint"])
    cycle = {"cycle": 0, "n_train": spec.n_train, "n_selected": 0,
             "cumulative_epochs": result["cumulative_epoch_end"], "checkpoint": str(checkpoint),
             "checkpoint_sha256": _sha256(checkpoint), **_timings(result),
             "metrics_full": result["metrics_full"]}
    _update_manifest(run_dir, cycles=[cycle], cycle0_checkpoint=str(checkpoint),
                     cycle0_checkpoint_sha256=cycle["checkpoint_sha256"], status="complete")
    log_phase("DONE", f"cycle 0 baseline: {checkpoint}")
    return checkpoint


def _timings(result: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(result[k]) for k in ("train_seconds", "validation_seconds",
                                          "gpu_seconds", "wall_seconds")}


def _resume_epoch(checkpoint_dir: Path, state: TrainState, spec: CycleSpec) -> int:
    """Mid-cycle resume from ``last.pth`` when a cycle was interrupted."""
    last = checkpoint_dir / "last.pth"
    if not last.exists():
        return 0
    bookkeeping = restore_train_state(state, last)
    extra = bookkeeping.get("extra", {})
    if extra.get("cycle") != spec.cycle:
        raise RuntimeError(f"{last} belongs to cycle {extra.get('cycle')}, not {spec.cycle}")
    epoch = int(bookkeeping["epoch"]) + 1
    log_phase("INIT", f"resuming cycle {spec.cycle} at epoch {epoch} from {last}")
    return epoch


# ── Stage 3: one strategy ───────────────────────────────────────────────────


def _cycle_dir(run_dir: Path, cycle: int) -> Path:
    path = run_dir / "cycles" / f"cycle_{cycle:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _score_and_select(cfg: RetroConfig, scorer, pool: pd.DataFrame, n: int, cycle: int,
                      cycle_dir: Path) -> Dict[str, Any]:
    """Score the remaining pool, select ``n``; write both tables; return the record."""
    selection_csv = cycle_dir / "selection.csv"
    if selection_csv.exists():
        selected = pd.read_csv(selection_csv)["sample_id"].astype(str).tolist()
        record = json.loads((cycle_dir / "selection.json").read_text())
        logger.info("cycle %d: reusing the %d ids already selected", cycle, len(selected))
        return {**record, "selected_ids": selected, "reused": True}
    started = perf_counter()
    scored = scorer.score(pool) if cfg.strategy != "random" else pool.assign(score=np.nan)
    scoring_seconds = perf_counter() - started
    scored.to_csv(cycle_dir / "pool_scores.csv", index=False)
    rng = np.random.default_rng([cfg.seed, cycle])
    selected = select(scored, n, cfg.strategy, rng)
    fingerprint = selection_fingerprint(scored, selected)
    chosen = scored[scored["sample_id"].isin(set(selected))]
    chosen[["sample_id", "score"] + [c for c in ("energy_mev", "patient", "anatomy",
                                               "peak_inside_crop") if c in chosen]].to_csv(
        selection_csv, index=False)
    record = {"scorer": getattr(scorer, "describe", lambda: {"name": "none"})()
              if cfg.strategy != "random" else {"name": "none"},
              "score_distribution": score_distribution(scored) if cfg.strategy != "random" else None,
              "scoring_seconds": scoring_seconds, "selection_fingerprint": fingerprint,
              "selection_csv": str(selection_csv), "pool_scores_csv": str(cycle_dir / "pool_scores.csv")}
    (cycle_dir / "selection.json").write_text(json.dumps(record, indent=2, cls=NumpyEncoder))
    return {**record, "selected_ids": selected, "reused": False}


def run_strategy(cfg: RetroConfig, run_dir: Path, cycle0_run: Path) -> List[Dict[str, Any]]:
    """Run every cycle of one strategy from the shared cycle-0 checkpoint,
    resuming whatever this run directory already holds."""
    inputs = load_run_inputs(cfg)
    cycle0_manifest = json.loads((cycle0_run / "manifest.json").read_text())
    if cycle0_manifest["inputs"]["splits_fingerprint"] != inputs.splits.fingerprint():
        raise RuntimeError("the cycle-0 run was trained on different splits than "
                           f"{cfg.splits_dir}; refusing to continue from it")
    checkpoint = Path(cycle0_manifest["cycle0_checkpoint"])
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != cycle0_manifest["cycle0_checkpoint_sha256"]:
        raise RuntimeError(f"{checkpoint} changed since its manifest was written")

    metrics_path = run_dir / "metrics.jsonl"
    resuming = (run_dir / "manifest.json").exists()
    if not resuming:
        _write_run_manifest(run_dir, cfg, inputs, strategy=cfg.strategy, role="strategy",
                            cycle0_run=str(cycle0_run), cycle0_checkpoint=str(checkpoint),
                            cycle0_checkpoint_sha256=checkpoint_sha, status="running")
        shutil.copy2(cycle0_run / "metrics.jsonl", run_dir / "cycle0_metrics.jsonl")
    metrics_log = MetricsLog(metrics_path)
    if not resuming:
        n = _copy_metrics(cycle0_run / "metrics.jsonl", metrics_log, str(cycle0_run))
        logger.info("copied %d cycle-0 rows into %s", n, metrics_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    done = {c["cycle"]: c for c in manifest.get("cycles", [])}

    train_cfg = cfg.training_config()
    state = _build_state(cfg)
    restore_train_state(state, checkpoint)
    n_tensors = verify_state_matches_checkpoint(state, checkpoint)
    log_phase("INIT", f"resumed cycle-0 state from {checkpoint} ({n_tensors} tensors verified "
                      f"bit for bit, sha256 {checkpoint_sha[:12]})")
    validation = build_validation_bundle(train_cfg, inputs.splits.validation,
                                         inputs.subsample_ids, cfg.metric_settings())
    scorer = (build_scorer(dataset_path=cfg.dataset_path, **dict(cfg.scorer))
              if cfg.strategy != "random" else None)

    training_ids = list(inputs.splits.initial)
    pool = inputs.metadata[inputs.metadata["sample_id"].isin(set(inputs.splits.pool))]
    pool = pool.reset_index(drop=True)
    cumulative = int(cycle0_manifest["cycles"][0]["cumulative_epochs"])
    n_initial, batch = len(inputs.splits.initial), inputs.batch_size
    cycles: List[Dict[str, Any]] = [dict(cycle0_manifest["cycles"][0], inherited=True)]

    for cycle in range(1, cfg.n_cycles + 1):
        cycle_dir = _cycle_dir(run_dir, cycle)
        wall_started = perf_counter()
        selection = _score_and_select(cfg, scorer, pool, batch, cycle, cycle_dir)
        selected = selection.pop("selected_ids")
        training_ids.extend(selected)
        pool = pool[~pool["sample_id"].isin(set(selected))].reset_index(drop=True)
        assert_schedule(len(training_ids), n_initial, cycle, batch)
        if len(set(training_ids)) != len(training_ids):
            raise AssertionError("a record was added to the training set twice")
        pd.DataFrame({"sample_id": training_ids}).to_csv(cycle_dir / "training_ids.csv", index=False)

        if cycle in done and Path(done[cycle]["checkpoint"]).exists():
            restore_train_state(state, Path(done[cycle]["checkpoint"]))
            cumulative = int(done[cycle]["cumulative_epochs"])
            cycles.append(done[cycle])
            log_phase("INIT", f"cycle {cycle} already complete; restored {done[cycle]['checkpoint']}")
            continue

        loader = training_loader(train_cfg, training_ids, inputs.splits.validation, cycle)
        spec = CycleSpec(cycle=cycle, strategy=cfg.strategy, n_train=len(training_ids),
                         epochs=cfg.epochs_per_cycle, cumulative_epoch_start=cumulative,
                         eval_every=cfg.eval_every_n_epochs, extra={"n_pool": int(len(pool))})
        checkpoint_dir = run_dir / "checkpoints" / f"cycle_{cycle:02d}"
        spec.start_epoch = _resume_epoch(checkpoint_dir, state, spec)
        log_phase("INIT", f"cycle {cycle}: +{batch} records by {cfg.strategy} -> "
                          f"{spec.n_train} training records, {len(pool)} left in the pool")
        result = train_cycle(state, loader, validation, spec, run_dir=run_dir,
                             metrics_log=metrics_log, checkpoint_dir=checkpoint_dir,
                             checkpoint_every=cfg.checkpoint_every_n_epochs)
        cumulative = int(result["cumulative_epoch_end"])
        record = {"cycle": cycle, "strategy": cfg.strategy, "n_train": spec.n_train,
                  "n_selected": batch, "n_pool_after": int(len(pool)),
                  "schedule_ok": True, "cumulative_epochs": cumulative,
                  "checkpoint": result["checkpoint"],
                  "checkpoint_sha256": _sha256(Path(result["checkpoint"])),
                  **_timings(result), "cycle_wall_seconds": perf_counter() - wall_started,
                  "scoring_seconds": selection.get("scoring_seconds", 0.0),
                  "selection_csv": selection.get("selection_csv"),
                  "selection_fingerprint": selection.get("selection_fingerprint"),
                  "score_distribution": selection.get("score_distribution"),
                  "metrics_full": result["metrics_full"]}
        (cycle_dir / "manifest.json").write_text(json.dumps(record, indent=2, cls=NumpyEncoder))
        cycles.append(record)
        _update_manifest(run_dir, cycles=cycles)
        m = result["metrics_full"] or {}
        log_phase("CYCLE", f"cycle {cycle} done | n_train {spec.n_train} | cumulative epochs "
                           f"{cumulative} | GPR {m.get('gpr_mean', float('nan')):.4f} "
                           f"(p05 {m.get('gpr_p05', float('nan')):.4f}) | "
                           f"{record['cycle_wall_seconds'] / 60:.1f} min")

    _update_manifest(run_dir, cycles=cycles, status="complete")
    return cycles
