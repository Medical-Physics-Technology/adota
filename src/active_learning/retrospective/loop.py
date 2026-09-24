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
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.active_learning.retrospective.config import RetroConfig
from src.active_learning.retrospective.dataset import (
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
from src.active_learning.retrospective.patient_split import assert_groups_disjoint, build_patient_splits
from src.active_learning.retrospective.sampling import select, selection_fingerprint
from src.active_learning.retrospective.scoring import build_scorer, score_distribution
from src.active_learning.retrospective.trainer import (
    CycleSpec,
    TrainState,
    build_train_state,
    build_validation_bundle,
    restore_train_state,
    train_cycle,
    training_loader,
    verify_state_matches_checkpoint,
)
from src.active_learning.retrospective.validation import draw_eval_subsample
from src.evaluation.cli import resolve_device
from src.training.logging_utils import log_phase
from src.training.run_dir import MetricsLog, save_resolved_config, write_manifest
from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)


# ── Stage 1: splits ─────────────────────────────────────────────────────────


def prepare_splits(cfg: RetroConfig, overwrite: bool = False) -> Dict[str, Any]:
    """Apply the exclusion list, draw ``V`` and the cycle-0 set, write them, and
    write the per-record metadata the fingerprints need.

    Refuses to write into a ``splits_dir`` that already holds a ``splits.json``
    unless ``overwrite`` is set: the splits are frozen once drawn, every run
    manifest pins their fingerprint, and some (the v3 d30 splits) are the v2
    ids materialised by hand rather than a draw this function would reproduce.
    """
    existing = Path(cfg.splits_dir) / "splits.json"
    if existing.exists() and not overwrite:
        raise FileExistsError(f"{existing} exists; the splits are frozen. Pass overwrite "
                              "(--overwrite on the CLI) to redraw them deliberately.")
    excluded = read_exclusion_list(Path(cfg.exclude_indexes_path))
    check = cross_check_exclusions(Path(cfg.exclude_indexes_path))
    all_ids = read_record_ids(Path(cfg.dataset_path))
    kept = apply_exclusions(all_ids, excluded)
    n_full = len(kept)
    if not 0.0 < cfg.data_fraction <= 1.0:
        raise ValueError(f"data_fraction must lie in (0, 1], got {cfg.data_fraction}")
    if cfg.data_fraction < 1.0:
        n_keep = int(round(cfg.data_fraction * len(kept)))
        rng = np.random.RandomState(cfg.data_fraction_seed)
        picked = sorted(rng.choice(len(kept), size=n_keep, replace=False).tolist())
        kept = [kept[i] for i in picked]
        logger.info("data_fraction %.2f: D reduced from %d to %d records (seed %d)",
                    cfg.data_fraction, n_full, len(kept), cfg.data_fraction_seed)
    if cfg.max_records is not None and cfg.max_records < len(kept):
        rng = np.random.RandomState(cfg.seed)
        picked = sorted(rng.choice(len(kept), size=cfg.max_records, replace=False).tolist())
        kept = [kept[i] for i in picked]
        logger.warning("max_records: subsampled D to %d records (smoke test only)", len(kept))
    spec = SplitSpec(val_fraction=cfg.val_fraction, initial_fraction=cfg.initial_fraction,
                     val_seed=cfg.split_seed, initial_seed=cfg.initial_seed)
    provenance = Path(cfg.record_provenance_csv) if cfg.record_provenance_csv else None
    split_extra: Dict[str, Any] = {"val_split": cfg.val_split}
    meta = None
    if cfg.val_split == "patient":
        # A patient split needs every record's group before V is drawn.
        meta = record_metadata(Path(cfg.dataset_path), kept, provenance,
                               scale=cfg.training_config().scale)
        splits, held_out = build_patient_splits(
            kept, meta, cfg.val_group_column, cfg.val_stratify_column,
            cfg.val_groups_per_stratum, cfg.split_seed, cfg.initial_fraction, cfg.initial_seed)
        groups = meta.set_index("sample_id")[cfg.val_group_column].astype(str)
        split_extra.update(val_group_column=cfg.val_group_column,
                           val_stratify_column=cfg.val_stratify_column,
                           val_groups_per_stratum=dict(cfg.val_groups_per_stratum),
                           held_out_groups=held_out,
                           n_groups_validation=int(groups.loc[splits.validation].nunique()),
                           n_groups_training=int(groups.loc[splits.training].nunique()))
    else:
        splits = build_splits(kept, spec)
    out = Path(cfg.splits_dir)
    summary_path = write_splits(splits, out, spec, extra={**split_extra,
        "dataset_path": cfg.dataset_path, "exclude_indexes_path": cfg.exclude_indexes_path,
        "n_records_file": len(all_ids), "n_excluded_listed": len(excluded),
        "n_after_exclusion": n_full, "data_fraction": cfg.data_fraction,
        "data_fraction_seed": cfg.data_fraction_seed, "n_after_data_fraction": len(kept),
        "max_records": cfg.max_records, "exclusion_cross_check": check})
    if meta is None:
        meta = record_metadata(Path(cfg.dataset_path), splits.training + splits.validation,
                               provenance, scale=cfg.training_config().scale)
    else:
        meta = meta.set_index("sample_id").loc[splits.training + splits.validation].reset_index()
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
                "data_fraction": self.splits_summary.get("data_fraction", 1.0),
                "n_after_data_fraction": self.splits_summary.get("n_after_data_fraction"),
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
    if summary.get("val_split") == "patient":
        assert_groups_disjoint(splits, metadata, summary["val_group_column"])
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
    state = build_train_state(train_cfg, device, lr_schedule=cfg.lr_schedule, lr_min=cfg.lr_min,
                              warmup_epochs=cfg.warmup_epochs)
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
    _write_run_manifest(run_dir, cfg, inputs, strategy="cycle0", role="cycle0_baseline",
                        lr_schedule=cfg.lr_schedule, lr_min=cfg.lr_min,
                        warmup_epochs=cfg.warmup_epochs)
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
    selected = select(scored, n, cfg.strategy, rng, **cfg.strategy_params)
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


def _log_cycle0_lr(cycle0_run: Path, cycle0_manifest: Dict[str, Any], cfg: RetroConfig) -> None:
    """Log what LR schedule and values the shared cycle-0 run actually used,
    and warn (never raise) when this strategy trains at a fixed LR on top of a
    cycle-0 baseline that was not itself trained at a constant LR."""
    cycle0_schedule = cycle0_manifest.get("config", {}).get("lr_schedule", "plateau")
    rows = [json.loads(line) for line in
            (cycle0_run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    lrs = sorted({row["lr"] for row in rows if "lr" in row})
    log_phase("INIT", f"cycle-0 run {cycle0_run} trained under lr_schedule={cycle0_schedule!r}, "
                      f"lr values seen: {lrs}")
    if cfg.lr_schedule != "plateau" and len(lrs) > 1:
        logger.warning("this strategy run uses lr_schedule=%r but the shared cycle-0 baseline "
                       "%s was not trained at a constant LR (values %s); the fixed schedule "
                       "starts cycle 1 from %.3e regardless (lr0, or lr_min under a warmup of "
                       "%d epochs)", cfg.lr_schedule, cycle0_run, lrs,
                       cfg.lr_min if cfg.warmup_epochs else cfg.training_config().learning_rate,
                       cfg.warmup_epochs)


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
                            cycle0_checkpoint_sha256=checkpoint_sha, status="running",
                            lr_schedule=cfg.lr_schedule, lr_min=cfg.lr_min,
                            warmup_epochs=cfg.warmup_epochs)
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
    _log_cycle0_lr(cycle0_run, cycle0_manifest, cfg)
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
