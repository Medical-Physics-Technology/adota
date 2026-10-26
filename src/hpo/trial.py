"""One HPO trial: its training config, a training segment up to a rung budget,
and the full metric set for a finalist.

A trial lives in its own directory under the HPO run (on ``/scratch``):

    trial.json            the trial (mixer, parameters, unit point, parameter count)
    config.yaml           the resolved TrainingConfig
    checkpoints/last.pth  full training state, the next segment resumes from it
    checkpoints/best.pth  the weights with the lowest validation loss so far
    metrics.jsonl         one row per epoch
    segment_<epochs>.json the result of the segment that ended at that budget
    finalist_<gamma>.csv  per-record metrics of best.pth, finalists only

Nothing here is a new training loop: the state comes from
:func:`build_train_state` / :func:`restore_train_state`, an epoch is
:func:`train_one_epoch`, the loss is :func:`evaluate_loss`, and checkpoints carry
the full RNG state through :class:`CheckpointManager`. The learning rate is
constant (EXP-0013: no divergence, and a schedule tied to the total budget would
make a trial's early epochs depend on the rung it ends at). The loss weights
are the static ``initial_weight_*`` pair, so the validation loss is comparable
across trials.
"""
from __future__ import annotations

import json
import logging
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from src.active_learning.retrospective.trainer import (
    build_train_state,
    restore_train_state,
    training_loader,
    validation_loader,
)
from src.active_learning.retrospective.validation import (
    MetricSettings,
    evaluate_loss,
    evaluate_metrics,
    summarise_metrics,
)
from src.schemas.configs import TrainingConfig
from src.training.checkpoints import CheckpointManager
from src.training.factory import build_adota_model
from src.training.loop import resolve_weights, train_one_epoch
from src.training.run_dir import MetricsLog

logger = logging.getLogger(__name__)

OBJECTIVE = "loss_combined_mean"


def trial_config(training: Mapping[str, Any], params: Mapping[str, Any], *, dataset_path: str,
                 excluded_indexes_file: str, mixer: str) -> TrainingConfig:
    """The base ``training`` block with the trial's parameters and mixer on top."""
    valid = {f.name for f in fields(TrainingConfig)}
    merged = {k: v for k, v in training.items() if k in valid}
    unknown = sorted(set(params) - valid)
    if unknown:
        raise KeyError(f"trial parameters that are not TrainingConfig fields: {unknown}")
    merged.update(params)
    merged.update(dataset_path=dataset_path, excluded_indexes_file=excluded_indexes_file,
                  sequence_mixer=mixer)
    if isinstance(merged.get("input_shape"), list):
        merged["input_shape"] = tuple(merged["input_shape"])
    if isinstance(merged.get("gpr_resolution_mm"), list):
        merged["gpr_resolution_mm"] = tuple(merged["gpr_resolution_mm"])
    config = TrainingConfig(**merged)
    if config.loss_mode == "mse_idd" and config.adaptive_after_epoch < 10_000:
        raise ValueError("adaptive loss weights make the validation loss incomparable across "
                         "trials; keep adaptive_after_epoch disabled")
    return config


def count_parameters(config: TrainingConfig) -> int:
    """The trainable parameters of the model ``config`` builds, counted on the
    meta device (no memory, no weights)."""
    with torch.device("meta"):
        model = build_adota_model(config, torch.device("meta"))
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def best_so_far(rows: Sequence[Mapping[str, Any]], end_epoch: int) -> Dict[str, Any]:
    """The lowest validation loss among epochs ``< end_epoch`` and its epoch."""
    usable = [r for r in rows if r["epoch"] < end_epoch]
    if not usable:
        return {"objective": float("nan"), "objective_epoch": None}
    best = min(usable, key=lambda r: (r["val_loss"][OBJECTIVE], r["epoch"]))
    return {"objective": float(best["val_loss"][OBJECTIVE]), "objective_epoch": int(best["epoch"])}


def run_segment(config: TrainingConfig, trial_dir: Path, train_ids: Sequence[str],
                val_ids: Sequence[str], end_epoch: int, device: torch.device,
                settings: MetricSettings) -> Dict[str, Any]:
    """Train from the trial's last checkpoint (or from scratch) to ``end_epoch``
    epochs in total; write and return ``segment_<end_epoch>.json``."""
    trial_dir = Path(trial_dir)
    checkpoint_dir = trial_dir / "checkpoints"
    started = perf_counter()
    state = build_train_state(config, device, lr_schedule="constant")
    start = 0
    if (checkpoint_dir / "last.pth").exists():
        start = int(restore_train_state(state, checkpoint_dir / "last.pth")["epoch"]) + 1
    logger.info("segment %s: epochs %d..%d", trial_dir.name, start, end_epoch - 1)
    manager = CheckpointManager(checkpoint_dir, save_every_n_epochs=10 ** 9)
    log = MetricsLog(trial_dir / "metrics.jsonl")
    if start < end_epoch:
        # The shuffle seed is offset by the start epoch, so a resumed segment does
        # not replay the shuffle of the first one.
        train_loader = training_loader(config, train_ids, list(val_ids[:1]), cycle=start)
        val_loader = validation_loader(config, val_ids)
    for epoch in range(start, end_epoch):
        t0 = perf_counter()
        w_mse, w_ps = resolve_weights(config, epoch, state.balancer, None, device)
        stats = train_one_epoch(model=state.model, train_loader=train_loader,
                                optimizer=state.optimizer, loss_mse_fn=state.loss_mse_fn,
                                loss_ps_fn=state.loss_ps_fn, weight_mse=w_mse, weight_ps=w_ps,
                                device=device, epoch=epoch, run_dir=trial_dir, max_batches=None,
                                loss_mode=config.loss_mode)
        t_train = perf_counter() - t0
        val = evaluate_loss(state.model, val_loader, device=device, weight_mse=float(w_mse),
                            weight_ps=float(w_ps), settings=settings)
        is_best = val[OBJECTIVE] < state.best_val_loss
        if is_best:
            state.best_val_loss = val[OBJECTIVE]
        log.log({"epoch": epoch, "timestamp": datetime.now().isoformat(timespec="seconds"),
                 "lr": float(state.optimizer.param_groups[0]["lr"]), "train": stats,
                 "val_loss": val, "is_best": is_best, "train_time_s": t_train,
                 "epoch_time_s": perf_counter() - t0})
        manager.save(model=state.model, optimizer=state.optimizer, scheduler=None, epoch=epoch,
                     best_val_loss=state.best_val_loss, patience_counter=0,
                     balancer=state.balancer, is_best=is_best, extra={"trial": trial_dir.name})
        logger.info("%s epoch %d: train %.4e val %.4e%s (%.0f s)", trial_dir.name, epoch,
                    stats.get(OBJECTIVE, float("nan")), val[OBJECTIVE], " *" if is_best else "",
                    perf_counter() - t0)
    result = {"trial": trial_dir.name, "end_epoch": int(end_epoch), "start_epoch": int(start),
              "seconds": perf_counter() - started, "failed": None,
              **best_so_far(_read_rows(trial_dir / "metrics.jsonl"), end_epoch)}
    write_segment(trial_dir, result)
    return result


def write_segment(trial_dir: Path, result: Mapping[str, Any]) -> Path:
    path = Path(trial_dir) / f"segment_{int(result['end_epoch']):03d}.json"
    path.write_text(json.dumps(dict(result), indent=2, default=str))
    return path


def read_segment(trial_dir: Path, end_epoch: int) -> Optional[Dict[str, Any]]:
    path = Path(trial_dir) / f"segment_{int(end_epoch):03d}.json"
    return json.loads(path.read_text()) if path.exists() else None


def evaluate_finalist(config: TrainingConfig, trial_dir: Path, ids: Sequence[str],
                      device: torch.device, settings: MetricSettings) -> Dict[str, Any]:
    """Per-record gamma, MAPE, RDE and dR80 of ``best.pth`` on ``ids``; writes
    ``finalist_<gamma label>.csv`` and returns the summary."""
    trial_dir = Path(trial_dir)
    state = build_train_state(replace(config, compile=False), device, lr_schedule="constant")
    CheckpointManager.load_weights_only(trial_dir / "checkpoints" / "best.pth",
                                        model=state.model, device=device)
    frame = evaluate_metrics(state.model, validation_loader(config, ids), list(ids),
                             device=device, settings=settings, desc=f"finalist {trial_dir.name}")
    frame.to_csv(trial_dir / f"finalist_{settings.gamma_label}.csv", index=False)
    summary = summarise_metrics(frame)
    (trial_dir / f"finalist_{settings.gamma_label}.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary
