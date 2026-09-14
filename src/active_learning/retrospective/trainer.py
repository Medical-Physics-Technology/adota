"""One cycle of training, on top of :mod:`src.training`.

Nothing here is a new training loop: the model, optimizer and scheduler come from
:mod:`src.training.factory`, an epoch is :func:`src.training.loop.train_one_epoch`,
the loss weights are :func:`src.training.loop.resolve_weights`, checkpoints are
written by :class:`src.training.checkpoints.CheckpointManager` with full RNG
state, and the metrics go through :class:`src.training.run_dir.MetricsLog`. What
this module adds is the cadence of Section 4 of the experiment design: the loss on
the whole validation set every epoch, the full metric set on the fixed subsample
every ``eval_every`` epochs, the full metric set on the whole validation set at the
cycle boundary, and on every row the cycle index, the cumulative epoch and the
current training set size, so epochs-to-quality can be recovered afterwards.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.active_learning.retrospective.validation import (
    MetricSettings,
    evaluate_loss,
    evaluate_metrics,
    summarise_metrics,
)
from src.loaders.generator import H5PYGenerator
from src.schemas.configs import TrainingConfig
from src.training.checkpoints import CheckpointManager
from src.training.data import build_dataloaders, collate_h5
from src.training.factory import (
    build_adota_model,
    build_optimizer_scheduler,
    configure_backends,
    maybe_compile_model,
    set_determinism,
)
from src.training.logging_utils import format_duration, log_phase
from src.training.loop import resolve_weights, train_one_epoch
from src.training.losses import LMSE, LPS, TwoObjectiveBalancer
from src.training.run_dir import MetricsLog
from src.training.utils import get_lr

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    """The mutable training state a cycle continues from."""

    config: TrainingConfig
    device: torch.device
    base_model: torch.nn.Module
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    balancer: TwoObjectiveBalancer
    loss_mse_fn: LMSE
    loss_ps_fn: LPS
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    prev_val: Optional[Dict[str, float]] = None


def build_train_state(config: TrainingConfig, device: torch.device) -> TrainState:
    """Fresh weights, fresh optimizer, seeded RNGs: the cycle-0 starting point."""
    set_determinism(config.seed)
    configure_backends(config)
    base_model = build_adota_model(config, device)
    optimizer, scheduler = build_optimizer_scheduler(base_model, config)
    model = maybe_compile_model(base_model, config)
    return TrainState(config=config, device=device, base_model=base_model, model=model,
                      optimizer=optimizer, scheduler=scheduler,
                      balancer=TwoObjectiveBalancer(smoothing=config.balancer_smoothing),
                      loss_mse_fn=LMSE(), loss_ps_fn=LPS(dx=config.lps_dx_mm, dy=config.lps_dy_mm))


def restore_train_state(state: TrainState, checkpoint: Path) -> Dict[str, Any]:
    """Full resume: weights, optimizer, scheduler, balancer, RNG and bookkeeping."""
    bookkeeping = CheckpointManager.load(
        Path(checkpoint), model=state.model, optimizer=state.optimizer,
        scheduler=state.scheduler, balancer=state.balancer, device=state.device)
    state.best_val_loss = float(bookkeeping["best_val_loss"])
    state.patience_counter = int(bookkeeping["patience_counter"])
    return bookkeeping


def verify_state_matches_checkpoint(state: TrainState, checkpoint: Path) -> int:
    """Assert that every weight now in the model equals the checkpoint bit for
    bit; returns the number of tensors compared."""
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)["model"]
    live = getattr(state.model, "_orig_mod", state.model).state_dict()
    if set(saved) != set(live):
        raise AssertionError("checkpoint and model disagree on the parameter names")
    for name, tensor in saved.items():
        if not torch.equal(tensor, live[name].detach().cpu()):
            raise AssertionError(f"parameter {name} differs from the checkpoint")
    return len(saved)


# ── Loaders ─────────────────────────────────────────────────────────────────


def validation_loader(config: TrainingConfig, ids: Sequence[str]) -> DataLoader:
    """A validation loader over ``ids``, built exactly as ``build_dataloaders``
    builds its validation side (no augmentation, cropped around the peak)."""
    dataset = H5PYGenerator(
        file_path=config.dataset_path, indexes=list(ids), augmentation=False, cropp=True,
        normalize=False, normalize_flux_only=config.normalize_flux_only,
        flux_mode=config.flux_mode, centerline_sidecar=config.centerline_sidecar,
        indexes_to_exclude_list=config.excluded_indexes_file)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                      num_workers=config.num_workers, pin_memory=True,
                      persistent_workers=config.num_workers > 0, collate_fn=collate_h5)


def training_loader(config: TrainingConfig, train_ids: Sequence[str], val_ids: Sequence[str],
                    cycle: int) -> DataLoader:
    """The training loader of one cycle through ``build_dataloaders``; the loader
    seed is offset by the cycle so successive cycles do not replay one shuffle."""
    cycle_config = replace(config, seed=config.seed + cycle)
    train_loader, _ = build_dataloaders(cycle_config, list(train_ids), list(val_ids))
    return train_loader


@dataclass
class ValidationBundle:
    """The frozen validation set and its fixed subsample, as loaders."""

    full_ids: List[str]
    full_loader: DataLoader
    subsample_ids: List[str]
    subsample_loader: DataLoader
    settings: MetricSettings


def build_validation_bundle(config: TrainingConfig, full_ids: Sequence[str],
                            subsample_ids: Sequence[str],
                            settings: MetricSettings) -> ValidationBundle:
    return ValidationBundle(
        full_ids=list(full_ids), full_loader=validation_loader(config, full_ids),
        subsample_ids=list(subsample_ids),
        subsample_loader=validation_loader(config, subsample_ids), settings=settings)


# ── One cycle ───────────────────────────────────────────────────────────────


@dataclass
class CycleSpec:
    """What one cycle is: its index, its training set size, its epochs."""

    cycle: int
    strategy: str
    n_train: int
    epochs: int
    cumulative_epoch_start: int
    eval_every: int
    start_epoch: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_cycle(state: TrainState, train_loader: DataLoader, validation: ValidationBundle,
                spec: CycleSpec, *, run_dir: Path, metrics_log: MetricsLog,
                checkpoint_dir: Path, checkpoint_every: int) -> Dict[str, Any]:
    """Train ``spec.epochs`` epochs from the current state; returns the cycle summary."""
    config, device = state.config, state.device
    manager = CheckpointManager(checkpoint_dir, save_every_n_epochs=checkpoint_every)
    val_dir = run_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    gamma_label = validation.settings.gamma_label
    started = perf_counter()
    train_seconds = validation_seconds = 0.0
    last_full: Optional[Dict] = None
    last_sub: Optional[Dict] = None

    for epoch in range(spec.start_epoch, spec.epochs):
        cumulative = spec.cumulative_epoch_start + epoch
        epoch_started = perf_counter()
        w_mse, w_ps = resolve_weights(config, cumulative, state.balancer, state.prev_val, device)
        log_phase("EPOCH", f"cycle {spec.cycle} epoch {epoch}/{spec.epochs - 1} "
                           f"(cumulative {cumulative}) | n_train={spec.n_train} "
                           f"lr={get_lr(state.optimizer):.2e}")

        train_stats = train_one_epoch(
            model=state.model, train_loader=train_loader, optimizer=state.optimizer,
            loss_mse_fn=state.loss_mse_fn, loss_ps_fn=state.loss_ps_fn,
            weight_mse=w_mse, weight_ps=w_ps, device=device, epoch=cumulative,
            run_dir=run_dir, max_batches=None, loss_mode=config.loss_mode)
        _sync(device)
        t_train = perf_counter() - epoch_started
        train_seconds += t_train

        # ── Validation: loss every epoch, metrics on the cadence ──────────
        t_val = perf_counter()
        val_loss = evaluate_loss(state.model, validation.full_loader, device=device,
                                 weight_mse=float(w_mse.item()), weight_ps=float(w_ps.item()),
                                 settings=validation.settings)
        boundary = epoch == spec.epochs - 1
        due = (cumulative + 1) % max(1, spec.eval_every) == 0
        metrics_full = metrics_sub = None
        if boundary:
            frame = evaluate_metrics(state.model, validation.full_loader, validation.full_ids,
                                     device=device, settings=validation.settings,
                                     desc=f"cycle {spec.cycle} full V")
            frame.to_csv(val_dir / f"cycle_{spec.cycle:02d}_full_gamma_{gamma_label}.csv",
                         index=False)
            metrics_full = summarise_metrics(frame)
            metrics_sub = summarise_metrics(frame, validation.subsample_ids)
            last_full = metrics_full
        elif due:
            frame = evaluate_metrics(state.model, validation.subsample_loader,
                                     validation.subsample_ids, device=device,
                                     settings=validation.settings,
                                     desc=f"cycle {spec.cycle} epoch {epoch} subsample")
            frame.to_csv(val_dir / f"cycle_{spec.cycle:02d}_epoch_{cumulative:04d}_"
                                   f"subsample_gamma_{gamma_label}.csv", index=False)
            metrics_sub = summarise_metrics(frame)
        if metrics_sub is not None:
            last_sub = metrics_sub
        validation_seconds += perf_counter() - t_val

        # ── Schedule, bookkeeping, checkpoint ─────────────────────────────
        state.scheduler.step(val_loss["loss_combined_mean"])
        is_best = val_loss["loss_combined_mean"] < state.best_val_loss
        if is_best:
            state.best_val_loss = val_loss["loss_combined_mean"]
            state.patience_counter = 0
        else:
            state.patience_counter += 1
        state.prev_val = val_loss

        record = {
            "cycle": spec.cycle, "epoch_in_cycle": epoch, "cumulative_epoch": cumulative,
            "n_train": spec.n_train, "strategy": spec.strategy,
            "cycle_boundary": boundary, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "lr": get_lr(state.optimizer),
            "weights": {"w_mse": float(w_mse.item()), "w_ps": float(w_ps.item())},
            "train": train_stats, "val_loss": val_loss,
            "metrics_subsample": metrics_sub, "metrics_full": metrics_full,
            "gamma_label": gamma_label, "epoch_time_s": perf_counter() - epoch_started,
            "train_time_s": t_train, "is_best": is_best, **spec.extra,
        }
        metrics_log.log(record)
        manager.save(model=state.model, optimizer=state.optimizer, scheduler=state.scheduler,
                     epoch=epoch, best_val_loss=state.best_val_loss,
                     patience_counter=state.patience_counter, balancer=state.balancer,
                     is_best=is_best,
                     extra={"cycle": spec.cycle, "cumulative_epoch": cumulative,
                            "n_train": spec.n_train, "strategy": spec.strategy})

        summary = (f"train {train_stats.get('loss_combined_mean', float('nan')):.4e} | "
                   f"val {val_loss['loss_combined_mean']:.4e}")
        if metrics_sub is not None:
            summary += (f" | GPR {metrics_sub['gpr_mean']:.4f} (p05 {metrics_sub['gpr_p05']:.4f})"
                        f" MAPE {metrics_sub['mape_pct_mean']:.2f}%"
                        f" |dR80| p95 {metrics_sub['abs_dr80_p95_mm']:.2f} mm"
                        + (" [full V]" if boundary else " [subsample]"))
        log_phase("EPOCH", f"done in {format_duration(perf_counter() - epoch_started)} | "
                           + summary)

    return {"checkpoint": str(checkpoint_dir / "last.pth"),
            "cumulative_epoch_end": spec.cumulative_epoch_start + spec.epochs,
            "epochs_trained": spec.epochs - spec.start_epoch,
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "gpu_seconds": train_seconds + validation_seconds,
            "wall_seconds": perf_counter() - started,
            "metrics_full": last_full, "metrics_subsample": last_sub}
