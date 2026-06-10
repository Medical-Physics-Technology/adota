"""ADoTA training entry point.

Usage:
    uv run python scripts/train_adota.py --config scripts/config_train_adota.yaml

The script:

1. Loads a YAML config into a :class:`TrainingConfig`; CLI flags override.
2. Sets a deterministic RNG seed across torch/cuda/numpy/python.
3. Creates a standardized run directory under ``runs/`` containing a
   reproducibility manifest, the resolved config, a streaming JSONL
   metrics log, a ``training.log`` file, and subdirectories for
   checkpoints, attention snapshots, validation artifacts, and NaN dumps.
4. Builds train / val :class:`H5PYGenerator` datasets and DataLoaders.
5. Builds :class:`DoTA3D_v3`, an AdamW optimizer, and a
   :class:`ReduceLROnPlateau` scheduler. Optionally resumes from a
   prior checkpoint.
6. Trains with two losses (:class:`LMSE`, :class:`LPS`), either with
   static or adaptive (softmax-over-log) balancing.
7. After every epoch: validates with full per-sample RMSE / MAPE / RDE
   on the de-normalized dose, per-energy breakdowns (fixed + quantile
   bins), worst-K sample logging, gradient and parameter norms.
   Every ``gpr_every_n_epochs``: gamma pass rate on a random subset of
   the val set. Every ``attention_every_n_epochs``: attention snapshot
   on a fixed canary sample.
8. Saves checkpoints (best, last, every Nth) with full RNG / optimizer /
   scheduler / balancer state for deterministic resume.
9. Honors SIGINT / SIGTERM (graceful shutdown after the current epoch)
   and an optional ``--max-hours`` wall-time budget.
10. ``--smoke-test`` runs 2 epochs with 4 train / 4 val batches, exits 0.
"""

from __future__ import annotations

import logging
import random
import shutil
import sys
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import load_yaml_config  # noqa: E402
from src.adota.models import DoTA3D_v3  # noqa: E402
from src.loaders.generator import H5PYGenerator  # noqa: E402
from src.schemas.configs import TrainingConfig  # noqa: E402
from src.training.losses import LMSE, LPS, TwoObjectiveBalancer  # noqa: E402
from src.training.run import (  # noqa: E402
    CheckpointManager,
    GracefulShutdown,
    MetricsLog,
    compute_grad_norm,
    compute_param_norm,
    dump_nan_context,
    format_duration,
    log_banner,
    log_phase,
    log_section,
    save_resolved_config,
    setup_training_logging,
    setup_training_run_directory,
    silence_pymedphys,
    write_manifest,
)
from src.training.utils import get_lr, validate_tensor_ranges  # noqa: E402
from src.training.validation import (  # noqa: E402
    evaluate_validation,
    save_attention_snapshot,
)

logger = logging.getLogger(__name__)
app = typer.Typer(help="ADoTA training tool")


# ── Determinism ─────────────────────────────────────────────────────────────


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config plumbing ─────────────────────────────────────────────────────────


def _build_config_from_yaml(
    yaml_path: Optional[Path],
    overrides: Dict[str, Any],
) -> TrainingConfig:
    """Load YAML config, apply CLI overrides, return ``TrainingConfig``."""
    yaml_cfg: Dict[str, Any] = {}
    if yaml_path is not None:
        yaml_cfg = load_yaml_config(yaml_path)

    valid_keys = {f.name for f in fields(TrainingConfig)}
    merged = {k: v for k, v in yaml_cfg.items() if k in valid_keys}
    for k, v in overrides.items():
        if v is not None:
            merged[k] = v

    # Normalize tuple-shaped fields that YAML loads as lists.
    if "input_shape" in merged and isinstance(merged["input_shape"], list):
        merged["input_shape"] = tuple(merged["input_shape"])
    if "gpr_resolution_mm" in merged and isinstance(merged["gpr_resolution_mm"], list):
        merged["gpr_resolution_mm"] = tuple(merged["gpr_resolution_mm"])

    return TrainingConfig(**merged)


# ── Data loading helpers ────────────────────────────────────────────────────


def _train_val_split(
    record_ids: List[str], test_size: float, random_state: int
) -> Tuple[List[str], List[str]]:
    """Deterministic shuffle then split; matches sklearn semantics."""
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(record_ids))
    rng.shuffle(indices)
    n_val = int(round(len(record_ids) * test_size))
    val_idx = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())
    return [record_ids[i] for i in train_idx], [record_ids[i] for i in val_idx]


def _load_record_ids(dataset_path: Path, excluded_indexes_path: Optional[Path]) -> List[str]:
    """Read the H5 keys and remove anything listed in the exclusion file."""
    with h5py.File(str(dataset_path), "r") as ds:
        record_ids = list(ds.keys())

    excluded: List[str] = []
    if excluded_indexes_path is not None and excluded_indexes_path.exists():
        with open(excluded_indexes_path, "r") as f:
            excluded = [line.strip() for line in f if line.strip()]
        logger.info(
            "Loaded %d excluded indexes from %s", len(excluded), excluded_indexes_path
        )

    excluded_set = set(excluded)
    return [rid for rid in record_ids if rid not in excluded_set]


def _collate_h5(batch):
    """Stack H5PYGenerator samples into contiguous batched tensors."""
    xs, es, ys = zip(*batch)
    X = torch.stack([t.contiguous() if not t.is_contiguous() else t for t in xs], dim=0)
    E = torch.stack([e.view(1) for e in es], dim=0)
    Y = torch.stack([t.contiguous() if not t.is_contiguous() else t for t in ys], dim=0)
    return X, E, Y


def _build_dataloaders(
    config: TrainingConfig,
    train_indexes: List[str],
    val_indexes: List[str],
) -> Tuple[DataLoader, DataLoader]:
    train_ds = H5PYGenerator(
        file_path=config.dataset_path,
        indexes=train_indexes,
        augmentation=config.augmentation,
        normalize=False,
        normalize_flux_only=config.normalize_flux_only,
        flux_mode=config.flux_mode,
        indexes_to_exclude_list=config.excluded_indexes_file,
    )
    val_ds = H5PYGenerator(
        file_path=config.dataset_path,
        indexes=val_indexes,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=config.normalize_flux_only,
        flux_mode=config.flux_mode,
        indexes_to_exclude_list=config.excluded_indexes_file,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        generator=generator,
        collate_fn=_collate_h5,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=_collate_h5,
    )
    return train_loader, val_loader


# ── Model / optimizer ───────────────────────────────────────────────────────


def _build_model(config: TrainingConfig, device: torch.device) -> DoTA3D_v3:
    model = DoTA3D_v3(
        input_shape=tuple(config.input_shape),
        num_transformers=config.num_transformers,
        num_heads=config.num_heads,
        dim_feedforward=config.dim_feedforward,
        num_levels=config.num_levels,
        enc_features=config.enc_features,
        kernel_size=config.kernel_size,
        convolutional_steps=config.convolutional_steps,
        conv_hidden_channels=config.conv_hidden_channels,
        dropout_rate=config.dropout_rate,
        causal=config.causal,
        zero_padding=config.zero_padding,
        last_activation=config.last_activation,
        num_forward=config.num_forward,
        transformer_residual=config.transformer_residual,
        conv_residual=config.conv_residual,
        weight_standardization=config.weight_standardization,
        norm_layer=config.norm_layer,
        weight_init=config.weight_init,
    ).to(device)
    return model


def _build_optimizer_scheduler(
    model: torch.nn.Module, config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )
    return optimizer, scheduler


# ── Canary / GPR subset selection ───────────────────────────────────────────


def _pick_canary(val_dataset: H5PYGenerator, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Return ``(x, energy, sample_id)`` for a fixed canary validation sample."""
    x, energy, _ = val_dataset[0]
    canary_id = val_dataset.record_ids[0]
    return x.unsqueeze(0).to(device), energy.view(1, 1).to(device), canary_id


def _pick_gpr_subset(n_val_samples: int, subset_size: int, rng: np.random.RandomState) -> List[int]:
    if subset_size >= n_val_samples:
        return list(range(n_val_samples))
    return sorted(rng.choice(n_val_samples, size=subset_size, replace=False).tolist())


# ── Training loop ───────────────────────────────────────────────────────────


def _train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_mse_fn: LMSE,
    loss_ps_fn: LPS,
    weight_mse: torch.Tensor,
    weight_ps: torch.Tensor,
    device: torch.device,
    epoch: int,
    run_dir: Path,
    max_batches: Optional[int],
    loss_mode: str,
) -> Dict[str, Any]:
    """Run one training epoch; returns aggregate stats."""
    model.train()
    sum_loss = 0.0
    sum_loss_mse = 0.0
    sum_loss_ps = 0.0
    timings = {"load_s": [], "forward_s": [], "backward_s": []}
    n_batches = 0
    last_grad_norm: Optional[float] = None

    # Heartbeat: aim for ~10 [TRAIN] lines per epoch regardless of size.
    try:
        total_batches = len(train_loader)
    except TypeError:
        total_batches = 0
    if max_batches is not None:
        total_batches = (
            min(total_batches, max_batches) if total_batches else max_batches
        )
    heartbeat_every = max(1, total_batches // 10) if total_batches else 1

    t_load_start = perf_counter()
    for batch_idx, (x, e, y) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        timings["load_s"].append(perf_counter() - t_load_start)

        if not validate_tensor_ranges(x, y, e):
            log_phase(
                "ERROR",
                f"batch {batch_idx}: tensors out of [0, 1]: "
                f"x=[{x.min().item():.3f}, {x.max().item():.3f}] "
                f"y=[{y.min().item():.3f}, {y.max().item():.3f}] "
                f"e=[{e.min().item():.3f}, {e.max().item():.3f}]",
                level=logging.WARNING,
            )

        optimizer.zero_grad()
        t_fwd = perf_counter()
        outputs = model(x, e)[0]  # forward returns (dose, attention)
        timings["forward_s"].append(perf_counter() - t_fwd)

        loss_mse = loss_mse_fn(outputs, y)
        if loss_mode == "mse_idd":
            loss_ps = loss_ps_fn(outputs, y)
            loss = weight_mse * loss_mse + weight_ps * loss_ps
        else:
            loss_ps = torch.zeros(1, device=device)
            loss = loss_mse

        if not torch.isfinite(loss):
            grad_norm = compute_grad_norm(model)
            fail_dir = dump_nan_context(
                run_dir,
                epoch=epoch,
                batch_idx=batch_idx,
                x=x,
                energy=e,
                y=y,
                outputs=outputs,
                loss_components={
                    "loss_mse": float(loss_mse.item()) if torch.isfinite(loss_mse) else float("nan"),
                    "loss_ps": float(loss_ps.item()) if torch.isfinite(loss_ps) else float("nan"),
                },
                weights={"w_mse": float(weight_mse.item()), "w_ps": float(weight_ps.item())},
                grad_norm=grad_norm,
            )
            log_phase(
                "ERROR",
                f"Non-finite loss at epoch={epoch} batch={batch_idx}. "
                f"Dump: {fail_dir}",
                level=logging.ERROR,
            )
            raise RuntimeError(
                f"Non-finite loss at epoch={epoch}, batch={batch_idx}. "
                f"Tensors and context saved under {run_dir / 'failures'}."
            )

        t_bwd = perf_counter()
        loss.backward()
        last_grad_norm = compute_grad_norm(model)
        optimizer.step()
        timings["backward_s"].append(perf_counter() - t_bwd)

        sum_loss += float(loss.item())
        sum_loss_mse += float(loss_mse.item())
        sum_loss_ps += float(loss_ps.item())
        n_batches += 1

        if (
            total_batches
            and ((batch_idx + 1) % heartbeat_every == 0 or batch_idx == total_batches - 1)
        ):
            log_phase(
                "TRAIN",
                f"batch {batch_idx + 1}/{total_batches}  "
                f"loss={loss.item():.4e} (mse={loss_mse.item():.4e} "
                f"ps={loss_ps.item():.4e})",
            )

        t_load_start = perf_counter()

    if n_batches == 0:
        return {"n_batches": 0}

    return {
        "n_batches": n_batches,
        "loss_combined_mean": sum_loss / n_batches,
        "loss_mse_mean": sum_loss_mse / n_batches,
        "loss_ps_mean": sum_loss_ps / n_batches,
        "t_load_mean_s": float(np.mean(timings["load_s"])) if timings["load_s"] else 0.0,
        "t_forward_mean_s": float(np.mean(timings["forward_s"])) if timings["forward_s"] else 0.0,
        "t_backward_mean_s": float(np.mean(timings["backward_s"])) if timings["backward_s"] else 0.0,
        "grad_norm_last": last_grad_norm,
        "param_norm": compute_param_norm(model),
    }


# ── Loss-balancing schedule ─────────────────────────────────────────────────


def _resolve_weights(
    config: TrainingConfig,
    epoch: int,
    balancer: TwoObjectiveBalancer,
    prev_val: Optional[Dict[str, float]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-epoch loss weights (static or adaptive)."""
    if epoch < config.adaptive_after_epoch or prev_val is None:
        return (
            torch.tensor(config.initial_weight_mse, device=device),
            torch.tensor(config.initial_weight_ps, device=device),
        )
    l_mse = torch.tensor(prev_val["loss_mse_mean"], device=device)
    l_ps = torch.tensor(prev_val["loss_ps_mean"], device=device)
    return balancer.get_weights(l_mse, l_ps)


# ── CLI ─────────────────────────────────────────────────────────────────────


@app.command()
def main(
    config: Annotated[
        Optional[Path], typer.Option(help="Path to YAML training config.")
    ] = None,
    resume_from: Annotated[
        Optional[Path],
        typer.Option(help="Path to a previous checkpoint .pth to resume from."),
    ] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index. -1 for CPU.")
    ] = None,
    max_hours: Annotated[
        Optional[float],
        typer.Option(help="Wall-time budget in hours. Loop exits cleanly when exceeded."),
    ] = None,
    smoke_test: Annotated[
        bool, typer.Option(help="2 epochs, 4 train/val batches each. For CI / sanity.")
    ] = False,
    max_records: Annotated[
        Optional[int],
        typer.Option(
            help="If set, subsample the record list to N samples before splitting. "
            "Useful for tiny end-to-end runs that exercise every artifact path."
        ),
    ] = None,
    num_epochs: Annotated[
        Optional[int],
        typer.Option(
            help="Override num_epochs from the config. Handy for short smoke "
            "runs that reuse a full training config."
        ),
    ] = None,
    config_name: Annotated[
        Optional[str], typer.Option(help="Override config_name (used in run dir name).")
    ] = None,
    runs_dir: Annotated[
        Optional[str],
        typer.Option(help="Override the base directory under which the run directory is created."),
    ] = None,
    verbose: Annotated[bool, typer.Option(help="Enable DEBUG-level logging.")] = False,
) -> None:
    overrides = {
        "resume_from": str(resume_from) if resume_from is not None else None,
        "device_index": device_index,
        "max_hours": max_hours,
        "smoke_test": smoke_test if smoke_test else None,  # leave default if False
        "config_name": config_name,
        "num_epochs": num_epochs,
        "runs_dir": runs_dir,
    }
    cfg = _build_config_from_yaml(config, overrides)

    if not cfg.dataset_path:
        raise typer.BadParameter(
            "dataset_path is required (in YAML config or via --config)."
        )

    # ── Run dir + logging ────────────────────────────────────────────
    runs_dir = Path(cfg.runs_dir)
    run_dir = setup_training_run_directory(runs_dir, cfg.config_name)

    import time

    run_start_wall = time.time()
    log_file = setup_training_logging(
        run_dir, start_time=run_start_wall, verbose=verbose
    )
    silence_pymedphys()

    log_banner("ADoTA TRAINING RUN")
    log_phase("INIT", f"Run dir   : {run_dir}")
    log_phase("INIT", f"Log file  : {log_file}")
    log_phase("INIT", f"Config    : {cfg.config_name}")

    if config is not None and config.exists():
        shutil.copy2(config, run_dir / "config_input.yaml")
    save_resolved_config(cfg, run_dir / "config.yaml")

    # ── Determinism ──────────────────────────────────────────────────
    _set_determinism(cfg.seed)

    # ── Device ────────────────────────────────────────────────────────
    if cfg.device_index >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.device_index}")
        gpu_name = torch.cuda.get_device_name(cfg.device_index)
        cap = torch.cuda.get_device_capability(cfg.device_index)
        log_phase("INIT", f"Device    : {device} ({gpu_name}, sm_{cap[0]}{cap[1]})")
    else:
        device = torch.device("cpu")
        log_phase("INIT", f"Device    : {device}")

    # ── Dataset discovery + split ────────────────────────────────────
    dataset_path = Path(cfg.dataset_path)
    excluded_path = (
        Path(cfg.excluded_indexes_file) if cfg.excluded_indexes_file else None
    )

    log_phase("INIT", f"Dataset   : {dataset_path.name}")
    record_ids = _load_record_ids(dataset_path, excluded_path)
    n_total_after_exclusion = len(record_ids)
    if max_records is not None and max_records < len(record_ids):
        rng = np.random.RandomState(cfg.seed)
        picked = sorted(
            rng.choice(len(record_ids), size=max_records, replace=False).tolist()
        )
        record_ids = [record_ids[i] for i in picked]
        log_phase(
            "INIT",
            f"Subsampled record list to {len(record_ids)} (from "
            f"{n_total_after_exclusion}) via --max-records.",
            level=logging.WARNING,
        )
    train_ids, val_ids = _train_val_split(
        record_ids,
        test_size=cfg.train_test_split,
        random_state=cfg.split_random_state,
    )
    log_phase(
        "INIT",
        f"Records   : {n_total_after_exclusion} usable "
        f"| {len(train_ids)} train | {len(val_ids)} val",
    )

    # Manifest
    write_manifest(
        run_dir,
        config=cfg,
        dataset_path=dataset_path,
        excluded_indexes_path=excluded_path,
        extra={"n_train": len(train_ids), "n_val": len(val_ids)},
    )

    # ── Smoke-test overrides ─────────────────────────────────────────
    if cfg.smoke_test:
        log_phase(
            "INIT",
            "Smoke-test mode: limiting to 2 epochs, 4 train/val batches each.",
            level=logging.WARNING,
        )
        cfg.num_epochs = 2
        cfg.gpr_every_n_epochs = 1
        cfg.gpr_subset_size = 2
        cfg.attention_every_n_epochs = 1
        cfg.checkpoint_every_n_epochs = 1

    # ── Data loaders ─────────────────────────────────────────────────
    train_loader, val_loader = _build_dataloaders(cfg, train_ids, val_ids)

    # ── Model / optimizer / scheduler ───────────────────────────────
    model = _build_model(cfg, device)
    optimizer, scheduler = _build_optimizer_scheduler(model, cfg)
    balancer = TwoObjectiveBalancer(smoothing=cfg.balancer_smoothing)
    loss_mse_fn = LMSE()
    loss_ps_fn = LPS(dx=cfg.lps_dx_mm, dy=cfg.lps_dy_mm)

    n_params = sum(p.numel() for p in model.parameters())
    log_phase(
        "INIT",
        f"Model     : DoTA3D_v3, {n_params / 1e6:.2f} M params "
        f"({cfg.num_transformers} transformer block(s), {cfg.num_heads} heads)",
    )
    log_phase(
        "INIT",
        f"Loss      : {cfg.initial_weight_mse:.3f}*LMSE + "
        f"{cfg.initial_weight_ps:.3f}*LPS  "
        f"(adaptive after epoch {cfg.adaptive_after_epoch})",
    )
    log_phase(
        "INIT",
        f"Optimizer : AdamW lr={cfg.learning_rate:.2e} wd={cfg.weight_decay:.2e}",
    )
    log_phase(
        "INIT",
        f"Schedule  : ReduceLROnPlateau(factor={cfg.lr_factor}, "
        f"patience={cfg.lr_patience})",
    )
    log_phase(
        "INIT",
        f"Validation: RMSE/MAPE/RDE every epoch | GPR every "
        f"{cfg.gpr_every_n_epochs} ep (subset={cfg.gpr_subset_size})",
    )
    log_phase(
        "INIT",
        f"Checkpts  : best + last + every {cfg.checkpoint_every_n_epochs} epochs",
    )
    budget = (
        f"{cfg.max_hours}h wall-time" if cfg.max_hours is not None else "unlimited"
    )
    log_phase(
        "INIT",
        f"Budget    : {budget} | early stop after "
        f"{cfg.patience} epochs without improvement",
    )

    # Save model hyperparams for inference scripts to consume.
    import json

    with open(run_dir / "hyperparams.json", "w") as f:
        json.dump(model.to_dict(), f, indent=2, default=str)

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    patience_counter = 0
    if cfg.resume_from is not None:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise typer.BadParameter(f"resume_from path does not exist: {resume_path}")
        log_phase("INIT", f"Resuming from checkpoint: {resume_path}")
        bookkeeping = CheckpointManager.load(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            balancer=balancer,
            device=device,
        )
        start_epoch = int(bookkeeping["epoch"]) + 1
        best_val_loss = float(bookkeeping["best_val_loss"])
        patience_counter = int(bookkeeping["patience_counter"])
        log_phase(
            "INIT",
            f"Resumed: start_epoch={start_epoch} "
            f"best_val_loss={best_val_loss:.6e} "
            f"patience_counter={patience_counter}",
        )

    # ── Bookkeeping objects ──────────────────────────────────────────
    metrics_log = MetricsLog(run_dir / "metrics.jsonl")
    checkpoints = CheckpointManager(
        run_dir / "checkpoints",
        save_every_n_epochs=cfg.checkpoint_every_n_epochs,
    )
    shutdown = GracefulShutdown()

    # ── Canary + GPR subset (fixed for the run) ──────────────────────
    canary_x, canary_e, canary_id = _pick_canary(val_loader.dataset, device)
    log_phase("INIT", f"Canary    : {canary_id}")
    rng_subset = np.random.RandomState(cfg.seed)
    gpr_subset_indices = _pick_gpr_subset(
        len(val_loader.dataset), cfg.gpr_subset_size, rng_subset
    )

    # ── Training loop ────────────────────────────────────────────────
    run_start = perf_counter()
    prev_val: Optional[Dict[str, float]] = None
    smoke_train_batches = 4 if cfg.smoke_test else None
    smoke_val_batches = 4 if cfg.smoke_test else None
    stop_reason: str = "completed"
    epochs_completed = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        if shutdown.requested:
            log_phase(
                "STOP",
                f"Shutdown requested before epoch {epoch}; exiting loop.",
                level=logging.WARNING,
            )
            stop_reason = "shutdown_signal"
            break
        if cfg.max_hours is not None:
            elapsed_h = (perf_counter() - run_start) / 3600.0
            if elapsed_h >= cfg.max_hours:
                log_phase(
                    "STOP",
                    f"Wall-time budget exceeded ({elapsed_h:.2f}h >= "
                    f"{cfg.max_hours:.2f}h); stopping.",
                    level=logging.WARNING,
                )
                stop_reason = "wall_time"
                break

        epoch_start = perf_counter()
        w_mse, w_ps = _resolve_weights(cfg, epoch, balancer, prev_val, device)
        log_section(
            f"EPOCH {epoch} / {cfg.num_epochs - 1}   "
            f"w=({w_mse.item():.3f}, {w_ps.item():.3f})  "
            f"lr={get_lr(optimizer):.2e}"
        )

        train_stats = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_mse_fn=loss_mse_fn,
            loss_ps_fn=loss_ps_fn,
            weight_mse=w_mse,
            weight_ps=w_ps,
            device=device,
            epoch=epoch,
            run_dir=run_dir,
            max_batches=smoke_train_batches,
            loss_mode=cfg.loss_mode,
        )

        if train_stats.get("n_batches", 0) > 0:
            train_time = (
                train_stats["t_load_mean_s"]
                + train_stats["t_forward_mean_s"]
                + train_stats["t_backward_mean_s"]
            ) * train_stats["n_batches"]
            samples_per_s = (
                (train_stats["n_batches"] * cfg.batch_size) / max(train_time, 1e-6)
            )
            gpu_mem_gb = (
                torch.cuda.max_memory_allocated(device) / 1e9
                if device.type == "cuda"
                else 0.0
            )
            log_phase(
                "TRAIN",
                f"done | mean_loss={train_stats['loss_combined_mean']:.4e} "
                f"(mse={train_stats['loss_mse_mean']:.4e} "
                f"ps={train_stats['loss_ps_mean']:.4e}) "
                f"| grad_norm={train_stats['grad_norm_last']:.3e} "
                f"param_norm={train_stats['param_norm']:.3e} "
                f"| {format_duration(train_time)} "
                f"({samples_per_s:.1f} samples/s) | gpu_mem={gpu_mem_gb:.1f} GB",
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        # ── Validation ────────────────────────────────────────────
        compute_gpr_this_epoch = (epoch + 1) % cfg.gpr_every_n_epochs == 0
        val_loader_for_eval = val_loader
        val_sample_ids = val_loader.dataset.record_ids
        if smoke_val_batches is not None:
            val_sample_ids = val_sample_ids[: smoke_val_batches * cfg.batch_size]

        log_phase(
            "VAL",
            f"running on {len(val_sample_ids)} samples"
            + (f" (+GPR on {cfg.gpr_subset_size})" if compute_gpr_this_epoch else "")
            + "...",
        )

        val_stats = evaluate_validation(
            model=model,
            val_dataloader=_limited_loader(val_loader_for_eval, smoke_val_batches),
            val_sample_ids=val_sample_ids,
            device=device,
            config=cfg,
            weight_mse=w_mse,
            weight_ps=w_ps,
            compute_gpr=compute_gpr_this_epoch,
            gpr_subset_indices=gpr_subset_indices,
            run_dir=run_dir,
            epoch=epoch,
        )

        log_phase(
            "VAL",
            f"done | loss={val_stats.get('loss_combined_mean', float('nan')):.4e} "
            f"(mse={val_stats.get('loss_mse_mean', float('nan')):.4e} "
            f"ps={val_stats.get('loss_ps_mean', float('nan')):.4e}) "
            f"| rmse={val_stats.get('rmse_gy_mean', float('nan')):.4e} Gy "
            f"mape={val_stats.get('mape_pct_mean', float('nan')):.3f}% "
            f"rde={val_stats.get('rde_pct_mean', float('nan')):.3f}%",
        )

        if compute_gpr_this_epoch and "gpr_mean" in val_stats:
            log_phase(
                "GPR",
                f"mean={val_stats['gpr_mean']:.4f} "
                f"+/- {val_stats.get('gpr_std', 0.0):.4f} "
                f"(N={val_stats.get('gpr_n_samples', 0)})",
            )

        # ── Attention snapshot ────────────────────────────────────
        if (epoch + 1) % cfg.attention_every_n_epochs == 0:
            attn_path = save_attention_snapshot(
                model=model,
                canary_x=canary_x,
                canary_energy=canary_e,
                run_dir=run_dir,
                epoch=epoch,
            )
            if attn_path is not None:
                log_phase("CKPT", f"attention saved: {attn_path.name}")

        scheduler.step(val_stats.get("loss_combined_mean", float("inf")))

        # ── Early stopping bookkeeping ───────────────────────────
        val_loss = val_stats.get("loss_combined_mean", float("inf"))
        is_best = val_loss < best_val_loss
        if is_best:
            previous_best = best_val_loss
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            log_phase(
                "BEST",
                f"*** NEW BEST val loss {best_val_loss:.4e} "
                f"(was {previous_best:.4e})",
            )
        else:
            patience_counter += 1

        # ── Persist ──────────────────────────────────────────────
        epoch_time = perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "lr": get_lr(optimizer),
            "weights": {"w_mse": float(w_mse.item()), "w_ps": float(w_ps.item())},
            "train": train_stats,
            "val": val_stats,
            "epoch_time_s": epoch_time,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "is_best": is_best,
        }
        metrics_log.log(record)

        checkpoints.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            balancer=balancer,
            is_best=is_best,
        )
        saved_files = ["last.pth"]
        if is_best:
            saved_files.append("best.pth")
        if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
            saved_files.append(f"epoch_{epoch:04d}.pth")
        log_phase("CKPT", "saved: " + ", ".join(saved_files))

        epochs_completed = epoch + 1 - start_epoch
        avg_epoch_time = (perf_counter() - run_start) / max(1, epochs_completed)
        remaining_epochs = cfg.num_epochs - (epoch + 1)
        eta_s = avg_epoch_time * remaining_epochs
        best_str = (
            f"best {best_val_loss:.4e} @ ep {best_epoch}"
            if best_epoch is not None
            else "no best yet"
        )
        log_phase(
            "EPOCH",
            f"done in {format_duration(epoch_time)} | {best_str} "
            f"| patience {patience_counter}/{cfg.patience} "
            f"| ETA {format_duration(eta_s)}",
        )

        prev_val = val_stats

        if patience_counter >= cfg.patience:
            log_phase(
                "STOP",
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {cfg.patience} epochs).",
                level=logging.WARNING,
            )
            stop_reason = "early_stop"
            break

    shutdown.restore()

    # ── Closing summary ──────────────────────────────────────────────
    total_time = perf_counter() - run_start
    log_banner("RUN SUMMARY")
    log_phase(
        "DONE",
        f"Epochs completed : {epochs_completed} / "
        f"{cfg.num_epochs - start_epoch}   (stop: {stop_reason})",
    )
    log_phase("DONE", f"Total wall time  : {format_duration(total_time)}")
    if best_epoch is not None:
        log_phase(
            "DONE",
            f"Best val loss    : {best_val_loss:.6e}  @ epoch {best_epoch}",
        )
        log_phase(
            "DONE",
            f"Best checkpoint  : {run_dir / 'checkpoints' / 'best.pth'}",
        )
    else:
        log_phase("DONE", "Best val loss    : (no completed epoch)")
    log_phase("DONE", f"Metrics file     : {run_dir / 'metrics.jsonl'}")
    log_phase("DONE", f"Run directory    : {run_dir}")


# ── DataLoader slicing for smoke test ───────────────────────────────────────


class _LimitedLoader:
    """Wrap a DataLoader and stop after ``max_batches`` iterations."""

    def __init__(self, loader: DataLoader, max_batches: Optional[int]):
        self._loader = loader
        self._max = max_batches
        self.dataset = loader.dataset

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if self._max is not None and i >= self._max:
                break
            yield batch

    def __len__(self) -> int:
        if self._max is None:
            return len(self._loader)
        return min(self._max, len(self._loader))


def _limited_loader(loader: DataLoader, max_batches: Optional[int]):
    if max_batches is None:
        return loader
    return _LimitedLoader(loader, max_batches)


if __name__ == "__main__":
    app()
