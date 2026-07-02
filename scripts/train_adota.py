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
   :class:`ReduceLROnPlateau` scheduler. Optionally wraps the model with
   ``torch.compile`` and resumes from a prior checkpoint.
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

The data, model/optimizer factory, training-step, and validation-selection
helpers live under :mod:`src.training` (``data``, ``factory``, ``loop``,
``validation``); this script is the orchestration layer that wires them
together.
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Dict, Optional

import numpy as np
import torch
import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.cli import resolve_device  # noqa: E402
from src.training.data import (  # noqa: E402
    build_dataloaders,
    limited_loader,
    load_record_ids,
    train_val_split,
)
from src.training.factory import (  # noqa: E402
    build_adota_model,
    build_config_from_yaml,
    build_optimizer_scheduler,
    configure_backends,
    maybe_compile_model,
    set_determinism,
)
from src.training.gpr_pool import (  # noqa: E402
    build_gpr_pool,
    load_gpr_pool,
    pool_to_indices,
    save_gpr_pool,
)
from src.training.loop import resolve_weights, train_one_epoch  # noqa: E402
from src.training.losses import LMSE, LPS, TwoObjectiveBalancer  # noqa: E402
from src.training.run import (  # noqa: E402
    CheckpointManager,
    GracefulShutdown,
    MetricsLog,
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
from src.training.utils import get_lr  # noqa: E402
from src.training.validation import (  # noqa: E402
    evaluate_validation,
    pick_canary,
    save_attention_snapshot,
)

logger = logging.getLogger(__name__)
app = typer.Typer(help="ADoTA training tool")


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
    weights_only: Annotated[
        bool,
        typer.Option(
            help="Warm-start: load only model weights from --resume-from "
            "(fresh optimizer/scheduler/epoch). For fine-tuning a prior best."
        ),
    ] = False,
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
        "weights_only": weights_only if weights_only else None,  # leave default if False
        "device_index": device_index,
        "max_hours": max_hours,
        "smoke_test": smoke_test if smoke_test else None,  # leave default if False
        "config_name": config_name,
        "num_epochs": num_epochs,
        "runs_dir": runs_dir,
    }
    cfg = build_config_from_yaml(config, overrides)

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

    # ── Determinism + backends ───────────────────────────────────────
    set_determinism(cfg.seed)
    configure_backends(cfg)

    # ── Device ────────────────────────────────────────────────────────
    device = resolve_device(cfg.device_index)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device.index)
        cap = torch.cuda.get_device_capability(device.index)
        log_phase("INIT", f"Device    : {device} ({gpu_name}, sm_{cap[0]}{cap[1]})")
    else:
        log_phase("INIT", f"Device    : {device}")

    # ── Dataset discovery + split ────────────────────────────────────
    dataset_path = Path(cfg.dataset_path)
    excluded_path = (
        Path(cfg.excluded_indexes_file) if cfg.excluded_indexes_file else None
    )

    log_phase("INIT", f"Dataset   : {dataset_path.name}")
    record_ids = load_record_ids(dataset_path, excluded_path)
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
    train_ids, val_ids = train_val_split(
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
        cfg.gpr_comparable_size = 1
        cfg.attention_every_n_epochs = 1
        cfg.checkpoint_every_n_epochs = 1

    # ── Data loaders ─────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(cfg, train_ids, val_ids)

    # ── Model / optimizer / scheduler ───────────────────────────────
    # Build the optimizer from the eager module, then optionally compile: the
    # compiled wrapper shares the same parameter tensors, and checkpoints are
    # saved/loaded via the unwrapped module (see CheckpointManager).
    base_model = build_adota_model(cfg, device)
    optimizer, scheduler = build_optimizer_scheduler(base_model, cfg)
    model = maybe_compile_model(base_model, cfg)
    balancer = TwoObjectiveBalancer(smoothing=cfg.balancer_smoothing)
    loss_mse_fn = LMSE()
    loss_ps_fn = LPS(dx=cfg.lps_dx_mm, dy=cfg.lps_dy_mm)

    n_params = sum(p.numel() for p in base_model.parameters())
    log_phase(
        "INIT",
        f"Model     : DoTA3D_v3, {n_params / 1e6:.2f} M params "
        f"({cfg.num_transformers} transformer block(s), {cfg.num_heads} heads)",
    )
    log_phase(
        "INIT",
        f"Accel     : compile={'on (' + cfg.compile_mode + ')' if cfg.compile else 'off'} "
        f"| tf32={'on' if cfg.allow_tf32 else 'off'}",
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
        json.dump(base_model.to_dict(), f, indent=2, default=str)

    # ── Resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch: Optional[int] = None
    patience_counter = 0
    if cfg.resume_from is not None:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise typer.BadParameter(f"resume_from path does not exist: {resume_path}")
        if cfg.weights_only:
            CheckpointManager.load_weights_only(
                resume_path, model=model, device=device
            )
            log_phase(
                "INIT",
                f"Warm-start (weights only) from: {resume_path} "
                f"| fresh optimizer/scheduler/epoch",
            )
        else:
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

    # ── Canary (fixed for the run) ───────────────────────────────────
    canary_x, canary_e, canary_id = pick_canary(val_loader.dataset, device)
    log_phase("INIT", f"Canary    : {canary_id}")

    # ── Gamma (GPR) pool: nested comparable-in-pool, pinned by record id ──
    # Priority: explicit gpr_samples_file > inherit parent run's pool on
    # resume/warm-start > draw fresh (deterministic) and persist.
    val_record_ids = val_loader.dataset.record_ids
    gpr_samples_path = run_dir / "gpr_samples.json"
    if cfg.gpr_samples_file:
        gpr_pool = load_gpr_pool(Path(cfg.gpr_samples_file))
        log_phase("INIT", f"Gamma pool: pinned from {cfg.gpr_samples_file}")
    else:
        inherited = None
        if cfg.resume_from is not None:
            parent_pool = Path(cfg.resume_from).resolve().parent.parent / "gpr_samples.json"
            if parent_pool.exists():
                inherited = parent_pool
        if inherited is not None:
            gpr_pool = load_gpr_pool(inherited)
            log_phase("INIT", f"Gamma pool: inherited from {inherited}")
        else:
            gpr_pool = build_gpr_pool(
                val_record_ids,
                pool_size=cfg.gpr_subset_size,
                comparable_size=cfg.gpr_comparable_size,
                seed=cfg.seed,
                source_run=cfg.resume_from,
            )
            log_phase("INIT", "Gamma pool: drawn fresh (nested comparable-in-pool)")
    save_gpr_pool(gpr_pool, gpr_samples_path)
    gpr_subset_indices, gpr_comparable_indices = pool_to_indices(
        gpr_pool, val_record_ids
    )
    gpr_comparable_ids = list(gpr_pool["comparable_ids"])
    log_phase(
        "INIT",
        f"Gamma pool: {len(gpr_subset_indices)} pool "
        f"({len(gpr_comparable_indices)} comparable) | {gpr_samples_path.name}",
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
        w_mse, w_ps = resolve_weights(cfg, epoch, balancer, prev_val, device)
        log_section(
            f"EPOCH {epoch} / {cfg.num_epochs - 1}   "
            f"w=({w_mse.item():.3f}, {w_ps.item():.3f})  "
            f"lr={get_lr(optimizer):.2e}"
        )

        train_stats = train_one_epoch(
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
            val_dataloader=limited_loader(val_loader_for_eval, smoke_val_batches),
            val_sample_ids=val_sample_ids,
            device=device,
            config=cfg,
            weight_mse=w_mse,
            weight_ps=w_ps,
            compute_gpr=compute_gpr_this_epoch,
            gpr_subset_indices=gpr_subset_indices,
            gpr_comparable_ids=gpr_comparable_ids,
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
            gpr_line = (
                f"pool={val_stats['gpr_mean']:.4f} "
                f"+/- {val_stats.get('gpr_std', 0.0):.4f} "
                f"(N={val_stats.get('gpr_n_samples', 0)})"
            )
            if "gpr_comparable_mean" in val_stats:
                gpr_line += (
                    f" | comparable={val_stats['gpr_comparable_mean']:.4f} "
                    f"(N={val_stats.get('gpr_comparable_n_samples', 0)})"
                )
            log_phase("GPR", gpr_line)

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


if __name__ == "__main__":
    app()
