"""Checkpoint saving, retention and resume.

Flow:
1. :func:`_rng_state` / :func:`_restore_rng_state` snapshot and restore the
   python, numpy and torch (CPU + CUDA) generators, so a resumed run continues
   the same random sequence.
2. :class:`CheckpointManager` writes model, optimizer, scheduler, loss-balancer
   and counter state, and enforces the "best + last + every Nth" retention
   policy.
3. :func:`load_weights_only` supports starting a new run from a prior model
   without inheriting its optimizer or schedule.

``torch.compile`` wrappers are unwrapped before saving so a checkpoint is
loadable regardless of whether the next run compiles.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.training.losses import TwoObjectiveBalancer

logger = logging.getLogger(__name__)

# ── Checkpoint manager ──────────────────────────────────────────────────────


def _rng_state() -> Dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])


def _unwrap_compiled(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module behind a ``torch.compile`` wrapper.

    ``torch.compile`` returns an ``OptimizedModule`` whose ``state_dict`` keys
    carry an ``_orig_mod.`` prefix. Saving / loading via the unwrapped module
    keeps checkpoints format-identical whether or not the model was compiled,
    so they stay loadable by inference scripts and across compile settings.
    """
    return getattr(model, "_orig_mod", model)


class CheckpointManager:
    """Save and load full training snapshots.

    Each snapshot contains everything needed to deterministically resume:
    model + optimizer + scheduler state dicts, epoch counter, best val
    loss, early-stopping patience counter, RNG state for all relevant
    generators, and the loss balancer's running state.

    Retention policy: ``best.pth`` and ``last.pth`` are always rewritten;
    additionally an ``epoch_NNNN.pth`` snapshot is kept every
    ``save_every_n_epochs`` epochs (1-indexed).
    """

    def __init__(self, checkpoint_dir: Path, save_every_n_epochs: int):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        best_val_loss: float,
        patience_counter: int,
        balancer: Optional[TwoObjectiveBalancer] = None,
        is_best: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist a training snapshot. Returns the ``last.pth`` path."""
        state: Dict[str, Any] = {
            "epoch": epoch,
            "model": _unwrap_compiled(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "rng": _rng_state(),
            "balancer_running_weights": (
                balancer.running_weights.detach().cpu()
                if balancer is not None and balancer.running_weights is not None
                else None
            ),
        }
        if extra:
            state["extra"] = extra

        last_path = self.dir / "last.pth"
        torch.save(state, last_path)

        if is_best:
            torch.save(state, self.dir / "best.pth")

        if (epoch + 1) % self.save_every_n_epochs == 0:
            torch.save(state, self.dir / f"epoch_{epoch:04d}.pth")

        return last_path

    @staticmethod
    def load(
        path: Path,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        balancer: Optional[TwoObjectiveBalancer] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Restore a snapshot in-place; returns the bookkeeping fields."""
        map_location = device if device is not None else "cpu"
        # weights_only=False is required: the snapshot stores NumPy/Python
        # RNG state (and a balancer tensor) alongside the state dicts.
        state = torch.load(path, map_location=map_location, weights_only=False)

        _unwrap_compiled(model).load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if balancer is not None and state.get("balancer_running_weights") is not None:
            balancer.running_weights = state["balancer_running_weights"]
        if state.get("rng") is not None:
            _restore_rng_state(state["rng"])

        return {
            "epoch": state["epoch"],
            "best_val_loss": state["best_val_loss"],
            "patience_counter": state["patience_counter"],
            "extra": state.get("extra", {}),
        }

    @staticmethod
    def load_weights_only(
        path: Path,
        *,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        """Warm-start: load only the model weights from a checkpoint.

        Optimizer, scheduler, balancer, RNG, and the epoch / best-loss
        bookkeeping are deliberately *not* restored, so the caller starts a
        fresh training run (fresh schedule, epoch counter at 0) from a prior
        set of weights. Unwraps ``torch.compile`` so a compiled model loads an
        eager checkpoint and vice versa.
        """
        map_location = device if device is not None else "cpu"
        state = torch.load(path, map_location=map_location, weights_only=False)
        _unwrap_compiled(model).load_state_dict(state["model"])
