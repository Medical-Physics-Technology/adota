"""Shutdown handling and numerical diagnostics for the training loop.

Flow:
1. :class:`GracefulShutdown` traps SIGINT/SIGTERM and sets a flag, letting the
   loop finish the current epoch and checkpoint instead of dying mid-write.
2. :func:`dump_nan_context` persists the offending batch and surrounding state
   the first time a non-finite loss appears, so it can be reproduced offline.
3. :func:`compute_grad_norm` / :func:`compute_param_norm` are the cheap
   per-epoch health metrics logged alongside the losses.
"""

from __future__ import annotations

import json
import logging
import signal
from pathlib import Path
from types import FrameType
from typing import Dict, List, Optional

import numpy as np
import torch

from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)

# ── Signal handling ─────────────────────────────────────────────────────────


class GracefulShutdown:
    """Trap SIGINT / SIGTERM so the training loop can exit cleanly.

    Usage::

        shutdown = GracefulShutdown()
        for epoch in range(...):
            if shutdown.requested:
                break
            ...
    """

    def __init__(self) -> None:
        self.requested: bool = False
        self._previous_int = signal.signal(signal.SIGINT, self._handler)
        self._previous_term = signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, frame: Optional[FrameType]) -> None:
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning(
            "Received %s; will finish current epoch then shut down cleanly.", name
        )
        self.requested = True

    def restore(self) -> None:
        """Restore original handlers (call at end of training)."""
        signal.signal(signal.SIGINT, self._previous_int)
        signal.signal(signal.SIGTERM, self._previous_term)


# ── NaN / Inf context dump ──────────────────────────────────────────────────


def dump_nan_context(
    run_dir: Path,
    epoch: int,
    batch_idx: int,
    x: torch.Tensor,
    energy: torch.Tensor,
    y: torch.Tensor,
    outputs: torch.Tensor,
    loss_components: Dict[str, float],
    weights: Dict[str, float],
    grad_norm: Optional[float] = None,
    sample_ids: Optional[List[str]] = None,
) -> Path:
    """Save tensors + diagnostic context for a non-finite batch.

    Files written under ``run_dir/failures/epoch_NNNN_batch_NNNN/``:
    ``x.npy``, ``energy.npy``, ``y.npy``, ``outputs.npy``,
    ``context.json``. Returns the directory path.
    """
    fail_dir = run_dir / "failures" / f"epoch_{epoch:04d}_batch_{batch_idx:04d}"
    fail_dir.mkdir(parents=True, exist_ok=True)

    np.save(fail_dir / "x.npy", x.detach().cpu().numpy())
    np.save(fail_dir / "energy.npy", energy.detach().cpu().numpy())
    np.save(fail_dir / "y.npy", y.detach().cpu().numpy())
    np.save(fail_dir / "outputs.npy", outputs.detach().cpu().numpy())

    context = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "loss_components": loss_components,
        "weights": weights,
        "grad_norm": grad_norm,
        "sample_ids": sample_ids,
        "x_min": float(x.min().item()),
        "x_max": float(x.max().item()),
        "y_min": float(y.min().item()),
        "y_max": float(y.max().item()),
        "energy_min": float(energy.min().item()),
        "energy_max": float(energy.max().item()),
    }
    with open(fail_dir / "context.json", "w") as f:
        json.dump(context, f, indent=2, cls=NumpyEncoder)

    return fail_dir


# ── Training diagnostics ────────────────────────────────────────────────────


def compute_grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of all gradients currently attached to model parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return float(total**0.5)


def compute_param_norm(model: torch.nn.Module) -> float:
    """L2 norm of all model parameters (trainable or not)."""
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().norm(2).item()) ** 2
    return float(total**0.5)
