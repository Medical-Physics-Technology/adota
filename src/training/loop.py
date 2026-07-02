"""The ADoTA per-epoch training step and loss-weight schedule.

:func:`train_one_epoch` runs one full pass over the training loader -
forward, two-objective loss, backward, optimizer step - while collecting
timing, loss, and gradient/parameter-norm statistics and guarding against
non-finite losses (dumping context for a post-mortem). :func:`resolve_weights`
picks the per-epoch ``(w_mse, w_ps)`` loss weights, either static or via the
adaptive :class:`TwoObjectiveBalancer`.

Both used to live inline in ``scripts/train_adota.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.training.losses import LMSE, LPS, TwoObjectiveBalancer
from src.training.run import (
    compute_grad_norm,
    compute_param_norm,
    dump_nan_context,
    log_phase,
)
from src.training.utils import validate_tensor_ranges
from src.schemas.configs import TrainingConfig

logger = logging.getLogger(__name__)


def train_one_epoch(
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


def resolve_weights(
    config: TrainingConfig,
    epoch: int,
    balancer: TwoObjectiveBalancer,
    prev_val: Optional[Dict[str, float]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-epoch loss weights (static or adaptive).

    Before ``config.adaptive_after_epoch`` (or when there is no previous
    validation result yet) the static ``initial_weight_*`` values are used;
    afterwards the balancer derives weights from the previous epoch's loss
    components.
    """
    if epoch < config.adaptive_after_epoch or prev_val is None:
        return (
            torch.tensor(config.initial_weight_mse, device=device),
            torch.tensor(config.initial_weight_ps, device=device),
        )
    l_mse = torch.tensor(prev_val["loss_mse_mean"], device=device)
    l_ps = torch.tensor(prev_val["loss_ps_mean"], device=device)
    return balancer.get_weights(l_mse, l_ps)
