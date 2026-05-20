"""Training-loop utilities for ADoTA.

Helpers in this module operate on objects that only exist inside the
training pipeline (input mini-batches, PyTorch optimizers). Generic
serialization helpers live in :mod:`src.utils.serialization`.
"""

from __future__ import annotations

from typing import List

import torch


def validate_tensor_ranges(
    *tensors: torch.Tensor,
    low: float = 0.0,
    high: float = 1.0,
) -> bool:
    """Return ``True`` if every tensor lies within ``[low, high]``.

    Used as a guard at the top of each training step to catch
    de-normalized inputs before they reach the model.

    Args:
        *tensors: One or more tensors to check.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        ``True`` if all tensors satisfy ``low <= t.min() and t.max() <= high``,
        else ``False``. Empty input returns ``True``.
    """
    for t in tensors:
        if t.numel() == 0:
            continue
        if t.min().item() < low or t.max().item() > high:
            return False
    return True


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate of an optimizer's first parameter group.

    Args:
        optimizer: Any PyTorch optimizer.

    Returns:
        The current learning rate as a Python ``float``.

    Raises:
        ValueError: If the optimizer has no parameter groups.
    """
    if not optimizer.param_groups:
        raise ValueError("Optimizer has no parameter groups.")
    return float(optimizer.param_groups[0]["lr"])


def get_all_lrs(optimizer: torch.optim.Optimizer) -> List[float]:
    """Return the learning rate of every parameter group.

    Useful when the optimizer has multiple groups with distinct schedules
    (e.g. encoder vs. decoder LRs); :func:`get_lr` only reads the first.

    Args:
        optimizer: Any PyTorch optimizer.

    Returns:
        List of learning rates, one per parameter group, in the order
        ``optimizer.param_groups`` exposes them.
    """
    return [float(g["lr"]) for g in optimizer.param_groups]
