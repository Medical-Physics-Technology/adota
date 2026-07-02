"""Shared fixtures / helpers for the ``src.training`` unit tests.

A tiny CPU-friendly :class:`DoTA3D_v3` configuration (input ``(2, 16, 16, 16)``,
2 levels, 8 encoder features ~ 1.7M params) keeps these tests fast and lets
them run anywhere without the H5 dataset or a GPU.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.schemas.configs import TrainingConfig

TINY_INPUT_SHAPE = (2, 16, 16, 16)

# Model hyperparameters that keep DoTA3D_v3 small and CPU-fast.
TINY_MODEL_KWARGS: Dict[str, Any] = dict(
    input_shape=list(TINY_INPUT_SHAPE),
    num_transformers=1,
    num_heads=2,
    num_levels=2,
    enc_features=8,
    kernel_size=3,
    convolutional_steps=1,
    conv_hidden_channels=8,
    dropout_rate=0.0,  # deterministic
    causal=True,
    zero_padding=True,
    last_activation=False,
    num_forward=1,
)


def make_tiny_config(**overrides: Any) -> TrainingConfig:
    """Return a :class:`TrainingConfig` wired for the tiny test model."""
    params: Dict[str, Any] = dict(TINY_MODEL_KWARGS)
    params.update(overrides)
    return TrainingConfig(**params)


def make_synthetic_batches(
    n_batches: int = 4,
    batch_size: int = 2,
    device: torch.device | None = None,
    seed: int = 0,
):
    """Build a small, fixed list of ``(x, energy, y)`` batches in ``[0, 1]``.

    ``y`` is a deterministic smooth function of ``x`` so the model has a real
    signal to overfit. Returned as a plain list, which ``train_one_epoch``
    consumes like any iterable supporting ``len``.
    """
    device = device or torch.device("cpu")
    g = torch.Generator().manual_seed(seed)
    c, d, h, w = TINY_INPUT_SHAPE
    batches = []
    for _ in range(n_batches):
        x = torch.rand(batch_size, c, d, h, w, generator=g)
        e = torch.rand(batch_size, 1, generator=g)
        # Target: lateral-smoothed first channel, kept in [0, 1].
        y = (x[:, :1] * 0.5 + 0.25).clamp(0.0, 1.0)
        batches.append((x.to(device), e.to(device), y.to(device)))
    return batches
