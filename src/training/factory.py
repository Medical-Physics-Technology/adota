"""Construction helpers for the ADoTA training run.

Builds the objects the training loop needs from a resolved
:class:`TrainingConfig`: the config itself (YAML + CLI overrides),
deterministic RNG seeding, backend (TF32 / cuDNN) configuration, the
model, the optimizer + LR scheduler, and an optional ``torch.compile``
wrapper around the model.

These helpers used to live inline in ``scripts/train_adota.py``.

Note on naming: :func:`build_adota_model` is specific to
:class:`DoTA3D_v3`. The optimizer / compile / backend helpers are
model-agnostic, so a future architecture (e.g. a Fourier neural operator
with a different input shape) gets its own ``build_<arch>_model`` factory
while reusing everything else here.
"""

from __future__ import annotations

import logging
import random
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.adota.config import load_yaml_config
from src.adota.models import DoTA3D_v3
from src.schemas.configs import TrainingConfig

logger = logging.getLogger(__name__)


# ── Config plumbing ─────────────────────────────────────────────────────────


def build_config_from_yaml(
    yaml_path: Optional[Path],
    overrides: Dict[str, Any],
) -> TrainingConfig:
    """Load YAML config, apply CLI overrides, return ``TrainingConfig``.

    Only keys that are real :class:`TrainingConfig` fields are kept from the
    YAML. CLI ``overrides`` win when their value is not ``None``. Tuple-shaped
    fields that YAML loads as lists are normalized back to tuples.
    """
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


# ── Determinism / backends ──────────────────────────────────────────────────


def set_determinism(seed: int) -> None:
    """Seed every RNG and force deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_backends(config: TrainingConfig) -> None:
    """Enable optional TF32 / cuDNN autotuning based on the config.

    When ``allow_tf32`` is set, TF32 matmul/conv kernels are enabled (a small
    numeric change for a large throughput gain on Ampere+). When ``compile``
    is set, cuDNN autotuning is turned on so kernel selection can specialize.

    When neither flag is set this is a no-op, so the default training path is
    byte-for-byte identical to the pre-acceleration behavior.
    """
    if config.allow_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config.compile or config.allow_tf32:
        # Autotuning conflicts with cudnn.deterministic; only relax it when the
        # user has explicitly opted into compile/TF32.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ── Model / optimizer ───────────────────────────────────────────────────────


def build_adota_model(config: TrainingConfig, device: torch.device) -> DoTA3D_v3:
    """Instantiate :class:`DoTA3D_v3` from the config and move it to ``device``.

    Specific to the ADoTA transformer architecture. A future model with a
    different input shape (e.g. a Fourier neural operator) should add its own
    ``build_<arch>_model`` factory rather than overloading this one.
    """
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


def maybe_compile_model(
    model: torch.nn.Module, config: TrainingConfig
) -> torch.nn.Module:
    """Optionally wrap ``model`` with :func:`torch.compile`.

    Model-agnostic. Returns ``model`` unchanged when ``config.compile`` is
    false. When true, returns ``torch.compile(model, mode=config.compile_mode)``.
    If compilation is unavailable (e.g. missing Triton/inductor backend), logs
    a warning and falls back to the eager model so a run never fails because of
    compile.

    The optimizer must be built from the *underlying* module's parameters; the
    compiled wrapper shares the same parameter tensors, and checkpoints are
    saved/loaded via the unwrapped module (see ``CheckpointManager``).
    """
    if not config.compile:
        return model
    try:
        compiled = torch.compile(model, mode=config.compile_mode)
        logger.info("torch.compile enabled (mode=%s)", config.compile_mode)
        return compiled
    except Exception as exc:  # pragma: no cover - backend-dependent
        logger.warning(
            "torch.compile failed (%s); falling back to eager execution.", exc
        )
        return model


def build_optimizer_scheduler(
    model: torch.nn.Module, config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """Build an AdamW optimizer + ReduceLROnPlateau scheduler from the config."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )
    return optimizer, scheduler
