"""Unit tests for :mod:`src.training.factory`."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.adota.models import DoTA3D_v3
from src.training.factory import (
    build_adota_model,
    build_optimizer_scheduler,
    maybe_compile_model,
    set_determinism,
)

from .conftest import TINY_INPUT_SHAPE, make_synthetic_batches, make_tiny_config


# ── build_optimizer_scheduler ───────────────────────────────────────────────


def test_build_optimizer_scheduler_reads_config() -> None:
    cfg = make_tiny_config(
        learning_rate=3e-4,
        weight_decay=5e-3,
        lr_factor=0.7,
        lr_patience=11,
    )
    model = torch.nn.Linear(4, 2)
    opt, sched = build_optimizer_scheduler(model, cfg)

    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 3e-4
    assert opt.param_groups[0]["weight_decay"] == 5e-3
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert sched.factor == 0.7
    assert sched.patience == 11


# ── build_adota_model ───────────────────────────────────────────────────────


def test_build_adota_model_forward_shape() -> None:
    cfg = make_tiny_config()
    device = torch.device("cpu")
    model = build_adota_model(cfg, device)

    assert isinstance(model, DoTA3D_v3)
    assert sum(p.numel() for p in model.parameters()) > 0

    c, d, h, w = TINY_INPUT_SHAPE
    x = torch.rand(2, c, d, h, w)
    e = torch.rand(2, 1)
    out = model(x, e)[0]
    assert out.shape == (2, 1, d, h, w)


# ── set_determinism ─────────────────────────────────────────────────────────


def test_set_determinism_reproduces_draws() -> None:
    set_determinism(123)
    a = (torch.rand(3), np.random.rand(3), random.random())
    set_determinism(123)
    b = (torch.rand(3), np.random.rand(3), random.random())

    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
    assert a[2] == b[2]
    assert torch.backends.cudnn.deterministic is True


# ── maybe_compile_model ─────────────────────────────────────────────────────


def test_maybe_compile_off_returns_same_object() -> None:
    cfg = make_tiny_config(compile=False)
    model = build_adota_model(cfg, torch.device("cpu"))
    assert maybe_compile_model(model, cfg) is model


def test_maybe_compile_on_is_numerically_equivalent() -> None:
    cfg = make_tiny_config(compile=True)
    model = build_adota_model(cfg, torch.device("cpu")).eval()

    x, e, _ = make_synthetic_batches(n_batches=1)[0]
    with torch.no_grad():
        eager = model(x, e)[0]

    compiled = maybe_compile_model(model, cfg)
    try:
        with torch.no_grad():
            got = compiled(x, e)[0]
    except Exception as exc:  # pragma: no cover - backend-dependent
        import pytest

        pytest.skip(f"torch.compile backend unavailable: {exc}")

    # The wrapper exposes the original module, and its output matches eager.
    assert getattr(compiled, "_orig_mod", None) is model
    assert torch.allclose(eager, got, atol=1e-4, rtol=1e-4)
