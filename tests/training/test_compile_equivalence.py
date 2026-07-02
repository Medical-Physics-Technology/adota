"""Numerical-equivalence and checkpoint-compatibility tests for torch.compile.

The acceleration is opt-in; these tests guard the two correctness properties
that matter when it is enabled:

1. ``torch.compile`` does not change the model's output.
2. Checkpoints written from a compiled model are format-identical to eager
   ones (no ``_orig_mod.`` key prefix) and load cleanly into an eager model.

All compile-dependent assertions skip cleanly when no compile backend is
available (e.g. CPU-only CI without Triton/inductor).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.training.factory import build_adota_model, maybe_compile_model
from src.training.run import CheckpointManager

from .conftest import make_synthetic_batches, make_tiny_config


def _compile_or_skip(model, cfg, sample):
    """Compile ``model`` and run one forward; skip if the backend is missing."""
    compiled = maybe_compile_model(model, cfg)
    if compiled is model:
        pytest.skip("torch.compile returned eager fallback (no backend).")
    x, e, _ = sample
    try:
        with torch.no_grad():
            out = compiled(x, e)[0]
    except Exception as exc:  # pragma: no cover - backend-dependent
        pytest.skip(f"torch.compile backend unavailable: {exc}")
    return compiled, out


def test_compiled_forward_matches_eager() -> None:
    cfg = make_tiny_config(compile=True)
    model = build_adota_model(cfg, torch.device("cpu")).eval()
    sample = make_synthetic_batches(n_batches=1)[0]

    x, e, _ = sample
    with torch.no_grad():
        eager = model(x, e)[0]

    _, compiled_out = _compile_or_skip(model, cfg, sample)
    assert torch.allclose(eager, compiled_out, atol=1e-4, rtol=1e-4)


def test_checkpoint_from_compiled_loads_into_eager(tmp_path: Path) -> None:
    cfg = make_tiny_config(compile=True)
    model = build_adota_model(cfg, torch.device("cpu")).eval()
    sample = make_synthetic_batches(n_batches=1)[0]
    compiled, _ = _compile_or_skip(model, cfg, sample)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path / "ckpts", save_every_n_epochs=1)
    ckpt_path = mgr.save(
        model=compiled,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        best_val_loss=1.0,
        patience_counter=0,
    )

    # No torch.compile key prefix leaked into the snapshot.
    state = torch.load(ckpt_path, weights_only=False)
    assert all(not k.startswith("_orig_mod.") for k in state["model"])

    # Loads cleanly into a fresh eager model and reproduces the output.
    eager_cfg = make_tiny_config(compile=False)
    fresh = build_adota_model(eager_cfg, torch.device("cpu")).eval()
    CheckpointManager.load(ckpt_path, model=fresh)

    x, e, _ = sample
    with torch.no_grad():
        out_compiled = compiled(x, e)[0]
        out_fresh = fresh(x, e)[0]
    assert torch.allclose(out_compiled, out_fresh, atol=1e-5, rtol=1e-5)
