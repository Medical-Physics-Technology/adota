"""Tests for :class:`src.training.run.CheckpointManager`.

Coverage:

- Round-trip fidelity: a saved checkpoint, when loaded back into a fresh
  model + optimizer + scheduler + balancer, restores every tracked piece
  of state (weights, optimizer state, scheduler counters, balancer
  running weights, training bookkeeping).
- RNG state survives a save/load cycle (the next ``torch.randn`` produces
  the same value as before the cycle).
- Retention policy: ``best.pth`` is written only when ``is_best=True``,
  ``last.pth`` is always rewritten, and an ``epoch_NNNN.pth`` snapshot
  is kept only on the configured cadence.
- Partial restore: ``CheckpointManager.load`` can restore weights into a
  fresh model without passing the optimizer / scheduler / balancer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest
import torch

from src.training.losses import TwoObjectiveBalancer
from src.training.run import CheckpointManager


# ── Fixtures ────────────────────────────────────────────────────────────────


def _build_state() -> Tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    TwoObjectiveBalancer,
]:
    """Build a tiny, fully-stateful training context."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    # A backward pass to populate optimizer momentum buffers.
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    # Step the scheduler so it has non-default internal state.
    scheduler.step(0.7)
    scheduler.step(0.8)
    balancer = TwoObjectiveBalancer(smoothing=0.7)
    balancer.get_weights(torch.tensor(0.4), torch.tensor(0.6))
    return model, optimizer, scheduler, balancer


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


# ── Round-trip fidelity ─────────────────────────────────────────────────────


def test_save_load_round_trip_restores_all_state(checkpoint_dir: Path) -> None:
    model, optimizer, scheduler, balancer = _build_state()
    mgr = CheckpointManager(checkpoint_dir, save_every_n_epochs=1)

    # Snapshot reference state.
    ref_weights = {k: v.clone() for k, v in model.state_dict().items()}
    ref_balancer = balancer.running_weights.clone()
    ref_lr = optimizer.param_groups[0]["lr"]
    ref_sched_bad = scheduler.num_bad_epochs

    mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=7,
        best_val_loss=0.123,
        patience_counter=3,
        balancer=balancer,
        is_best=True,
    )

    # Fresh objects, perturbed where possible to prove restore actually fired.
    new_model, new_optimizer, new_scheduler, new_balancer = _build_state()
    for p in new_model.parameters():
        p.data.zero_()
    for g in new_optimizer.param_groups:
        g["lr"] = 7.0
    new_balancer.reset()

    bookkeeping = CheckpointManager.load(
        checkpoint_dir / "last.pth",
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        balancer=new_balancer,
        device=torch.device("cpu"),
    )

    assert bookkeeping["epoch"] == 7
    assert bookkeeping["best_val_loss"] == pytest.approx(0.123)
    assert bookkeeping["patience_counter"] == 3

    for k, ref in ref_weights.items():
        assert torch.equal(new_model.state_dict()[k], ref), f"weight mismatch: {k}"

    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(ref_lr)
    assert new_scheduler.num_bad_epochs == ref_sched_bad
    assert new_balancer.running_weights is not None
    assert torch.allclose(new_balancer.running_weights, ref_balancer)


def test_rng_state_round_trips(checkpoint_dir: Path) -> None:
    """After load, the next ``torch.randn`` matches the pre-save draw."""
    model, optimizer, scheduler, balancer = _build_state()
    mgr = CheckpointManager(checkpoint_dir, save_every_n_epochs=1)
    mgr.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        best_val_loss=1.0,
        patience_counter=0,
        balancer=balancer,
        is_best=False,
    )

    # Reference draw right after save.
    expected = torch.randn(5)

    # Perturb global RNG, then reload and re-draw.
    torch.manual_seed(999_999)
    _ = torch.randn(100)
    CheckpointManager.load(
        checkpoint_dir / "last.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        balancer=balancer,
        device=torch.device("cpu"),
    )
    redrawn = torch.randn(5)

    assert torch.equal(expected, redrawn)


# ── Retention policy ────────────────────────────────────────────────────────


def test_retention_policy_writes_last_best_and_periodic(checkpoint_dir: Path) -> None:
    model, optimizer, scheduler, balancer = _build_state()
    mgr = CheckpointManager(checkpoint_dir, save_every_n_epochs=3)

    # Epoch 0: not best, not periodic (0+1 % 3 != 0) → only last.pth.
    mgr.save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=0, best_val_loss=1.0, patience_counter=0,
        balancer=balancer, is_best=False,
    )
    assert (checkpoint_dir / "last.pth").exists()
    assert not (checkpoint_dir / "best.pth").exists()
    assert not (checkpoint_dir / "epoch_0000.pth").exists()

    # Epoch 1: best → best.pth too.
    mgr.save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=1, best_val_loss=0.5, patience_counter=0,
        balancer=balancer, is_best=True,
    )
    assert (checkpoint_dir / "best.pth").exists()

    # Epoch 2: periodic (2+1 = 3, 3 % 3 == 0) → epoch_0002.pth.
    mgr.save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=2, best_val_loss=0.5, patience_counter=1,
        balancer=balancer, is_best=False,
    )
    assert (checkpoint_dir / "epoch_0002.pth").exists()

    # Epoch 3: not periodic, not best. last.pth rewritten.
    last_mtime_before = (checkpoint_dir / "last.pth").stat().st_mtime
    mgr.save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=3, best_val_loss=0.5, patience_counter=2,
        balancer=balancer, is_best=False,
    )
    assert (checkpoint_dir / "last.pth").stat().st_mtime >= last_mtime_before
    assert not (checkpoint_dir / "epoch_0003.pth").exists()


# ── Partial restore ────────────────────────────────────────────────────────


def test_load_works_without_optimizer_scheduler_balancer(checkpoint_dir: Path) -> None:
    """Inference-only restore: pass only the model."""
    model, optimizer, scheduler, balancer = _build_state()
    mgr = CheckpointManager(checkpoint_dir, save_every_n_epochs=1)
    mgr.save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=4, best_val_loss=0.2, patience_counter=0,
        balancer=balancer, is_best=False,
    )

    ref_weights = {k: v.clone() for k, v in model.state_dict().items()}
    new_model, _, _, _ = _build_state()
    for p in new_model.parameters():
        p.data.zero_()

    bookkeeping = CheckpointManager.load(
        checkpoint_dir / "last.pth",
        model=new_model,
        device=torch.device("cpu"),
    )
    assert bookkeeping["epoch"] == 4
    for k, ref in ref_weights.items():
        assert torch.equal(new_model.state_dict()[k], ref)


# ── torch.compile compatibility ────────────────────────────────────────────


def test_save_from_compiled_model_strips_orig_mod_prefix(checkpoint_dir: Path) -> None:
    """A compiled model must produce eager-compatible checkpoints.

    ``torch.compile`` wraps the module so its ``state_dict`` keys gain an
    ``_orig_mod.`` prefix; ``CheckpointManager`` must unwrap it so snapshots
    stay loadable by eager models / inference scripts.
    """
    model, optimizer, scheduler, balancer = _build_state()
    try:
        compiled = torch.compile(model)
    except Exception as exc:  # pragma: no cover - backend-dependent
        pytest.skip(f"torch.compile unavailable: {exc}")

    mgr = CheckpointManager(checkpoint_dir, save_every_n_epochs=1)
    ckpt = mgr.save(
        model=compiled, optimizer=optimizer, scheduler=scheduler,
        epoch=2, best_val_loss=0.3, patience_counter=1,
        balancer=balancer, is_best=False,
    )

    state = torch.load(ckpt, weights_only=False)
    assert all(not k.startswith("_orig_mod.") for k in state["model"])

    # And it loads straight into a fresh eager model.
    fresh, _, _, _ = _build_state()
    for p in fresh.parameters():
        p.data.zero_()
    CheckpointManager.load(ckpt, model=fresh, device=torch.device("cpu"))
    for k, ref in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], ref)


def test_load_weights_only_restores_weights_not_bookkeeping(checkpoint_dir: Path) -> None:
    """Warm-start loads weights only; optimizer/epoch state is not returned."""
    model, optimizer, scheduler, balancer = _build_state()
    mgr = CheckpointManager(checkpoint_dir, save_every_n_epochs=1)
    mgr.save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=7, best_val_loss=0.1, patience_counter=3,
        balancer=balancer, is_best=True,
    )

    ref_weights = {k: v.clone() for k, v in model.state_dict().items()}
    fresh, _, _, _ = _build_state()
    for p in fresh.parameters():
        p.data.zero_()

    # Returns None (no bookkeeping); only weights are copied over.
    out = CheckpointManager.load_weights_only(
        checkpoint_dir / "best.pth", model=fresh, device=torch.device("cpu")
    )
    assert out is None
    for k, ref in ref_weights.items():
        assert torch.equal(fresh.state_dict()[k], ref)


# ── Edge case: balancer with no running state ──────────────────────────────


def test_save_load_with_empty_balancer(checkpoint_dir: Path) -> None:
    """A balancer that has never produced weights survives the round trip."""
    model, optimizer, scheduler, _ = _build_state()
    empty_balancer = TwoObjectiveBalancer()
    assert empty_balancer.running_weights is None

    mgr = CheckpointManager(checkpoint_dir, save_every_n_epochs=1)
    mgr.save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=0, best_val_loss=1.0, patience_counter=0,
        balancer=empty_balancer, is_best=False,
    )

    new_balancer = TwoObjectiveBalancer()
    new_balancer.get_weights(torch.tensor(0.5), torch.tensor(0.5))
    assert new_balancer.running_weights is not None

    CheckpointManager.load(
        checkpoint_dir / "last.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        balancer=new_balancer,
        device=torch.device("cpu"),
    )
    # Loader leaves the balancer untouched when the saved running state
    # was None; the existing running state on ``new_balancer`` persists.
    assert new_balancer.running_weights is not None
