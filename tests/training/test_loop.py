"""Unit tests for :mod:`src.training.loop`."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.training.factory import build_adota_model, build_optimizer_scheduler
from src.training.losses import LMSE, LPS, TwoObjectiveBalancer
from src.training.loop import resolve_weights, train_one_epoch

from .conftest import make_synthetic_batches, make_tiny_config


def _run_epochs(loss_mode: str, n_epochs: int, run_dir: Path):
    torch.manual_seed(0)
    cfg = make_tiny_config(learning_rate=1e-3)
    device = torch.device("cpu")
    model = build_adota_model(cfg, device)
    optimizer, _ = build_optimizer_scheduler(model, cfg)
    batches = make_synthetic_batches(n_batches=4, batch_size=2, device=device)

    w_mse = torch.tensor(0.7)
    w_ps = torch.tensor(0.3)
    stats = []
    for epoch in range(n_epochs):
        s = train_one_epoch(
            model=model,
            train_loader=batches,
            optimizer=optimizer,
            loss_mse_fn=LMSE(),
            loss_ps_fn=LPS(),
            weight_mse=w_mse,
            weight_ps=w_ps,
            device=device,
            epoch=epoch,
            run_dir=run_dir,
            max_batches=None,
            loss_mode=loss_mode,
        )
        stats.append(s)
    return stats


def test_train_one_epoch_reports_stats(tmp_path: Path) -> None:
    stats = _run_epochs("mse_idd", n_epochs=1, run_dir=tmp_path)[0]
    assert stats["n_batches"] == 4
    for key in (
        "loss_combined_mean",
        "loss_mse_mean",
        "loss_ps_mean",
        "grad_norm_last",
        "param_norm",
    ):
        assert key in stats
    assert math.isfinite(stats["grad_norm_last"])
    assert math.isfinite(stats["param_norm"])
    assert stats["grad_norm_last"] >= 0.0


def test_train_one_epoch_overfits(tmp_path: Path) -> None:
    stats = _run_epochs("mse_idd", n_epochs=6, run_dir=tmp_path)
    first = stats[0]["loss_combined_mean"]
    last = stats[-1]["loss_combined_mean"]
    assert last < first, f"loss did not decrease: {[s['loss_combined_mean'] for s in stats]}"


def test_train_one_epoch_mse_only_ignores_ps(tmp_path: Path) -> None:
    stats = _run_epochs("mse_only", n_epochs=1, run_dir=tmp_path)[0]
    # In mse_only mode the PS term is never added.
    assert stats["loss_ps_mean"] == 0.0
    assert stats["loss_combined_mean"] == stats["loss_mse_mean"]


# ── resolve_weights ─────────────────────────────────────────────────────────


def test_resolve_weights_static_before_adaptive() -> None:
    cfg = make_tiny_config(
        initial_weight_mse=0.6,
        initial_weight_ps=0.4,
        adaptive_after_epoch=10,
    )
    balancer = TwoObjectiveBalancer(smoothing=0.9)
    w_mse, w_ps = resolve_weights(cfg, epoch=3, balancer=balancer, prev_val=None,
                                  device=torch.device("cpu"))
    assert float(w_mse) == pytest.approx(0.6)
    assert float(w_ps) == pytest.approx(0.4)


def test_resolve_weights_static_when_no_prev_val() -> None:
    cfg = make_tiny_config(adaptive_after_epoch=0,
                           initial_weight_mse=0.5, initial_weight_ps=0.5)
    balancer = TwoObjectiveBalancer(smoothing=0.9)
    # Past the adaptive epoch but no prev_val yet -> still static.
    w_mse, w_ps = resolve_weights(cfg, epoch=5, balancer=balancer, prev_val=None,
                                  device=torch.device("cpu"))
    assert float(w_mse) == pytest.approx(0.5)
    assert float(w_ps) == pytest.approx(0.5)


def test_resolve_weights_adaptive_uses_balancer() -> None:
    cfg = make_tiny_config(adaptive_after_epoch=0)
    balancer = TwoObjectiveBalancer(smoothing=0.9)
    prev_val = {"loss_mse_mean": 0.4, "loss_ps_mean": 0.6}
    w_mse, w_ps = resolve_weights(cfg, epoch=5, balancer=balancer, prev_val=prev_val,
                                  device=torch.device("cpu"))
    # Balancer weights are a normalized pair summing to 1.
    assert math.isclose(float(w_mse) + float(w_ps), 1.0, abs_tol=1e-5)
