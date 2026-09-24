"""The fixed learning-rate schedules: the pure function on its own (with and
without the EXP-0012 warmup), config validation and round trip, and (marked
``slow``, like the neighbouring end-to-end tests in ``test_loop.py``) the
retrospective loop actually holding to them across cycles, including a resume
from a cycle-0 baseline trained under ``"plateau"`` (the exact EXP-0009 ->
EXP-0010 situation).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from src.active_learning.retrospective.config import RetroConfig
from src.active_learning.retrospective.loop import prepare_splits, run_cycle0, run_strategy
from src.active_learning.retrospective.lr_schedule import fixed_lr
from src.training.run_dir import save_resolved_config

from .test_loop import TINY_TRAINING, make_config


def test_fixed_lr_values():
    for epoch in (0, 1, 25, 49):
        assert fixed_lr("constant", 5e-4, 0.0, epoch, 50) == 5e-4

    assert fixed_lr("cosine_per_cycle", 1e-3, 1e-5, 0, 50) == 1e-3
    assert fixed_lr("cosine_per_cycle", 1e-3, 1e-5, 49, 50) == pytest.approx(1e-5)

    values = [fixed_lr("cosine_per_cycle", 1e-3, 1e-5, e, 50) for e in range(50)]
    assert all(a >= b for a, b in zip(values, values[1:]))  # monotone non-increasing

    assert fixed_lr("cosine_per_cycle", 1e-3, 1e-5, 0, 1) == 1e-3
    assert fixed_lr("constant", 1e-3, 1e-5, 0, 1) == 1e-3

    with pytest.raises(ValueError):
        fixed_lr("linear", 1e-3, 0.0, 0, 50)


def _cosine_before_warmup(lr0: float, lr_min: float, epoch: int, epochs: int) -> float:
    """The EXP-0011 formula, copied verbatim, to pin ``warmup_epochs=0`` to it."""
    if epochs <= 1:
        return float(lr0)
    progress = epoch / (epochs - 1)
    return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + math.cos(math.pi * progress)))


def test_zero_warmup_is_the_old_formula_bit_for_bit():
    for epochs in (1, 2, 3, 50):
        for epoch in range(epochs):
            old = _cosine_before_warmup(5e-4, 5e-5, epoch, epochs)
            assert fixed_lr("cosine_per_cycle", 5e-4, 5e-5, epoch, epochs) == old
            assert fixed_lr("cosine_per_cycle", 5e-4, 5e-5, epoch, epochs, warmup_epochs=0) == old
            assert fixed_lr("constant", 5e-4, 5e-5, epoch, epochs, warmup_epochs=0) == 5e-4


def test_cosine_warmup_values():
    lr0, lr_min, w, n = 5e-4, 5e-5, 5, 50
    values = [fixed_lr("cosine_per_cycle", lr0, lr_min, e, n, warmup_epochs=w) for e in range(n)]
    assert values[0] == lr_min                                  # no jump from the previous cycle's end
    assert values[w - 1] == pytest.approx(lr_min + (lr0 - lr_min) * (w - 1) / w)
    assert values[w] == pytest.approx(lr0)                      # first epoch at the peak
    assert values[-1] == pytest.approx(lr_min)
    assert values[:w + 1] == sorted(values[:w + 1])             # rising
    assert all(a >= b for a, b in zip(values[w:], values[w + 1:]))  # then falling
    assert max(values) == pytest.approx(lr0)
    # Warmup until the last epoch: no room to decay, the last epoch is at lr0.
    assert fixed_lr("cosine_per_cycle", lr0, lr_min, 2, 3, warmup_epochs=2) == pytest.approx(lr0)


def test_constant_warmup_rises_then_stays_flat():
    lr0, lr_min = 1e-3, 1e-4
    values = [fixed_lr("constant", lr0, lr_min, e, 10, warmup_epochs=4) for e in range(10)]
    assert values[0] == lr_min
    assert values[1:4] == pytest.approx([lr_min + (lr0 - lr_min) * k / 4 for k in (1, 2, 3)])
    assert values[4:] == pytest.approx([lr0] * 6)


@pytest.mark.parametrize("warmup", [-1, 50, 51])
def test_fixed_lr_rejects_bad_warmup(warmup):
    with pytest.raises(ValueError, match="warmup_epochs"):
        fixed_lr("cosine_per_cycle", 5e-4, 5e-5, 0, 50, warmup_epochs=warmup)


@pytest.mark.parametrize("raw, match", [
    ({"lr_schedule": "plateau", "warmup_epochs": 5}, "plateau"),
    ({"lr_schedule": "cosine_per_cycle", "warmup_epochs": -1}, ">= 0"),
    ({"lr_schedule": "cosine_per_cycle", "warmup_epochs": 50, "epochs_per_cycle": 50}, "smaller"),
    ({"lr_schedule": "constant", "warmup_epochs": 3, "epochs_per_cycle": 2}, "smaller"),
])
def test_retro_config_rejects_bad_warmup(raw, match):
    with pytest.raises(ValueError, match=match):
        RetroConfig.from_dict(raw)


def test_retro_config_warmup_round_trip(tmp_path):
    assert RetroConfig().warmup_epochs == 0
    assert RetroConfig.from_dict({"lr_schedule": "plateau", "warmup_epochs": 0}).warmup_epochs == 0
    cfg = RetroConfig.from_dict({"lr_schedule": "cosine_per_cycle", "lr_min": 5e-5,
                                 "warmup_epochs": 5})
    save_resolved_config(cfg, tmp_path / "config.yaml")
    back = RetroConfig.from_dict(yaml.safe_load((tmp_path / "config.yaml").read_text()))
    assert back == cfg
    assert asdict(back)["warmup_epochs"] == 5


def test_shipped_loop_config_is_exp0012():
    config = Path(__file__).resolve().parents[2] / "scripts" / "config_al_retro_loop.yaml"
    raw = yaml.safe_load(config.read_text())
    cfg = RetroConfig.from_dict(raw)
    assert cfg.experiment == "EXP-0012"
    assert (cfg.lr_schedule, cfg.lr_min, cfg.warmup_epochs) == ("cosine_per_cycle", 5e-5, 5)
    assert cfg.runs_dir.endswith("/d30/exp0012")


def test_retro_config_rejects_unknown_lr_schedule():
    with pytest.raises(ValueError):
        RetroConfig.from_dict({"lr_schedule": "linear"})


@pytest.mark.slow
def test_constant_schedule_keeps_lr_fixed_across_cycles(tmp_path):
    cfg = make_config(tmp_path, "random")
    cfg.lr_schedule = "constant"
    cfg.epochs_per_cycle = 2
    prepare_splits(cfg)

    cycle0_dir = tmp_path / "runs" / "cycle0"
    cycle0_dir.mkdir(parents=True)
    run_cycle0(cfg, cycle0_dir)

    run_dir = tmp_path / "runs" / "constant"
    run_dir.mkdir()
    run_strategy(cfg, run_dir, cycle0_dir)

    lr0 = TINY_TRAINING["learning_rate"]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows
    assert all(r["lr"] == pytest.approx(lr0) for r in rows)
    assert all(r["lr_schedule"] == "constant" for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["lr_schedule"] == "constant"


@pytest.mark.slow
def test_cosine_schedule_restarts_every_cycle(tmp_path):
    cfg = make_config(tmp_path, "random")
    cfg.lr_schedule = "cosine_per_cycle"
    cfg.lr_min = 1e-5
    cfg.epochs_per_cycle = 3
    prepare_splits(cfg)

    cycle0_dir = tmp_path / "runs" / "cycle0"
    cycle0_dir.mkdir(parents=True)
    run_cycle0(cfg, cycle0_dir)

    run_dir = tmp_path / "runs" / "cosine"
    run_dir.mkdir()
    run_strategy(cfg, run_dir, cycle0_dir)

    lr0 = TINY_TRAINING["learning_rate"]
    mid = fixed_lr("cosine_per_cycle", lr0, cfg.lr_min, 1, 3)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    by_cycle: dict = {}
    for row in rows:
        by_cycle.setdefault(row["cycle"], []).append(row["lr"])
    for cycle, lrs in by_cycle.items():
        assert lrs == pytest.approx([lr0, mid, cfg.lr_min], abs=1e-9), f"cycle {cycle}"


@pytest.mark.slow
def test_fixed_schedule_resumes_from_plateau_cycle0(tmp_path):
    cfg = make_config(tmp_path, "random")
    cfg.epochs_per_cycle = 2
    prepare_splits(cfg)

    # Cycle 0 trains under the default "plateau" schedule.
    cycle0_dir = tmp_path / "runs" / "cycle0"
    cycle0_dir.mkdir(parents=True)
    run_cycle0(cfg, cycle0_dir)

    # The strategy resumes from it under a fixed schedule: must not raise.
    cfg.lr_schedule = "constant"
    run_dir = tmp_path / "runs" / "constant_from_plateau"
    run_dir.mkdir()
    run_strategy(cfg, run_dir, cycle0_dir)

    lr0 = TINY_TRAINING["learning_rate"]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    strategy_rows = [r for r in rows if r.get("lr_schedule") == "constant"]
    assert strategy_rows
    assert all(r["lr"] == pytest.approx(lr0) for r in strategy_rows)


@pytest.mark.slow
def test_cosine_warmup_holds_across_cycles(tmp_path):
    cfg = make_config(tmp_path, "random")
    cfg.lr_schedule = "constant"
    cfg.epochs_per_cycle = 3
    prepare_splits(cfg)

    # Cycle 0 at a constant LR, as the shared EXP-0009 baseline.
    cycle0_dir = tmp_path / "runs" / "cycle0"
    cycle0_dir.mkdir(parents=True)
    run_cycle0(cfg, cycle0_dir)

    cfg.lr_schedule = "cosine_per_cycle"
    cfg.lr_min = 1e-5
    cfg.warmup_epochs = 1
    run_dir = tmp_path / "runs" / "cosine_warmup"
    run_dir.mkdir()
    run_strategy(cfg, run_dir, cycle0_dir)

    lr0 = TINY_TRAINING["learning_rate"]
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    by_cycle: dict = {}
    for row in rows:
        if row["cycle"] > 0:
            by_cycle.setdefault(row["cycle"], []).append(row["lr"])
    assert by_cycle
    for cycle, lrs in by_cycle.items():
        assert lrs == pytest.approx([cfg.lr_min, lr0, cfg.lr_min], abs=1e-9), f"cycle {cycle}"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["warmup_epochs"] == 1
