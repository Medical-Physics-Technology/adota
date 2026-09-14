"""The fixed learning-rate schedules: the pure function on its own, config
validation, and (marked ``slow``, like the neighbouring end-to-end tests in
``test_loop.py``) the retrospective loop actually holding to them across
cycles, including a resume from a cycle-0 baseline trained under
``"plateau"`` (the exact EXP-0009 -> EXP-0010 situation).
"""
from __future__ import annotations

import json

import pytest

from src.active_learning.retrospective.config import RetroConfig
from src.active_learning.retrospective.loop import prepare_splits, run_cycle0, run_strategy
from src.active_learning.retrospective.lr_schedule import fixed_lr

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
