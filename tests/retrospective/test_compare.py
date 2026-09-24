"""The after-the-fact readers of ``compare.py``: divergence flag (boundaries
included), restart ratios, trajectory estimate and seed aggregation, on
synthetic run frames and a synthetic ``metrics.jsonl`` (no training)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.active_learning.retrospective.compare import (
    METRIC_KEYS,
    RunData,
    aggregate_over_seeds,
    boundary_table,
    divergence_table,
    read_run,
    restart_table,
    trajectory_table,
    unique_labels,
)


def _rows(seed: int, strategy: str = "s", spike_at=None, n_cycles: int = 2,
          epochs: int = 10) -> pd.DataFrame:
    """Per-epoch rows of a run: a smoothly falling loss, subsample metrics every
    5 epochs and a full-V block at the boundary; ``spike_at=(cycle, epoch)``
    multiplies the training loss by ten from that epoch on."""
    rng = np.random.default_rng(seed)
    rows = []
    cumulative = 0
    for cycle in range(1, n_cycles + 1):
        for epoch in range(epochs):
            loss = 0.5 * np.exp(-0.05 * cumulative)
            if spike_at and (cycle, epoch) >= spike_at and cycle == spike_at[0]:
                loss *= 10.0
            boundary = epoch == epochs - 1
            cadence = epoch % 5 == 4
            row = {"cycle": cycle, "epoch_in_cycle": epoch, "cumulative_epoch": cumulative,
                   "n_train": 100 * cycle, "cycle_boundary": boundary, "strategy": strategy,
                   "lr": 1e-3, "train_loss": loss, "val_loss": loss * 1.1,
                   "val_loss_mse": loss, "val_loss_ps": loss, "epoch_time_s": 1.0,
                   "gamma_label": "g"}
            for prefix, present in (("sub", cadence), ("full", boundary)):
                for metric in METRIC_KEYS:
                    row[f"{prefix}_{metric}"] = (0.8 + 0.01 * cumulative + rng.normal(0, 0.01)
                                                 if present else np.nan)
            rows.append(row)
            cumulative += 1
    return pd.DataFrame(rows)


def _run(label: str, strategy: str, seed: int, **kwargs) -> RunData:
    return RunData(run_dir=Path(f"/nonexistent/{label}"), label=label, strategy=strategy,
                   manifest={"config": {"seed": seed}}, rows=_rows(seed, strategy, **kwargs),
                   seed=seed)


def test_divergence_flags_only_the_spiked_cycle():
    clean = _run("a", "random", 1)
    spiked = _run("b", "score_topk", 2, spike_at=(2, 3))
    table = divergence_table([clean, spiked], ratio=2.0)
    assert not table[table["label"] == "a"]["diverged"].any()
    flagged = table[(table["label"] == "b") & table["diverged"]]
    assert flagged["cycle"].tolist() == [2]
    assert flagged["epoch_in_cycle"].iloc[0] == 3
    assert flagged["max_rise"].iloc[0] == pytest.approx(10.0, rel=0.1)


def test_trajectory_is_the_median_of_the_last_evaluations():
    run = _run("a", "random", 1)
    table = trajectory_table([run], last_k=2)
    assert table["cycle"].tolist() == [1, 2]
    assert (table["n_evaluations"] == 2).all()
    cadence = run.rows[run.rows["sub_gpr_mean"].notna() & (run.rows["cycle"] == 2)]
    expected = cadence["sub_gpr_mean"].iloc[-2:].median()
    assert table.loc[table["cycle"] == 2, "sub_gpr_mean"].iloc[0] == pytest.approx(expected)
    assert table.loc[table["cycle"] == 2, "n_train"].iloc[0] == 200


def test_aggregate_over_seeds_gives_mean_and_range_per_strategy():
    runs = unique_labels([_run("random", "random", s) for s in (1, 2, 3)]
                         + [_run("score_topk", "score_topk", 1)])
    assert [r.label for r in runs] == ["random_seed1", "random_seed2", "random_seed3", "score_topk"]
    table = boundary_table(runs)
    agg = aggregate_over_seeds(table)
    random_c2 = agg[(agg["strategy"] == "random") & (agg["cycle"] == 2)].iloc[0]
    values = table[(table["strategy"] == "random") & (table["cycle"] == 2)]["full_gpr_mean"]
    assert random_c2["n_runs"] == 3
    assert random_c2["full_gpr_mean"] == pytest.approx(values.mean())
    assert random_c2["full_gpr_mean_min"] == pytest.approx(values.min())
    assert random_c2["full_gpr_mean_max"] == pytest.approx(values.max())
    assert random_c2["n_train"] == 200 and random_c2["label"] == "random"
    topk = agg[agg["strategy"] == "score_topk"]
    assert (topk["n_runs"] == 1).all()
    assert (topk["full_gpr_mean_min"] == topk["full_gpr_mean"]).all()


def test_unique_labels_refuses_the_same_strategy_and_seed_twice():
    with pytest.raises(ValueError, match="explicit labels"):
        unique_labels([_run("random", "random", 1), _run("random", "random", 1)])


def _write_run(run_dir: Path, losses: dict, strategy: str = "random", seed: int = 7) -> Path:
    """A run directory as the loop writes it: ``losses[cycle]`` is the list of
    (train, val) combined losses of that cycle's epochs, cycle 0 included (the
    inherited rows)."""
    run_dir.mkdir(parents=True)
    rows, cumulative = [], 0
    for cycle in sorted(losses):
        for epoch, (train, val) in enumerate(losses[cycle]):
            rows.append({"cycle": cycle, "epoch_in_cycle": epoch, "cumulative_epoch": cumulative,
                         "n_train": 100 + 10 * cycle, "strategy": strategy if cycle else "cycle0",
                         "cycle_boundary": epoch == len(losses[cycle]) - 1, "lr": 1e-3,
                         "train": {"loss_combined_mean": train},
                         "val_loss": {"loss_combined_mean": val, "loss_mse_mean": val,
                                      "loss_ps_mean": val},
                         "metrics_subsample": None, "metrics_full": None, "gamma_label": "g"})
            cumulative += 1
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps({"strategy": strategy,
                                                       "config": {"seed": seed}}))
    return run_dir


def _falling(start: float, n: int = 12) -> list:
    return [(start * 0.9 ** k, 2 * start * 0.9 ** k) for k in range(n)]


def test_restart_ratio_and_boundary_divergence_on_a_metrics_log(tmp_path):
    cycle0 = _falling(1.0)
    end0 = cycle0[-1][0]
    # Cycle 1: a warmup-like start, no jump at epoch 0 but a delayed peak at epoch 3.
    cycle1 = [(end0, 2 * end0), (1.2 * end0, 2 * end0), (1.5 * end0, 3 * end0),
              (3.0 * end0, 8 * end0)] + _falling(end0, 8)
    end1 = cycle1[-1][0]
    # Cycle 2: a warm-restart shock, x40 at epoch 0, then recovery.
    cycle2 = [(40 * end1, 10 * 2 * end1)] + _falling(end1, 11)
    run = read_run(_write_run(tmp_path / "run", {0: cycle0, 1: cycle1, 2: cycle2}))

    restarts = restart_table([run], peak_epochs=10).set_index("cycle")
    assert list(restarts.index) == [1, 2]
    assert restarts.loc[1, "restart_ratio"] == pytest.approx(1.0)
    assert restarts.loc[1, "restart_peak_ratio"] == pytest.approx(3.0)
    assert restarts.loc[1, "restart_peak_epoch"] == 3
    assert restarts.loc[1, "val_restart_peak_ratio"] == pytest.approx(8 * end0 / cycle0[-1][1])
    assert restarts.loc[2, "restart_ratio"] == pytest.approx(40.0)
    assert restarts.loc[2, "restart_peak_ratio"] == pytest.approx(40.0)
    assert restarts.loc[2, "restart_peak_epoch"] == 0
    assert restarts.loc[2, "val_restart_ratio"] == pytest.approx(10.0)
    assert restarts.loc[2, "train_loss_prev_end"] == pytest.approx(end1)

    # A peak window shorter than the delay misses the cycle-1 peak.
    short = restart_table([run], peak_epochs=2).set_index("cycle")
    assert short.loc[1, "restart_peak_ratio"] == pytest.approx(1.2)

    table = divergence_table([run], ratio=2.0).set_index("cycle")
    # The x40 step from cycle 1's last epoch into cycle 2 belongs to cycle 2.
    assert table.loc[2, "diverged"]
    assert table.loc[2, "at_restart"]
    assert table.loc[2, "epoch_in_cycle"] == 0
    assert table.loc[2, "max_rise"] == pytest.approx(40.0)
    # Cycle 1's largest rise is x2 (1.5 -> 3.0), inside the cycle and not above the ratio.
    assert not table.loc[1, "at_restart"]
    assert table.loc[1, "max_rise"] == pytest.approx(2.0)
    assert not table.loc[1, "diverged"]
    assert not table.loc[0, "diverged"]


def test_divergence_grouped_by_cycle_would_have_missed_the_restart():
    run = _run("a", "random", 1)
    rows = run.rows.copy()
    first_of_2 = (rows["cycle"] == 2) & (rows["epoch_in_cycle"] == 0)
    rows.loc[rows["cycle"] == 2, "train_loss"] *= 50.0     # the whole cycle sits x50 higher
    run = RunData(**{**run.__dict__, "rows": rows})
    table = divergence_table([run]).set_index("cycle")
    assert table.loc[2, "diverged"] and table.loc[2, "at_restart"]
    assert table.loc[2, "train_loss_after"] == pytest.approx(rows.loc[first_of_2, "train_loss"].iloc[0])
    assert not table.loc[1, "diverged"]
