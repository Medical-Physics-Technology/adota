"""The after-the-fact readers of ``compare.py``: divergence flag, trajectory
estimate and seed aggregation, on synthetic run frames (no training)."""
from __future__ import annotations

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
