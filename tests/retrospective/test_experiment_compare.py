"""Run-by-run comparison of two experiments (``experiment_compare.py``): pairing,
selection identity, the per-cycle table and the paired differences, on synthetic
runs (no training), plus a smoke test of the four figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.active_learning.retrospective.compare import METRIC_KEYS, RunData
from src.active_learning.retrospective.experiment_compare import (
    cycle_table,
    loss_curves,
    pair_runs,
    paired_deltas,
    selection_identity,
    strategy_runs,
    summarize_deltas,
)


def _rows(seed: int, strategy: str, shift: float = 0.0, epochs: int = 10) -> pd.DataFrame:
    """Two cycles plus the inherited cycle 0; subsample metrics every 5 epochs,
    full-V metrics at the boundary, pass rates raised by ``shift``."""
    rng = np.random.default_rng(seed)
    rows, cumulative = [], 0
    for cycle in range(3):
        for epoch in range(epochs):
            loss = 0.5 * np.exp(-0.05 * cumulative)
            boundary = epoch == epochs - 1
            row = {"cycle": cycle, "epoch_in_cycle": epoch, "cumulative_epoch": cumulative,
                   "n_train": 100 * (cycle + 1), "cycle_boundary": boundary,
                   "strategy": strategy if cycle else "cycle0", "lr": 1e-4,
                   "train_loss": loss, "val_loss": loss * (1.5 if boundary else 1.0),
                   "val_loss_mse": loss, "val_loss_ps": loss, "epoch_time_s": 1.0, "gamma_label": "g"}
            for prefix, present in (("sub", epoch % 5 == 4), ("full", boundary)):
                for metric in METRIC_KEYS:
                    row[f"{prefix}_{metric}"] = (0.8 + 0.01 * cumulative + shift + rng.normal(0, 1e-3)
                                                 if present else np.nan)
            rows.append(row)
            cumulative += 1
    return pd.DataFrame(rows)


def _run(tmp: Path, experiment: str, strategy: str, seed: int, shift: float = 0.0,
         selected: tuple = ("a", "b")) -> RunData:
    run_dir = tmp / experiment / f"{strategy}_seed{seed}"
    for cycle in (1, 2):
        folder = run_dir / "cycles" / f"cycle_{cycle:02d}"
        folder.mkdir(parents=True)
        pd.DataFrame({"sample_id": [f"{s}{cycle}" for s in selected]}).to_csv(
            folder / "selection.csv", index=False)
        pd.DataFrame({"sample_id": ["x", "y"] + [f"{s}{cycle}" for s in selected]}).to_csv(
            folder / "training_ids.csv", index=False)
    return RunData(run_dir=run_dir, label=f"{strategy}_seed{seed}", strategy=strategy,
                   manifest={"role": "strategy", "config": {"seed": seed}},
                   rows=_rows(seed, strategy, shift), seed=seed)


def test_pair_runs_matches_on_strategy_and_seed_and_refuses_an_orphan(tmp_path):
    a = [_run(tmp_path, "A", "random", 1), _run(tmp_path, "A", "score_topk", 1)]
    b = [_run(tmp_path, "B", "score_topk", 1), _run(tmp_path, "B", "random", 1)]
    pairs = pair_runs(a, b)
    assert list(pairs) == [("random", 1), ("score_topk", 1)]
    assert pairs[("random", 1)][1].run_dir.parent.name == "B"
    with pytest.raises(ValueError, match="without a partner"):
        pair_runs(a, b[:1])


def test_strategy_runs_drops_the_cycle0_baseline(tmp_path):
    run = _run(tmp_path, "A", "random", 1)
    base = RunData(**{**run.__dict__, "strategy": "cycle0", "manifest": {"role": "cycle0_baseline"}})
    assert strategy_runs([run, base]) == [run]


def test_selection_identity_catches_a_reordered_batch(tmp_path):
    a = [_run(tmp_path, "A", "random", 1)]
    same = pair_runs(a, [_run(tmp_path, "B", "random", 1)])
    assert selection_identity(same)["selection_identical"].all()
    swapped = pair_runs(a, [_run(tmp_path, "C", "random", 1, selected=("b", "a"))])
    identity = selection_identity(swapped)
    assert not identity["selection_identical"].any()
    assert identity["training_set_identical"].all()


def test_cycle_table_and_paired_deltas(tmp_path):
    a = [_run(tmp_path, "A", "random", s) for s in (1, 2)]
    b = [_run(tmp_path, "B", "random", s, shift=0.02) for s in (1, 2)]
    ta, tb = cycle_table(a, "A", last_k=2), cycle_table(b, "B", last_k=2)
    assert set(ta["cycle"]) == {0, 1, 2}
    assert ta["val_last_over_min"].between(1.0, 2.0).all()
    assert not ta["diverged"].any()
    assert ta["traj_gpr_mean"].notna().all()
    paired = paired_deltas(ta[ta["cycle"] >= 1], tb[tb["cycle"] >= 1], ["full_gpr_mean", "val_loss"])
    assert len(paired) == 4
    assert paired["delta_full_gpr_mean"].to_numpy() == pytest.approx(0.02, abs=5e-3)
    summary = summarize_deltas(paired, ["full_gpr_mean", "val_loss"], lower_is_better=["val_loss"])
    row = summary[(summary["cycle"] == 2) & (summary["metric"] == "full_gpr_mean")].iloc[0]
    assert row["n_b_better"] == 2 and row["n_pairs"] == 2


def test_the_four_figures_are_written(tmp_path):
    from src.figures.al_experiment_comparison import (
        gamma_text,
        paired_quality_figure,
        schedule_and_loss_figure,
        stability_figure,
        strategy_endpoint_figure,
    )

    assert gamma_text("2pct_2mm_cutoff10pct") == "2%/2mm, 10% dose cutoff"
    runs = {e: [_run(tmp_path, e, s, seed, shift=0.01 * (e == "B")) for s in ("random", "score_topk")
                for seed in (1, 2)] for e in ("A", "B")}
    table = pd.concat([cycle_table(runs[e], e) for e in runs], ignore_index=True)
    scored = table[table["cycle"] >= 1]
    paired = paired_deltas(scored[scored["experiment"] == "A"], scored[scored["experiment"] == "B"],
                           ["full_gpr_mean"])
    legend = {"A": "A", "B": "B"}
    panel = [("full_gpr_mean", "Mean GPR", "GPR [%]", 100.0)]
    curves = pd.concat([loss_curves(runs[e], e) for e in runs], ignore_index=True)
    out = [schedule_and_loss_figure(curves, runs["A"][0].boundaries(), ["A", "B"], legend,
                                    str(tmp_path / "g1"), "t", caption="c", shared_until=9),
           paired_quality_figure(scored, paired, panel, ["A", "B"], legend, str(tmp_path / "g2"), "t"),
           strategy_endpoint_figure(table, 2, panel, ["A", "B"], legend, str(tmp_path / "g3"), "t"),
           stability_figure(table, ["A", "B"], legend, str(tmp_path / "g4"), "t")]
    for paths in out:
        assert [p.suffix for p in paths] == [".svg", ".pdf", ".png"]
        assert all(p.stat().st_size > 0 for p in paths)
