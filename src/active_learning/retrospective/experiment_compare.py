"""Comparing two retrospective experiments run by run (``scripts/al_compare_experiments.py``).

:mod:`src.active_learning.retrospective.compare` compares the strategies inside
one experiment. This module compares two experiments that differ in one training
setting (EXP-0012 warmup plus cosine against EXP-0013 constant learning rate) and
share everything else. The strategies are model-free, so a strategy under a seed
selects the same records in both experiments; :func:`selection_identity` checks
that from the per-cycle ``selection.csv`` and ``training_ids.csv``, and when it
holds, the run of experiment A and the run of experiment B with the same
``(strategy, seed)`` are a pair that differs only in the training setting.
:func:`paired_deltas` takes B minus A per pair and cycle.

:func:`cycle_table` gathers per run and cycle what the comparison reads: the
full-V metrics at the boundary, the median of the last subsample evaluations
(the boundary estimate that does not depend on which epoch the cycle ended on),
how far the boundary validation loss sits above the cycle's best
(``val_last_over_min``, the snapshot noise) and whether the cycle diverged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.active_learning.retrospective.compare import (
    METRIC_KEYS,
    RunData,
    boundary_table,
    divergence_table,
    trajectory_table,
)

PairKey = Tuple[str, int]


def strategy_runs(runs: Sequence[RunData]) -> List[RunData]:
    """Drop the cycle-0 baseline run an experiment directory also holds."""
    return [r for r in runs if r.manifest.get("role", "strategy") != "cycle0_baseline"
            and r.strategy != "cycle0"]


def pair_runs(runs_a: Sequence[RunData], runs_b: Sequence[RunData]
              ) -> Dict[PairKey, Tuple[RunData, RunData]]:
    """Match the runs of two experiments on ``(strategy, seed)``; raise when a
    key is missing on one side or appears twice on one side."""
    sides = []
    for runs in (runs_a, runs_b):
        keyed: Dict[PairKey, RunData] = {}
        for run in runs:
            key = (run.strategy, int(run.seed))
            if key in keyed:
                raise ValueError(f"two runs with strategy {key[0]} and seed {key[1]}: "
                                 f"{keyed[key].run_dir} and {run.run_dir}")
            keyed[key] = run
        sides.append(keyed)
    only_a = sorted(set(sides[0]) - set(sides[1]))
    only_b = sorted(set(sides[1]) - set(sides[0]))
    if only_a or only_b:
        raise ValueError(f"runs without a partner: only in A {only_a}, only in B {only_b}")
    return {key: (sides[0][key], sides[1][key]) for key in sorted(sides[0])}


def _cycle_ids(run_dir: Path, cycle: int, name: str) -> List[str]:
    path = Path(run_dir) / "cycles" / f"cycle_{cycle:02d}" / name
    return pd.read_csv(path)["sample_id"].astype(str).tolist() if path.exists() else []


def selection_identity(pairs: Dict[PairKey, Tuple[RunData, RunData]]) -> pd.DataFrame:
    """Per pair and scored cycle: is the selected batch the same records in the
    same order, and is the training set the same set of records."""
    rows = []
    for (strategy, seed), (a, b) in pairs.items():
        cycles = sorted(int(c) for c in set(a.rows["cycle"]) & set(b.rows["cycle"]) if c >= 1)
        for cycle in cycles:
            sel_a, sel_b = (_cycle_ids(r.run_dir, cycle, "selection.csv") for r in (a, b))
            ids_a, ids_b = (_cycle_ids(r.run_dir, cycle, "training_ids.csv") for r in (a, b))
            rows.append({"strategy": strategy, "seed": seed, "cycle": cycle,
                         "n_selected": len(sel_b), "n_train": len(ids_b),
                         "selection_identical": bool(sel_a) and sel_a == sel_b,
                         "training_set_identical": bool(ids_a) and set(ids_a) == set(ids_b)})
    return pd.DataFrame(rows)


def cycle_table(runs: Sequence[RunData], experiment: str, last_k: int = 3,
                divergence_ratio: float = 2.0) -> pd.DataFrame:
    """Per run and cycle: boundary full-V metrics, last-``last_k`` subsample
    medians (``traj_`` columns), boundary validation loss over the cycle's
    minimum and maximum over minimum, and the divergence flag."""
    full = boundary_table(runs)
    traj = trajectory_table(runs, last_k=last_k)
    traj = traj[["label", "cycle"] + [f"sub_{m}" for m in METRIC_KEYS if f"sub_{m}" in traj]]
    traj = traj.rename(columns=lambda c: c.replace("sub_", "traj_", 1))
    div = divergence_table(runs, ratio=divergence_ratio)
    div = div[["label", "cycle", "max_rise", "diverged"]] if not div.empty else \
        pd.DataFrame(columns=["label", "cycle", "max_rise", "diverged"])
    noise = []
    for run in runs:
        for cycle, group in run.rows.groupby("cycle"):
            val = group["val_loss"].to_numpy(dtype=float)
            marks = group[group["cycle_boundary"]]
            if marks.empty or not np.isfinite(val).any():
                continue
            best = float(np.nanmin(val))
            noise.append({"label": run.label, "cycle": int(cycle),
                          "val_min": best, "val_last_over_min": float(marks["val_loss"].iloc[-1]) / best,
                          "val_peak_over_min": float(np.nanmax(val)) / best})
    table = full.merge(traj, on=["label", "cycle"], how="left") \
        .merge(pd.DataFrame(noise), on=["label", "cycle"], how="left") \
        .merge(div, on=["label", "cycle"], how="left")
    seeds = {r.label: r.seed for r in runs}
    table.insert(0, "experiment", experiment)
    table.insert(3, "seed", table["label"].map(seeds))
    table["diverged"] = table["diverged"].fillna(False).astype(bool)
    return table


def paired_deltas(table_a: pd.DataFrame, table_b: pd.DataFrame, metrics: Sequence[str]
                  ) -> pd.DataFrame:
    """Per ``(strategy, seed, cycle)``: each metric in A, in B and B minus A."""
    keys = ["strategy", "seed", "cycle"]
    a = table_a.set_index(keys)[list(metrics)]
    b = table_b.set_index(keys)[list(metrics)]
    delta = (b - a).dropna(how="all")
    out = pd.concat([a.add_suffix("_a"), b.add_suffix("_b"), delta.add_prefix("delta_")],
                    axis=1, join="inner")
    return out.reset_index().sort_values(keys).reset_index(drop=True)


def summarize_deltas(paired: pd.DataFrame, metrics: Sequence[str],
                     lower_is_better: Sequence[str] = ()) -> pd.DataFrame:
    """Per cycle and metric: mean of A, of B, the mean, min and max of B minus A,
    and on how many pairs B was the better one."""
    rows = []
    for cycle, group in paired.groupby("cycle", sort=True):
        for metric in metrics:
            delta = group[f"delta_{metric}"].dropna()
            better = (delta < 0) if metric in lower_is_better else (delta > 0)
            rows.append({"cycle": int(cycle), "metric": metric,
                         "mean_a": float(group[f"{metric}_a"].mean()),
                         "mean_b": float(group[f"{metric}_b"].mean()),
                         "delta_mean": float(delta.mean()), "delta_min": float(delta.min()),
                         "delta_max": float(delta.max()), "n_b_better": int(better.sum()),
                         "n_pairs": int(len(delta))})
    return pd.DataFrame(rows)


def loss_curves(runs: Sequence[RunData], experiment: str) -> pd.DataFrame:
    """Long per-epoch table of the learning rate and both losses, one block per run."""
    parts = [r.rows[["cycle", "epoch_in_cycle", "cumulative_epoch", "n_train", "lr",
                     "train_loss", "val_loss"]].assign(experiment=experiment, label=r.label,
                                                       strategy=r.strategy, seed=r.seed)
             for r in runs]
    return pd.concat(parts, ignore_index=True)
