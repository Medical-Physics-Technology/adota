"""Reading several strategy runs back for the comparison.

Each strategy is an independent run, so the comparison is a separate step: read
the ``metrics.jsonl`` and ``manifest.json`` of every run, check that the runs
are comparable (same splits, same cycle-0 checkpoint, same evaluation subsample,
same training set size at every cycle), then build the tables the figures of
``scripts/al_compare.py`` draw and the epochs-to-quality table, which is derived
from the logged per-epoch metrics after the fact rather than decided in advance.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METRIC_KEYS = ("gpr_mean", "gpr_p05", "gpr_frac_below_95", "mape_pct_mean", "mape_pct_p95",
               "rde_pct_mean", "dr80_median_mm", "abs_dr80_median_mm", "abs_dr80_p95_mm",
               "dr80_defined_fraction", "n_gpr", "n_dr80")

DEFAULT_THRESHOLDS = {"sub_gpr_mean": (0.95, "ge"), "sub_gpr_p05": (0.90, "ge"),
                      "sub_gpr_frac_below_95": (0.10, "le"), "sub_mape_pct_mean": (5.0, "le"),
                      "sub_abs_dr80_p95_mm": (2.0, "le")}
"""Metric, level and direction for the epochs-to-quality table. Defaults only;
the config of ``al_compare.py`` sets the real ones after the numbers exist."""


@dataclass
class RunData:
    """One run, read back."""

    run_dir: Path
    label: str
    strategy: str
    manifest: Dict
    rows: pd.DataFrame

    @property
    def cycles(self) -> List[Dict]:
        return list(self.manifest.get("cycles", []))

    def boundaries(self) -> pd.DataFrame:
        """``(cycle, cumulative_epoch, n_train)`` at the end of every cycle."""
        marks = self.rows[self.rows["cycle_boundary"]]
        return marks[["cycle", "cumulative_epoch", "n_train"]].drop_duplicates("cycle") \
            .sort_values("cycle").reset_index(drop=True)


def _flatten(row: Dict) -> Dict:
    out = {"cycle": int(row["cycle"]), "epoch_in_cycle": int(row.get("epoch_in_cycle", -1)),
           "cumulative_epoch": int(row["cumulative_epoch"]), "n_train": int(row["n_train"]),
           "cycle_boundary": bool(row.get("cycle_boundary", False)),
           "strategy": row.get("strategy"), "lr": row.get("lr"),
           "train_loss": (row.get("train") or {}).get("loss_combined_mean"),
           "val_loss": (row.get("val_loss") or {}).get("loss_combined_mean"),
           "val_loss_mse": (row.get("val_loss") or {}).get("loss_mse_mean"),
           "val_loss_ps": (row.get("val_loss") or {}).get("loss_ps_mean"),
           "epoch_time_s": row.get("epoch_time_s"), "gamma_label": row.get("gamma_label")}
    for prefix, key in (("sub", "metrics_subsample"), ("full", "metrics_full")):
        block = row.get(key) or {}
        for metric in METRIC_KEYS:
            out[f"{prefix}_{metric}"] = block.get(metric, np.nan)
    return out


def read_run(run_dir: Path, label: Optional[str] = None) -> RunData:
    """Read one run; later duplicates of an epoch (a resumed cycle) win."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [_flatten(json.loads(line))
            for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows).drop_duplicates(["cycle", "cumulative_epoch"], keep="last")
    frame = frame.sort_values("cumulative_epoch").reset_index(drop=True)
    strategy = str(manifest.get("strategy", "unknown"))
    return RunData(run_dir=run_dir, label=label or strategy, strategy=strategy,
                   manifest=manifest, rows=frame)


def check_consistency(runs: Sequence[RunData]) -> Dict[str, object]:
    """Raise unless every run shares the splits, the cycle-0 checkpoint, the
    evaluation subsample and the training set size at every cycle index."""
    if not runs:
        raise ValueError("no runs to compare")
    problems: List[str] = []
    keys = {
        "splits_fingerprint": [r.manifest["inputs"]["splits_fingerprint"] for r in runs],
        "cycle0_checkpoint_sha256": [r.manifest.get("cycle0_checkpoint_sha256") for r in runs],
        "eval_subsample_sha256": [r.manifest["inputs"]["eval_subsample_sha256"] for r in runs],
        "gamma_label": [r.rows["gamma_label"].dropna().iloc[-1] for r in runs],
    }
    for name, values in keys.items():
        if len(set(values)) != 1:
            problems.append(f"{name} differs across runs: {values}")
    sizes = {r.label: {int(c["cycle"]): int(c["n_train"]) for c in r.cycles} for r in runs}
    cycles = sorted(set.union(*(set(s) for s in sizes.values())))
    for cycle in cycles:
        present = {label: s.get(cycle) for label, s in sizes.items() if cycle in s}
        if len(set(present.values())) != 1:
            problems.append(f"training set size at cycle {cycle} differs: {present}")
    if problems:
        raise AssertionError("runs are not comparable:\n  " + "\n  ".join(problems))
    return {"n_runs": len(runs), "labels": [r.label for r in runs], "cycles": cycles,
            "n_train_per_cycle": {c: sizes[runs[0].label][c] for c in cycles
                                  if c in sizes[runs[0].label]},
            **{k: v[0] for k, v in keys.items()}}


def boundary_table(runs: Sequence[RunData]) -> pd.DataFrame:
    """Full-validation-set metrics at every cycle boundary, per run."""
    parts = []
    for run in runs:
        marks = run.rows[run.rows["cycle_boundary"]].copy()
        marks.insert(0, "label", run.label)
        cols = ["label", "strategy", "cycle", "cumulative_epoch", "n_train", "val_loss"] + \
               [f"full_{m}" for m in METRIC_KEYS]
        parts.append(marks[cols])
    return pd.concat(parts, ignore_index=True)


def quality_curves(runs: Sequence[RunData]) -> Dict[str, pd.DataFrame]:
    """Per run, the rows carrying subsample metrics (the cadence rows and the
    boundaries), which is what the quality figures draw."""
    out = {}
    for run in runs:
        frame = run.rows[run.rows["sub_gpr_mean"].notna()].copy()
        out[run.label] = frame.reset_index(drop=True)
    return out


def epochs_to_quality(runs: Sequence[RunData],
                      thresholds: Dict[str, Tuple[float, str]] = DEFAULT_THRESHOLDS
                      ) -> pd.DataFrame:
    """The first cumulative epoch at which each run's subsample metric crosses
    each threshold, and the training set size at that point; NaN when it never
    does. Derived from the log after the fact."""
    rows = []
    for run in runs:
        frame = run.rows[run.rows["sub_gpr_mean"].notna()]
        for metric, (level, direction) in thresholds.items():
            if metric not in frame:
                continue
            values = frame[metric].to_numpy(dtype=float)
            hit = values >= level if direction == "ge" else values <= level
            first = int(np.argmax(hit)) if hit.any() else None
            rows.append({"label": run.label, "strategy": run.strategy, "metric": metric,
                         "threshold": level, "direction": direction,
                         "reached": bool(hit.any()),
                         "cumulative_epoch": (int(frame["cumulative_epoch"].iloc[first])
                                              if first is not None else np.nan),
                         "n_train": (int(frame["n_train"].iloc[first])
                                     if first is not None else np.nan),
                         "cycle": (int(frame["cycle"].iloc[first]) if first is not None
                                   else np.nan)})
    return pd.DataFrame(rows)


def fingerprints(runs: Sequence[RunData]) -> Dict[str, List[Dict]]:
    """The per-cycle selection fingerprints of every run, cycle 0 excluded."""
    return {run.label: [dict(c["selection_fingerprint"], cycle=int(c["cycle"]))
                        for c in run.cycles if c.get("selection_fingerprint")]
            for run in runs}


def fingerprint_table(prints: Dict[str, List[Dict]], key: str) -> pd.DataFrame:
    """Long table: label, cycle, category, selected share, pool share."""
    rows = []
    for label, cycles in prints.items():
        for entry in cycles:
            selected = entry["selected"].get(key, {})
            pool = entry["pool"].get(key, {})
            for category in sorted(set(selected) | set(pool), key=str):
                rows.append({"label": label, "cycle": entry["cycle"], "category": category,
                             "selected_share": selected.get(category, 0.0),
                             "pool_share": pool.get(category, 0.0)})
    return pd.DataFrame(rows)
