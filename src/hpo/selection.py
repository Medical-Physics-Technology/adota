"""Reading an HPO run's results and choosing the EXP-0014 members from them.

- **Best per mixer.** The trial that reached the highest rung with the lowest
  validation loss there; it becomes C0 (transformer), X0 (mlstm) or U0 (none).
- **A distant second (C1).** Among the transformer trials promoted at least
  once, excluding C0, the one farthest from C0 in the unit cube of the search
  space (ties to the lower loss). Promotion is the quality floor here; the
  quality gate proper is applied to the trained member later.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.hpo.space import unit_distance


def results_table(trials: Sequence[Mapping[str, Any]], segments: Mapping[str, Mapping[int, Mapping]],
                  budgets: Sequence[int]) -> pd.DataFrame:
    """One row per trial: its rung reached and objective at every rung."""
    rows = []
    for t in trials:
        row = {"trial": t["trial_id"], "mixer": t["mixer"], "n_params": t["n_params"],
               "fixed": bool(t.get("extra", {}).get("fixed", False)), **{f"p_{k}": v for k, v in t["params"].items()}}
        reached = -1
        for r, budget in enumerate(budgets):
            seg = segments.get(t["trial_id"], {}).get(budget)
            row[f"objective_r{r}"] = None if seg is None else seg.get("objective")
            row[f"failed_r{r}"] = None if seg is None else seg.get("failed")
            if seg is not None and seg.get("failed") is None:
                reached = r
        row["rung_reached"] = reached
        rows.append(row)
    return pd.DataFrame(rows)


def best_per_mixer(table: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for mixer, part in table.groupby("mixer"):
        top = int(part["rung_reached"].max())
        if top < 0:
            continue
        at_top = part[part["rung_reached"] == top].copy()
        at_top["obj"] = at_top[f"objective_r{top}"].astype(float)
        out[mixer] = at_top.sort_values(["obj", "trial"]).iloc[0]["trial"]
    return out


def distant_second(table: pd.DataFrame, unit_points: Mapping[str, Sequence[float]], best: str,
                   mixer: str = "transformer") -> Optional[Dict[str, Any]]:
    """The promoted trial of ``mixer`` farthest from ``best``; ``None`` if none."""
    pool = table[(table["mixer"] == mixer) & (table["rung_reached"] >= 1) & (table["trial"] != best)]
    if pool.empty:
        return None
    rows = []
    for row in pool.itertuples():
        rows.append({"trial": row.trial,
                     "distance": unit_distance(unit_points[row.trial], unit_points[best]),
                     "objective_r1": float(getattr(row, "objective_r1"))})
    ranked = sorted(rows, key=lambda r: (-r["distance"], r["objective_r1"], r["trial"]))
    return {**ranked[0], "candidates": ranked}


def member_roles(table: pd.DataFrame, unit_points: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    """``{"C0": ..., "C1": ..., "X0": ..., "U0": ...}`` with the trial ids chosen."""
    best = best_per_mixer(table)
    roles: Dict[str, Any] = {}
    names = {"transformer": "C0", "mlstm": "X0", "none": "U0"}
    for mixer, role in names.items():
        if mixer in best:
            roles[role] = best[mixer]
    if "C0" in roles:
        second = distant_second(table, unit_points, roles["C0"])
        if second is not None:
            roles["C1"] = second["trial"]
            roles["C1_candidates"] = second["candidates"]
    return roles


def finalists(table: pd.DataFrame, roles: Mapping[str, Any]) -> List[str]:
    """Trials that get the full metric set: every top-rung trial and C1."""
    out = []
    for _, part in table.groupby("mixer"):
        top = int(part["rung_reached"].max())
        if top >= 0:
            out += part.loc[part["rung_reached"] == top, "trial"].tolist()
    if roles.get("C1") and roles["C1"] not in out:
        out.append(roles["C1"])
    return sorted(out)
