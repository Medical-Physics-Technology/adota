"""Sampling strategies over a scored candidate table.

Every strategy sees the same valid candidates and spends the same Monte Carlo
budget, so the arms differ only in what they choose. ``random`` is the control, and
it is not a straw man: it is the comparison most published active-learning results
fail to beat.

Difficulty is used **conditional on energy**. The score is dominated by path length,
so an unconstrained draw picks only the deepest beamlets and the batch collapses onto
one corner of the input space; splitting the budget equally over the energy layers
present, and capping how much of a layer any one patient may supply, keeps the batch
spread over the anatomy the model actually has to predict. ``score_topk`` is kept as
the ablation that shows that collapse rather than preventing it.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STRATEGIES = ("random", "score", "score_topk", "stratified_score")


def _draw(frame: pd.DataFrame, k: int, rng: np.random.Generator,
          weights: Optional[np.ndarray] = None,
          caps: Optional[Dict[str, int]] = None) -> List[int]:
    """Draw ``k`` row positions without replacement, honouring per-patient caps.

    ``weights`` of ``None`` means uniform. Drawing is sequential because a cap can
    remove a whole patient mid-draw; ``k`` is in the thousands, so this is cheap.
    """
    n = len(frame)
    k = min(k, n)
    if k <= 0:
        return []
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).copy()
    w = np.clip(w, 0.0, None)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(n, dtype=float)
    patients = frame["patient_id"].to_numpy()
    used: Dict[str, int] = {}
    picked: List[int] = []
    for _ in range(k):
        total = w.sum()
        if total <= 0:
            break
        idx = int(rng.choice(n, p=w / total))
        picked.append(idx)
        w[idx] = 0.0
        if caps is not None:
            pid = patients[idx]
            used[pid] = used.get(pid, 0) + 1
            if used[pid] >= caps.get(pid, 10 ** 9):
                w[patients == pid] = 0.0
    return picked


def _per_energy_budget(frame: pd.DataFrame, n: int) -> Dict[float, int]:
    """Split ``n`` as evenly as the layers allow, never asking a layer for more
    candidates than it has; the shortfall spills to the layers that can take it."""
    layers = sorted(frame["energy_mev"].unique())
    available = {e: int((frame["energy_mev"] == e).sum()) for e in layers}
    budget = {e: 0 for e in layers}
    remaining = n
    open_layers = [e for e in layers if available[e] > 0]
    while remaining > 0 and open_layers:
        share = max(1, remaining // len(open_layers))
        for e in list(open_layers):
            take = min(share, available[e] - budget[e], remaining)
            budget[e] += take
            remaining -= take
            if budget[e] >= available[e]:
                open_layers.remove(e)
            if remaining <= 0:
                break
    return budget


def select(
    table: pd.DataFrame,
    strategy: str,
    n: int,
    *,
    rng: np.random.Generator,
    score_column: str = "score_full",
    alpha: float = 2.0,
    max_per_patient_frac: float = 0.20,
    per_energy_quota: bool = True,
) -> pd.DataFrame:
    """Choose ``n`` candidates to simulate.

    Args:
        table: The scored candidate table; invalid rows are dropped here, so a
            caller never has to remember to.
        strategy: One of :data:`STRATEGIES`.
        n: The batch size, in beamlets.
        rng: The run's generator; the only source of randomness.
        score_column: Which score drives ``score`` / ``score_topk`` /
            ``stratified_score``. The deployed default is the 30-metric score.
        alpha: Sampling sharpness for ``score``: probability proportional to
            ``score ** alpha``. 1 is proportional, larger is greedier.
        max_per_patient_frac: Cap on the share of any energy layer's budget one
            patient may supply.
        per_energy_quota: Split the budget over the energy layers. Off, the draw is
            global and the deep layers dominate.

    Returns:
        The selected rows, with a ``selection_rank`` column in draw order.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    valid = table[table["valid"].astype(bool)].reset_index(drop=True)
    if strategy != "random" and score_column not in valid.columns:
        raise ValueError(f"score column {score_column!r} missing from the candidate table")
    if len(valid) < n:
        raise ValueError(
            f"only {len(valid)} valid candidates for a batch of {n}; enlarge the "
            "candidate pool (n_per_field or n_gantry_per_ct) rather than the budget")

    if strategy == "score_topk":
        chosen = valid.nlargest(n, score_column).reset_index(drop=True)
        chosen["selection_rank"] = np.arange(len(chosen))
        return chosen

    groups: Sequence[pd.DataFrame]
    budgets: Sequence[int]
    if strategy == "stratified_score":
        deciles = pd.qcut(valid[score_column], 10, labels=False, duplicates="drop")
        groups = [valid[deciles == d] for d in sorted(pd.unique(deciles))]
        base = n // len(groups)
        budgets = [base + (1 if i < n - base * len(groups) else 0) for i in range(len(groups))]
    elif per_energy_quota:
        budget = _per_energy_budget(valid, n)
        layers = sorted(budget)
        groups = [valid[valid["energy_mev"] == e] for e in layers]
        budgets = [budget[e] for e in layers]
    else:
        groups, budgets = [valid], [n]

    picked: List[pd.DataFrame] = []
    shortfall = 0
    for group, want in zip(groups, budgets):
        want += shortfall
        caps = None
        if max_per_patient_frac < 1.0:
            cap = max(1, int(np.ceil(want * max_per_patient_frac)))
            caps = {pid: cap for pid in group["patient_id"].unique()}
        weights = None
        if strategy in ("score", "stratified_score"):
            scores = group[score_column].to_numpy(dtype=float)
            span = np.nanmax(scores) - np.nanmin(scores)
            unit = (scores - np.nanmin(scores)) / span if span > 0 else np.ones_like(scores)
            weights = np.power(np.clip(unit, 1e-6, None), alpha)
        rows = _draw(group.reset_index(drop=True), want, rng, weights, caps)
        shortfall = want - len(rows)
        picked.append(group.reset_index(drop=True).iloc[rows])

    chosen = pd.concat(picked, ignore_index=True)
    if shortfall > 0:  # a capped layer ran dry; top up from whatever is left
        rest = valid[~valid["candidate_id"].isin(set(chosen["candidate_id"]))]
        extra = _draw(rest.reset_index(drop=True), shortfall, rng)
        chosen = pd.concat([chosen, rest.reset_index(drop=True).iloc[extra]], ignore_index=True)
    chosen["selection_rank"] = np.arange(len(chosen))
    logger.info("%s: selected %d of %d valid candidates (%d patients, %d energy layers)",
                strategy, len(chosen), len(valid), chosen["patient_id"].nunique(),
                chosen["energy_mev"].nunique())
    return chosen


def selection_fingerprint(chosen: pd.DataFrame, score_column: str = "score_full") -> dict:
    """What a strategy actually bought: the summary every cycle manifest records."""
    return {
        "n": int(len(chosen)),
        "n_patients": int(chosen["patient_id"].nunique()),
        "per_anatomy": {str(k): int(v) for k, v in chosen["anatomy"].value_counts().items()},
        "per_energy": {str(k): int(v) for k, v in chosen["energy_mev"].value_counts().items()},
        "score_mean": float(chosen[score_column].mean()) if score_column in chosen else None,
        "score_p90": (float(chosen[score_column].quantile(0.9))
                      if score_column in chosen else None),
    }
