"""The strategy registry and the three strategies of the retrospective benchmark.

A strategy is a function from a scored pool table, a batch size and a random
generator to a list of selected identifiers. Strategies are registered by name
and picked from the YAML config, so a fourth strategy is one decorated function
here and one line of config; the loop does not change.

The three strategies:

- ``random``: uniform over the remaining pool. The control, not a straw man; it
  ignores the scores entirely.
- ``score_topk``: the ``N`` highest-scoring records. Pure exploitation.
- ``score_topk_mixed``: a ``top_fraction`` share of the batch is the top-scoring
  prefix of ``score_topk``, the rest is drawn uniformly from everything else
  (scored or not). Exploitation plus coverage.

``stratified_score`` (equal counts per score decile, uniform inside a decile)
was dropped after EXP-0009 showed it is uniform sampling over the score
distribution, i.e. a second ``random`` run rather than a coverage strategy.

Records without a score (the reference study could not score records above
250 MeV or without flux) remain in the pool: ``random`` and the remainder half
of ``score_topk_mixed`` can draw them, ``score_topk`` cannot. The selection
fingerprint counts them.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A strategy takes the pool, the batch size and a random generator, plus any
# strategy-specific keyword parameters forwarded by ``select``.
Strategy = Callable[..., List[str]]
STRATEGIES: Dict[str, Strategy] = {}

N_DECILES = 10
ENERGY_EDGES_MEV = tuple(float(e) for e in range(70, 280, 20))


def register_strategy(name: str) -> Callable[[Strategy], Strategy]:
    """Register ``fn`` under ``name``; refuse to overwrite silently."""

    def wrap(fn: Strategy) -> Strategy:
        if name in STRATEGIES:
            raise ValueError(f"strategy {name!r} is already registered")
        STRATEGIES[name] = fn
        return fn

    return wrap


def available_strategies() -> List[str]:
    return sorted(STRATEGIES)


def score_deciles(scores: Sequence[float], n_deciles: int = N_DECILES) -> np.ndarray:
    """Decile index per record over the finite scores (0 easiest, 9 hardest),
    ``-1`` for a missing score. Rank-based, so tied scores never collapse a
    decile the way ``pd.qcut`` would."""
    values = np.asarray(scores, dtype=float)
    out = np.full(values.shape, -1, dtype=int)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return out
    order = np.argsort(values[finite], kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    out[finite] = np.minimum((ranks * n_deciles) // n, n_deciles - 1)
    return out


def _check(pool: pd.DataFrame, n: int) -> None:
    if n <= 0:
        raise ValueError(f"batch size must be positive, got {n}")
    if len(pool) < n:
        raise ValueError(f"the pool holds {len(pool)} records, fewer than the batch of {n}")
    if not pool["sample_id"].is_unique:
        raise ValueError("the pool has duplicate sample ids")


@register_strategy("random")
def random_strategy(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> List[str]:
    """Uniform without replacement over the whole remaining pool."""
    _check(pool, n)
    picked = rng.choice(len(pool), size=n, replace=False)
    return pool["sample_id"].to_numpy()[np.sort(picked)].tolist()


def _top_scored(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
    """The pool row indices of the ``n`` highest finite scores. Ties are broken
    by a random permutation so the draw stays a function of the seed rather
    than of the file order. Shared by ``score_topk`` and ``score_topk_mixed``
    so their ranking cannot drift apart."""
    scores = pool["score"].to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if int(finite.sum()) < n:
        raise ValueError(f"only {int(finite.sum())} scored records for a batch of {n}")
    shuffled = rng.permutation(len(pool))
    ranked = shuffled[np.argsort(-scores[shuffled], kind="stable")]
    top = [i for i in ranked if finite[i]][:n]
    return np.asarray(top, dtype=int)


@register_strategy("score_topk")
def score_topk_strategy(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> List[str]:
    """The ``n`` highest scores. Ties are broken by a random permutation so the
    draw stays a function of the seed rather than of the file order."""
    _check(pool, n)
    top = _top_scored(pool, n, rng)
    return pool["sample_id"].to_numpy()[sorted(top)].tolist()


@register_strategy("score_topk_mixed")
def score_topk_mixed_strategy(pool: pd.DataFrame, n: int, rng: np.random.Generator,
                               top_fraction: float = 0.5) -> List[str]:
    """``top_fraction`` of the batch is the top-scoring prefix of ``score_topk``,
    the rest is drawn uniformly without replacement from everything else in the
    pool (scored or not), exactly as ``random`` draws. Exploitation plus
    coverage, and a direct test of whether pure top-k overshoots on cycle one."""
    _check(pool, n)
    if not (0.0 <= top_fraction <= 1.0):
        raise ValueError(f"top_fraction must be in [0, 1], got {top_fraction}")
    n_top = int(round(top_fraction * n))
    top = _top_scored(pool, n_top, rng) if n_top else np.asarray([], dtype=int)
    remaining = np.setdiff1d(np.arange(len(pool)), top, assume_unique=True)
    n_rest = n - n_top
    rest = remaining[rng.choice(len(remaining), size=n_rest, replace=False)] if n_rest else np.asarray([], dtype=int)
    chosen = np.concatenate([top, rest]).astype(int)
    return pool["sample_id"].to_numpy()[sorted(chosen)].tolist()


def select(pool: pd.DataFrame, n: int, strategy: str, rng: np.random.Generator, **params) -> List[str]:
    """Run the named strategy and check its answer: exactly ``n`` distinct ids,
    all from the pool. Extra keyword ``params`` are forwarded to the strategy;
    ``random`` and ``score_topk`` take none, so an unknown one simply raises
    ``TypeError`` from the call below."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; registered: {available_strategies()}")
    ids = list(STRATEGIES[strategy](pool, n, rng, **params))
    if len(ids) != n or len(set(ids)) != n:
        raise AssertionError(f"strategy {strategy} returned {len(ids)} ids "
                             f"({len(set(ids))} distinct) for a batch of {n}")
    missing = set(ids) - set(pool["sample_id"].astype(str))
    if missing:
        raise AssertionError(f"strategy {strategy} selected {len(missing)} ids not in the pool")
    logger.info("%s: selected %d of %d pool records", strategy, n, len(pool))
    return ids


# ── The selection fingerprint ───────────────────────────────────────────────


def _shares(values: pd.Series) -> Dict[str, float]:
    counts = values.value_counts(dropna=False)
    total = float(counts.sum()) or 1.0
    return {str(k): float(v) / total for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}


def _energy_bin(energy: pd.Series) -> pd.Series:
    edges = list(ENERGY_EDGES_MEV)
    labels = [f"{int(lo)}-{int(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]
    return pd.cut(energy, bins=edges, labels=labels, right=False, include_lowest=True).astype(str)


def selection_fingerprint(pool: pd.DataFrame, selected_ids: Sequence[str]) -> Dict:
    """The distribution of the selected batch over energy, anatomy, patient and
    score decile, against the same distribution over the pool it was drawn from.

    This is the guard against the distributional bias that made active learning
    lose to random in the quantum liquid water study; every cycle records it.
    """
    frame = pool.copy()
    has_scores = "score" in frame and bool(np.isfinite(frame["score"]).any())
    deciles = score_deciles(frame["score"].to_numpy(dtype=float)) if has_scores else None
    frame["score_decile"] = (np.where(deciles < 0, "unscored", deciles.astype(str))
                             if deciles is not None else "unscored")
    frame["energy_bin"] = _energy_bin(frame["energy_mev"])
    picked = frame[frame["sample_id"].isin(set(selected_ids))]
    out = {"n_selected": int(len(picked)), "n_pool": int(len(frame)),
           "n_selected_unscored": int((picked["score_decile"] == "unscored").sum()),
           "selected": {}, "pool": {}}
    for column, key in (("energy_bin", "energy"), ("anatomy", "anatomy"),
                        ("patient", "patient"), ("score_decile", "score_decile")):
        if column not in frame or (key == "score_decile" and not has_scores):
            continue
        out["selected"][key] = _shares(picked[column])
        out["pool"][key] = _shares(frame[column])
    if "score" in frame:
        scores = picked["score"].to_numpy(dtype=float)
        finite = scores[np.isfinite(scores)]
        out["selected_score_mean"] = float(finite.mean()) if finite.size else float("nan")
        pool_scores = frame["score"].to_numpy(dtype=float)
        pool_finite = pool_scores[np.isfinite(pool_scores)]
        out["pool_score_mean"] = float(pool_finite.mean()) if pool_finite.size else float("nan")
    out["n_selected_patients"] = int(picked["patient"].nunique()) if "patient" in picked else 0
    return out
