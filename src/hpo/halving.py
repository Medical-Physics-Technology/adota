"""Synchronous successive halving: the budget schedule and the promotions.

Every trial of a mixer trains to the first rung's epoch budget; the best
``1 / eta`` of them (at least one) continue, from where they stopped, to the
next rung, and so on. A rung is promoted only when every trial in it has
finished, so the outcome depends on the results alone, never on which GPU
finished first. This is the synchronous form of ASHA: with two GPUs it costs
little idle time and keeps the search reproducible.

The objective is smaller-is-better (the validation loss); ties go to the
lower trial id, so the ranking is total.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


def rung_budgets(min_epochs: int, eta: int, n_rungs: int) -> List[int]:
    """Cumulative epochs at each rung: ``min_epochs * eta**r``."""
    if min_epochs < 1 or eta < 2 or n_rungs < 1:
        raise ValueError(f"need min_epochs >= 1, eta >= 2, n_rungs >= 1; got {min_epochs}, {eta}, {n_rungs}")
    return [int(min_epochs * eta ** r) for r in range(n_rungs)]


def n_promoted(n_in_rung: int, eta: int) -> int:
    return max(1, n_in_rung // eta)


def rank(objectives: Mapping[str, float]) -> List[str]:
    """Trial ids from best to worst; a non-finite objective ranks last."""
    def key(item):
        trial, value = item
        finite = value == value and value not in (float("inf"), float("-inf"))
        return (not finite, value if finite else 0.0, trial)
    return [trial for trial, _ in sorted(objectives.items(), key=key)]


def promote(objectives: Mapping[str, float], eta: int) -> List[str]:
    """The trials of a finished rung that continue to the next one."""
    ordered = rank(objectives)
    survivors = [t for t in ordered[:n_promoted(len(ordered), eta)]]
    return [t for t in survivors if objectives[t] == objectives[t]]  # never promote a NaN


def schedule(trial_ids: Sequence[str], eta: int, n_rungs: int) -> Dict[int, int]:
    """How many trials each rung holds when all of ``trial_ids`` start at rung 0."""
    counts, n = {}, len(trial_ids)
    for r in range(n_rungs):
        counts[r] = n
        n = n_promoted(n, eta)
    return counts


def total_epochs(n_trials: int, budgets: Sequence[int], eta: int) -> int:
    """Epochs a mixer costs: each rung's trials train from the previous budget."""
    total, n, previous = 0, n_trials, 0
    for budget in budgets:
        total += n * (budget - previous)
        previous, n = budget, n_promoted(n, eta)
    return total
