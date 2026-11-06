"""Paired Monte Carlo realisations of new beamlets: the noise floor of EXP-0014.

The pool was simulated once, at 1e6 primaries. Every model's error is measured
against that one realisation, so its noise is part of every error, and it is
shared: two models scored against the same noisy dose agree partly because of
the noise. The noise set measures it on new beamlets that are simulated twice,
at the pool's 1e6 primaries and at 1e7 (about 3.2 times less noise per voxel):

- :func:`select_patients` draws the patients, an equal number per anatomy from
  one split of the pool registry (the test split: out of sample for every
  engine, and never used by the active learning loop);
- the beamlets come from the pool's own planner
  (:func:`src.active_learning.beamlet_pool.planning.plan_patient`) with another
  seed, so they follow the pool's population (energies, steering, overshoot gate)
  without repeating its beamlets;
- :func:`patient_batches` puts one patient in each batch, so a field's beamlets
  run in one beamlet-mode call and two processes get the same anatomy mix;
- :func:`pair_config_differences` checks that the two realisations' configs
  differ only in the primaries and the storage locations.

Both realisations keep ``robustness.rng_seed: 0``. In beamlet mode MCsquare runs
each beamlet on one thread and seeds it, at the beamlet's start, with
``seed + t`` for thread ``t`` (``compute_simulation_beamlet.c``), so a fixed seed
would give every beamlet a thread simulates the same random stream; seed 0 takes
``time(NULL) + t`` instead, as the pool did. The two runs of a beamlet start
hours apart, far more than the 24 threads' offsets, so their streams differ.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

PAIR_KEYS = {("robustness", "num_primaries"), ("batches", "store_root"), ("batches", "work_root")}
"""The only config keys the two realisations may differ in."""


def select_patients(registry: pd.DataFrame, per_anatomy: Mapping[str, int], split: str = "test",
                    seed: int = 0) -> pd.DataFrame:
    """``per_anatomy[a]`` patients of anatomy ``a`` from ``split`` of the registry,
    drawn without replacement, in draw order within each anatomy."""
    pool = registry[registry["split"] == split]
    rng = np.random.default_rng(seed)
    picked = []
    for anatomy in sorted(per_anatomy):
        ids = np.sort(pool.loc[pool["anatomy"] == anatomy, "patient_id"].astype(str).unique())
        n = int(per_anatomy[anatomy])
        if n > len(ids):
            raise ValueError(f"{n} {anatomy} patients asked, {len(ids)} in split {split!r}")
        for rank, pid in enumerate(rng.choice(ids, size=n, replace=False)):
            picked.append({"patient_id": str(pid), "anatomy": anatomy, "rank": rank})
    return pd.DataFrame(picked)


def patient_batches(plan: pd.DataFrame, selection: pd.DataFrame) -> pd.Series:
    """Batch per row: one patient per batch, numbered anatomy by anatomy in draw
    order, so each anatomy's patients alternate between even and odd batches and
    two processes (``b % 2``) get the same mix when it is drawn in even numbers."""
    order = selection.sort_values(["anatomy", "rank"], kind="stable")["patient_id"].tolist()
    return plan["patient_id"].astype(str).map({pid: i for i, pid in enumerate(order)})


def _flatten(cfg: Mapping[str, Any], prefix: tuple = ()) -> Dict[tuple, Any]:
    out: Dict[tuple, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, Mapping):
            out.update(_flatten(value, prefix + (key,)))
        else:
            out[prefix + (key,)] = value
    return out


def pair_config_differences(a: Mapping[str, Any], b: Mapping[str, Any]) -> List[str]:
    """Problems with a pair of realisation configs: a key outside
    :data:`PAIR_KEYS` that differs, or a key of :data:`PAIR_KEYS` that does not."""
    fa, fb = _flatten(a), _flatten(b)
    problems = [f"{'.'.join(k)} differs: {fa.get(k)!r} vs {fb.get(k)!r}"
                for k in sorted(set(fa) | set(fb)) if k not in PAIR_KEYS and fa.get(k) != fb.get(k)]
    problems += [f"{'.'.join(k)} must differ" for k in sorted(PAIR_KEYS) if fa.get(k) == fb.get(k)]
    return problems
