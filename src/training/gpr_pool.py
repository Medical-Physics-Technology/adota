"""Selection and persistence of the fixed gamma (GPR) evaluation pool.

Gamma pass rate is expensive, so it is computed on a fixed subset of the
validation set rather than all of it. For fair longitudinal comparison the
subset must be (a) frozen for the lifetime of a run, (b) pinned by record id
rather than by position, (c) persisted so it is auditable and replayable, and
(d) inheritable across a resume / warm-start.

The pool is *nested*: a small ``comparable`` set (apples-to-apples with prior
runs that used that exact set) sits inside a larger ``pool`` (a lower-variance
estimate). Both are reported every GPR epoch.

The selection is deliberately model-independent (record ids only, no model
error), so the evaluation set never shifts with the model being measured.

Persistence schema (``gpr_samples.json``)::

    {
      "schema": 1,
      "selection": {"method": "random", "seed": 1234, "source_run": null},
      "comparable_ids": [...],     # the nested, prior-run-comparable subset
      "pool_ids": [...],           # superset of comparable_ids
      "scores": {}                 # reserved: per-id heterogeneity score / stratum
    }

``method`` is ``"random"`` today. A future acquisition function (stratified
over input-grid heterogeneity) changes only how ``pool_ids`` is chosen and
fills ``scores``; everything downstream (persist / replay / nest / inherit /
two-number reporting) is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.training.validation import pick_gpr_subset

SCHEMA_VERSION = 1


def build_gpr_pool(
    val_record_ids: Sequence[str],
    *,
    pool_size: int,
    comparable_size: int,
    seed: int,
    source_run: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a nested ``comparable``-in-``pool`` gamma set, by record id.

    The ``comparable`` set reproduces the legacy positional draw
    (:func:`pick_gpr_subset` seeded with ``seed``), so a run that shares the
    same validation split and ``seed`` selects the *same* records a prior run
    did. The pool extends it with additional distinct records drawn from a
    separate deterministic stream.

    Args:
        val_record_ids: Validation record ids, in iteration order.
        pool_size: Target size of the (stable) pool.
        comparable_size: Target size of the nested comparable subset.
        seed: Seed for the deterministic draws.
        source_run: Optional provenance string recorded in the artifact.

    Returns:
        The pool dict following the persistence schema above.
    """
    ids = list(val_record_ids)
    n = len(ids)
    pool_size = min(pool_size, n)
    comparable_size = min(comparable_size, pool_size)

    comp_pos = pick_gpr_subset(n, comparable_size, np.random.RandomState(seed))
    comparable_ids = sorted(ids[i] for i in comp_pos)

    comp_set = set(comparable_ids)
    remaining = [r for r in ids if r not in comp_set]
    n_extra = pool_size - len(comparable_ids)
    extra_ids: List[str] = []
    if n_extra > 0 and remaining:
        rng = np.random.RandomState(seed + 1)
        take = min(n_extra, len(remaining))
        sel = rng.choice(len(remaining), size=take, replace=False)
        extra_ids = [remaining[i] for i in sel.tolist()]
    pool_ids = sorted(comp_set | set(extra_ids))

    return {
        "schema": SCHEMA_VERSION,
        "selection": {"method": "random", "seed": seed, "source_run": source_run},
        "comparable_ids": comparable_ids,
        "pool_ids": pool_ids,
        "scores": {},
    }


def save_gpr_pool(pool: Dict[str, Any], path: Path) -> None:
    """Persist a gamma pool artifact as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pool, f, indent=2)


def load_gpr_pool(path: Path) -> Dict[str, Any]:
    """Load a gamma pool artifact, validating the required keys."""
    with open(path, "r") as f:
        pool = json.load(f)
    for key in ("comparable_ids", "pool_ids"):
        if key not in pool:
            raise ValueError(f"gamma pool file {path} is missing '{key}'")
    pool.setdefault("schema", SCHEMA_VERSION)
    pool.setdefault("selection", {"method": "unknown"})
    pool.setdefault("scores", {})
    return pool


def pool_to_indices(
    pool: Dict[str, Any], val_record_ids: Sequence[str]
) -> Tuple[List[int], List[int]]:
    """Map pool / comparable record ids to positions in ``val_record_ids``.

    Records that are not present in the current validation ordering are
    skipped (robust to a changed split / exclusion file).

    Returns:
        ``(pool_indices, comparable_indices)`` as sorted, de-duplicated
        positional index lists into the validation iteration order.
    """
    id_to_pos = {rid: i for i, rid in enumerate(val_record_ids)}
    pool_indices = sorted({id_to_pos[r] for r in pool["pool_ids"] if r in id_to_pos})
    comparable_indices = sorted(
        {id_to_pos[r] for r in pool["comparable_ids"] if r in id_to_pos}
    )
    return pool_indices, comparable_indices
