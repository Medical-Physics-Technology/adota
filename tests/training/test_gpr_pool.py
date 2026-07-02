"""Unit tests for :mod:`src.training.gpr_pool` (nested gamma pool)."""

from __future__ import annotations

from pathlib import Path

from src.training.gpr_pool import (
    build_gpr_pool,
    load_gpr_pool,
    pool_to_indices,
    save_gpr_pool,
)
from src.training.validation import pick_gpr_subset

import numpy as np

VAL_IDS = [f"rec_{i:04d}" for i in range(300)]


def test_pool_is_nested_and_sized() -> None:
    pool = build_gpr_pool(VAL_IDS, pool_size=50, comparable_size=20, seed=1234)
    assert len(pool["comparable_ids"]) == 20
    assert len(pool["pool_ids"]) == 50
    # comparable nests inside pool.
    assert set(pool["comparable_ids"]).issubset(set(pool["pool_ids"]))
    # ids are real validation records.
    assert set(pool["pool_ids"]).issubset(set(VAL_IDS))


def test_pool_is_deterministic() -> None:
    a = build_gpr_pool(VAL_IDS, pool_size=50, comparable_size=20, seed=1234)
    b = build_gpr_pool(VAL_IDS, pool_size=50, comparable_size=20, seed=1234)
    assert a == b
    c = build_gpr_pool(VAL_IDS, pool_size=50, comparable_size=20, seed=99)
    assert c["pool_ids"] != a["pool_ids"]


def test_comparable_matches_legacy_positional_draw() -> None:
    """The comparable set must reproduce the old pick_gpr_subset selection.

    This is what makes a new run apples-to-apples with a prior run that used
    the legacy positional draw at the same seed and split.
    """
    seed = 1234
    pool = build_gpr_pool(VAL_IDS, pool_size=50, comparable_size=20, seed=seed)
    legacy_pos = pick_gpr_subset(len(VAL_IDS), 20, np.random.RandomState(seed))
    legacy_ids = sorted(VAL_IDS[i] for i in legacy_pos)
    assert pool["comparable_ids"] == legacy_ids


def test_pool_to_indices_maps_to_positions() -> None:
    pool = build_gpr_pool(VAL_IDS, pool_size=50, comparable_size=20, seed=7)
    pool_idx, comp_idx = pool_to_indices(pool, VAL_IDS)
    assert len(pool_idx) == 50
    assert len(comp_idx) == 20
    assert set(comp_idx).issubset(set(pool_idx))
    # Positions resolve back to the right ids.
    assert sorted(VAL_IDS[i] for i in comp_idx) == pool["comparable_ids"]


def test_pool_to_indices_skips_absent_ids() -> None:
    pool = build_gpr_pool(VAL_IDS, pool_size=50, comparable_size=20, seed=7)
    # A different val ordering missing half the records.
    other = VAL_IDS[:150]
    pool_idx, comp_idx = pool_to_indices(pool, other)
    # Only ids present in `other` survive.
    assert all(other[i] in set(pool["pool_ids"]) for i in pool_idx)
    assert all(i < len(other) for i in pool_idx)


def test_save_load_round_trip(tmp_path: Path) -> None:
    pool = build_gpr_pool(VAL_IDS, pool_size=40, comparable_size=10, seed=3)
    path = tmp_path / "gpr_samples.json"
    save_gpr_pool(pool, path)
    loaded = load_gpr_pool(path)
    assert loaded["comparable_ids"] == pool["comparable_ids"]
    assert loaded["pool_ids"] == pool["pool_ids"]
    assert loaded["selection"]["seed"] == 3


def test_pool_caps_at_validation_size() -> None:
    small = [f"v{i}" for i in range(8)]
    pool = build_gpr_pool(small, pool_size=50, comparable_size=20, seed=1)
    # Cannot exceed the available records.
    assert len(pool["pool_ids"]) == 8
    assert len(pool["comparable_ids"]) <= 8
    assert set(pool["comparable_ids"]).issubset(set(pool["pool_ids"]))
