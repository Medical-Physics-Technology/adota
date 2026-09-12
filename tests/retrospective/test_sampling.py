"""The strategy registry and the three strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.active_learning.retrospective import sampling
from src.active_learning.retrospective.sampling import (
    STRATEGIES,
    available_strategies,
    register_strategy,
    score_deciles,
    select,
    selection_fingerprint,
)


def make_pool(n: int = 500, n_unscored: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = pd.DataFrame({
        "sample_id": [f"s{i:04d}" for i in range(n)],
        "energy_mev": rng.uniform(70.0, 250.0, size=n),
        "patient": rng.choice([f"P{k}" for k in range(8)], size=n),
        "anatomy": rng.choice(["thorax", "pelvic"], size=n),
        "score": rng.normal(size=n),
    })
    pool.loc[pool.index[:n_unscored], "score"] = np.nan
    return pool


@pytest.mark.parametrize("strategy", ["random", "score_topk", "stratified_score"])
def test_every_strategy_spends_exactly_its_budget_from_the_pool(strategy):
    pool = make_pool()
    chosen = select(pool, 120, strategy, np.random.default_rng(1))
    assert len(chosen) == 120 and len(set(chosen)) == 120
    assert set(chosen) <= set(pool["sample_id"])


@pytest.mark.parametrize("strategy", ["random", "score_topk", "stratified_score"])
def test_selection_is_a_function_of_the_seed(strategy):
    pool = make_pool()
    a = select(pool, 50, strategy, np.random.default_rng([3, 1]))
    b = select(pool, 50, strategy, np.random.default_rng([3, 1]))
    assert a == b


def test_random_ignores_the_scores_entirely():
    pool = make_pool()
    shuffled = pool.assign(score=pool["score"].sample(frac=1, random_state=5).to_numpy())
    assert select(pool, 80, "random", np.random.default_rng(0)) == \
        select(shuffled, 80, "random", np.random.default_rng(0))


def test_score_topk_takes_the_highest_scores():
    pool = make_pool()
    chosen = select(pool, 30, "score_topk", np.random.default_rng(0))
    expected = pool.dropna(subset=["score"]).nlargest(30, "score")["sample_id"]
    assert set(chosen) == set(expected)


def test_stratified_score_draws_equal_counts_per_decile():
    pool = make_pool(n=520, n_unscored=20)
    chosen = select(pool, 100, "stratified_score", np.random.default_rng(0))
    deciles = pd.Series(score_deciles(pool["score"]), index=pool["sample_id"])
    counts = deciles.loc[chosen].value_counts()
    assert sorted(counts.index) == list(range(10)) and counts.max() == counts.min() == 10


def test_stratified_score_tops_up_thin_deciles():
    pool = make_pool(n=15, n_unscored=0)
    chosen = select(pool, 12, "stratified_score", np.random.default_rng(0))
    assert len(chosen) == 12 and len(set(chosen)) == 12


def test_unscored_records_reach_random_but_never_the_score_strategies():
    pool = make_pool(n=60, n_unscored=50)
    unscored = set(pool.loc[pool["score"].isna(), "sample_id"])
    assert set(select(pool, 55, "random", np.random.default_rng(0))) & unscored
    for strategy in ("score_topk", "stratified_score"):
        assert not set(select(pool, 10, strategy, np.random.default_rng(0))) & unscored
        with pytest.raises(ValueError, match="scored records"):
            select(pool, 11, strategy, np.random.default_rng(0))


def test_registry_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="unknown strategy"):
        select(make_pool(), 5, "greedy", np.random.default_rng(0))
    with pytest.raises(ValueError, match="already registered"):
        register_strategy("random")(lambda pool, n, rng: [])


def test_a_new_strategy_is_one_registered_function():
    name = "first_k_test"
    try:
        @register_strategy(name)
        def first_k(pool, n, rng):
            return pool["sample_id"].head(n).tolist()

        assert name in available_strategies()
        assert select(make_pool(), 3, name, np.random.default_rng(0)) == ["s0000", "s0001", "s0002"]
    finally:
        STRATEGIES.pop(name, None)


def test_select_checks_the_strategy_answer():
    name = "broken_test"
    try:
        register_strategy(name)(lambda pool, n, rng: ["s0000"] * n)
        with pytest.raises(AssertionError, match="distinct"):
            select(make_pool(), 3, name, np.random.default_rng(0))
    finally:
        STRATEGIES.pop(name, None)


def test_score_deciles_are_rank_based_and_flag_missing():
    scores = np.array([np.nan, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    deciles = score_deciles(scores)
    assert deciles[0] == -1 and deciles[1:].min() == 0 and deciles[1:].max() == 9
    assert list(deciles[1:]) == sorted(deciles[1:])


def test_fingerprint_shares_cover_selected_and_pool():
    pool = make_pool()
    chosen = select(pool, 48, "score_topk", np.random.default_rng(0))   # one decile of 480
    print_ = selection_fingerprint(pool, chosen)
    assert print_["n_selected"] == 48 and print_["n_pool"] == 500
    for side in ("selected", "pool"):
        for key in ("energy", "anatomy", "patient", "score_decile"):
            assert abs(sum(print_[side][key].values()) - 1.0) < 1e-9
    assert print_["selected_score_mean"] > print_["pool_score_mean"]
    assert print_["selected"]["score_decile"] == {"9": 1.0}
    assert "unscored" in print_["pool"]["score_decile"]
    unscored = selection_fingerprint(pool.assign(score=np.nan), chosen)
    assert "score_decile" not in unscored["selected"]   # random never scores
    assert sampling.N_DECILES == 10
