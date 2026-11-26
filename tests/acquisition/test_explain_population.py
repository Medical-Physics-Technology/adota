"""The population view of the difficulty explanation: example selection, vectorised decomposition, deciles."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.acquisition.explain.contributions import decompose, group_contributions
from src.acquisition.explain.population import (
    ALL,
    decile_profiles,
    leading_group_share,
    population_groups,
    profile_agreement,
    score_deciles,
    select_examples,
    variance_share,
    within_stratum_pairs,
)
from src.acquisition.scorer import FULL, DifficultyScorer, percentile_grids

METRICS = ["m_a", "m_b", "m_c"]
GROUPS = {"m_a": "G1", "m_b": "G1", "m_c": "G2"}


def _reference(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"m_a": rng.normal(0, 1, n), "m_b": rng.lognormal(0, 1, n), "m_c": rng.uniform(0, 1, n)},
                        index=[f"s{i:04d}" for i in range(n)])


def _scorer(ref: pd.DataFrame, scale: float = 1.0) -> DifficultyScorer:
    weights = {"m_a": 1.2 * scale, "m_b": -0.5 * scale, "m_c": 0.8}
    return DifficultyScorer(variant=FULL, intercept=0.1, weights=weights, grids=percentile_grids(ref, METRICS))


def _meta(index: pd.Index) -> pd.DataFrame:
    anatomy = np.array(["thoracic", "pelvic", "abdominal"])[np.arange(len(index)) % 3]
    patient = [f"{a}-{i % 20:02d}" for i, a in enumerate(anatomy)]
    return pd.DataFrame({"patient_id": patient, "anatomy": anatomy}, index=index)


def test_selection_is_per_stratum_nearest_percentile_with_distinct_patients():
    ref = _reference()
    scores = pd.Series(_scorer(ref).score(ref), index=ref.index)
    meta = _meta(ref.index)
    sel = select_examples(scores, meta, [10, 50, 90], ["easy", "medium", "hard"])
    assert len(sel) == 9 and sel["patient_id"].is_unique
    assert set(sel["label"]) == {f"{a}_{lv}" for a in ("thoracic", "pelvic", "abdominal")
                                 for lv in ("easy", "medium", "hard")}
    for r in sel.itertuples():
        assert abs(r.stratum_percentile - r.target_percentile) < 3.0
        assert meta.at[r.sample_id, "anatomy"] == r.stratum
    # the order within a stratum follows the score
    for _, sub in sel.groupby("stratum"):
        assert list(sub.sort_values("score")["level"]) == ["easy", "medium", "hard"]
    assert select_examples(scores, meta, [10, 50, 90], ["easy", "medium", "hard"]).equals(sel)  # deterministic


def test_selection_skips_a_used_patient_and_fails_when_none_is_left():
    idx = pd.Index(["a", "b", "c"])
    meta = pd.DataFrame({"patient_id": ["p1", "p1", "p2"], "anatomy": ["x", "x", "x"]}, index=idx)
    scores = pd.Series([0.0, 0.1, 5.0], index=idx)
    sel = select_examples(scores, meta, [0, 1], ["easy", "next"])
    assert list(sel["sample_id"]) == ["a", "c"]          # "b" is nearer but shares p1
    with pytest.raises(ValueError, match="no unused patient"):
        select_examples(scores, meta, [0, 1, 2], ["l1", "l2", "l3"])


def test_within_stratum_pairs_pair_consecutive_levels():
    sel = pd.DataFrame({"label": ["t_e", "t_m", "t_h", "p_e", "p_m", "p_h"], "stratum": list("tttppp"),
                        "level": ["easy", "medium", "hard"] * 2})
    assert within_stratum_pairs(sel, ["easy", "medium", "hard"]) == [("t_m", "t_e"), ("t_h", "t_m"),
                                                                     ("p_m", "p_e"), ("p_h", "p_m")]


def test_population_groups_equal_the_per_beamlet_decomposition():
    ref = _reference()
    scorer = _scorer(ref)
    grouped, total, base = population_groups(scorer, ref, ref, GROUPS)
    np.testing.assert_allclose(grouped.sum(axis=1), total, atol=1e-12)
    sub = ref.iloc[:7]
    slow = group_contributions(decompose(scorer, sub, ref, "s").table, GROUPS)
    slow = slow.pivot(index="beamlet_id", columns="group", values="contribution")
    np.testing.assert_allclose(grouped.loc[sub.index, slow.columns].to_numpy(), slow.loc[sub.index].to_numpy(),
                               atol=1e-12)
    assert abs(total.mean()) < 1e-12 and np.isfinite(base)


def test_deciles_are_equal_count_and_profiles_add_up():
    ref = _reference()
    grouped, total, _ = population_groups(_scorer(ref), ref, ref, GROUPS)
    deciles = score_deciles(total, 10)
    assert deciles.value_counts().eq(60).all() and deciles.min() == 1 and deciles.max() == 10
    assert total[deciles == 10].min() >= total[deciles == 9].max()
    prof = decile_profiles(grouped, total, deciles, _meta(ref.index))
    assert set(prof["panel"]) == {ALL, "thoracic", "pelvic", "abdominal"}
    sums = prof.groupby(["panel", "decile"]).agg(s=("mean_contribution", "sum"), t=("mean_total", "first"))
    np.testing.assert_allclose(sums["s"], sums["t"], atol=1e-12)
    assert prof[prof["panel"] == ALL].groupby("decile")["n_beamlets"].first().eq(60).all()


def test_leading_share_rows_sum_to_one_and_agreement_is_perfect_for_the_same_scorer():
    ref = _reference()
    meta = _meta(ref.index)
    grouped, total, _ = population_groups(_scorer(ref), ref, ref, GROUPS)
    deciles = score_deciles(total)
    share = leading_group_share(grouped, deciles)
    np.testing.assert_allclose(share.drop(columns="decile").sum(axis=1), 1.0)
    prof = decile_profiles(grouped, total, deciles, meta)
    g2, t2, _ = population_groups(_scorer(ref, scale=0.5), ref, ref, GROUPS)
    agree = profile_agreement({"ens": prof, "same": prof, "other": decile_profiles(g2, t2, deciles, meta)}, "ens")
    same = agree.set_index("scorer").loc["same"]
    assert same["pearson_vs_reference"] == pytest.approx(1.0) and same["max_abs_difference"] == 0.0
    assert bool(same["same_top_decile_leader"])
    assert agree.set_index("scorer").loc["other", "max_abs_difference"] > 0


def test_variance_shares_add_up_to_one_per_panel_and_match_a_single_group():
    ref = _reference()
    meta = _meta(ref.index)
    grouped, total, _ = population_groups(_scorer(ref), ref, ref, GROUPS)
    share = variance_share(grouped, total, meta)
    np.testing.assert_allclose(share.groupby("panel")["variance_share"].sum(), 1.0, atol=1e-12)
    one = variance_share(grouped[["G1"]] * 0 + total.to_frame("G1").to_numpy(), total, meta)
    np.testing.assert_allclose(one["variance_share"], 1.0, atol=1e-12)
