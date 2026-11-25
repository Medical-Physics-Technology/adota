"""Exact additive contributions of the linear difficulty score, and the feature groups around them."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.acquisition.explain.contributions import (
    baseline,
    decompose,
    ensemble_consistency,
    group_contributions,
    mean_decomposition,
    pair_contrast,
    percentile_frame,
    sum_check,
)
from src.acquisition.explain.groups import FAMILY, feature_groups
from src.acquisition.scorer import FULL, DifficultyScorer, percentile_grids
from src.active_learning.ensemble.independence import fit_scorer

METRICS = ["m_normal", "m_skewed", "m_uniform", "m_discrete"]


def _reference(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "m_normal": rng.normal(10.0, 3.0, n),
        "m_skewed": rng.lognormal(0.0, 1.2, n),
        "m_uniform": rng.uniform(-5.0, 5.0, n),
        "m_discrete": rng.integers(0, 6, n).astype(float),
    }, index=[f"r{i}" for i in range(n)])


def _scorer(ref: pd.DataFrame) -> DifficultyScorer:
    weights = {"m_normal": 1.5, "m_skewed": -0.7, "m_uniform": 0.4, "m_discrete": 2.0}
    return DifficultyScorer(variant=FULL, intercept=0.3, weights=weights, grids=percentile_grids(ref, METRICS))


def _beamlets(seed: int = 1, n: int = 5) -> pd.DataFrame:
    return _reference(n, seed).set_axis([f"b{i}" for i in range(n)])


def test_contributions_match_an_independent_computation_and_sum_to_score_minus_baseline():
    ref, scorer, beam = _reference(), None, _beamlets()
    scorer = _scorer(ref)
    dec = decompose(scorer, beam, ref, "s")
    for m in METRICS:
        p_ref = np.mean([scorer.percentile(m, ref[m].to_numpy())])
        p_x = scorer.percentile(m, beam[m].to_numpy())
        expected = scorer.weights[m] * (p_x - p_ref)
        got = dec.table[dec.table["feature"] == m].set_index("beamlet_id")["contribution"].loc[beam.index]
        np.testing.assert_allclose(got.to_numpy(), expected, atol=1e-12, rtol=0)
    check = sum_check([dec])
    assert len(check) == len(beam) and check["residual"].abs().max() < 1e-12
    assert set(dec.table.columns) == {"beamlet_id", "scorer", "feature", "percentile", "mean_percentile_ref",
                                      "weight", "contribution"}


def test_baseline_is_the_reference_mean_score_and_the_reference_decomposes_to_zero_mean():
    ref = _reference()
    scorer = _scorer(ref)
    assert baseline(scorer, ref) == pytest.approx(float(np.mean(scorer.score(ref))), abs=1e-12)
    dec = decompose(scorer, ref, ref, "s")
    assert dec.baseline == pytest.approx(baseline(scorer, ref), abs=1e-12)
    per_beamlet = dec.table.groupby("beamlet_id")["contribution"].sum()
    assert abs(per_beamlet.mean()) < 1e-12
    per_feature = dec.table.groupby("feature")["contribution"].mean()
    assert per_feature.abs().max() < 1e-12


def test_percentile_frame_is_in_unit_interval_and_follows_the_scorer_transform():
    ref = _reference()
    scorer = _scorer(ref)
    frame = percentile_frame(scorer, ref)
    assert list(frame.columns) == METRICS and frame.index.equals(ref.index)
    assert frame.to_numpy().min() >= 0.0 and frame.to_numpy().max() <= 1.0
    sub = percentile_frame(scorer, ref, ["m_skewed"])
    np.testing.assert_array_equal(sub["m_skewed"].to_numpy(), frame["m_skewed"].to_numpy())


def test_sum_check_assertion_fires_when_the_score_is_not_the_linear_form():
    ref = _reference()
    scorer = _scorer(ref)

    class Offset(DifficultyScorer):
        def score(self, features):
            return np.asarray(super().score(features)) + 0.01

    broken = Offset(variant=scorer.variant, intercept=scorer.intercept, weights=scorer.weights,
                    grids=scorer.grids)
    # A constant offset moves the baseline too, so it cancels; offset only the beamlet scores.

    class Shifted(Offset):
        def score(self, features):
            out = np.asarray(DifficultyScorer.score(self, features))
            return out + (0.01 if len(features) < len(ref) else 0.0)

    shifted = Shifted(variant=scorer.variant, intercept=scorer.intercept, weights=scorer.weights,
                      grids=scorer.grids)
    with pytest.raises(AssertionError, match="do not add up"):
        decompose(shifted, _beamlets(), ref, "bad")
    # and the pure constant offset is absorbed by the baseline, as it must be
    decompose(broken, _beamlets(), ref, "const")


def test_ridge_members_average_to_the_ensemble_and_so_do_their_contributions():
    ref = _reference()
    rng = np.random.default_rng(7)
    ys = [rng.normal(size=len(ref)) + 0.5 * ref["m_normal"].to_numpy() for _ in range(5)]
    members = [fit_scorer(ref, METRICS, y, FULL, f"m{k}") for k, y in enumerate(ys)]
    ensemble = fit_scorer(ref, METRICS, np.mean(ys, axis=0), FULL, "ens")
    for m in METRICS:
        assert np.mean([s.weights.get(m, 0.0) for s in members]) == pytest.approx(ensemble.weights[m], abs=1e-10)
    assert np.mean([s.intercept for s in members]) == pytest.approx(ensemble.intercept, abs=1e-10)

    beam = _beamlets()
    dec_ens = decompose(ensemble, beam, ref, "ens")
    dec_members = [decompose(s, beam, ref, f"m{k}") for k, s in enumerate(members)]
    assert ensemble_consistency(dec_ens, dec_members) < 1e-10
    # a different ensemble is detected
    other = fit_scorer(ref, METRICS, ys[0], FULL, "other")
    assert ensemble_consistency(decompose(other, beam, ref, "o"), dec_members) > 1e-6


def test_mean_decomposition_averages_scores_baselines_and_contributions():
    ref = _reference()
    beam = _beamlets()
    a = _scorer(ref)
    b = DifficultyScorer(variant=FULL, intercept=-1.0, weights={"m_normal": -0.5, "m_uniform": 1.0, "m_discrete": 0.2,
                                                               "m_skewed": 0.9}, grids=a.grids)
    da, db = decompose(a, beam, ref, "a"), decompose(b, beam, ref, "b")
    mean = mean_decomposition([da, db], name="avg")
    assert mean.scorer == "avg"
    np.testing.assert_allclose(mean.scores.to_numpy(), (da.scores.to_numpy() + db.scores.to_numpy()) / 2, atol=1e-12)
    assert mean.baseline == pytest.approx((da.baseline + db.baseline) / 2, abs=1e-12)
    key = ["beamlet_id", "feature"]
    ta, tb = da.table.set_index(key)["contribution"], db.table.set_index(key)["contribution"]
    tm = mean.table.set_index(key)["contribution"]
    np.testing.assert_allclose(tm.loc[ta.index].to_numpy(), ((ta + tb) / 2).loc[ta.index].to_numpy(),
                               atol=1e-12)
    assert (mean.table["scorer"] == "avg").all()
    assert sum_check([mean])["residual"].abs().max() < 1e-12


def _table() -> pd.DataFrame:
    rows = []
    contrib = {"b1": {"f1": 1.0, "f2": 0.5, "f3": -0.25, "f4": 2.0},
               "b2": {"f1": 0.25, "f2": 0.5, "f3": 0.75, "f4": -1.0}}
    for scorer in ("s1", "s2"):
        for bid, d in contrib.items():
            for f, c in d.items():
                rows.append({"beamlet_id": bid, "scorer": scorer, "feature": f, "percentile": 0.5,
                             "mean_percentile_ref": 0.4, "weight": 1.0,
                             "contribution": c * (1 if scorer == "s1" else 2)})
    return pd.DataFrame(rows)


def test_group_contributions_sum_within_groups_and_keep_ungrouped_features_alone():
    table = _table()
    groups = {"f1": "G", "f2": "G", "f3": "H"}          # f4 has no group
    out = group_contributions(table, groups)
    s1b1 = out[(out["scorer"] == "s1") & (out["beamlet_id"] == "b1")].set_index("group")
    assert s1b1.loc["G", "contribution"] == pytest.approx(1.5) and s1b1.loc["G", "n_features"] == 2
    assert s1b1.loc["H", "contribution"] == pytest.approx(-0.25) and s1b1.loc["H", "n_features"] == 1
    assert s1b1.loc["f4", "contribution"] == pytest.approx(2.0) and s1b1.loc["f4", "n_features"] == 1
    assert set(s1b1.loc["G", "features"].split()) == {"f1", "f2"}
    by_group = out.groupby(["beamlet_id", "scorer"])["contribution"].sum()
    by_feature = table.groupby(["beamlet_id", "scorer"])["contribution"].sum()
    np.testing.assert_allclose(by_group.loc[by_feature.index].to_numpy(), by_feature.to_numpy(), atol=1e-12)
    assert out["n_features"].sum() == len(table)


def test_pair_contrast_at_group_and_feature_level():
    table = _table()
    feat = pair_contrast(table, "b1", "b2", level="feature", top_k=2)
    assert set(feat["scorer"]) == {"s1", "s2"} and (feat["level"] == "feature").all()
    s1 = feat[feat["scorer"] == "s1"].sort_values("rank")
    # differences b1 - b2: f4 3.0, f1 0.75, f2 0.0, f3 -1.0
    assert list(s1["item"]) == ["f4", "f1", "f2", "f3"]
    np.testing.assert_allclose(s1["difference"], [3.0, 0.75, 0.0, -1.0])
    assert s1["share_of_gap"].sum() == pytest.approx(1.0) and s1["score_gap"].iloc[0] == pytest.approx(2.75)
    assert list(s1["top"]) == [True, True, False, False]
    for _, sub in feat.groupby("scorer"):
        assert sub["top"].sum() == 2 and sub["share_of_gap"].sum() == pytest.approx(1.0)
    s2 = feat[feat["scorer"] == "s2"].sort_values("rank")
    np.testing.assert_allclose(s2["difference"], 2 * s1["difference"].to_numpy())
    assert (s1["harder"] == "b1").all() and (s1["easier"] == "b2").all()

    grouped = group_contributions(table, {"f1": "G", "f2": "G", "f3": "H"})
    grp = pair_contrast(grouped, "b1", "b2", level="group", top_k=1)
    g1 = grp[grp["scorer"] == "s1"].sort_values("rank")
    assert list(g1["item"]) == ["f4", "G", "H"] and (g1["level"] == "group").all()
    np.testing.assert_allclose(g1["difference"], [3.0, 0.75, -1.0])
    assert g1["share_of_gap"].sum() == pytest.approx(1.0) and list(g1["top"]) == [True, False, False]
    for _, sub in grp.groupby("scorer"):
        assert sub["top"].sum() == 1


def _blocks(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    noise = 0.4

    def col(z):
        return z + noise * rng.normal(size=n)

    data = {
        "bp_range_max_mm": col(z1), "energy_mev": col(z1), "wepl_mean": -col(z1),      # wepl_mean flipped
        "sobel_dw_mean": col(z2), "sobel_dw_edge_energy": col(z2), "sobel_th_mean": col(z2),
        "ct_max_hu": rng.normal(size=n),
    }
    frame = pd.DataFrame(data)
    return frame.rank(pct=True)


def test_feature_groups_recover_blocks_a_singleton_and_family_names():
    pct = _blocks()
    groups = feature_groups(pct, threshold=0.7)
    names = groups.names
    block1 = {"bp_range_max_mm", "energy_mev", "wepl_mean"}
    block2 = {"sobel_dw_mean", "sobel_dw_edge_energy", "sobel_th_mean"}
    assert len({names[m] for m in block1}) == 1 and len({names[m] for m in block2}) == 1
    assert len(set(names.values())) == 3                       # two blocks and one singleton
    assert names["ct_max_hu"] == "densest voxel in the crop [ct_max_hu]"
    n1, n2 = names["energy_mev"], names["sobel_dw_mean"]
    assert n1 != n2
    for name, members in ((n1, block1), (n2, block2)):
        family, medoid = name[:-1].split(" [")
        assert medoid in members and family == FAMILY[medoid]
        assert name == f"{family} [{medoid}]"
    assert n1.startswith("depth and energy [") and n2.startswith("edges around the peak [")
    # the negatively correlated member joined and is recorded with a negative sign
    assert groups.clusters.signs["wepl_mean"] != groups.clusters.signs[groups.clusters.medoid[
        groups.clusters.assignment["wepl_mean"]]] or groups.clusters.medoid[
        groups.clusters.assignment["wepl_mean"]] == "wepl_mean"


def test_feature_groups_threshold_validation_and_table():
    pct = _blocks(n=600)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            feature_groups(pct, threshold=bad)
    groups = feature_groups(pct, threshold=0.7)
    assert groups.threshold == pytest.approx(0.7)
    table = groups.table()
    assert np.allclose(table["abs_rho_threshold"], 0.7) and len(table) == 3
    assert set(table["group"]) == set(groups.names.values())
    # a threshold of 1 keeps every feature apart unless perfectly rank-correlated
    strict = feature_groups(pct, threshold=1.0)
    assert len(set(strict.names.values())) == pct.shape[1]
