"""Univariate and grouped-importance analysis of engine errors on synthetic data with a known signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.engine_drivers.importance import (
    TARGETS,
    Target,
    grouped_importance,
    partial_dependence_table,
    score_relation,
    univariate,
)

FEATURES = ["a", "b", "c", "d"]
GROUPS_OF = {"a": "G_ab", "b": "G_ab", "c": "G_c", "d": "G_d"}
N_PATIENTS = 60
ROWS_PER_PATIENT = 20
RDE = Target("rde", "rde_pct", "RDE", "log")


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n = N_PATIENTS * ROWS_PER_PATIENT
    a = rng.normal(size=n)
    d = rng.normal(size=n)
    frame = pd.DataFrame({
        "a": a,
        "b": a + 0.1 * rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": d,
    })
    # log1p of the target recovers a linear signal in a and d
    frame["rde_pct"] = np.expm1(1.0 * a + 0.5 * d + 0.2 * rng.normal(size=n) + 3.0)
    frame["score"] = a
    patients = np.repeat(np.arange(N_PATIENTS), ROWS_PER_PATIENT)
    return frame, patients


@pytest.fixture(scope="module", autouse=True)
def _few_threads():
    """Small boosted fits are slower, not faster, on dozens of OpenMP threads."""
    with threadpool_limits(limits=2):
        yield


@pytest.fixture(scope="module")
def importance(synthetic):
    frame, patients = synthetic
    return grouped_importance(frame, FEATURES, GROUPS_OF, RDE, patients, n_folds=3, n_repeats=2, n_boot=50)


def test_target_fail_log_orients_gamma_as_failing_share():
    t = Target("fail", "g", "label", "fail_log")
    x = pd.DataFrame({"g": [100.0, 98.0, 50.0]})
    assert np.allclose(t.values(x), np.log1p([0.0, 2.0, 50.0]))


def test_target_fail_log_clips_above_100_at_zero():
    t = Target("fail", "g", "label", "fail_log")
    assert t.values(pd.DataFrame({"g": [101.0, 120.0]})).tolist() == [0.0, 0.0]


def test_target_log_is_log1p_and_clips_negatives():
    t = Target("l", "x", "label", "log")
    assert np.allclose(t.values(pd.DataFrame({"x": [0.0, 3.0, -2.0]})), [0.0, np.log1p(3.0), 0.0])


def test_target_none_is_identity_and_keeps_sign():
    t = Target("n", "x", "label", "none")
    assert t.values(pd.DataFrame({"x": [-1.5, 0.0, 2.5]})).tolist() == [-1.5, 0.0, 2.5]


def test_targets_table_is_as_documented():
    by_name = {t.name: t for t in TARGETS}
    assert [t.name for t in TARGETS] == ["fail_2_2", "fail_3_3", "mape", "rde", "dr80"]
    assert by_name["fail_2_2"].column == "gpr_2pct_2mm_10pct"
    assert by_name["fail_3_3"].column == "gpr_3pct_3mm_10pct"
    assert by_name["mape"].column == "mape_pct"
    assert by_name["rde"].column == "rde_pct"
    assert by_name["dr80"].column == "dr80_mm"
    assert [t.transform for t in TARGETS] == ["fail_log", "fail_log", "log", "log", "none"]


def test_target_columns_match_scoring_criterion_key():
    from src.engine_drivers.scoring import criterion_key

    assert criterion_key((2, 2, 10)) == TARGETS[0].column
    assert criterion_key((3, 3, 10)) == TARGETS[1].column


def test_univariate_columns_and_signal(synthetic):
    frame, patients = synthetic
    out = univariate(frame, FEATURES, RDE, patients, n_boot=50)
    for col in ["feature", "spearman", "ci_low", "ci_high", "ci_excludes_zero", "target"]:
        assert col in out.columns
    assert set(out["feature"]) == set(FEATURES)
    assert (out["target"] == "rde").all()
    row = out.set_index("feature")
    assert row.loc["a", "spearman"] > 0.5
    assert row.loc["a", "ci_low"] > 0
    assert bool(row.loc["a", "ci_excludes_zero"])
    assert row.loc["c", "ci_low"] < 0 < row.loc["c", "ci_high"]
    assert not bool(row.loc["c", "ci_excludes_zero"])
    assert (out["ci_low"] <= out["ci_high"]).all()


def test_univariate_is_sorted_by_absolute_correlation(synthetic):
    frame, patients = synthetic
    out = univariate(frame, FEATURES, RDE, patients, n_boot=20)
    mags = out["spearman"].abs().to_numpy()
    assert (np.diff(mags) <= 1e-12).all()
    assert out["feature"].iloc[-1] == "c"


def test_grouped_importance_ranks_correlated_group_first(importance):
    imp = importance.importance
    assert imp["group"].iloc[0] == "G_ab"
    assert imp["rank"].tolist() == [1, 2, 3]
    assert set(imp["group"]) == {"G_ab", "G_c", "G_d"}
    row = imp.set_index("group")
    assert row["delta_mse"].idxmin() == "G_c"
    assert abs(row.loc["G_c", "delta_mse"]) < 0.05 * row.loc["G_ab", "delta_mse"]
    assert row.loc["G_ab", "delta_mse"] > row.loc["G_d", "delta_mse"] > row.loc["G_c", "delta_mse"]
    assert row.loc["G_ab", "members"] == "a b"


def test_grouped_importance_shares_and_intervals(importance):
    imp = importance.importance
    assert imp["share"].sum() == pytest.approx(1.0)
    assert (imp["share"].dropna() >= 0).all()
    assert (imp["ci_low"] <= imp["delta_mse"] + 1e-12).all()
    assert (imp["delta_mse"] <= imp["ci_high"] + 1e-12).all()
    assert (imp["target"] == "rde").all()
    assert "relative_to_baseline_mse" in imp.columns


def test_grouped_importance_fit_summary(importance):
    fit = importance.fit
    assert fit["n"] == N_PATIENTS * ROWS_PER_PATIENT
    assert fit["n_patients"] == N_PATIENTS
    assert fit["oof_r2"] > 0.3
    assert fit["oof_spearman"] > 0.5
    assert fit["mse_baseline"] > 0


def test_partial_dependence_rises_with_lead_feature(synthetic, importance):
    frame, _ = synthetic
    table = partial_dependence_table(importance, frame, FEATURES, ["a"], RDE, grid_points=10)
    assert set(table["feature"]) == {"a"}
    assert (table["target"] == "rde").all()
    assert len(table) == 10
    assert table["value"].is_monotonic_increasing
    pdv = table["partial_dependence"].to_numpy()
    assert pdv[-1] > pdv[0] + 1.0


def test_partial_dependence_has_one_block_per_lead_feature(synthetic, importance):
    frame, _ = synthetic
    table = partial_dependence_table(importance, frame, FEATURES, ["a", "d"], RDE, grid_points=5)
    assert table.groupby("feature").size().to_dict() == {"a": 5, "d": 5}


def test_score_relation_correlation_and_deciles(synthetic):
    frame, patients = synthetic
    corr, table = score_relation(frame, "score", RDE, patients, n_boot=50, deciles=None)
    assert corr["spearman"] > 0
    assert corr["feature"] == "score"
    assert len(table) == 10
    for col in ["decile", "n", "median", "q25", "q75", "mean", "target"]:
        assert col in table.columns
    assert table["decile"].tolist() == list(range(1, 11))
    assert table["n"].sum() == len(frame)
    assert (table["q25"] <= table["median"]).all() and (table["median"] <= table["q75"]).all()
    assert table["median"].iloc[-1] > table["median"].iloc[0]


def test_score_relation_uses_supplied_deciles(synthetic):
    frame, patients = synthetic
    custom = pd.Series(np.where(frame["score"] > 0, 2, 1), index=frame.index)
    _, table = score_relation(frame, "score", RDE, patients, n_boot=20, deciles=custom)
    assert table["decile"].tolist() == [1, 2]
    assert table["n"].sum() == len(frame)
