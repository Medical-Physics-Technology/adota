"""The EXP-0014 closing analysis on synthetic data: the leave-one-member-out gap
is near zero when every member shares one difficulty and positive when one
member has its own; the scan bootstrap, the decision rule, the energy baseline,
the scorer JSON round trip, the noise set's patients, batches and config pair, and the shared-noise
diagnostics."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from src.acquisition.scorer import FULL, SPARSE, DifficultyScorer
from src.active_learning.beamlet_pool.noise import pair_config_differences, patient_batches, select_patients
from src.active_learning.ensemble.independence import (
    EvalSet,
    Standardiser,
    energy_baseline,
    fit_scorer,
    leave_one_member_out,
    pearson_from_sums,
    scan_resample_counts,
    scan_sums,
    scorer_json,
    verdict,
    within_test_cv_gap,
)
from src.active_learning.ensemble.noise_floor import cross_reference_agreement, floor_ratio, mc_pair_errors
from src.active_learning.retrospective.validation import MetricSettings
from src.adota.config import DEFAULT_SCALE

METRICS = ["m0", "m1", "m2", "m3"]


def synthetic(n_scans: int = 12, per_scan: int = 60, seed: int = 0, own_feature_member: str = ""):
    """Features, and members whose log-errors are a shared function of m0 + m1
    plus member noise; ``own_feature_member`` also depends strongly on m3."""
    rng = np.random.default_rng(seed)
    n = n_scans * per_scan
    feats = pd.DataFrame(rng.normal(size=(n, len(METRICS))), columns=METRICS,
                         index=[f"s{i}" for i in range(n)])
    scans = np.repeat([f"scan{i}" for i in range(n_scans)], per_scan)
    shared = feats["m0"] + 0.5 * feats["m1"]
    errors = {}
    for k, member in enumerate(["A", "B", "C", "D"]):
        own = 1.5 * feats["m3"] if member == own_feature_member else 0.0
        errors[member] = 1.0 + 0.1 * k + shared + own + 0.3 * rng.normal(size=n)
    return feats, pd.DataFrame(errors, index=feats.index), scans


def split(feats, z, scans, n_fit_scans: int = 8):
    fit = np.isin(scans, [f"scan{i}" for i in range(n_fit_scans)])
    test = EvalSet("test", feats[~fit], z[~fit], scans[~fit])
    return feats[fit], z[fit], test


def test_standardiser_uses_the_fit_rows_only():
    errors = pd.DataFrame({"A": [1.0, 3.0, 100.0]})
    std = Standardiser.fit(errors.iloc[:2])
    np.testing.assert_allclose(std(errors)["A"].to_numpy(), [-1.0, 1.0, 98.0])


def test_pearson_from_sums_and_bootstrap_identity():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=200), rng.normal(size=200)
    scans = np.repeat(["a", "b", "c", "d"], 50)
    sums = scan_sums(x + y, y, scans, ["a", "b", "c", "d"])
    assert pearson_from_sums(sums.sum(axis=0)) == pytest.approx(np.corrcoef(x + y, y)[0, 1])
    assert pearson_from_sums(np.ones((1, 4)) @ sums)[0] == pytest.approx(np.corrcoef(x + y, y)[0, 1])
    counts = scan_resample_counts(4, 50, seed=0)
    assert counts.shape == (50, 4) and np.all(counts.sum(axis=1) == 4)


def test_fit_scorer_finds_the_planted_metrics_and_the_lasso_drops_the_rest():
    feats, errors, _ = synthetic()
    y = errors["A"].to_numpy()
    ridge = fit_scorer(feats, METRICS, y, FULL)
    assert np.corrcoef(ridge.score(feats), y)[0, 1] > 0.9
    sparse = fit_scorer(feats, METRICS, (y - y.mean()) / y.std() * 0.07, SPARSE)
    assert {"m0", "m1"} <= set(sparse.weights) and "m2" not in sparse.weights
    with pytest.raises(ValueError):
        fit_scorer(feats, METRICS, y, "gbm")


def test_shared_difficulty_gives_no_gap_and_an_idiosyncratic_member_does():
    feats, errors, scans = synthetic()
    fit_f, fit_z, test = split(feats, Standardiser.fit(errors)(errors), scans)
    rows, summary = leave_one_member_out(fit_f, fit_z, METRICS, [test], ["A", "B", "C", "D"], n_boot=200)
    assert summary["median_gap"].iloc[0] < 0.03
    assert rows["gap"].abs().max() < 0.03
    assert verdict(summary["median_gap"].iloc[0], summary["ci_hi"].iloc[0]) == "model-independent"

    feats, errors, scans = synthetic(own_feature_member="D")
    fit_f, fit_z, test = split(feats, Standardiser.fit(errors)(errors), scans)
    rows, _ = leave_one_member_out(fit_f, fit_z, METRICS, [test], ["A", "B", "C", "D"], n_boot=200)
    gaps = rows.set_index("member")["gap"]
    assert gaps["D"] > 0.2 and gaps["D"] == gaps.max()


def test_verdict_follows_the_preregistered_thresholds():
    assert verdict(0.02, 0.04) == "model-independent"
    assert verdict(0.02, 0.06) == "approximately model-independent"
    assert verdict(0.09, 0.2) == "approximately model-independent"
    assert verdict(0.11, 0.12) == "model-specific"


def test_energy_baseline_is_the_bin_mean_and_falls_back_to_the_nearest_bin():
    pred = energy_baseline(np.array([71.0, 75.0, 95.0]), np.array([1.0, 3.0, 10.0]), np.array([72.0, 99.0, 240.0]))
    np.testing.assert_allclose(pred, [2.0, 10.0, 10.0])


def test_within_test_cv_gap_covers_engines_outside_the_ladder():
    feats, errors, scans = synthetic(own_feature_member="D")
    z = Standardiser.fit(errors)(errors)
    test = EvalSet("test", feats, z, scans)
    cv = within_test_cv_gap(test, METRICS, ["A", "D"], ["A", "B", "C"], n_splits=4).set_index("engine")
    assert not cv.loc["D", "in_ladder"] and cv.loc["D", "gap"] > cv.loc["A", "gap"]


def test_scorer_json_round_trips_through_the_deployed_loader(tmp_path):
    feats, errors, _ = synthetic()
    y = errors["A"].to_numpy()
    fits = {FULL: fit_scorer(feats, METRICS, y, FULL), SPARSE: fit_scorer(feats, METRICS, y, SPARSE)}
    path = tmp_path / "scorer.json"
    path.write_text(json.dumps(scorer_json(fits, "test")))
    loaded = DifficultyScorer.load(path, variant=FULL)
    np.testing.assert_allclose(loaded.score(feats), fits[FULL].score(feats))


def _registry() -> pd.DataFrame:
    rows = [{"patient_id": f"{a[0]}{i}", "anatomy": a, "split": split}
            for a in ("thoracic", "abdominal", "pelvic") for i in range(10)
            for split in [("test" if i < 6 else "train")]]
    return pd.DataFrame(rows)


def test_select_patients_draws_per_anatomy_from_one_split_only():
    picked = select_patients(_registry(), {"thoracic": 4, "abdominal": 4, "pelvic": 4}, "test", seed=1)
    assert picked.groupby("anatomy").size().to_dict() == {"abdominal": 4, "pelvic": 4, "thoracic": 4}
    assert picked["patient_id"].is_unique and picked["patient_id"].str[1:].astype(int).lt(6).all()
    assert picked.equals(select_patients(_registry(), {"thoracic": 4, "abdominal": 4, "pelvic": 4}, "test", seed=1))
    with pytest.raises(ValueError):
        select_patients(_registry(), {"thoracic": 7}, "test")


@pytest.mark.parametrize("anatomies", [("thoracic", "abdominal", "pelvic"), ("thoracic", "abdominal")])
def test_patient_batches_give_two_processes_the_same_anatomy_mix(anatomies):
    picked = select_patients(_registry(), {a: 4 for a in anatomies}, "test", seed=2)
    plan = pd.DataFrame({"patient_id": np.repeat(picked["patient_id"].to_numpy(), 3)})
    plan["anatomy"] = plan["patient_id"].map(dict(zip(picked["patient_id"], picked["anatomy"])))
    plan["batch"] = patient_batches(plan, picked).to_numpy()
    assert plan.groupby("patient_id")["batch"].nunique().eq(1).all()
    assert plan.groupby("batch")["patient_id"].nunique().eq(1).all()
    per_shard = plan.drop_duplicates("patient_id").groupby([plan["batch"] % 2, "anatomy"]).size()
    assert per_shard.eq(2).all() and len(per_shard) == 2 * len(anatomies)


def test_pair_configs_may_differ_only_in_primaries_and_storage():
    a = {"robustness": {"num_primaries": 1e6, "rng_seed": 0}, "batches": {"store_root": "/s/a", "work_root": "/w/a"},
         "plan": {"seed": 1}}
    b = {"robustness": {"num_primaries": 1e7, "rng_seed": 0}, "batches": {"store_root": "/s/b", "work_root": "/w/b"},
         "plan": {"seed": 1}}
    assert pair_config_differences(a, b) == []
    b_seed = {**b, "plan": {"seed": 2}}
    assert pair_config_differences(a, b_seed) == ["plan.seed differs: 1 vs 2"]
    same_store = {**b, "batches": {"store_root": "/s/a", "work_root": "/w/b"}}
    assert pair_config_differences(a, same_store) == ["batches.store_root must differ"]


def test_shared_reference_noise_shows_up_as_same_minus_cross_agreement():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.normal(size=n)
    noise_a, noise_b = rng.normal(size=n), rng.normal(size=n)
    # Errors on the log scale are the engine's own error plus the reference's noise.
    err = lambda own, noise: np.expm1(np.abs(truth + own + noise))  # noqa: E731
    own = {e: 0.5 * rng.normal(size=n) for e in ("x", "y")}
    errors_a = pd.DataFrame({e: err(own[e], noise_a) for e in own})
    errors_b = pd.DataFrame({e: err(own[e], noise_b) for e in own})
    row = cross_reference_agreement(errors_a, errors_b).iloc[0]
    assert row["pearson_same_reference"] > row["pearson_cross_reference"] and row["shared_noise_share"] > 0.05


def test_floor_ratio():
    errors = pd.DataFrame({"e": [1.0, 2.0]}, index=["a", "b"])
    out = floor_ratio(pd.Series([0.5, 0.5], index=["a", "b"]), errors).iloc[0]
    assert out["median_ratio"] == pytest.approx(0.375) and out["ratio_of_means"] == pytest.approx(1.0 / 3.0)
    assert out["ratio_of_means_one_realisation"] == pytest.approx(1.0 / 3.0 / np.sqrt(2.0))


def test_mc_pair_errors_is_zero_for_one_realisation_against_itself(tmp_path):
    rng = np.random.default_rng(0)
    shape = (8, 8, 40)                        # storage (y, x, z)
    dose = np.zeros(shape, dtype=np.float32)
    dose[3:5, 3:5, 5:30] = np.linspace(0.1, 0.9, 25, dtype=np.float32)
    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("s0")
        g.create_dataset("ct", data=rng.random(shape, dtype=np.float32))
        g.create_dataset("flux", data=rng.random(shape, dtype=np.float32))
        g.create_dataset("dose", data=dose)
        g.attrs["initial_energy"] = 0.2
    settings = MetricSettings(scale=dict(DEFAULT_SCALE), gamma_params={
        "dose_percent_threshold": 2, "distance_mm_threshold": 2, "interp_fraction": 10, "max_gamma": 2,
        "lower_percent_dose_cutoff": 10, "random_subset": None, "local_gamma": False, "quiet": True},
        gamma_backend="pymedphys")
    out = mc_pair_errors(str(path), str(path), ["s0"], settings).iloc[0]
    assert out["rde_pct"] == pytest.approx(0.0) and out["dr80_mm"] == pytest.approx(0.0)
