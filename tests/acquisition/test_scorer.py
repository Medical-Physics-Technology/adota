"""The frozen scorer applies the study's transform and reads both file layouts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.acquisition.scorer import FULL, SPARSE, DifficultyScorer


def _study_layout(tmp_path):
    grids = {"a": np.linspace(0, 100, 101).tolist(), "b": np.linspace(-1, 1, 101).tolist()}
    data = {"seed": 42, "percentile_grids_dev": grids,
            "weights": {SPARSE: {"intercept": 0.5, "coefficients": {"a": 2.0}},
                        FULL: {"intercept": 0.1, "coefficients": {"a": 1.0, "b": -1.0}}}}
    path = tmp_path / "frozen_final_scorer.json"
    path.write_text(json.dumps(data))
    return path


def test_percentile_matches_the_study_transform(tmp_path):
    scorer = DifficultyScorer.load(_study_layout(tmp_path), SPARSE)
    grid = scorer.grids["a"]
    values = np.array([-5.0, 0.0, 50.0, 100.0, 500.0])
    expected = np.clip(np.searchsorted(grid, values, side="right") / 101, 0, 1)
    np.testing.assert_allclose(scorer.percentile("a", values), expected)
    assert scorer.percentile("a", np.array([500.0]))[0] == 1.0      # clipped, robust to outliers


def test_score_is_intercept_plus_weighted_percentiles(tmp_path):
    scorer = DifficultyScorer.load(_study_layout(tmp_path), FULL)
    features = {"a": 50.0, "b": 0.0, "unused": 123.0}
    p_a, p_b = scorer.percentile("a", np.array([50.0]))[0], scorer.percentile("b", np.array([0.0]))[0]
    assert scorer.score(features) == pytest.approx(0.1 + 1.0 * p_a - 1.0 * p_b)
    assert sum(scorer.contributions(features).values()) + scorer.intercept == pytest.approx(scorer.score(features))


def test_frame_scoring_matches_row_scoring(tmp_path):
    scorer = DifficultyScorer.load(_study_layout(tmp_path), FULL)
    frame = pd.DataFrame({"a": [10.0, 60.0, 90.0], "b": [-0.5, 0.0, 0.9]})
    np.testing.assert_allclose(scorer.score(frame), [scorer.score(row) for row in frame.to_dict("records")])


def test_higher_metric_means_harder(tmp_path):
    scorer = DifficultyScorer.load(_study_layout(tmp_path), SPARSE)
    assert scorer.score({"a": 90.0}) > scorer.score({"a": 10.0})


def test_arms_layout_and_missing_metrics(tmp_path):
    inner = json.loads(_study_layout(tmp_path).read_text())
    path = tmp_path / "analytic_scorer.json"
    path.write_text(json.dumps({"features": ["a", "b"], "arms": {"analytic/all": inner}}))
    with pytest.raises(KeyError):
        DifficultyScorer.load(path, SPARSE)                        # arm required for this layout
    scorer = DifficultyScorer.load(path, FULL, arm="analytic/all")
    assert scorer.metrics == ("a", "b") and scorer.missing(["a"]) == ("b",)


def test_real_frozen_scorer_if_present():
    path = "/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/frozen_final_scorer.json"
    try:
        scorer = DifficultyScorer.load(path, SPARSE)
    except FileNotFoundError:
        pytest.skip(f"study scorer not on this machine: {path}")
    assert len(scorer.metrics) == 14 and len(scorer.grids) == 30


def test_deployed_scorer_loads_and_is_the_30_metric_analytic_fit():
    from src.acquisition.features import FEATURE_NAMES
    scorer = DifficultyScorer.load()
    assert scorer.variant == FULL and len(scorer.metrics) == 30 and set(scorer.grids) == set(FEATURE_NAMES)
    sparse = DifficultyScorer.load(variant=SPARSE)
    assert 5 <= len(sparse.metrics) <= 14
    # a deep, heterogeneous beamlet scores above a shallow homogeneous one
    hard = {m: float(np.percentile(scorer.grids[m], 90)) for m in FEATURE_NAMES}
    easy = {m: float(np.percentile(scorer.grids[m], 10)) for m in FEATURE_NAMES}
    assert scorer.score(hard) > scorer.score(easy)
