"""The input-only guarantee of the scoring path.

Scoring may read the CT, the flux and the energy of a record, plus the analytic
surrogate dose. It must never read the stored ground-truth dose, nor anything
derived from it. Two tests pin that: one hands the scoring path a record whose
``dose`` raises when opened, the other scores the same records with the dose
replaced by noise and demands identical scores.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from src.acquisition.features import FEATURE_NAMES
from src.active_learning.retrospective.scoring import (
    DifficultyPoolScorer,
    build_scorer,
    input_only_features,
    read_stored_inputs,
    score_distribution,
)

from .conftest import write_dataset

IDS = ["a", "b", "c"]
ENERGIES = [100.0, 120.0, 140.0]


class GuardedGroup:
    """An HDF5 group whose ``dose`` cannot be opened."""

    def __init__(self, group):
        self._group = group
        self.attrs = group.attrs

    def __getitem__(self, key):
        if key == "dose":
            raise AssertionError("the scoring path opened the ground-truth dose")
        return self._group[key]


def test_input_only_features_never_open_the_dose(tmp_path):
    path = write_dataset(tmp_path / "ds.h5", IDS, energies=ENERGIES)
    with h5py.File(path, "r") as handle:
        record = read_stored_inputs(GuardedGroup(handle["a"]), "a")
        assert record.dose is None
        features = input_only_features(GuardedGroup(handle["b"]), "b")
    assert features is not None
    assert all(np.isfinite(features[name]) for name in FEATURE_NAMES)
    assert features["mode"] == "analytic"


def test_scores_do_not_depend_on_the_stored_dose(tmp_path):
    real = write_dataset(tmp_path / "real.h5", IDS, seed=3, energies=ENERGIES, dose_kind="bragg")
    noise = write_dataset(tmp_path / "noise.h5", IDS, seed=3, energies=ENERGIES,
                          dose_kind="garbage")
    pool = pd.DataFrame({"sample_id": IDS, "energy_mev": ENERGIES})
    scored_real = DifficultyPoolScorer(real, n_workers=1).score(pool)
    scored_noise = DifficultyPoolScorer(noise, n_workers=1).score(pool)
    assert scored_real["scored"].all()
    assert np.isfinite(scored_real["score"]).all()
    np.testing.assert_array_equal(scored_real["score"], scored_noise["score"])
    assert scored_real["sample_id"].tolist() == IDS
    summary = score_distribution(scored_real)
    assert summary["n_scored"] == 3 and summary["p00"] <= summary["p50"] <= summary["p100"]


def test_records_above_250_mev_are_left_unscored(tmp_path):
    path = write_dataset(tmp_path / "ds.h5", ["hi", "lo"], energies=[260.0, 110.0])
    pool = pd.DataFrame({"sample_id": ["hi", "lo"]})
    scored = DifficultyPoolScorer(path, n_workers=1).score(pool)
    assert not bool(scored.loc[0, "scored"]) and np.isnan(scored.loc[0, "score"])
    assert bool(scored.loc[1, "scored"])


def test_scorer_registry(tmp_path):
    path = write_dataset(tmp_path / "ds.h5", ["a"])
    assert build_scorer("difficulty", path, n_workers=1).name == "difficulty"
    with pytest.raises(ValueError, match="unknown scorer"):
        build_scorer("oracle", path)
