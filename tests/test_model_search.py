"""Tests of :mod:`src.model_search` (the single-model HPO behind
``scripts/model_hpo_training.py``): the space, the candidate checks, the record
sets and the winner rule. CPU only, no dataset, no training."""

from __future__ import annotations

from pathlib import Path

import h5py
import optuna
import pandas as pd
import pytest

from src.model_search.runs import prepare_ids, resolve_ids, scan_overlap, select_winner
from src.model_search.space import build_candidate, rejection, suggest, validate_space
from src.model_search.study import n_trained, top_trials
from src.schemas.configs import TrainingConfig

V11 = dict(enc_features=32, num_levels=4, convolutional_steps=1, conv_hidden_channels=64, dropout_rate=0.1,
           num_transformers=1, num_heads=4, learning_rate=5e-4)


def _base() -> TrainingConfig:
    return TrainingConfig(input_shape=[2, 160, 30, 30], kernel_size=3, causal=True, num_forward=2)


def test_validate_space_rejects_malformed_and_unknown():
    validate_space({"num_levels": {"choices": [3, 4]}, "warmup_epochs": {"int": [0, 3]},
                    "dim_feedforward_factor": {"choices": [1, 2]}})
    with pytest.raises(ValueError, match="not a TrainingConfig field"):
        validate_space({"no_such_field": {"choices": [1]}})
    with pytest.raises(ValueError, match="exactly one of"):
        validate_space({"num_levels": {"choices": [3], "int": [1, 2]}})
    with pytest.raises(ValueError, match="invalid range"):
        validate_space({"learning_rate": {"loguniform": [0.0, 1e-3]}})
    with pytest.raises(ValueError, match="listed before"):
        validate_space({"weight_standardization": {"choices": [True], "when": {"norm_layer": "group"}},
                        "norm_layer": {"choices": ["group"]}})


def test_suggest_honours_conditions():
    space = {"norm_layer": {"choices": ["batch"]},
             "weight_standardization": {"choices": [False, True], "when": {"norm_layer": "group"},
                                        "otherwise": False},
             "learning_rate": {"loguniform": [1e-4, 1e-3]}}
    trial = optuna.create_study().ask()
    params = suggest(trial, space)
    assert params["norm_layer"] == "batch" and params["weight_standardization"] is False
    assert "weight_standardization" not in trial.params          # never sampled
    assert 1e-4 <= params["learning_rate"] <= 1e-3


def test_candidate_reproduces_v11_and_applies_the_cap():
    c = build_candidate(_base(), V11)
    assert c.n_params == 14_927_651                               # v11's parameter count
    assert c.token_size == 128 and c.config.dim_feedforward is None
    assert rejection(c, 2 * 14_927_651) is None
    assert "over the cap" in rejection(c, 10_000_000)
    wide = build_candidate(_base(), {**V11, "dim_feedforward_factor": 2, "warmup_epochs": 2})
    assert wide.config.dim_feedforward == 256 and wide.warmup_epochs == 2
    assert wide.n_params > c.n_params


def test_candidate_refuses_heads_that_do_not_divide_the_token():
    with pytest.raises(ValueError, match="does not divide"):
        build_candidate(_base(), {**V11, "num_heads": 3})


def test_resolve_ids_from_csv_and_table(tmp_path: Path):
    pd.DataFrame({"sample_id": ["a", "b"]}).to_csv(tmp_path / "ids.csv", index=False)
    pd.DataFrame({"sample_id": ["a", "b", "c"], "role": ["R", "F", "F+V"]}).to_csv(tmp_path / "records.csv",
                                                                                index=False)
    assert resolve_ids("ids.csv", tmp_path) == ["a", "b"]
    assert resolve_ids({"table": "records.csv", "column": "role", "values": ["F", "F+V"]}, tmp_path) == ["b", "c"]


def test_prepare_ids_drops_excluded_and_high_energy(tmp_path: Path):
    h5 = tmp_path / "d.h5"
    with h5py.File(h5, "w") as f:
        for rid, energy in (("a", 100.0), ("b", 260.0), ("c", 150.0)):
            f.create_group(rid).attrs["energy_mev"] = energy
    (tmp_path / "exclude.txt").write_text("c\n")
    sets, counts = prepare_ids({"train": ["a", "b", "c"]}, h5, tmp_path / "exclude.txt", 250.0, None)
    assert sets["train"] == ["a"]
    row = counts.iloc[0]
    assert (row["listed"], row["excluded"], row["above_energy_cut"], row["kept"]) == (3, 1, 1, 1)


def test_scan_overlap_finds_shared_scans():
    scans = {"a": "s1", "b": "s2", "c": "s1"}
    out = scan_overlap({"train": ["a", "b"], "test": ["c"]}, scans, [("train", "test")])
    assert out == {"train|test": ["s1"]}


def test_select_winner_lowest_loss_unless_the_gamma_guard_fires():
    rows = [{"label": "t1", "final_loss": 0.030, "gpr_mean": 0.950},
            {"label": "t2", "final_loss": 0.031, "gpr_mean": 0.954},
            {"label": "t3", "final_loss": 0.040, "gpr_mean": 0.940}]
    assert select_winner(rows, gpr_guard_pp=0.5)["label"] == "t1"   # +0.4 pp is inside the guard
    rows[1]["gpr_mean"] = 0.957
    picked = select_winner(rows, gpr_guard_pp=0.5)
    assert picked["label"] == "t2" and "gamma guard" in picked["rule"]
    with pytest.raises(ValueError):
        select_winner([], 0.5)


def test_trained_count_and_top_trials():
    study = optuna.create_study()
    for value, trained in ((0.3, True), (0.1, True), (None, False), (0.2, True)):
        trial = study.ask()
        trial.set_user_attr("trained", trained)
        if value is None:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, value)
    assert n_trained(study) == 3
    assert [t.value for t in top_trials(study, 2)] == [0.1, 0.2]
