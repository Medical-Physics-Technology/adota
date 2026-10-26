"""The EXP-0014 HPO: the search space and its Sobol trials under a parameter cap,
successive halving, member selection, the trial config guard, and (slow) a
training segment on a tiny synthetic HDF5 that resumes from its checkpoint."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest
import torch

from src.active_learning.retrospective.validation import MetricSettings
from src.adota.config import DEFAULT_GAMMA_PARAMS, DEFAULT_SCALE
from src.hpo.halving import n_promoted, promote, rank, rung_budgets, schedule, total_epochs
from src.hpo.selection import finalists, member_roles, results_table
from src.hpo.space import decode, draw_trials, encode, unit_distance
from src.hpo.trial import best_so_far, count_parameters, read_segment, run_segment, trial_config
from tests.retrospective.conftest import write_dataset, write_exclusion_file

SPACE = {"enc_features": {"choices": [16, 32, 48]},
         "learning_rate": {"loguniform": [2e-4, 1e-3]},
         "dropout_rate": {"uniform": [0.0, 0.2]}}


def test_decode_and_encode_round_trip_and_cover_the_choices():
    values = decode([0.5, 0.0, 0.5], SPACE)
    assert values == {"enc_features": 32, "learning_rate": pytest.approx(2e-4), "dropout_rate": 0.1}
    assert decode([0.999, 0.999, 0.0], SPACE)["enc_features"] == 48
    u = encode({"enc_features": 48, "learning_rate": 1e-3, "dropout_rate": 0.0}, SPACE)
    assert u == pytest.approx([5 / 6, 1.0, 0.0])
    with pytest.raises(ValueError):
        encode({"enc_features": 20, "learning_rate": 1e-3, "dropout_rate": 0.0}, SPACE)
    assert unit_distance([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)


def test_draw_trials_is_deterministic_respects_the_cap_and_puts_fixed_first():
    def count(params):
        return params["enc_features"] * 1000

    fixed = [{"enc_features": 16, "learning_rate": 5e-4, "dropout_rate": 0.1}]
    a = draw_trials("transformer", SPACE, 6, seed=7, count_params=count, max_params=32_000, fixed=fixed)
    b = draw_trials("transformer", SPACE, 6, seed=7, count_params=count, max_params=32_000, fixed=fixed)
    assert [t.params for t in a] == [t.params for t in b]
    assert a[0].trial_id == "transformer_fixed0" and a[0].sobol_index is None
    assert all(t.n_params <= 32_000 for t in a)
    assert all(t.params["enc_features"] != 48 for t in a)
    assert len({t.trial_id for t in a}) == 6
    c = draw_trials("transformer", SPACE, 6, seed=8, count_params=count, max_params=32_000)
    assert [t.params for t in c] != [t.params for t in a[1:]] + [None]
    with pytest.raises(RuntimeError):
        draw_trials("x", SPACE, 5, seed=1, count_params=count, max_params=1, max_draws=64)
    with pytest.raises(ValueError):
        draw_trials("x", SPACE, 5, seed=1, count_params=count, max_params=1,
                    fixed=[{"enc_features": 48, "learning_rate": 5e-4, "dropout_rate": 0.0}])


def test_successive_halving_budgets_promotions_and_cost():
    assert rung_budgets(3, 3, 3) == [3, 9, 27]
    assert schedule(range(24), 3, 3) == {0: 24, 1: 8, 2: 2}
    assert total_epochs(24, [3, 9, 27], 3) == 24 * 3 + 8 * 6 + 2 * 18
    assert n_promoted(2, 3) == 1
    objectives = {"a": 0.3, "b": 0.1, "c": float("nan"), "d": 0.1, "e": 0.5, "f": 0.2}
    assert rank(objectives) == ["b", "d", "f", "a", "e", "c"]
    assert promote(objectives, 3) == ["b", "d"]
    assert promote({"x": float("nan")}, 3) == []


def _table():
    trials, segments = [], {}
    for mixer, offset in (("transformer", 0.0), ("mlstm", 0.05), ("none", 0.1)):
        for k in range(6):
            tid = f"{mixer}_{k}"
            trials.append({"trial_id": tid, "mixer": mixer, "n_params": 1, "params": {"k": k},
                           "extra": {"fixed": False}})
            segments[tid] = {3: {"objective": offset + 0.1 * k, "failed": None}}
            if k < 2:
                segments[tid][9] = {"objective": offset + 0.05 * (2 - k), "failed": None}
    return trials, results_table(trials, segments, [3, 9])


def test_roles_take_the_best_of_the_top_rung_and_the_farthest_promoted_transformer():
    trials, table = _table()
    points = {t["trial_id"]: [0.1 * int(t["trial_id"][-1])] for t in trials}
    roles = member_roles(table, points)
    assert roles["C0"] == "transformer_1" and roles["X0"] == "mlstm_1" and roles["U0"] == "none_1"
    assert roles["C1"] == "transformer_0"
    assert finalists(table, roles) == sorted(f"{m}_{k}" for m in ("transformer", "mlstm", "none")
                                             for k in (0, 1))


def test_best_so_far_and_the_trial_config_guards():
    rows = [{"epoch": e, "val_loss": {"loss_combined_mean": v}} for e, v in enumerate([3, 1, 2, 0.5])]
    assert best_so_far(rows, 3) == {"objective": 1.0, "objective_epoch": 1}
    assert best_so_far(rows, 4)["objective_epoch"] == 3
    assert math.isnan(best_so_far([], 3)["objective"])
    base = {"batch_size": 2, "adaptive_after_epoch": 10_000}
    config = trial_config(base, {"enc_features": 16}, dataset_path="d.h5", excluded_indexes_file="x",
                          mixer="mlstm")
    assert config.sequence_mixer == "mlstm" and config.enc_features == 16
    with pytest.raises(KeyError):
        trial_config(base, {"no_such_field": 1}, dataset_path="d", excluded_indexes_file="x", mixer="none")
    with pytest.raises(ValueError):
        trial_config({**base, "adaptive_after_epoch": 5}, {}, dataset_path="d", excluded_indexes_file="x",
                     mixer="none")
    assert count_parameters(config) > count_parameters(
        trial_config(base, {"enc_features": 16}, dataset_path="d", excluded_indexes_file="x", mixer="none"))


TINY = {"batch_size": 2, "num_workers": 0, "compile": False, "allow_tf32": False,
        "num_transformers": 1, "num_heads": 2, "num_levels": 2, "enc_features": 8,
        "kernel_size": 3, "convolutional_steps": 1, "conv_hidden_channels": 8,
        "dropout_rate": 0.0, "num_forward": 1, "input_shape": [2, 160, 30, 30]}


@pytest.mark.slow
@pytest.mark.parametrize("mixer", ["transformer", "mlstm"])
def test_a_segment_trains_checkpoints_and_resumes(tmp_path, mixer):
    ids = [f"rec{i}" for i in range(6)]
    dataset = write_dataset(tmp_path / "ds.h5", ids, seed=3)
    excluded = write_exclusion_file(tmp_path / "exclude.txt")
    config = trial_config(TINY, {"learning_rate": 1e-3}, dataset_path=str(dataset),
                          excluded_indexes_file=str(excluded), mixer=mixer)
    settings = MetricSettings(scale=dict(DEFAULT_SCALE), gamma_params=dict(DEFAULT_GAMMA_PARAMS))
    trial_dir = tmp_path / "trial"
    device = torch.device("cpu")
    first = run_segment(config, trial_dir, ids[:4], ids[4:], 1, device, settings)
    assert first["start_epoch"] == 0 and first["objective_epoch"] == 0
    second = run_segment(config, trial_dir, ids[:4], ids[4:], 3, device, settings)
    assert second["start_epoch"] == 1 and second["end_epoch"] == 3
    rows = [json.loads(line) for line in (trial_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert second["objective"] == min(r["val_loss"]["loss_combined_mean"] for r in rows)
    assert read_segment(trial_dir, 1)["end_epoch"] == 1
    assert (trial_dir / "checkpoints" / "best.pth").exists()
    assert pd.notna(second["objective"])


def test_a_resolved_mapping_config_is_written_as_readable_yaml(tmp_path):
    """The HPO workers read the driver's config.yaml back; a mapping must not be
    written as its repr (the first smoke run failed on exactly that)."""
    import yaml

    from src.training.run_dir import save_resolved_config

    cfg = {"training": {"batch_size": 56, "scale": {"min_ds": 0.0}}, "mixers": ["none"], "x": None}
    save_resolved_config(cfg, tmp_path / "config.yaml")
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == cfg
