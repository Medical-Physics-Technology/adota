"""End to end on CPU with a synthetic HDF5 set and a tiny model: the three
stages, the schedule, the manifests, the cycle-0 restore and the resume.

The two end-to-end tests are marked ``slow``: the metric set runs gamma on the
full crop on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.active_learning.retrospective.compare import (
    boundary_table,
    check_consistency,
    epochs_to_quality,
    fingerprints,
    read_run,
)
from src.active_learning.retrospective.loop import (
    RetroConfig,
    load_run_inputs,
    prepare_splits,
    run_cycle0,
    run_strategy,
)

from .conftest import write_dataset, write_exclusion_file

IDS = [f"rec{i:03d}" for i in range(26)]
TINY_TRAINING = {
    "batch_size": 4, "num_workers": 0, "compile": False, "allow_tf32": False,
    "learning_rate": 1e-3, "num_transformers": 1, "num_heads": 2, "num_levels": 2,
    "enc_features": 8, "kernel_size": 3, "convolutional_steps": 1, "conv_hidden_channels": 8,
    "dropout_rate": 0.0, "num_forward": 1, "input_shape": [2, 160, 30, 30],
}


def make_config(tmp_path: Path, strategy: str = "random") -> RetroConfig:
    dataset = write_dataset(tmp_path / "ds.h5", IDS, seed=1)
    excluded = write_exclusion_file(tmp_path / "exclude.txt", ["rec025", "rec024"])
    provenance = tmp_path / "prov.csv"
    pd.DataFrame({"sample_id": IDS, "anatomy": ["thorax"] * 13 + ["pelvic"] * 13,
                  "patient_key": [f"P{i % 4}" for i in range(26)]}).to_csv(provenance, index=False)
    return RetroConfig(
        experiment="EXP-TEST", dataset_path=str(dataset), exclude_indexes_path=str(excluded),
        record_provenance_csv=str(provenance), splits_dir=str(tmp_path / "splits"),
        runs_dir=str(tmp_path / "runs"), val_fraction=0.15, initial_fraction=0.20,
        batch_fraction=0.10, n_cycles=2, epochs_per_cycle=1, eval_every_n_epochs=1,
        eval_subsample_size=2, strategy=strategy, seed=7, device_index=-1,
        checkpoint_every_n_epochs=1, scorer={"name": "difficulty", "n_workers": 1},
        training=dict(TINY_TRAINING))


@pytest.mark.slow
def test_the_three_stages_end_to_end(tmp_path):
    cfg = make_config(tmp_path, "random")
    summary = prepare_splits(cfg)
    # 26 records, 2 excluded -> D = 24; V = round(3.6) = 4; T = 20; cycle-0 = 4; N = 2.
    assert summary["n_after_exclusion"] == 24 and summary["n_validation"] == 4
    assert summary["n_training"] == 20 and summary["n_initial"] == 4
    inputs = load_run_inputs(cfg)
    assert inputs.batch_size == 2 and len(inputs.subsample_ids) == 2
    assert not set(inputs.splits.validation) & set(inputs.splits.pool)

    cycle0_dir = tmp_path / "runs" / "cycle0"
    cycle0_dir.mkdir(parents=True)
    checkpoint = run_cycle0(cfg, cycle0_dir)
    manifest = json.loads((cycle0_dir / "manifest.json").read_text())
    assert manifest["cycles"][0]["n_train"] == 4 and checkpoint.exists()
    assert manifest["cycle0_checkpoint_sha256"]
    rows = [json.loads(line) for line in (cycle0_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["cycle_boundary"] and rows[0]["metrics_full"]["n"] == 4.0

    run_dir = tmp_path / "runs" / "random"
    run_dir.mkdir()
    cycles = run_strategy(cfg, run_dir, cycle0_dir)
    assert [c["n_train"] for c in cycles] == [4, 6, 8]
    assert [c["cumulative_epochs"] for c in cycles] == [1, 2, 3]
    assert all(c["schedule_ok"] for c in cycles[1:])
    training_ids = pd.read_csv(run_dir / "cycles" / "cycle_02" / "training_ids.csv")["sample_id"]
    assert len(training_ids) == 8 and training_ids.is_unique
    assert not set(training_ids) & set(inputs.splits.validation)
    log = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["cumulative_epoch"] for r in log] == [0, 1, 2]
    assert log[0]["inherited_from"] == str(cycle0_dir)
    assert all(r["n_train"] == n for r, n in zip(log, [4, 6, 8]))
    fingerprint = cycles[1]["selection_fingerprint"]
    assert fingerprint["n_selected"] == 2 and set(fingerprint["selected"]) >= {"energy", "patient"}
    assert (run_dir / "validation").glob("cycle_02_full_gamma_2pct_2mm_cutoff10pct.csv")

    # A rerun on the same directory resumes: nothing retrained, same cycles reported.
    before = torch.load(cycles[2]["checkpoint"], map_location="cpu", weights_only=False)
    again = run_strategy(cfg, run_dir, cycle0_dir)
    assert [c["cumulative_epochs"] for c in again] == [1, 2, 3]
    after = torch.load(cycles[2]["checkpoint"], map_location="cpu", weights_only=False)
    assert all(torch.equal(before["model"][k], after["model"][k]) for k in before["model"])

    # The comparison reads it back and the numbers line up with the manifest.
    run = read_run(run_dir)
    check_consistency([run, read_run(run_dir, "again")])
    table = boundary_table([run])
    assert table["n_train"].tolist() == [4, 6, 8]
    assert np.isfinite(table["full_gpr_mean"]).all()
    assert len(fingerprints([run])[run.label]) == 2
    e2q = epochs_to_quality([run], {"sub_gpr_mean": (0.0, "ge"), "sub_mape_pct_mean": (-1.0, "le")})
    assert e2q.loc[e2q["metric"] == "sub_gpr_mean", "reached"].item()
    assert not e2q.loc[e2q["metric"] == "sub_mape_pct_mean", "reached"].item()


@pytest.mark.slow
def test_a_score_strategy_scores_the_pool_each_cycle(tmp_path):
    cfg = make_config(tmp_path, "score_topk")
    prepare_splits(cfg)
    cycle0_dir = tmp_path / "runs" / "cycle0"
    cycle0_dir.mkdir(parents=True)
    run_cycle0(cfg, cycle0_dir)
    run_dir = tmp_path / "runs" / "topk"
    run_dir.mkdir()
    cycles = run_strategy(cfg, run_dir, cycle0_dir)
    first = pd.read_csv(run_dir / "cycles" / "cycle_01" / "pool_scores.csv")
    second = pd.read_csv(run_dir / "cycles" / "cycle_02" / "pool_scores.csv")
    assert len(first) == 16 and len(second) == 14          # the pool shrinks by N each cycle
    chosen = pd.read_csv(run_dir / "cycles" / "cycle_01" / "selection.csv")
    assert set(chosen["sample_id"]) == set(first.nlargest(2, "score")["sample_id"])
    assert cycles[1]["scoring_seconds"] > 0 and cycles[1]["score_distribution"]["n_scored"] == 16


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_data_fraction_subsamples_d_before_any_split(tmp_path, fraction):
    cfg = make_config(tmp_path)
    cfg.data_fraction = fraction
    summary = prepare_splits(cfg)
    assert summary["n_after_exclusion"] == 24
    assert summary["n_after_data_fraction"] == round(24 * fraction)
    inputs = load_run_inputs(cfg)
    total = len(inputs.splits.validation) + len(inputs.splits.training)
    assert total == round(24 * fraction)
    again = prepare_splits(cfg)
    assert again["fingerprint"] == summary["fingerprint"]        # same seed, same subset
    if fraction < 1.0:
        cfg.data_fraction_seed += 1
        assert prepare_splits(cfg)["fingerprint"] != summary["fingerprint"]
