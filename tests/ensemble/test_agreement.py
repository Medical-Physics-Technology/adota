"""Ensemble agreement (EXP-0014 stage 0b) on synthetic members: pair levels from
training ids, within-cell rank agreement against pooled agreement, the scan
bootstrap, the held-out check, member resolution from CSVs, and the root-disk
guard of the script. The one inference test runs a tiny model on CPU and is
marked ``slow``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import typer
import yaml

from scripts.active_learning.ensemble_agreement import assert_off_root_filesystem
from src.active_learning.ensemble.agreement import (
    LEVEL_DATA,
    LEVEL_SEED,
    LEVEL_UNKNOWN,
    agreement_matrix,
    assign_cells,
    cell_correlations,
    error_matrix,
    pair_levels,
    pair_table,
    scan_bootstrap,
)
from src.active_learning.ensemble.members import (
    EvaluationSettings,
    Member,
    held_out_table,
    resolve_member,
)
from src.adota.config import DEFAULT_GAMMA_PARAMS
from src.schemas.configs import TrainingConfig
from src.training.factory import build_adota_model
from tests.retrospective.conftest import write_dataset, write_exclusion_file

SCANS = ["s0", "s1", "s2", "s3"]
BINS = [80.0, 120.0, 160.0]   # three 20 MeV bins
PER_CELL = 20


def _metadata() -> pd.DataFrame:
    rows = [{"sample_id": f"{s}_{int(e)}_{k:02d}", "patient_key": s, "energy_mev": e + 1.0}
            for s in SCANS for e in BINS for k in range(PER_CELL)]
    return pd.DataFrame(rows)


def _member(label: str, rde: np.ndarray, ids, training=None, seed=None) -> Member:
    errors = pd.DataFrame({"sample_id": ids, "rde_pct": rde, "gpr": 1.0 - rde / (rde.max() + 1),
                           "mape_pct": rde, "dr80_mm": -rde})
    return Member(label=label, errors=errors, training_ids=training, source="csv",
                  provenance={"seed": seed})


def test_pair_level_comes_from_the_training_ids_not_the_names():
    ids = ["a", "b"]
    x = np.array([1.0, 2.0])
    members = [_member("topk_seed1", x, ids, ["t1", "t2"], 1),
               _member("topk_seed2", x, ids, ["t2", "t1"], 2),
               _member("random_seed1", x, ids, ["t1", "t3"], 1),
               _member("external", x, ids, None)]
    levels = pair_levels(members).set_index(["a", "b"])
    assert levels.loc[("topk_seed1", "topk_seed2"), "level"] == LEVEL_SEED
    assert levels.loc[("topk_seed1", "topk_seed2"), "training_jaccard"] == 1.0
    assert levels.loc[("topk_seed1", "random_seed1"), "level"] == LEVEL_DATA
    assert levels.loc[("topk_seed1", "random_seed1"), "training_jaccard"] == pytest.approx(1 / 3)
    assert levels.loc[("topk_seed1", "external"), "level"] == LEVEL_UNKNOWN


def test_within_cell_agreement_removes_what_energy_alone_explains():
    meta = _metadata()
    ids = meta["sample_id"].tolist()
    rng = np.random.default_rng(0)
    energy = meta["energy_mev"].to_numpy()
    # Both members find high energies hard, but rank records inside a cell independently.
    a = _member("a", energy + rng.normal(0, 1, len(ids)), ids)
    b = _member("b", energy + rng.normal(0, 1, len(ids)), ids)
    # c agrees with a record by record.
    c = _member("c", a.errors["rde_pct"].to_numpy() * 2.0 + 5.0, ids)
    members = [a, b, c]
    matrix = error_matrix(members, "rde_pct")
    cells = assign_cells(meta, ids)
    assert cells["cell"].nunique() == len(SCANS) * len(BINS)
    corr, cell_frame = cell_correlations(matrix, cells, min_cell_size=10)
    table = pair_table(matrix, cells, pair_levels(members), corr, top_fraction=0.1)
    row = table.set_index(["a", "b"])
    assert row.loc[("a", "b"), "pooled_spearman"] > 0.8
    assert abs(row.loc[("a", "b"), "median_cell_spearman"]) < 0.3
    assert row.loc[("a", "c"), "median_cell_spearman"] == pytest.approx(1.0)
    assert row.loc[("a", "c"), "top_jaccard_within_cell"] == pytest.approx(1.0)
    assert (table["n_cells"] == len(cell_frame)).all()
    grid = agreement_matrix(table, ["a", "b", "c"], "median_cell_spearman")
    assert np.isnan(grid.loc["a", "a"]) and grid.loc["c", "a"] == grid.loc["a", "c"]


def test_small_cells_are_skipped_and_undefined_errors_dropped():
    meta = _metadata()
    ids = meta["sample_id"].tolist()
    x = np.arange(len(ids), dtype=float)
    a, b = _member("a", x, ids), _member("b", x, ids)
    b.errors.loc[0, "dr80_mm"] = np.nan
    assert len(error_matrix([a, b], "abs_dr80_mm")) == len(ids) - 1
    cells = assign_cells(meta, ids)
    _, kept = cell_correlations(error_matrix([a, b], "rde_pct"), cells, min_cell_size=PER_CELL)
    assert len(kept) == len(SCANS) * len(BINS)
    with pytest.raises(ValueError):
        cell_correlations(error_matrix([a, b], "rde_pct"), cells, min_cell_size=PER_CELL + 1)


def test_scan_bootstrap_is_seeded_and_reports_the_contrast():
    rng = np.random.default_rng(1)
    corr = np.column_stack([0.9 + rng.normal(0, 0.02, 12), 0.5 + rng.normal(0, 0.05, 12),
                            0.4 + rng.normal(0, 0.05, 12)])
    scans = np.repeat(["s0", "s1", "s2", "s3"], 3)
    levels = [LEVEL_SEED, LEVEL_DATA, LEVEL_DATA]
    ci1, lv1 = scan_bootstrap(corr, scans, levels, n_boot=200, seed=5)
    ci2, lv2 = scan_bootstrap(corr, scans, levels, n_boot=200, seed=5)
    pd.testing.assert_frame_equal(ci1, ci2)
    pd.testing.assert_frame_equal(lv1, lv2)
    by = lv1.set_index("level")
    assert by.loc[LEVEL_SEED, "n_pairs"] == 1 and by.loc[LEVEL_DATA, "n_pairs"] == 2
    contrast = by.loc[f"{LEVEL_DATA}_minus_{LEVEL_SEED}"]
    assert contrast["ci_hi"] < 0 and contrast["mean_median_cell_spearman"] == pytest.approx(-0.45, abs=0.05)
    assert (ci1["median_cell_spearman_lo"] <= ci1["median_cell_spearman_hi"]).all()
    assert (lv1["n_scans"] == 4).all()


def test_held_out_table_counts_shared_scans():
    meta = _metadata()
    group_of = meta.set_index("sample_id")["patient_key"]
    eval_ids = meta.loc[meta["patient_key"] == "s0", "sample_id"].tolist()
    clean = meta.loc[meta["patient_key"] != "s0", "sample_id"].tolist()
    x = np.zeros(len(eval_ids))
    table = held_out_table([_member("clean", x, eval_ids, clean),
                            _member("leaky", x, eval_ids, clean + eval_ids[:3]),
                            _member("unknown", x, eval_ids, None)], eval_ids, group_of)
    row = table.set_index("label")
    assert row.loc["clean", "n_shared_groups"] == 0
    assert row.loc["leaky", "n_shared_groups"] == 1 and row.loc["leaky", "n_shared_records"] == 3
    assert pd.isna(row.loc["unknown", "n_training"])


def _settings(dataset_path="unused.h5") -> EvaluationSettings:
    return EvaluationSettings(dataset_path=str(dataset_path), gamma_params=dict(DEFAULT_GAMMA_PARAMS))


def test_resolve_member_from_an_errors_csv_reorders_to_the_evaluation_ids(tmp_path):
    ids = ["r2", "r0", "r1"]
    csv = tmp_path / "errs.csv"
    pd.DataFrame({"sample_id": ["r0", "r1", "r2", "r9"], "rde_pct": [0.0, 1.0, 2.0, 9.0]}) \
        .to_csv(csv, index=False)
    train = tmp_path / "train.csv"
    pd.DataFrame({"sample_id": ["t0", "t1"]}).to_csv(train, index=False)
    member = resolve_member({"errors_csv": str(csv), "training_ids": str(train), "label": "m"},
                            ids, _settings(), source="csv")
    assert member.errors["sample_id"].tolist() == ids
    assert member.errors["rde_pct"].tolist() == [2.0, 0.0, 1.0]
    assert member.training_ids == ["t0", "t1"] and member.source == "csv"
    assert member.provenance["errors_csv_sha256"]
    with pytest.raises(FileNotFoundError):
        resolve_member({"errors_csv": str(csv)}, ["r0", "missing"], _settings(), source="csv")
    with pytest.raises(ValueError):
        resolve_member({"errors_csv": str(csv), "typo": 1}, ids, _settings())


def test_the_script_refuses_the_root_filesystem():
    with pytest.raises(typer.BadParameter):
        assert_off_root_filesystem(Path("/definitely/not/here"), "output_dir")


TINY = {"batch_size": 2, "num_workers": 0, "compile": False, "allow_tf32": False,
        "num_transformers": 1, "num_heads": 2, "num_levels": 2, "enc_features": 8,
        "kernel_size": 3, "convolutional_steps": 1, "conv_hidden_channels": 8,
        "dropout_rate": 0.0, "num_forward": 1, "input_shape": [2, 160, 30, 30]}


@pytest.mark.slow
def test_inference_from_a_checkpoint_matches_the_evaluation_order(tmp_path):
    ids = [f"rec{i}" for i in range(4)]
    dataset = write_dataset(tmp_path / "ds.h5", ids, seed=3)
    excluded = write_exclusion_file(tmp_path / "exclude.txt")
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump({**TINY, "excluded_indexes_file": str(excluded)}))
    config = TrainingConfig(**{**TINY, "input_shape": tuple(TINY["input_shape"])})
    checkpoint = tmp_path / "model.pth"
    torch.manual_seed(0)
    torch.save({"model": build_adota_model(config, torch.device("cpu")).state_dict()}, checkpoint)
    cache = tmp_path / "cache"
    cache.mkdir()
    spec = {"checkpoint": str(checkpoint), "model_config": str(model_config), "label": "tiny"}
    member = resolve_member(spec, ["rec2", "rec0"], _settings(dataset), source="auto",
                            device=torch.device("cpu"), cache_dir=cache)
    assert member.source == "inference"
    assert member.errors["sample_id"].tolist() == ["rec2", "rec0"]
    assert np.isfinite(member.errors["rde_pct"]).all()
    assert (cache / "tiny_errors.csv").exists() and member.provenance["checkpoint_sha256"]
    # The cached CSV is reused on the next resolve instead of inferring again.
    again = resolve_member({**spec, "errors_csv": str(cache / "tiny_errors.csv")}, ["rec0"],
                           _settings(dataset), source="csv")
    assert again.source == "csv"
    assert again.errors["rde_pct"].item() == pytest.approx(member.errors["rde_pct"].iloc[1])
