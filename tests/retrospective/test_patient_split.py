"""The patient-held-out validation split (EXP-0012)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.active_learning.retrospective.config import RetroConfig
from src.active_learning.retrospective.dataset import Splits
from src.active_learning.retrospective.loop import load_run_inputs, prepare_splits
from src.active_learning.retrospective.patient_split import (
    assert_groups_disjoint,
    build_patient_splits,
    select_held_out_groups,
)
from tests.retrospective.test_loop import make_config


def _metadata(n_thorax: int = 10, n_pelvis: int = 4, per_group: int = 5) -> pd.DataFrame:
    rows = []
    for source, n_groups in (("thorax", n_thorax), ("pelvis", n_pelvis)):
        for g in range(n_groups):
            for r in range(per_group):
                rows.append({"sample_id": f"{source}{g:02d}_{r}", "patient_key": f"{source}{g:02d}",
                             "source_dataset": source})
    return pd.DataFrame(rows)


def test_held_out_groups_per_stratum_and_seeded():
    meta = _metadata()
    held = select_held_out_groups(meta, "patient_key", "source_dataset",
                                  {"thorax": 3, "pelvis": 1}, seed=42)
    assert len(held["thorax"]) == 3 and len(held["pelvis"]) == 1
    assert all(g.startswith("thorax") for g in held["thorax"])
    assert held == select_held_out_groups(meta.sample(frac=1.0, random_state=0), "patient_key",
                                          "source_dataset", {"thorax": 3, "pelvis": 1}, seed=42)
    assert held != select_held_out_groups(meta, "patient_key", "source_dataset",
                                          {"thorax": 3, "pelvis": 1}, seed=43)


@pytest.mark.parametrize("counts, match", [
    ({"thorax": 3}, "missing"),
    ({"thorax": 3, "pelvis": 1, "abdomen": 1}, "unknown"),
    ({"thorax": 10, "pelvis": 1}, "hold out between"),
    ({"thorax": 0, "pelvis": 1}, "hold out between"),
])
def test_held_out_groups_rejects_bad_counts(counts, match):
    with pytest.raises(ValueError, match=match):
        select_held_out_groups(_metadata(), "patient_key", "source_dataset", counts, seed=1)


def test_a_group_spanning_two_strata_is_an_error():
    meta = _metadata()
    meta.loc[0, "source_dataset"] = "pelvis"
    with pytest.raises(ValueError, match="span several"):
        select_held_out_groups(meta, "patient_key", "source_dataset", {"thorax": 1, "pelvis": 1}, 1)


def test_missing_group_column_names_the_fix():
    with pytest.raises(KeyError, match="v3 HDF5"):
        select_held_out_groups(_metadata().drop(columns="patient_key"), "patient_key",
                               "source_dataset", {"thorax": 1, "pelvis": 1}, 1)


def test_patient_splits_hold_out_whole_groups():
    meta = _metadata()
    ids = meta["sample_id"].tolist()
    splits, held = build_patient_splits(ids, meta, "patient_key", "source_dataset",
                                        {"thorax": 3, "pelvis": 1}, val_seed=42,
                                        initial_fraction=0.2, initial_seed=7)
    held_groups = set(held["thorax"]) | set(held["pelvis"])
    group_of = meta.set_index("sample_id")["patient_key"]
    assert set(group_of[splits.validation]) == held_groups
    assert not held_groups & set(group_of[splits.training])
    assert len(splits.validation) == 4 * 5 and len(splits.training) == 10 * 5
    assert splits.validation == [r for r in ids if group_of[r] in held_groups]   # file order kept
    assert set(splits.initial) <= set(splits.training) and len(splits.initial) == 10


def test_disjointness_check_catches_a_shared_group():
    meta = _metadata()
    ids = meta["sample_id"].tolist()
    leaky = Splits(validation=ids[:3], training=ids[3:], initial=ids[3:5])
    with pytest.raises(AssertionError, match="both V and T"):
        assert_groups_disjoint(leaky, meta, "patient_key")


def test_config_validates_the_split_mode():
    with pytest.raises(ValueError, match="val_split"):
        RetroConfig(val_split="scan")
    with pytest.raises(ValueError, match="val_groups_per_stratum"):
        RetroConfig(val_split="patient")
    assert RetroConfig().val_split == "record"


def _patient_config(tmp_path) -> RetroConfig:
    cfg = make_config(tmp_path)
    # The synthetic dataset carries no v3 attrs, so a provenance map supplies the
    # groups as `patient` and the strata as `anatomy`: three patients per anatomy.
    ids = pd.read_csv(cfg.record_provenance_csv)["sample_id"].tolist()
    anatomy = ["thorax"] * 13 + ["pelvic"] * 13
    patient = [f"{a}{i % 3}" for i, a in enumerate(anatomy)]
    pd.DataFrame({"sample_id": ids, "anatomy": anatomy, "patient_key": patient}).to_csv(
        cfg.record_provenance_csv, index=False)
    cfg.val_split = "patient"
    cfg.val_group_column = "patient"
    cfg.val_stratify_column = "anatomy"
    cfg.val_groups_per_stratum = {"thorax": 1, "pelvic": 1}
    return cfg


def test_prepare_splits_patient_mode_end_to_end(tmp_path):
    cfg = _patient_config(tmp_path)
    summary = prepare_splits(cfg)
    assert summary["val_split"] == "patient"
    assert summary["n_groups_validation"] == 2 and summary["n_groups_training"] == 4
    inputs = load_run_inputs(cfg)
    group_of = inputs.metadata.set_index("sample_id")["patient"]
    assert not set(group_of[inputs.splits.validation]) & set(group_of[inputs.splits.training])
    assert set(inputs.metadata["sample_id"]) == set(inputs.splits.training + inputs.splits.validation)
    written = json.loads((tmp_path / "splits" / "splits.json").read_text())
    assert sorted(written["held_out_groups"]) == ["pelvic", "thorax"]


def test_load_run_inputs_rejects_a_tampered_patient_split(tmp_path):
    cfg = _patient_config(tmp_path)
    prepare_splits(cfg)
    val_csv = tmp_path / "splits" / "validation_ids.csv"
    train_csv = tmp_path / "splits" / "training_ids.csv"
    val = pd.read_csv(val_csv)
    train = pd.read_csv(train_csv)
    # Move one validation record into training: its patient is now on both sides.
    pd.concat([train, val.iloc[:1]]).to_csv(train_csv, index=False)
    val.iloc[1:].to_csv(val_csv, index=False)
    with pytest.raises(AssertionError, match="both V and T"):
        load_run_inputs(cfg)


def test_record_mode_is_unchanged(tmp_path):
    cfg = make_config(tmp_path)
    summary = prepare_splits(cfg)
    assert summary["val_split"] == "record" and "held_out_groups" not in summary
    assert summary["n_validation"] == 4 and summary["n_training"] == 20
