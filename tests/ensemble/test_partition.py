"""The EXP-0014 scan partition: F / V / R, the cross-fitting folds and the HPO split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.active_learning.ensemble.partition import (
    HPO_NONE,
    HPO_TRAIN,
    HPO_VAL,
    ROLE_F,
    ROLE_FV,
    ROLE_R,
    ROLE_V,
    assign_folds,
    check_partition,
    fold_ids,
    fold_summary,
    hpo_split,
    record_table,
    scan_table,
    scans_from_frozen_ids,
    scans_from_patients,
)


def _metadata(seed: int = 0) -> pd.DataFrame:
    """20 thorax and 9 pelvic scans with uneven record counts."""
    rng = np.random.RandomState(seed)
    rows = []
    for anatomy, n_scans in (("thorax", 20), ("pelvic_abdominal", 9)):
        for s in range(n_scans):
            scan = f"{anatomy[:2]}{s:02d}"
            for r in range(int(rng.randint(3, 40))):
                rows.append({"sample_id": f"{scan}_r{r}", "patient_key": scan,
                             "patient": f"{anatomy}|{scan}", "anatomy": anatomy,
                             "energy_mev": float(rng.uniform(70, 230))})
    return pd.DataFrame(rows)


def _partition(meta: pd.DataFrame, n_folds: int = 4, seed: int = 7):
    f = scans_from_frozen_ids(["th00_r0", "th01_r1", "pe00_r0"], meta)
    v = scans_from_patients(["thorax|th01", "thorax|th02", "pelvic_abdominal|pe01"], meta)
    scans = scan_table(meta, f, v)
    scans["fold"] = assign_folds(scans, n_folds, seed)
    scans["hpo"] = hpo_split(scans, {"thorax": 3, "pelvic_abdominal": 1}, seed)
    return scans, record_table(meta, scans)


def test_roles_come_from_whole_scans_and_mark_the_overlap():
    scans, records = _partition(_metadata())
    role = scans.set_index("scan")["role"]
    assert role["th00"] == ROLE_F and role["pe00"] == ROLE_F
    assert role["th01"] == ROLE_FV
    assert role["th02"] == ROLE_V and role["pe01"] == ROLE_V
    assert (scans["role"] == ROLE_R).sum() == len(scans) - 5
    # one frozen id puts every record of its scan in F
    assert (records.loc[records["scan"] == "th00", "role"] == ROLE_F).all()


def test_unknown_ids_and_patients_are_errors():
    meta = _metadata()
    with pytest.raises(KeyError):
        scans_from_frozen_ids(["th00_r0", "nope"], meta)
    with pytest.raises(KeyError):
        scans_from_patients(["thorax|zz99"], meta)


def test_folds_are_balanced_per_anatomy_and_cover_every_record_once():
    scans, records = _partition(_metadata())
    check_partition(records, 4)
    per = scans.groupby(["anatomy", "fold"]).size().unstack(fill_value=0)
    assert (per.max(axis=1) - per.min(axis=1) <= 1).all()
    evaluated = sorted(sum((fold_ids(records, k)["eval"] for k in range(4)), []))
    assert evaluated == sorted(records["sample_id"])
    for k in range(4):
        split = fold_ids(records, k)
        train_scans = set(records.loc[records["sample_id"].isin(split["train"]), "scan"])
        eval_scans = set(records.loc[records["sample_id"].isin(split["eval"]), "scan"])
        assert not train_scans & eval_scans


def test_folds_depend_on_the_seed_and_the_scans_not_the_row_order():
    meta = _metadata()
    scans, _ = _partition(meta)
    shuffled = scans.sample(frac=1.0, random_state=3)
    again = assign_folds(shuffled, 4, 7)
    assert again.reindex(scans.index).tolist() == scans["fold"].tolist()
    other = assign_folds(scans, 4, 8)
    assert other.tolist() != scans["fold"].tolist()


def test_hpo_split_uses_r_scans_only_with_the_requested_counts():
    scans, records = _partition(_metadata())
    assert (scans.loc[scans["role"] != ROLE_R, "hpo"] == HPO_NONE).all()
    val = scans[scans["hpo"] == HPO_VAL]
    assert val["anatomy"].value_counts().to_dict() == {"thorax": 3, "pelvic_abdominal": 1}
    assert ((scans["role"] == ROLE_R) == scans["hpo"].isin([HPO_TRAIN, HPO_VAL])).all()


def test_check_partition_rejects_a_leaking_hpo_split_and_a_split_scan():
    _, records = _partition(_metadata())
    leaking = records.copy()
    leaking.loc[leaking["role"] == ROLE_F, "hpo"] = HPO_TRAIN
    with pytest.raises(ValueError):
        check_partition(leaking, 4)
    split = records.copy()
    first = split.index[split["scan"] == "th05"][0]
    split.loc[first, "fold"] = (split.loc[first, "fold"] + 1) % 4
    with pytest.raises(ValueError):
        check_partition(split, 4)


def test_too_few_scans_for_the_folds_is_an_error():
    meta = _metadata()
    scans = scan_table(meta, [], [])
    with pytest.raises(ValueError):
        assign_folds(scans, 10, 0)


def test_fold_summary_counts_f_and_v_scans():
    _, records = _partition(_metadata())
    summary = fold_summary(records)
    assert summary["n_records"].sum() == len(records)
    assert summary["n_f_scans"].sum() == 3      # th00, th01 (F+V), pe00
    assert summary["n_v_scans"].sum() == 3      # th01 (F+V), th02, pe01
