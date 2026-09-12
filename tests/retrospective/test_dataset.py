"""The exclusion list, the order of operations, the split files and the schedule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.active_learning.retrospective.dataset import (
    SplitSpec,
    apply_exclusions,
    assert_schedule,
    batch_size_from_fraction,
    build_splits,
    cross_check_exclusions,
    read_exclusion_list,
    read_splits,
    record_metadata,
    scheduled_size,
    write_splits,
)
from src.training.data import train_val_split

from .conftest import write_dataset

IDS = [f"r{i:04d}" for i in range(1000)]


def test_missing_exclusion_list_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="refusing to run on an unfiltered set"):
        read_exclusion_list(tmp_path / "missing.txt")


def test_apply_exclusions_drops_listed_ids_in_file_order(tmp_path):
    listing = tmp_path / "exclude.txt"
    listing.write_text("r0002\n\nr0999\nnot-a-record\n")
    kept = apply_exclusions(IDS, read_exclusion_list(listing))
    assert len(kept) == 998 and "r0002" not in kept and "r0999" not in kept
    assert kept == [r for r in IDS if r not in ("r0002", "r0999")]


def test_cross_check_reports_a_mismatch_both_ways(tmp_path):
    primary, vendored = tmp_path / "a.txt", tmp_path / "b.txt"
    primary.write_text("x\ny\n")
    vendored.write_text("y\nz\n")
    report = cross_check_exclusions(primary, vendored)
    assert report["only_in_primary"] == ["x"] and report["only_in_vendored"] == ["z"]
    vendored.write_text("y\nx\n")
    report = cross_check_exclusions(primary, vendored)
    assert not report["only_in_primary"] and not report["only_in_vendored"]


def test_build_splits_follows_the_order_of_operations():
    spec = SplitSpec(val_fraction=0.15, initial_fraction=0.20, val_seed=42, initial_seed=7)
    splits = build_splits(IDS, spec)
    assert len(splits.validation) == 150 and len(splits.training) == 850
    assert len(splits.initial) == 170 and len(splits.pool) == 680
    assert not set(splits.validation) & set(splits.training)
    assert set(splits.initial) <= set(splits.training)
    assert not set(splits.pool) & set(splits.initial)
    # V is exactly what train_adota's split mechanism draws, no second mechanism.
    _, expected_val = train_val_split(IDS, test_size=0.15, random_state=42)
    assert splits.validation == expected_val


def test_splits_are_deterministic_and_seed_sensitive():
    a = build_splits(IDS, SplitSpec(val_seed=1, initial_seed=2))
    b = build_splits(IDS, SplitSpec(val_seed=1, initial_seed=2))
    c = build_splits(IDS, SplitSpec(val_seed=1, initial_seed=3))
    assert a.validation == b.validation and a.initial == b.initial
    assert a.validation == c.validation and a.initial != c.initial   # V frozen, start re-drawn


def test_write_read_roundtrip_keeps_the_fingerprint(tmp_path):
    spec = SplitSpec()
    splits = build_splits(IDS, spec)
    write_splits(splits, tmp_path, spec, extra={"note": "test"})
    back = read_splits(tmp_path)
    assert back.validation == splits.validation and back.initial == splits.initial
    assert back.fingerprint() == splits.fingerprint()
    assert (tmp_path / "splits.json").exists()


def test_read_splits_asserts_disjointness(tmp_path):
    splits = build_splits(IDS, SplitSpec())
    write_splits(splits, tmp_path, SplitSpec())
    leaked = pd.read_csv(tmp_path / "validation_ids.csv")
    leaked.loc[0, "sample_id"] = splits.training[0]
    leaked.to_csv(tmp_path / "validation_ids.csv", index=False)
    with pytest.raises(AssertionError, match="overlap"):
        read_splits(tmp_path)


def test_schedule_arithmetic_and_assertion():
    assert batch_size_from_fraction(59622, 0.10) == 5962
    assert scheduled_size(11924, 3, 5962) == 11924 + 3 * 5962
    assert_schedule(11924 + 2 * 5962, 11924, 2, 5962)
    with pytest.raises(AssertionError, match="schedule"):
        assert_schedule(11924 + 2 * 5962 + 1, 11924, 2, 5962)


def test_record_metadata_reads_attributes_and_joins_provenance(tmp_path):
    ids = ["a", "b", "c"]
    path = write_dataset(tmp_path / "ds.h5", ids, energies=[90.0, 120.0, 150.0])
    provenance = tmp_path / "prov.csv"
    pd.DataFrame({"sample_id": ["a", "c"], "anatomy": ["thorax", "pelvic"],
                  "patient_key": ["P1", "P2"]}).to_csv(provenance, index=False)
    meta = record_metadata(path, ids, provenance)
    assert meta["sample_id"].tolist() == ids
    np.testing.assert_allclose(meta["energy_mev"], [90.0, 120.0, 150.0], atol=1e-3)
    assert meta["patient"].tolist() == ["P1", "unknown", "P2"]
    assert meta["anatomy"].tolist() == ["thorax", "unknown", "pelvic"]
