"""The exclusion list, the order of operations, the split files and the schedule."""
from __future__ import annotations

import h5py
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


def test_record_metadata_v2_without_provenance_stays_unknown(tmp_path):
    ids = ["a", "b"]
    path = write_dataset(tmp_path / "ds_v2.h5", ids, energies=[90.0, 120.0])
    meta = record_metadata(path, ids)
    assert meta.columns.tolist() == ["sample_id", "energy_mev", "gantry_deg", "theta_x_deg",
                                     "theta_y_deg", "patient", "anatomy"]
    assert meta["patient"].tolist() == ["unknown", "unknown"]
    assert meta["anatomy"].tolist() == ["unknown", "unknown"]


def _add_v3_attrs(path, ids, *, energy_mev_by_id, source_dataset_by_id, patient_key_by_id,
                  spot_key_by_id, isocenter_by_id):
    """Layer v3 provenance attrs on top of the v2-shaped groups `write_dataset`
    wrote. `conftest.py` is read-only for this package, so this stays local to the
    test module rather than growing the shared fixture.
    """
    with h5py.File(path, "a") as handle:
        for sample_id in ids:
            group = handle[sample_id]
            group.attrs["schema_version"] = np.int64(3)
            group.attrs["energy_mev"] = np.float64(energy_mev_by_id[sample_id])
            group.attrs["source_dataset"] = source_dataset_by_id[sample_id]
            group.attrs["patient_key"] = patient_key_by_id[sample_id]
            group.attrs["spot_key"] = spot_key_by_id[sample_id]
            group.attrs["isocenter_mm"] = np.array(isocenter_by_id[sample_id], dtype=np.float64)


def test_record_metadata_v3_attrs_adds_provenance_columns_and_uses_energy_attr(tmp_path):
    ids = ["a", "b"]
    path = write_dataset(tmp_path / "ds_v3.h5", ids, energies=[90.0, 120.0])
    # energy_mev attrs deliberately differ from the denormalised initial_energy values,
    # to prove the v3 path reads the attr rather than denormalising.
    _add_v3_attrs(
        path, ids,
        energy_mev_by_id={"a": 201.5, "b": 55.25},
        source_dataset_by_id={"a": "trainset_pelvis", "b": "initial_test_one_ct"},
        patient_key_by_id={"a": "patientA", "b": "patientB"},
        spot_key_by_id={"a": "spotA", "b": "spotB"},
        isocenter_by_id={"a": (10.0, 20.0, 30.0), "b": (40.0, 50.0, 60.0)},
    )

    meta = record_metadata(path, ids)
    np.testing.assert_allclose(meta["energy_mev"], [201.5, 55.25])
    assert meta["source_dataset"].tolist() == ["trainset_pelvis", "initial_test_one_ct"]
    assert meta["patient_key"].tolist() == ["patientA", "patientB"]
    assert meta["spot_key"].tolist() == ["spotA", "spotB"]
    np.testing.assert_allclose(meta["isocenter_x_mm"], [10.0, 40.0])
    np.testing.assert_allclose(meta["isocenter_y_mm"], [20.0, 50.0])
    np.testing.assert_allclose(meta["isocenter_z_mm"], [30.0, 60.0])
    # no provenance CSV: patient/anatomy fall back to patient_key/source_dataset
    assert meta["patient"].tolist() == ["patientA", "patientB"]
    assert meta["anatomy"].tolist() == ["trainset_pelvis", "initial_test_one_ct"]

    provenance = tmp_path / "prov_v3.csv"
    pd.DataFrame({"sample_id": ["a", "b"], "anatomy": ["thorax", "pelvic"],
                  "patient_key": ["P1", "P2"]}).to_csv(provenance, index=False)
    meta_with_prov = record_metadata(path, ids, provenance)
    assert meta_with_prov["patient"].tolist() == ["P1", "P2"]
    assert meta_with_prov["anatomy"].tolist() == ["thorax", "pelvic"]


def test_record_metadata_reads_from_index_csv_when_present(tmp_path):
    ids = ["a", "b"]
    path = write_dataset(tmp_path / "ds_idx.h5", ids, energies=[90.0, 120.0])
    _add_v3_attrs(
        path, ids,
        energy_mev_by_id={"a": 1.0, "b": 2.0},
        source_dataset_by_id={"a": "trainset_pelvis", "b": "trainset_pelvis"},
        patient_key_by_id={"a": "attrs-patient-a", "b": "attrs-patient-b"},
        spot_key_by_id={"a": "attrs-spot-a", "b": "attrs-spot-b"},
        isocenter_by_id={"a": (1.0, 1.0, 1.0), "b": (2.0, 2.0, 2.0)},
    )

    index_path = path.with_name(path.stem + "_index.csv")
    pd.DataFrame({
        "sample_id": ["a", "b"],
        "energy_mev": [111.0, 222.0],
        "gantry_angle": [10.0, 20.0],
        "beamlet_angles_0": [0.1, 0.2],
        "beamlet_angles_1": [-0.1, -0.2],
        "source_dataset": ["initial_test_one_ct", "initial_test_one_ct"],
        "patient_key": ["csv-patient-a", "csv-patient-b"],
        "spot_key": ["csv-spot-a", "csv-spot-b"],
        "isocenter_mm_0": [100.0, 200.0],
        "isocenter_mm_1": [101.0, 201.0],
        "isocenter_mm_2": [102.0, 202.0],
    }).to_csv(index_path, index=False)

    # requested in reverse order, to prove the frame follows `ids`, not the CSV's order
    meta = record_metadata(path, ["b", "a"])
    assert meta["sample_id"].tolist() == ["b", "a"]
    np.testing.assert_allclose(meta["energy_mev"], [222.0, 111.0])
    np.testing.assert_allclose(meta["gantry_deg"], [20.0, 10.0])
    np.testing.assert_allclose(meta["theta_x_deg"], [0.2, 0.1])
    np.testing.assert_allclose(meta["theta_y_deg"], [-0.2, -0.1])
    assert meta["source_dataset"].tolist() == ["initial_test_one_ct", "initial_test_one_ct"]
    assert meta["patient_key"].tolist() == ["csv-patient-b", "csv-patient-a"]
    assert meta["spot_key"].tolist() == ["csv-spot-b", "csv-spot-a"]
    np.testing.assert_allclose(meta["isocenter_x_mm"], [200.0, 100.0])
    np.testing.assert_allclose(meta["isocenter_y_mm"], [201.0, 101.0])
    np.testing.assert_allclose(meta["isocenter_z_mm"], [202.0, 102.0])
    # patient/anatomy fall back to the index's patient_key/source_dataset, values that
    # differ from the h5 attrs above, proving the index -- not the attrs -- was read
    assert meta["patient"].tolist() == ["csv-patient-b", "csv-patient-a"]
    assert meta["anatomy"].tolist() == ["initial_test_one_ct", "initial_test_one_ct"]

    with pytest.raises(KeyError, match="missing-id"):
        record_metadata(path, ["a", "missing-id"])
