"""Splits for the retrospective benchmark: exclusions, the validation set, the
cycle-0 set, the pool and the growth schedule.

The order of operations is fixed and every run follows it:

1. apply the exclusion list to the full HDF5 set, giving ``D``;
2. draw the validation set ``V`` as a fraction of ``D`` with
   :func:`src.training.data.train_val_split`, the mechanism ``train_adota.py``
   already uses, so no second split mechanism exists;
3. the training split is ``T = D \\ V``;
4. draw the cycle-0 set as a fraction of ``T`` with the same mechanism;
5. the pool at the start of cycle 1 is ``T`` minus the cycle-0 set.

The identifiers of ``V``, ``T`` and the cycle-0 set are written to CSV once, and
every run reads those files rather than re-deriving them, so the strategies
provably share the same validation set and the same starting set.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

from src.adota.config import DEFAULT_SCALE, denormalize_energy
from src.training.data import train_val_split

logger = logging.getLogger(__name__)

EXCLUDE_FILE_NAME = (
    "IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt")
DEFAULT_EXCLUDE_PATH = f"/home/mstryja/projects/dota_pytorch/auxilary_files/{EXCLUDE_FILE_NAME}"
VENDORED_EXCLUDE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "excluded_indexes" / EXCLUDE_FILE_NAME)

SPLIT_FILES = {"validation": "validation_ids.csv", "training": "training_ids.csv",
               "initial": "cycle0_ids.csv"}


# ── Exclusions ──────────────────────────────────────────────────────────────


def read_exclusion_list(path: Path) -> List[str]:
    """The record ids to drop, one per line. A missing file is an error, never an
    unfiltered set."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"exclusion list {path} is missing; refusing to run on an unfiltered set. "
            f"Point exclude_indexes_path at the list (a copy is vendored at "
            f"{VENDORED_EXCLUDE_PATH}).")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def cross_check_exclusions(path: Path, vendored: Path = VENDORED_EXCLUDE_PATH) -> Dict:
    """Compare the configured list with the vendored copy; report any mismatch."""
    primary = set(read_exclusion_list(path))
    if not Path(vendored).exists():
        return {"vendored_present": False, "n_primary": len(primary)}
    copy = set(read_exclusion_list(vendored))
    report = {"vendored_present": True, "n_primary": len(primary), "n_vendored": len(copy),
              "only_in_primary": sorted(primary - copy), "only_in_vendored": sorted(copy - primary)}
    if report["only_in_primary"] or report["only_in_vendored"]:
        logger.warning("exclusion list mismatch against the vendored copy: %d only in %s, "
                       "%d only in %s", len(report["only_in_primary"]), path,
                       len(report["only_in_vendored"]), vendored)
    else:
        logger.info("exclusion list matches the vendored copy (%d ids)", len(primary))
    return report


def apply_exclusions(record_ids: Sequence[str], excluded: Sequence[str]) -> List[str]:
    """``D``: the record ids not on the exclusion list, in file order."""
    drop = set(excluded)
    kept = [r for r in record_ids if r not in drop]
    logger.info("records: %d before exclusion, %d after (%d dropped)",
                len(record_ids), len(kept), len(record_ids) - len(kept))
    return kept


def read_record_ids(dataset_path: Path) -> List[str]:
    with h5py.File(str(dataset_path), "r") as handle:
        return list(handle.keys())


# ── Splits ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SplitSpec:
    """How the splits are drawn. ``val_seed`` and ``initial_seed`` are separate so
    the validation set can stay frozen while the starting set is re-drawn."""

    val_fraction: float = 0.15
    initial_fraction: float = 0.20
    val_seed: int = 42
    initial_seed: int = 20260910


@dataclass
class Splits:
    """``V``, ``T`` and the cycle-0 set, as record ids."""

    validation: List[str]
    training: List[str]
    initial: List[str]

    @property
    def pool(self) -> List[str]:
        """The records a strategy may still select at the start of cycle 1."""
        start = set(self.initial)
        return [r for r in self.training if r not in start]

    def assert_consistent(self) -> None:
        val, train, init = set(self.validation), set(self.training), set(self.initial)
        if val & train:
            raise AssertionError(f"validation and training overlap on {len(val & train)} ids")
        if not init <= train:
            raise AssertionError("the cycle-0 set is not a subset of the training split")
        if len(val) != len(self.validation) or len(train) != len(self.training):
            raise AssertionError("duplicate ids inside a split")

    def fingerprint(self) -> str:
        """A hash over the three id lists, recorded in every run manifest."""
        digest = hashlib.sha256()
        for part in (self.validation, self.training, self.initial):
            digest.update("\n".join(part).encode())
            digest.update(b"\0")
        return digest.hexdigest()


def build_splits(record_ids: Sequence[str], spec: SplitSpec = SplitSpec()) -> Splits:
    """Steps 2 to 4 of the order of operations, on ``D``."""
    training, validation = train_val_split(list(record_ids), test_size=spec.val_fraction,
                                           random_state=spec.val_seed)
    _, initial = train_val_split(training, test_size=spec.initial_fraction,
                                 random_state=spec.initial_seed)
    splits = Splits(validation=validation, training=training, initial=initial)
    splits.assert_consistent()
    logger.info("splits: |D| = %d, |V| = %d, |T| = %d, cycle-0 set %d, pool %d",
                len(record_ids), len(validation), len(training), len(initial),
                len(training) - len(initial))
    return splits


def write_splits(splits: Splits, directory: Path, spec: SplitSpec,
                 extra: Optional[Dict] = None) -> Path:
    """Write the three id lists and a ``splits.json`` with the counts and seeds."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for attr, name in SPLIT_FILES.items():
        pd.DataFrame({"sample_id": getattr(splits, attr)}).to_csv(directory / name, index=False)
    summary = {"spec": asdict(spec), "n_validation": len(splits.validation),
               "n_training": len(splits.training), "n_initial": len(splits.initial),
               "n_pool": len(splits.pool), "fingerprint": splits.fingerprint(),
               **(extra or {})}
    (directory / "splits.json").write_text(json.dumps(summary, indent=2))
    return directory / "splits.json"


def read_splits(directory: Path) -> Splits:
    """Read the id lists a run must use; assert they are disjoint at load time."""
    directory = Path(directory)
    parts = {}
    for attr, name in SPLIT_FILES.items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run the `splits` stage first")
        parts[attr] = pd.read_csv(path)["sample_id"].astype(str).tolist()
    splits = Splits(**parts)
    splits.assert_consistent()
    return splits


# ── Schedule ────────────────────────────────────────────────────────────────


def batch_size_from_fraction(n_training: int, fraction: float) -> int:
    """``N``, the records added per cycle, as a fraction of ``|T|``."""
    return int(round(fraction * n_training))


def scheduled_size(n_initial: int, cycle: int, batch_size: int) -> int:
    """The training set size after ``cycle`` cycles: ``|cycle-0 set| + c * N``."""
    return n_initial + cycle * batch_size


def assert_schedule(n_current: int, n_initial: int, cycle: int, batch_size: int) -> None:
    expected = scheduled_size(n_initial, cycle, batch_size)
    if n_current != expected:
        raise AssertionError(
            f"training set holds {n_current} records after cycle {cycle}; the schedule "
            f"{n_initial} + {cycle} * {batch_size} says {expected}")


# ── Record metadata ─────────────────────────────────────────────────────────


def _v3_row_from_attrs(attrs) -> Dict[str, object]:
    """The v3 provenance columns for one record, from its HDF5 attrs (spec section 4)."""
    iso = np.asarray(attrs.get("isocenter_mm", [np.nan, np.nan, np.nan]), dtype=float)
    return {
        "source_dataset": str(attrs.get("source_dataset", "unknown")),
        "patient_key": str(attrs.get("patient_key", "unknown")),
        "spot_key": str(attrs.get("spot_key", "unknown")),
        "isocenter_x_mm": float(iso[0]) if iso.size > 0 else np.nan,
        "isocenter_y_mm": float(iso[1]) if iso.size > 1 else np.nan,
        "isocenter_z_mm": float(iso[2]) if iso.size > 2 else np.nan,
    }


def _frame_from_attrs(dataset_path: Path, ids: Sequence[str], scale: Dict[str, float]) -> pd.DataFrame:
    """Today's per-id attribute read (v2), extended with the v3 columns of
    :func:`_v3_row_from_attrs` when a group carries ``schema_version`` (v3).
    """
    rows = []
    with h5py.File(str(dataset_path), "r") as handle:
        for sample_id in ids:
            attrs = handle[sample_id].attrs
            angles = np.asarray(attrs.get("beamlet_angles", [np.nan, np.nan]), dtype=float)
            is_v3 = "schema_version" in attrs
            energy_mev = (float(attrs["energy_mev"]) if is_v3
                         else float(denormalize_energy(float(attrs["initial_energy"]), scale)))
            row = {
                "sample_id": sample_id,
                "energy_mev": energy_mev,
                "gantry_deg": float(attrs.get("gantry_angle", np.nan)),
                "theta_x_deg": float(angles[0]) if angles.size > 0 else np.nan,
                "theta_y_deg": float(angles[1]) if angles.size > 1 else np.nan,
            }
            if is_v3:
                row.update(_v3_row_from_attrs(attrs))
            rows.append(row)
    return pd.DataFrame(rows)


def _frame_from_index(index_csv_path: Path, ids: Sequence[str]) -> pd.DataFrame:
    """The frame built straight from ``<stem>_index.csv``, without opening the HDF5
    file (spec section 6.4). An id absent from the index is a hard error, never a
    silent NaN.
    """
    # round_trip: pandas' default float parser is off by an ulp on some values, which
    # would make the index path disagree with the attrs path on energy_mev.
    index = pd.read_csv(index_csv_path, float_precision="round_trip", dtype={
        "sample_id": str, "source_dataset": str, "patient_key": str, "spot_key": str})
    index = index.set_index("sample_id")
    missing = [sample_id for sample_id in ids if sample_id not in index.index]
    if missing:
        raise KeyError(f"{missing[0]!r} not found in index {index_csv_path}")
    rows = index.loc[list(ids)]
    return pd.DataFrame({
        "sample_id": list(ids),
        "energy_mev": rows["energy_mev"].astype(float).to_numpy(),
        "gantry_deg": rows["gantry_angle"].astype(float).to_numpy(),
        "theta_x_deg": rows["beamlet_angles_0"].astype(float).to_numpy(),
        "theta_y_deg": rows["beamlet_angles_1"].astype(float).to_numpy(),
        "source_dataset": rows["source_dataset"].astype(str).to_numpy(),
        "patient_key": rows["patient_key"].astype(str).to_numpy(),
        "spot_key": rows["spot_key"].astype(str).to_numpy(),
        "isocenter_x_mm": rows["isocenter_mm_0"].astype(float).to_numpy(),
        "isocenter_y_mm": rows["isocenter_mm_1"].astype(float).to_numpy(),
        "isocenter_z_mm": rows["isocenter_mm_2"].astype(float).to_numpy(),
    })


def record_metadata(dataset_path: Path, ids: Sequence[str],
                    provenance_csv: Optional[Path] = None,
                    scale: Dict[str, float] = DEFAULT_SCALE) -> pd.DataFrame:
    """Energy, gantry and steering per record, joined with the patient and anatomy
    of the study's provenance map when one is given.

    When ``<stem>_index.csv`` sits next to ``dataset_path`` the frame is built from
    that CSV alone (no HDF5 read); otherwise attributes are read per id, and a v3
    record (``schema_version`` present) contributes the extra provenance columns
    ``source_dataset``, ``patient_key``, ``spot_key``, ``isocenter_x_mm``,
    ``isocenter_y_mm``, ``isocenter_z_mm``, with ``energy_mev`` taken from the
    ``energy_mev`` attr rather than denormalised. On a v2 file the returned frame is
    exactly as before.

    Records missing from the provenance map keep ``patient`` and ``anatomy`` as
    ``"unknown"``, unless the frame carries the v3 columns and no provenance map was
    given (or it does not exist), in which case they fall back to ``patient_key``
    and ``source_dataset`` respectively.
    """
    dataset_path = Path(dataset_path)
    index_csv_path = dataset_path.with_name(dataset_path.stem + "_index.csv")
    if index_csv_path.exists():
        frame = _frame_from_index(index_csv_path, ids)
        logger.info("record metadata: %d records read from %s", len(frame), index_csv_path)
    else:
        frame = _frame_from_attrs(dataset_path, ids, scale)

    is_v3 = "source_dataset" in frame.columns
    have_provenance = provenance_csv is not None and Path(provenance_csv).exists()
    frame["patient"] = frame["patient_key"] if is_v3 and not have_provenance else "unknown"
    frame["anatomy"] = frame["source_dataset"] if is_v3 and not have_provenance else "unknown"
    if have_provenance:
        prov = pd.read_csv(provenance_csv).rename(columns={"patient_key": "patient"})
        prov = prov.drop_duplicates("sample_id").set_index("sample_id")
        hit = frame["sample_id"].isin(prov.index)
        frame.loc[hit, "patient"] = prov.loc[frame.loc[hit, "sample_id"], "patient"].to_numpy()
        frame.loc[hit, "anatomy"] = prov.loc[frame.loc[hit, "sample_id"], "anatomy"].to_numpy()
        logger.info("record metadata: %d of %d records found in %s", int(hit.sum()),
                    len(frame), provenance_csv)
    elif provenance_csv is not None:
        logger.warning("provenance map %s not found; patient and anatomy stay unknown",
                       provenance_csv)
    return frame
