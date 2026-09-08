"""The CT pool the loop samples from, and the validation CTs it never samples.

A patient enters the loop with an anatomy label and a **role**: ``validation`` CTs
carry the frozen yardstick, ``pool`` CTs are where candidates are generated. The two
never mix, and neither overlaps the training collections or the patients already
claimed by another experiment.

The leakage rule is the one the design fixed: training consumed the first roughly
fifty patients of each sorted collection, so only the tail is safe. This module takes
the last ``n_holdout`` of a collection, removes every patient id listed in the
exclusion registries, assigns the first ``n_validation`` of what remains to the
validation role and the rest to the pool. The result is written to a CSV that is the
provenance record for both sets: the loop reads that file, never the rule, so a
selection cannot silently change under a rerun.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.datasets.base import CTRecord
from src.datasets.tcia import TCIADataset

logger = logging.getLogger(__name__)

POOL_CSV_FIELDS = ["role", "anatomy", "dataset_name", "root", "collection",
                   "patient_id", "series_uid", "n_slices"]


@dataclass(frozen=True)
class PoolEntry:
    """One CT with its role. The CSV row, and what the loop iterates over."""

    role: str            # "validation" | "pool"
    anatomy: str
    dataset_name: str
    root: str
    collection: str
    patient_id: str
    series_uid: str = ""
    n_slices: int = 0


def read_excluded_patient_ids(paths: Iterable[str]) -> set:
    """Patient ids claimed by other experiments, from registry CSVs.

    Any column named ``patient_id`` in any of the files counts; a missing file is an
    error rather than an empty set, because silently not excluding a patient is the
    failure this guard exists to prevent.
    """
    excluded: set = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"exclusion registry not found: {p}")
        with p.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("patient_id"):
                    excluded.add(row["patient_id"].strip())
    return excluded


def _dataset_for(spec: dict, patient_ids: Optional[Sequence[str]] = None,
                 n_patients: Optional[int] = None, selection: str = "last") -> TCIADataset:
    return TCIADataset(
        root=spec["root"], collection=spec.get("collection"),
        anatomy=spec.get("anatomy", ""), name=spec.get("name"),
        patient_ids=list(patient_ids) if patient_ids is not None else None,
        n_patients=n_patients, selection=selection,
        qc=spec.get("qc"),
    )


def build_pool(
    dataset_specs: Sequence[dict],
    *,
    n_holdout: int = 30,
    n_validation: int = 5,
    exclude_files: Sequence[str] = (),
) -> List[PoolEntry]:
    """Assign roles across the configured collections.

    Args:
        dataset_specs: One per anatomy, in the shape
            :func:`src.datasets.registry.build_tcia_dataset` takes, plus an optional
            ``n_holdout`` / ``n_validation`` override per entry.
        n_holdout: How many patients from the tail of each collection are safe.
        n_validation: How many of those become validation CTs.
        exclude_files: Registry CSVs whose ``patient_id`` column is off limits.

    Returns:
        Every selected CT, validation entries first within each anatomy.
    """
    excluded = read_excluded_patient_ids(exclude_files)
    entries: List[PoolEntry] = []
    for spec in dataset_specs:
        hold = int(spec.get("n_holdout", n_holdout))
        n_val = int(spec.get("n_validation", n_validation))
        dataset = _dataset_for(spec, n_patients=hold, selection="last")
        kept: List[CTRecord] = []
        for i in range(len(dataset)):
            rec = dataset.record(i)
            if rec is None:
                continue
            if rec.patient_id in excluded:
                logger.info("  %s: excluded by registry", rec.patient_id)
                continue
            kept.append(rec)
        if len(kept) < n_val + 1:
            raise ValueError(
                f"{spec.get('name')}: only {len(kept)} usable patients in the last "
                f"{hold} after exclusions; need more than {n_val} to leave a pool")
        for i, rec in enumerate(kept):
            entries.append(PoolEntry(
                role="validation" if i < n_val else "pool",
                anatomy=rec.anatomy, dataset_name=spec.get("name", rec.dataset_name),
                root=spec["root"], collection=spec.get("collection", "") or "",
                patient_id=rec.patient_id, series_uid=rec.series_uid,
                n_slices=rec.n_slices))
        logger.info("%s (%s): %d validation + %d pool",
                    spec.get("name"), spec.get("anatomy"), n_val, len(kept) - n_val)
    return entries


def write_pool(entries: Sequence[PoolEntry], path: Path) -> Path:
    """Write the selection to its provenance CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POOL_CSV_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))
    return path


def read_pool(path: Path, role: Optional[str] = None) -> List[PoolEntry]:
    """Read the selection back, optionally only one role."""
    with Path(path).open() as handle:
        rows = [PoolEntry(role=r["role"], anatomy=r["anatomy"],
                          dataset_name=r["dataset_name"], root=r["root"],
                          collection=r["collection"], patient_id=r["patient_id"],
                          series_uid=r.get("series_uid", ""),
                          n_slices=int(r.get("n_slices") or 0))
                for r in csv.DictReader(handle)]
    return [r for r in rows if role is None or r.role == role]


class RecordResolver:
    """Turns :class:`PoolEntry` rows back into loadable :class:`CTRecord` objects.

    One :class:`TCIADataset` per collection is built lazily, pinned to exactly the
    patients that collection contributes, because constructing one scans a DICOM
    directory tree per patient and the loop asks for the same CTs repeatedly.
    """

    def __init__(self, entries: Sequence[PoolEntry]) -> None:
        self._entries = list(entries)
        self._cache: Dict[Tuple[str, str], Dict[str, CTRecord]] = {}

    def _collection(self, entry: PoolEntry) -> Dict[str, CTRecord]:
        key = (entry.root, entry.collection)
        if key not in self._cache:
            wanted = [e.patient_id for e in self._entries
                      if (e.root, e.collection) == key]
            dataset = _dataset_for(
                {"root": entry.root, "collection": entry.collection or None,
                 "anatomy": entry.anatomy, "name": entry.dataset_name},
                patient_ids=wanted)
            self._cache[key] = {dataset.record(i).patient_id: dataset.record(i)
                                for i in range(len(dataset))}
        return self._cache[key]

    def record(self, entry: PoolEntry) -> CTRecord:
        records = self._collection(entry)
        if entry.patient_id not in records:
            raise ValueError(
                f"no usable CT series for {entry.patient_id} under {entry.root}; "
                "the pool CSV and the collection on disk disagree")
        return records[entry.patient_id]
