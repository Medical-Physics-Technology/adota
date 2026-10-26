"""The scan-level partition of EXP-0014: F / V / R, the cross-fitting folds and the
HPO split of R.

Everything here works on whole scans (``patient_key`` on the v3 HDF5), never on
records, because the scan is the unit every evaluation in EXP-0014 holds out:

- **F** is the frozen test set of EXP-0004 and **V** the held-out scans of
  EXP-0012. They are read from their files and may overlap (one scan does).
  **R** is every other scan.
- **Folds.** Every one of the scans is assigned to one of ``n_folds`` folds,
  independently of F / V / R, so every record gets an out-of-sample error from
  each member. Within each anatomy the scans are drawn in a seeded order and
  each goes to the fold with the fewest scans of that anatomy, ties broken by
  the fewest records and then the lowest fold index. Per anatomy the fold sizes
  then differ by at most one scan.
- **HPO split.** Hyperparameter optimisation trains and validates on R only: a
  fixed number of R scans per anatomy is held out for HPO validation with
  :func:`src.active_learning.retrospective.patient_split.select_held_out_groups`.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.active_learning.retrospective.patient_split import select_held_out_groups

logger = logging.getLogger(__name__)

ROLE_F, ROLE_V, ROLE_FV, ROLE_R = "F", "V", "F+V", "R"
HPO_TRAIN, HPO_VAL, HPO_NONE = "train", "val", "none"


def scans_from_frozen_ids(frozen_ids: Sequence[str], metadata: pd.DataFrame,
                          group_column: str = "patient_key") -> List[str]:
    """The scans of F: every scan with at least one record among ``frozen_ids``.

    F was frozen as a list of record ids; its unit is the scan, so the whole
    scan belongs to F. An id absent from ``metadata`` is an error, since it
    would silently shrink F."""
    index = metadata.drop_duplicates("sample_id").set_index("sample_id")[group_column]
    missing = [i for i in frozen_ids if i not in index.index]
    if missing:
        raise KeyError(f"{len(missing)} frozen-test ids are not in the dataset, e.g. {missing[0]!r}")
    return sorted(index.loc[list(frozen_ids)].astype(str).unique())


def scans_from_patients(patients: Sequence[str], metadata: pd.DataFrame,
                        patient_column: str = "patient",
                        group_column: str = "patient_key") -> List[str]:
    """The scans of V, from the provenance patient strings EXP-0012 wrote."""
    lookup = metadata.drop_duplicates(patient_column).set_index(patient_column)[group_column]
    missing = [p for p in patients if p not in lookup.index]
    if missing:
        raise KeyError(f"{len(missing)} held-out patients are not in the dataset, e.g. {missing[0]!r}")
    return sorted(lookup.loc[list(patients)].astype(str).unique())


def scan_table(metadata: pd.DataFrame, f_scans: Sequence[str], v_scans: Sequence[str],
               group_column: str = "patient_key", stratify_column: str = "anatomy") -> pd.DataFrame:
    """One row per scan: its anatomy, record count and role in F / V / R."""
    per_scan = metadata.groupby(group_column)[stratify_column].nunique()
    if (per_scan > 1).any():
        raise ValueError(f"{int((per_scan > 1).sum())} scans span several {stratify_column!r} values")
    table = (metadata.groupby(group_column)
             .agg(anatomy=(stratify_column, "first"), n_records=("sample_id", "size"))
             .reset_index().rename(columns={group_column: "scan"}))
    table["scan"] = table["scan"].astype(str)
    in_f, in_v = table["scan"].isin(set(f_scans)), table["scan"].isin(set(v_scans))
    for name, scans, hit in (("F", f_scans, in_f), ("V", v_scans, in_v)):
        if int(hit.sum()) != len(set(scans)):
            raise KeyError(f"{len(set(scans)) - int(hit.sum())} {name} scans are not in the metadata")
    table["role"] = np.select([in_f & in_v, in_f, in_v], [ROLE_FV, ROLE_F, ROLE_V], ROLE_R)
    return table.sort_values("scan").reset_index(drop=True)


def assign_folds(scans: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """The fold of every scan (index aligned with ``scans``), stratified by anatomy.

    Strata are visited in sorted order with one seeded generator, and the scans
    of a stratum in a seeded permutation of their sorted keys, so the result
    depends on the seed and the scan set only, never on the file order."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    rng = np.random.RandomState(seed)
    fold = pd.Series(-1, index=scans.index, dtype=int)
    records = np.zeros(n_folds, dtype=np.int64)
    for stratum in sorted(scans["anatomy"].astype(str).unique()):
        members = scans[scans["anatomy"].astype(str) == stratum].sort_values("scan")
        if len(members) < n_folds:
            raise ValueError(f"stratum {stratum!r} has {len(members)} scans, fewer than {n_folds} folds")
        counts = np.zeros(n_folds, dtype=np.int64)
        for position in rng.permutation(len(members)):
            row = members.iloc[position]
            target = min(range(n_folds), key=lambda k: (counts[k], records[k], k))
            fold.loc[row.name] = target
            counts[target] += 1
            records[target] += int(row["n_records"])
    return fold


def hpo_split(scans: pd.DataFrame, val_scans_per_stratum: Mapping[str, int], seed: int) -> pd.Series:
    """``train`` or ``val`` for every R scan, ``none`` for every F or V scan."""
    r = scans[scans["role"] == ROLE_R]
    frame = pd.DataFrame({"sample_id": r["scan"], "scan": r["scan"], "anatomy": r["anatomy"]})
    held = select_held_out_groups(frame, "scan", "anatomy", dict(val_scans_per_stratum), seed)
    val = {s for group in held.values() for s in group}
    out = pd.Series(HPO_NONE, index=scans.index, dtype=object)
    out[scans["role"] == ROLE_R] = HPO_TRAIN
    out[scans["scan"].isin(val)] = HPO_VAL
    return out


def record_table(metadata: pd.DataFrame, scans: pd.DataFrame,
                 group_column: str = "patient_key") -> pd.DataFrame:
    """One row per record with its scan's role, fold and HPO role."""
    keep = ["sample_id", group_column, "energy_mev"]
    frame = metadata[keep].rename(columns={group_column: "scan"}).copy()
    frame["scan"] = frame["scan"].astype(str)
    columns = ["scan", "anatomy", "role", "fold", "hpo"]
    return frame.merge(scans[columns], on="scan", how="left", validate="many_to_one")


def check_partition(records: pd.DataFrame, n_folds: int) -> None:
    """Raise unless the folds cover every record once, no scan sits in two folds,
    and the HPO sets are disjoint scans of R only."""
    if records["sample_id"].duplicated().any():
        raise ValueError("duplicate sample ids in the partition")
    if records["fold"].isna().any() or not set(records["fold"]) <= set(range(n_folds)):
        raise ValueError("a record has no fold or a fold outside range")
    if (records.groupby("scan")["fold"].nunique() > 1).any():
        raise ValueError("a scan is split across folds")
    hpo = records[records["hpo"] != HPO_NONE]
    if (hpo["role"] != ROLE_R).any():
        raise ValueError("an F or V scan entered the HPO split")
    train, val = (set(hpo.loc[hpo["hpo"] == k, "scan"]) for k in (HPO_TRAIN, HPO_VAL))
    if train & val or not train or not val:
        raise ValueError("the HPO train and validation scans must be non-empty and disjoint")


def fold_ids(records: pd.DataFrame, fold: int) -> Dict[str, List[str]]:
    """``train``: every record outside ``fold``; ``eval``: every record in it."""
    inside = records["fold"] == fold
    return {"train": records.loc[~inside, "sample_id"].tolist(),
            "eval": records.loc[inside, "sample_id"].tolist()}


def fold_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Scans and records per fold and anatomy, with the F / V scans each fold holds."""
    rows = []
    for (fold, anatomy), part in records.groupby(["fold", "anatomy"]):
        rows.append({"fold": int(fold), "anatomy": anatomy, "n_scans": part["scan"].nunique(),
                     "n_records": len(part),
                     "n_f_scans": part.loc[part["role"].isin([ROLE_F, ROLE_FV]), "scan"].nunique(),
                     "n_v_scans": part.loc[part["role"].isin([ROLE_V, ROLE_FV]), "scan"].nunique()})
    return pd.DataFrame(rows)
