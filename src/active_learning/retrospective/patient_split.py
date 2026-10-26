"""A patient-held-out validation set for the retrospective benchmark.

The record-level split of :mod:`src.active_learning.retrospective.dataset`
draws ``V`` from ``D`` record by record, so ``V`` and ``T`` share almost every
patient (55 of 55 CT scans at d30) and the benchmark measures generalisation to
new beamlets of seen patients only. This module holds out whole patients
instead: within every stratum (the anatomy source) a fixed number of groups is
drawn with the split seed, every record of those groups forms ``V``, and ``T``
is everything else. The cycle-0 set is then drawn from ``T`` record by record,
exactly as in the record-level split.

A group is whatever ``group_column`` names in the per-record metadata. On the v3
HDF5 that is ``patient_key``, which identifies one CT scan by its geometry; a
patient scanned twice would count as two groups.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.active_learning.retrospective.dataset import Splits
from src.training.data import train_val_split

logger = logging.getLogger(__name__)


def select_held_out_groups(metadata: pd.DataFrame, group_column: str, stratify_column: str,
                           groups_per_stratum: Dict[str, int], seed: int) -> Dict[str, List[str]]:
    """The groups held out for ``V``, per stratum.

    Groups are sorted before the seeded permutation, so the draw depends on the
    seed and the set of groups, never on the file order. A group that spans two
    strata, a stratum missing from ``groups_per_stratum``, or a request for as
    many groups as a stratum holds (which would leave it out of ``T``) is an
    error.
    """
    for column in (group_column, stratify_column):
        if column not in metadata.columns:
            raise KeyError(f"per-record metadata has no {column!r} column; a patient split needs "
                           "the v3 HDF5 (or its index CSV) or a provenance map that supplies it")
    strata_per_group = metadata.groupby(group_column)[stratify_column].nunique()
    mixed = strata_per_group[strata_per_group > 1]
    if len(mixed):
        raise ValueError(f"{len(mixed)} groups span several {stratify_column!r} values, "
                         f"e.g. {mixed.index[0]!r}")
    present = sorted(metadata[stratify_column].astype(str).unique())
    unknown = sorted(set(groups_per_stratum) - set(present))
    missing = sorted(set(present) - set(groups_per_stratum))
    if unknown or missing:
        raise ValueError(f"groups_per_stratum must name exactly the strata present {present}; "
                         f"unknown {unknown}, missing {missing}")
    rng = np.random.RandomState(seed)
    held_out = {}
    for stratum in present:
        groups = sorted(metadata.loc[metadata[stratify_column].astype(str) == stratum,
                                     group_column].astype(str).unique())
        n = int(groups_per_stratum[stratum])
        if not 0 < n < len(groups):
            raise ValueError(f"stratum {stratum!r} has {len(groups)} groups; hold out between 1 "
                             f"and {len(groups) - 1}, got {n}")
        held_out[stratum] = sorted(groups[i] for i in rng.permutation(len(groups))[:n])
    return held_out


def build_patient_splits(record_ids: Sequence[str], metadata: pd.DataFrame, group_column: str,
                         stratify_column: str, groups_per_stratum: Dict[str, int], val_seed: int,
                         initial_fraction: float, initial_seed: int
                         ) -> Tuple[Splits, Dict[str, List[str]]]:
    """``V`` = every record of the held-out groups, ``T`` = the rest (both in the
    order of ``record_ids``), the cycle-0 set drawn from ``T`` as the record-level
    split draws it. Returns the splits and the held-out groups per stratum."""
    meta = metadata.set_index(metadata["sample_id"].astype(str))
    ids = [str(r) for r in record_ids]
    absent = [r for r in ids if r not in meta.index]
    if absent:
        raise KeyError(f"{len(absent)} records have no metadata, e.g. {absent[0]!r}")
    held_out = select_held_out_groups(meta.loc[ids].reset_index(drop=True), group_column,
                                      stratify_column, groups_per_stratum, val_seed)
    held = {g for groups in held_out.values() for g in groups}
    group_of = meta[group_column].astype(str)
    validation = [r for r in ids if group_of[r] in held]
    training = [r for r in ids if group_of[r] not in held]
    _, initial = train_val_split(training, test_size=initial_fraction, random_state=initial_seed)
    splits = Splits(validation=validation, training=training, initial=initial)
    splits.assert_consistent()
    assert_groups_disjoint(splits, metadata, group_column)
    logger.info("patient split: %d groups held out (%s); |V| = %d, |T| = %d, cycle-0 set %d",
                len(held), ", ".join(f"{k} {len(v)}" for k, v in held_out.items()),
                len(validation), len(training), len(initial))
    return splits, held_out


def assert_groups_disjoint(splits: Splits, metadata: pd.DataFrame, group_column: str) -> None:
    """Raise unless no ``group_column`` value has records in both ``V`` and ``T``."""
    if group_column not in metadata.columns:
        raise KeyError(f"per-record metadata has no {group_column!r} column")
    group_of = metadata.set_index(metadata["sample_id"].astype(str))[group_column].astype(str)
    shared = set(group_of.loc[splits.validation]) & set(group_of.loc[splits.training])
    if shared:
        raise AssertionError(f"{len(shared)} {group_column} values have records in both V and T, "
                             f"e.g. {sorted(shared)[0]!r}")
