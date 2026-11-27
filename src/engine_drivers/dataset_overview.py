"""Per-beamlet metadata of the pool test set and its per-site summary.

The engine-error driver analysis scores ADoTA and the pencil-beam engine on the
test split of the Apollo 1e6 beamlet pool. This module assembles what a reader
needs to know about that set before any error is shown: which patient, site and
field each beamlet comes from, its gantry and couch angle, its energy, where the
spot sits in the beam's eye view, and whether it enters the analysis (analytic
peak inside the crop).

Three inputs are joined on the sample id:

- the export index (``sample_id``, ``pool_patient_id``, ``pool_anatomy``,
  ``gantry_angle``, ``energy_mev``, ``bixel_shift_xy_mm_0/1``);
- the pool plan (``candidate_id`` equal to the export ``sample_id``,
  ``field_dir`` naming the field);
- the analytic feature table (``peak_inside_crop``).

Functions are pure: paths in, frames out, nothing is written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

SITE_PREFIXES: Dict[str, str] = {"LUNG1": "thoracic", "Lung_Dx": "thoracic",
                                 "StageII-Colorectal-CT": "abdominal", "Prostate-AEC": "pelvic"}
"""Patient-id prefix (the text before the patient number) to treatment site."""

SITE_ORDER = ("thoracic", "abdominal", "pelvic")
"""Row order of :func:`site_summary`."""

METADATA_COLUMNS = ("sample_id", "patient_id", "site", "field_id", "gantry_deg", "couch_deg",
                    "energy_mev", "spot_x_mm", "spot_y_mm", "in_analysis")

SUMMARY_COLUMNS = ("site", "patients", "fields", "beamlets", "beamlets_per_patient_min",
                   "beamlets_per_patient_median", "beamlets_per_patient_max", "energy_min_mev",
                   "energy_median_mev", "energy_max_mev", "spot_x_min_mm", "spot_x_max_mm", "spot_y_min_mm",
                   "spot_y_max_mm", "in_analysis")

GANTRY_DECIMALS = 6
"""Gantry angles are compared after rounding to 1e-6 deg when checking field consistency."""


def site_from_patient(patient_id: str) -> str:
    """Treatment site of a pool patient id, from its dataset prefix.

    A prefix matches when the id starts with the prefix followed by ``-`` (for
    example ``Prostate-AEC-074``); the longest matching prefix wins.

    Raises:
        ValueError: if no prefix in :data:`SITE_PREFIXES` matches.
    """
    matches = [prefix for prefix in SITE_PREFIXES if str(patient_id).startswith(prefix + "-")]
    if not matches:
        raise ValueError(f"unknown patient id prefix: {patient_id!r} (known: {sorted(SITE_PREFIXES)})")
    return SITE_PREFIXES[max(matches, key=len)]


def _check_fields(frame: pd.DataFrame) -> None:
    """Raise unless field_id and (patient_id, gantry rounded) map one to one."""
    key = frame["patient_id"].astype(str) + "@" + frame["gantry_deg"].round(GANTRY_DECIMALS).map(repr)
    pairs = pd.DataFrame({"field_id": frame["field_id"], "key": key}).drop_duplicates()
    per_field = pairs.groupby("field_id")["key"].nunique()
    per_key = pairs.groupby("key")["field_id"].nunique()
    bad_fields: List[str] = per_field[per_field > 1].index.tolist()
    bad_keys: List[str] = per_key[per_key > 1].index.tolist()
    if bad_fields:
        raise ValueError(f"field_id maps to more than one (patient, gantry): {bad_fields[:5]}")
    if bad_keys:
        raise ValueError(f"(patient, gantry) maps to more than one field_id: {bad_keys[:5]}")


def beamlet_metadata(index_csv: PathLike, plan_csv: PathLike, features_csv: PathLike) -> pd.DataFrame:
    """One row per exported test beamlet with the columns :data:`METADATA_COLUMNS`, sorted by sample id.

    - ``site`` comes from the patient-id prefix and must agree with the export's ``pool_anatomy``;
    - ``field_id`` is the plan's ``field_dir``; every sample must have one, and a field id must
      correspond to exactly one (patient, gantry) pair and vice versa;
    - ``gantry_deg`` is the export ``gantry_angle`` modulo 360;
    - ``couch_deg`` is 0.0 for every row: no couch angle is stored anywhere in the pool, and the
      MCsquare plan writer (:mod:`src.mc_generation.config_writer`) writes ``PatientSupportAngle``
      with its default of 0;
    - ``spot_x_mm`` and ``spot_y_mm`` are ``bixel_shift_xy_mm_0`` and ``bixel_shift_xy_mm_1``: the
      spot position on the isocentre plane in the beam's eye view, lateral ``X_g`` in the gantry
      rotation plane and ``Y_g`` along the patient axis;
    - ``in_analysis`` is the feature table's ``peak_inside_crop``; every sample needs a feature row.

    Raises:
        ValueError: on a duplicated sample id, an unknown prefix, a site/anatomy disagreement,
            a missing plan or feature row, or an inconsistent field mapping.
    """
    index = pd.read_csv(index_csv, usecols=["sample_id", "pool_patient_id", "pool_anatomy", "gantry_angle",
                                            "energy_mev", "bixel_shift_xy_mm_0", "bixel_shift_xy_mm_1"])
    if index["sample_id"].duplicated().any():
        raise ValueError("export index has duplicated sample_id")
    plan = pd.read_csv(plan_csv, usecols=["candidate_id", "patient_id", "field_dir"])
    plan = plan.drop_duplicates()
    if plan["candidate_id"].duplicated().any():
        raise ValueError("plan has more than one row per candidate_id")
    features = pd.read_csv(features_csv, usecols=["sample_id", "peak_inside_crop"])
    if features["sample_id"].duplicated().any():
        raise ValueError("feature table has duplicated sample_id")

    out = pd.DataFrame({"sample_id": index["sample_id"].astype(str),
                        "patient_id": index["pool_patient_id"].astype(str)})
    out["site"] = out["patient_id"].map(site_from_patient)
    disagree = out["site"].to_numpy() != index["pool_anatomy"].astype(str).to_numpy()
    if disagree.any():
        raise ValueError(f"site from prefix disagrees with pool_anatomy for {int(disagree.sum())} rows, "
                         f"e.g. {out.loc[disagree, 'sample_id'].head(3).tolist()}")

    plan = plan.assign(candidate_id=plan["candidate_id"].astype(str)).set_index("candidate_id")
    field = out["sample_id"].map(plan["field_dir"])
    if field.isna().any():
        raise ValueError(f"{int(field.isna().sum())} samples have no field_dir in the plan")
    plan_patient = out["sample_id"].map(plan["patient_id"]).astype(str)
    if (plan_patient != out["patient_id"]).any():
        raise ValueError("plan patient_id disagrees with export pool_patient_id")
    out["field_id"] = field.astype(str)
    out["gantry_deg"] = np.mod(index["gantry_angle"].to_numpy(dtype=float), 360.0)
    out["couch_deg"] = 0.0
    out["energy_mev"] = index["energy_mev"].to_numpy(dtype=float)
    out["spot_x_mm"] = index["bixel_shift_xy_mm_0"].to_numpy(dtype=float)
    out["spot_y_mm"] = index["bixel_shift_xy_mm_1"].to_numpy(dtype=float)
    _check_fields(out)

    peak = features.assign(sample_id=features["sample_id"].astype(str)).set_index("sample_id")["peak_inside_crop"]
    in_analysis = out["sample_id"].map(peak)
    if in_analysis.isna().any():
        raise ValueError(f"{int(in_analysis.isna().sum())} samples have no feature row")
    out["in_analysis"] = in_analysis.map(_as_bool).astype(bool)
    return out.loc[:, list(METADATA_COLUMNS)].sort_values("sample_id").reset_index(drop=True)


def _as_bool(value: object) -> bool:
    """Bool from a bool-like CSV cell (True/False, 1/0, "true"/"false")."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
        raise ValueError(f"not a bool-like value: {value!r}")
    return bool(value)


def check_join(metadata: pd.DataFrame, errors_csv: PathLike) -> int:
    """Number of rows when the metadata joins one to one with an engine's error table.

    Raises:
        ValueError: ``join lost rows`` if a sample id is duplicated on either side or present on
            only one side.
    """
    errors = pd.read_csv(errors_csv, usecols=["sample_id"])["sample_id"].astype(str)
    meta = metadata["sample_id"].astype(str)
    dup_meta, dup_err = int(meta.duplicated().sum()), int(errors.duplicated().sum())
    only_meta = sorted(set(meta) - set(errors))
    only_err = sorted(set(errors) - set(meta))
    if dup_meta or dup_err or only_meta or only_err:
        raise ValueError(f"join lost rows: {dup_meta} duplicated in metadata, {dup_err} duplicated in errors, "
                         f"{len(only_meta)} only in metadata (e.g. {only_meta[:3]}), "
                         f"{len(only_err)} only in errors (e.g. {only_err[:3]})")
    return int(meta.size)


def site_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per site (thoracic, abdominal, pelvic) with the columns :data:`SUMMARY_COLUMNS`.

    Counts are distinct patients and fields and all beamlets; ``in_analysis`` counts the beamlets
    that enter the analysis. Sites absent from ``metadata`` are left out.
    """
    rows = []
    for site in SITE_ORDER:
        part = metadata[metadata["site"] == site]
        if part.empty:
            continue
        per_patient = part.groupby("patient_id").size()
        rows.append({"site": site, "patients": int(part["patient_id"].nunique()),
                     "fields": int(part["field_id"].nunique()), "beamlets": int(len(part)),
                     "beamlets_per_patient_min": int(per_patient.min()),
                     "beamlets_per_patient_median": float(per_patient.median()),
                     "beamlets_per_patient_max": int(per_patient.max()),
                     "energy_min_mev": float(part["energy_mev"].min()),
                     "energy_median_mev": float(part["energy_mev"].median()),
                     "energy_max_mev": float(part["energy_mev"].max()),
                     "spot_x_min_mm": float(part["spot_x_mm"].min()), "spot_x_max_mm": float(part["spot_x_mm"].max()),
                     "spot_y_min_mm": float(part["spot_y_mm"].min()), "spot_y_max_mm": float(part["spot_y_mm"].max()),
                     "in_analysis": int(part["in_analysis"].astype(bool).sum())})
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
