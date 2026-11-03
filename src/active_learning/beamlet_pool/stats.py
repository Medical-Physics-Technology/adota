"""Reporting statistics of a finished beamlet pool, from its index and plan alone.

Three views, each a set of plain tables (figures: :mod:`src.figures.beamlet_pool`):

- **anatomy**: beamlets per anatomy, collection and split, what was lost on the
  way (planned, stored, missing, rejected) and the energy and range spectra;
- **patients**: beamlets and fields per patient against the split quota, and how
  concentrated the pool is on few patients (Gini, effective number of patients,
  share of the largest decile);
- **gantry frame**: where the beamlets sit in the IEC 61217 gantry coordinate
  system. The field angle is the gantry angle; the steering angles become the
  spot position on the isocentre plane, ``X_g`` (lateral, from ``theta_y`` and the
  SMX distance) and ``Y_g`` (along the gantry axis, from ``theta_x`` and the SMY
  distance), the convention of ``sweep_lateral_half_extents``.

Nothing here reads a dose: the index already carries every per-beamlet number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.beamlets.bdl import angles_to_spot_position

PATIENT_COLUMNS = ["patient_id", "collection", "manufacturer", "kvp", "z_extent_mm"]


def load_pool_index(store_root: Path, registry_csv: Optional[Path] = None) -> pd.DataFrame:
    """Every planned beamlet with its fate (``status``) and its collection.

    ``status`` is ``stored``, ``missing`` (generator QA) or ``rejected:<reason>``
    (post-MC check). The collection comes from the plan's candidates; scanner and
    z extent from the patient registry when it is given.
    """
    store_root = Path(store_root)
    index = pd.concat([pd.read_csv(p) for p in sorted((store_root / "index").glob("batch_*.csv"))],
                      ignore_index=True)
    plan = pd.read_csv(store_root / "plan" / "plan.csv", usecols=["candidate_id", "dataset_name"])
    index = index.merge(plan.rename(columns={"dataset_name": "collection"}), on="candidate_id", how="left")
    if registry_csv is not None:
        reg = pd.read_csv(registry_csv, dtype={"patient_id": str}, low_memory=False)
        reg = reg[reg["split"] != "excluded"][[c for c in PATIENT_COLUMNS if c != "collection"]]
        index = index.merge(reg.drop_duplicates("patient_id"), on="patient_id", how="left")
    return index


def stored(index: pd.DataFrame) -> pd.DataFrame:
    return index[index["status"] == "stored"]


def anatomy_tables(index: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Counts per anatomy (and collection) and split, fates, energy and R80 spectra."""
    kept = stored(index)
    fates = pd.crosstab([index["anatomy"], index["split"]], index["status"])
    fates.insert(0, "planned", fates.sum(axis=1))
    fates["stored_fraction"] = fates.get("stored", 0) / fates["planned"]
    spectra = kept.groupby("anatomy")[["energy_mev", "r80_mm", "dose_in_body_fraction"]] \
        .describe(percentiles=[0.05, 0.5, 0.95])
    spectra.columns = [f"{a}_{b}" for a, b in spectra.columns]
    return {
        "by_anatomy_split": pd.crosstab(kept["anatomy"], kept["split"], margins=True, margins_name="total"),
        "by_collection_split": pd.crosstab([kept["anatomy"], kept["collection"]], kept["split"], margins=True,
                                           margins_name="total"),
        "patients_by_collection_split": kept.groupby(["anatomy", "collection", "split"])["patient_id"].nunique()
        .unstack(fill_value=0),
        "fates": fates,
        "spectra": spectra,
    }


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of non-negative counts (0: all equal, towards 1: one holds all)."""
    x = np.sort(np.asarray(counts, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, x.size + 1)
    return float((2 * ranks - x.size - 1).dot(x) / (x.size * x.sum()))


def concentration(counts: np.ndarray) -> Dict[str, float]:
    """How evenly beamlets spread over patients."""
    x = np.asarray(counts, dtype=float)
    share = x / x.sum()
    top = np.sort(x)[::-1][:max(1, int(np.ceil(0.1 * x.size)))]
    return {"n_patients": int(x.size), "beamlets": int(x.sum()), "mean": float(x.mean()),
            "median": float(np.median(x)), "min": int(x.min()), "max": int(x.max()), "gini": gini(x),
            "effective_patients": float(1.0 / np.sum(share ** 2)), "top_decile_share": float(top.sum() / x.sum())}


def patient_tables(index: pd.DataFrame, quotas: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """One row per patient (planned, stored, fields, energy) and concentration per split."""
    keys = ["patient_id", "anatomy", "split"] + [c for c in PATIENT_COLUMNS[1:] if c in index.columns]
    per = index.groupby(keys, dropna=False).agg(planned=("candidate_id", "size"),
                                                 stored=("status", lambda s: int((s == "stored").sum())),
                                                 fields=("gantry_deg", "nunique")).reset_index()
    energy = stored(index).groupby("patient_id")["energy_mev"].agg(energy_median_mev="median",
                                                                   energy_max_mev="max")
    per = per.merge(energy, on="patient_id", how="left")
    if quotas is not None:
        per = per.merge(quotas[["patient_id", "quota", "drawn"]], on="patient_id", how="left")
    rows = [{"split": split, **concentration(part["stored"].to_numpy())}
            for split, part in per.groupby("split")]
    rows.append({"split": "all", **concentration(per["stored"].to_numpy())})
    return {"per_patient": per.sort_values(["anatomy", "split", "stored"], ascending=[True, True, False]),
            "concentration": pd.DataFrame(rows).set_index("split")}


def gantry_frame(index: pd.DataFrame, d_smx: float, d_smy: float) -> pd.DataFrame:
    """Stored beamlets with their spot position on the isocentre plane (mm).

    ``angles_to_spot_position`` maps its first angle onto the z (gantry-axis)
    component through ``d_smy`` and returns ``(y, z)``: ``theta_x`` gives ``Y_g``,
    ``theta_y`` gives ``X_g``.
    """
    kept = stored(index).copy()
    lateral, axial = angles_to_spot_position(kept["theta_x_deg"].to_numpy(), kept["theta_y_deg"].to_numpy(),
                                             d_smx, d_smy)
    kept["x_g_mm"], kept["y_g_mm"] = lateral, axial
    kept["gantry_deg"] = np.mod(kept["gantry_deg"], 360.0)
    return kept


def gantry_tables(frame: pd.DataFrame, gantry_bin_deg: float = 15.0) -> Dict[str, pd.DataFrame]:
    """Beamlets per gantry sector and anatomy, and per spot position (lattice cell)."""
    edges = np.arange(0.0, 360.0 + gantry_bin_deg, gantry_bin_deg)
    sector = pd.cut(frame["gantry_deg"], edges, right=False, labels=edges[:-1])
    by_gantry = pd.crosstab(sector, frame["anatomy"]).reindex(edges[:-1], fill_value=0)
    by_gantry.index.name = "gantry_bin_lo_deg"
    spots = frame.groupby(["anatomy", "x_g_mm", "y_g_mm"]).size().rename("beamlets").reset_index()
    fields = frame.groupby(["anatomy", "split", "patient_id", "gantry_deg"]).size().rename("beamlets").reset_index()
    return {"by_gantry_sector": by_gantry, "spot_positions": spots, "fields": fields,
            "uniformity": uniformity(frame, by_gantry)}


def _dispersion(counts: np.ndarray) -> Dict[str, float]:
    """Spread of cell counts against the counting noise of a uniform draw.

    ``cv`` is the coefficient of variation of the counts, ``poisson_cv`` the one a
    uniform draw of the same total would show by chance (``1/sqrt(mean)``); the
    ratio ``dispersion`` is about 1 when the cells differ by noise alone.
    """
    x = np.asarray(counts, dtype=float)
    mean = x.mean()
    cv = float(x.std(ddof=1) / mean) if mean > 0 and x.size > 1 else float("nan")
    poisson = float(1.0 / np.sqrt(mean)) if mean > 0 else float("nan")
    return {"cells": int(x.size), "mean_per_cell": float(mean), "min": int(x.min()), "max": int(x.max()),
            "cv": cv, "poisson_cv": poisson, "dispersion": cv / poisson if poisson else float("nan")}


def uniformity(frame: pd.DataFrame, by_gantry: pd.DataFrame) -> pd.DataFrame:
    """How far the gantry sectors and spot-lattice cells are from a uniform spread, per anatomy."""
    rows = []
    for anatomy, part in [("all", frame), *frame.groupby("anatomy", sort=True)]:
        sectors = by_gantry.sum(axis=1) if anatomy == "all" else by_gantry[anatomy]
        rows.append({"anatomy": anatomy, "view": "gantry_sector", **_dispersion(sectors.to_numpy())})
        # the angle is drawn per field, not per beamlet: judge the sectors by fields too
        fields = part.drop_duplicates(["patient_id", "gantry_deg"])["gantry_deg"]
        per_sector = np.histogram(fields, bins=np.append(by_gantry.index.to_numpy(dtype=float), 360.0))[0]
        rows.append({"anatomy": anatomy, "view": "gantry_sector_fields", **_dispersion(per_sector)})
        cells = part.groupby(["x_g_mm", "y_g_mm"]).size()
        full = cells.reindex(pd.MultiIndex.from_product([np.sort(frame["x_g_mm"].unique()),
                                                         np.sort(frame["y_g_mm"].unique())]), fill_value=0)
        rows.append({"anatomy": anatomy, "view": "spot_cell", **_dispersion(full.to_numpy())})
    return pd.DataFrame(rows).set_index(["anatomy", "view"])
