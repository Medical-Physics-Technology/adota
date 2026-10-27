"""Which patients the pool may use, and in which split.

Every CT series of the configured collections is screened from its DICOM headers
only (no pixel data), all of them: files are grouped by the series UID each one
carries, and a series must stack into a volume (:mod:`src.provenance.series_geometry`):

- the image-quality gates of :class:`src.datasets.tcia.TCIADataset`, on the measured
  slice count and z extent;
- the 1 mm grid of every CT series of the patient, computed the way
  :func:`src.mc_generation.geometry.resample_to_isotropic` will build it, compared
  with the scans of the training HDF5 (``*_index.csv``). The training set records a
  scan only by that grid (``patient_key`` hashes origin, size and spacing), so a
  match on the grid is how a training patient is found;
- claims of other experiments (registry CSVs, plan directories), which force a
  split or exclude the patient.

The remaining patients are split per collection, seeded, in the configured
fractions. The result is one CSV, the provenance of every split that is drawn from
it, on any machine.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.datasets.tcia import auto_collection, best_candidate, ct_series_candidates
from src.provenance.dicom_qc import gates_from_dict
from src.provenance.series_geometry import SeriesGeometry

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
# TCIADataset's built-in gates, which the configured ``qc`` refines.
_DATASET_DEFAULTS = dict(min_slices=50, max_slices=1000, require_ct=True, require_monochrome2=True)
EXCLUDED = "excluded"


@dataclass
class CollectionSpec:
    """One DICOM collection, in the shape of the generator configs' ``datasets`` entries."""

    name: str
    anatomy: str
    root: str
    collection: Optional[str] = None
    qc: Dict = field(default_factory=dict)


@dataclass
class Claim:
    """Patients another experiment owns: forced into ``role`` (a split, or
    ``excluded``), or mapped per row through ``role_map`` of a CSV column."""

    role: Optional[str] = None
    patient_ids: Sequence[str] = ()
    file: Optional[str] = None
    role_column: Optional[str] = None
    role_map: Dict[str, str] = field(default_factory=dict)
    plan_dir: Optional[str] = None          # patient ids parsed from plan directory names
    why: str = ""

    def assignments(self) -> Dict[str, str]:
        out: Dict[str, str] = {p: self.role for p in self.patient_ids}
        if self.file:
            table = pd.read_csv(self.file, dtype=str)
            for _, row in table.iterrows():
                role = self.role_map.get(row[self.role_column], self.role) if self.role_column else self.role
                if role:
                    out[str(row["patient_id"])] = role
        if self.plan_dir:
            pattern = re.compile(r"^((?:LUNG1|Prostate-AEC|Lung_Dx|StageII-Colorectal-CT)-?[A-Z]?\d+)")
            for entry in sorted(Path(self.plan_dir).iterdir()):
                match = pattern.match(entry.name)
                if match:
                    out[match.group(1)] = self.role
        bad = {r for r in out.values() if r not in (*SPLITS, EXCLUDED)}
        if bad:
            raise ValueError(f"claim roles must be one of {(*SPLITS, EXCLUDED)}, got {sorted(bad)}")
        return out


def grid_from_geometry(geom: SeriesGeometry, spacing_mm: float = 1.0) -> Dict[str, float]:
    """The series' grid and the grid it resamples to, from its checked headers."""
    sy, sx = (float(v) for v in geom.pixel_spacing_mm)
    n = geom.n_slices
    z_lo, z_hi = geom.z_ends_mm
    dz = (z_hi - z_lo) / max(n - 1, 1)
    return {"n_slices": n, "spacing_x_mm": sx, "spacing_y_mm": sy, "spacing_z_mm": float(dz),
            "origin_x_mm": geom.origin_mm[0], "origin_y_mm": geom.origin_mm[1], "origin_z_mm": z_lo,
            "origin_z_top_mm": z_hi, "grid_x": int(round(geom.columns * sx / spacing_mm)),
            "grid_y": int(round(geom.rows * sy / spacing_mm)), "grid_z": int(round(n * dz / spacing_mm)),
            "z_extent_mm": float(n * dz)}


def training_scans(index_csv: Path) -> pd.DataFrame:
    """One row per scan of the training HDF5: its key and grid."""
    cols = ["patient_key", "image_origin_mm_0", "image_origin_mm_1", "image_origin_mm_2", "image_size_vox_0",
            "image_size_vox_2", "image_spacing_mm_2", "source_dataset"]
    index = pd.read_csv(index_csv, usecols=cols, dtype={"patient_key": str})
    return index.drop_duplicates("patient_key").reset_index(drop=True)


def match_training(grid: Mapping[str, float], scans: pd.DataFrame, tol_mm: float = 2.0) -> List[str]:
    """Training scans with the same 1 mm grid: x size, z extent, x and y origin, and
    a z origin at either end of the series.

    The generator that built the training set recorded the z origin of the slice
    it read first, which for some collections is the highest one (StageII-Colorectal:
    the training z origin is the lowest slice plus the extent minus one slice), so
    both ends are accepted; the in-plane origins, which it kept, make the match
    specific.
    """
    z_extent = scans["image_size_vox_2"] * scans["image_spacing_mm_2"]
    z_top = grid.get("origin_z_top_mm", grid["origin_z_mm"])
    z_close = (((scans["image_origin_mm_2"] - grid["origin_z_mm"]).abs() <= tol_mm)
               | ((scans["image_origin_mm_2"] - z_top).abs() <= tol_mm))
    hit = scans[((scans["image_size_vox_0"] - grid["grid_x"]).abs() <= 2)
                & ((z_extent - grid["z_extent_mm"]).abs() <= 2 * tol_mm) & z_close
                & ((scans["image_origin_mm_0"] - grid["origin_x_mm"]).abs() <= tol_mm)
                & ((scans["image_origin_mm_1"] - grid["origin_y_mm"]).abs() <= tol_mm)]
    return hit["patient_key"].tolist()


def screen_patient(spec: CollectionSpec, patient_id: str, scans: pd.DataFrame,
                   spacing_mm: float = 1.0) -> Dict[str, object]:
    """One patient: every CT series checked from all of its headers, matched to the
    training scans, and the largest series passing QC selected.

    The training match runs on every CT series, whatever its QC: a training scan
    can be any series of the patient, and a patient whose series fails today's
    gates is still a training patient.
    """
    gates = gates_from_dict(spec.qc, **_DATASET_DEFAULTS)
    if gates.max_slice_gap_ratio is None:
        raise ValueError(f"{spec.name}: the screen needs qc.max_slice_gap_ratio")
    collection = spec.collection or auto_collection(spec.root)
    row: Dict[str, object] = {"collection": spec.name, "anatomy": spec.anatomy, "root": spec.root,
                              "dataset_collection": spec.collection or "", "patient_id": patient_id}
    try:
        candidates = ct_series_candidates(str(Path(spec.root) / collection / patient_id), gates)
    except Exception as exc:  # one unreadable patient must not end the screen
        candidates, row["header_error"] = [], repr(exc)
    matches = sorted({k for c in candidates for k in match_training(grid_from_geometry(c.geometry, spacing_mm), scans)})
    row.update(training_matches=" ".join(matches), n_training_matches=len(matches), n_ct_series=len(candidates),
               n_ct_series_not_stacking=sum(not c.geometry.ok for c in candidates),
               max_slice_gap_ratio=gates.max_slice_gap_ratio)
    best = best_candidate(candidates)
    if best is None:
        largest = max(candidates, key=lambda c: c.geometry.n_slices, default=None)
        reason = "no CT series" if largest is None else \
            f"no CT series passing QC; largest: {', '.join(largest.qc_reasons)}"
        row.update(qc_pass=False, qc_reason=reason)
        return row
    prov = best.provenance
    row.update(grid_from_geometry(best.geometry, spacing_mm), series_uid=best.geometry.series_uid,
               series_dir=best.series_dir, slice_step_mm=best.geometry.step_mm,
               max_slice_step_mm=best.geometry.max_step_mm, qc_pass=True, qc_reason="", kvp=prov.get("kvp"),
               kernel=prov.get("convolution_kernel"), manufacturer=prov.get("manufacturer"))
    return row


def _screen_task(args) -> Dict[str, object]:
    return screen_patient(*args)


def screen_collection(spec: CollectionSpec, scans: pd.DataFrame, spacing_mm: float = 1.0,
                      n_workers: int = 1) -> pd.DataFrame:
    """Every patient of one collection: training match, geometry, QC."""
    collection_dir = Path(spec.root) / (spec.collection or auto_collection(spec.root))
    patients = sorted(d.name for d in collection_dir.iterdir() if d.is_dir())
    tasks = [(spec, pid, scans, spacing_mm) for pid in patients]
    if n_workers > 1:
        with ProcessPoolExecutor(n_workers) as pool:
            rows = list(pool.map(_screen_task, tasks, chunksize=2))
    else:
        rows = [_screen_task(t) for t in tasks]
    frame = pd.DataFrame(rows)
    logger.info("%s: %d patients, %d pass QC, %d match a training scan, %d CT series do not stack",
                spec.name, len(frame), int(frame["qc_pass"].sum()), int((frame["n_training_matches"] > 0).sum()),
                int(frame["n_ct_series_not_stacking"].sum()))
    return frame


def matched_training_scans(registry: pd.DataFrame, scans: pd.DataFrame) -> pd.DataFrame:
    """Per training-set source, how many of its scans a screened patient matched."""
    hit = {key for keys in registry["training_matches"].fillna("") for key in str(keys).split() if key}
    return scans.assign(matched=scans["patient_key"].isin(hit)).groupby("source_dataset")["matched"] \
        .agg(matched="sum", scans="size").reset_index()


def assign_splits(screened: pd.DataFrame, claims: Sequence[Claim], fractions: Mapping[str, float],
                  seed: int, min_z_extent_mm: float = 0.0) -> pd.DataFrame:
    """Add ``split`` and ``split_reason`` to the screened table.

    Order of precedence: a training match excludes the patient whatever else
    claims it; then failed QC; then claims (forced split or exclusion); then too
    short a scan; the rest is split per collection with a seeded shuffle, in the
    configured fractions (largest-remainder rounding).
    """
    out = screened.copy()
    out["split"], out["split_reason"] = "", ""
    claimed: Dict[str, tuple] = {}
    for claim in claims:
        for pid, role in claim.assignments().items():
            claimed.setdefault(pid, (role, claim.why or claim.file or claim.plan_dir or "listed"))
    matches = pd.to_numeric(out.get("n_training_matches"), errors="coerce").fillna(0)
    for i, row in out.iterrows():
        if matches[i] > 0:
            out.loc[i, ["split", "split_reason"]] = [EXCLUDED, f"training scan {row['training_matches']}"]
        elif not bool(row.get("qc_pass", False)):
            out.loc[i, ["split", "split_reason"]] = [EXCLUDED, f"qc: {row.get('qc_reason', '')}"]
        elif row["patient_id"] in claimed:
            role, why = claimed[row["patient_id"]]
            out.loc[i, ["split", "split_reason"]] = [role, f"claim: {why}"]
        elif float(row.get("z_extent_mm", 0.0) or 0.0) < min_z_extent_mm:
            out.loc[i, ["split", "split_reason"]] = [EXCLUDED, f"z extent below {min_z_extent_mm:g} mm"]
    names = list(fractions)
    weights = np.asarray([float(fractions[n]) for n in names])
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    for collection, part in out[out["split"] == ""].groupby("collection", sort=True):
        ids = part.index.to_numpy()
        rng.shuffle(ids)
        raw = weights * len(ids)
        counts = np.floor(raw).astype(int)
        for k in np.argsort(-(raw - counts))[: len(ids) - counts.sum()]:
            counts[k] += 1
        start = 0
        for name, count in zip(names, counts):
            out.loc[ids[start:start + count], ["split", "split_reason"]] = [name, f"drawn (seed {seed})"]
            start += count
    return out


def split_summary(registry: pd.DataFrame) -> pd.DataFrame:
    """Patients per collection and split."""
    return registry.groupby(["collection", "split"]).size().unstack(fill_value=0)
