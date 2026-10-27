"""Which patients the pool may use, and in which split.

Every CT series of the configured collections is screened from its DICOM headers
only (no pixel data):

- the image-quality gates of :class:`src.datasets.tcia.TCIADataset`;
- its 1 mm grid, computed the way :func:`src.mc_generation.geometry.resample_to_isotropic`
  will build it, compared with the scans of the training HDF5 (``*_index.csv``).
  The training set records a scan only by that grid (``patient_key`` hashes origin,
  size and spacing), so a match on the grid is how a training patient is found;
- claims of other experiments (registry CSVs, plan directories), which force a
  split or exclude the patient.

The remaining patients are split per collection, seeded, in the configured
fractions. The result is one CSV, the provenance of every split that is drawn from
it, on any machine.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.datasets.tcia import TCIADataset

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
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


def grid_from_headers(series_dir: str, series_uid: str, spacing_mm: float = 1.0) -> Dict[str, float]:
    """The series' grid and the grid it resamples to, from its first and last headers."""
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(series_dir, series_uid)
    if not files:
        raise RuntimeError(f"no files for series {series_uid} in {series_dir}")
    heads = []
    for path in (files[0], files[-1]):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        heads.append(reader)
    first, last = heads
    sx, sy = first.GetSpacing()[:2]
    nx, ny = first.GetSize()[:2]
    n = len(files)
    dz = abs(last.GetOrigin()[2] - first.GetOrigin()[2]) / max(n - 1, 1) if n > 1 else first.GetSpacing()[2]
    origin = [float(v) for v in first.GetOrigin()]
    z_ends = sorted([origin[2], float(last.GetOrigin()[2])])
    return {"n_slices": n, "spacing_x_mm": float(sx), "spacing_y_mm": float(sy), "spacing_z_mm": float(dz),
            "origin_x_mm": origin[0], "origin_y_mm": origin[1], "origin_z_mm": z_ends[0], "origin_z_top_mm": z_ends[1],
            "grid_x": int(round(nx * sx / spacing_mm)), "grid_y": int(round(ny * sy / spacing_mm)),
            "grid_z": int(round(n * dz / spacing_mm)), "z_extent_mm": float(n * dz)}


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


def screen_collection(spec: CollectionSpec, scans: pd.DataFrame, spacing_mm: float = 1.0) -> pd.DataFrame:
    """Every patient of one collection: training match, grid and QC.

    The training match is computed for every patient whatever its QC, on the
    largest CT series without the configured gates and on the series the gates
    select, so a training patient that fails today's gates is still counted as
    matched (it cannot leak either way, but an unmatched training scan must mean
    that its collection is not on this machine, not that a gate hid it).
    """
    common = dict(root=spec.root, collection=spec.collection, anatomy=spec.anatomy, name=spec.name,
                  selection="first")
    ungated = {r.patient_id: r for r in (lambda d: [d.record(i) for i in range(len(d))])(TCIADataset(**common))}
    gated_ds = TCIADataset(**common, qc=spec.qc or None)
    gated = {r.patient_id: r for r in (gated_ds.record(i) for i in range(len(gated_ds)))}
    rows = []
    base = {"collection": spec.name, "anatomy": spec.anatomy, "root": spec.root,
            "dataset_collection": spec.collection or ""}
    for pid in sorted(d.name for d in Path(gated_ds.collection_dir).iterdir() if d.is_dir()):
        row = {**base, "patient_id": pid}
        matches: List[str] = []
        for rec in {id(r): r for r in (ungated.get(pid), gated.get(pid)) if r is not None}.values():
            try:
                matches += match_training(grid_from_headers(rec.series_dir, rec.series_uid, spacing_mm), scans)
            except Exception as exc:  # one unreadable series must not end the screen
                row["header_error"] = repr(exc)
        row.update(training_matches=" ".join(sorted(set(matches))), n_training_matches=len(set(matches)))
        rec = gated.get(pid)
        if rec is None:
            row.update(qc_pass=False, qc_reason="no CT series passing QC" if pid in ungated else "no CT series")
            rows.append(row)
            continue
        try:
            grid = grid_from_headers(rec.series_dir, rec.series_uid, spacing_mm)
        except Exception as exc:
            row.update(qc_pass=False, qc_reason=f"header: {exc}")
            rows.append(row)
            continue
        prov = rec.provenance or {}
        row.update(grid, series_uid=rec.series_uid, series_dir=rec.series_dir, qc_pass=True, qc_reason="",
                   kvp=prov.get("kvp"), kernel=prov.get("convolution_kernel"), manufacturer=prov.get("manufacturer"))
        rows.append(row)
    frame = pd.DataFrame(rows)
    logger.info("%s: %d patients, %d pass QC, %d match a training scan", spec.name, len(frame),
                int(frame["qc_pass"].sum()), int((frame["n_training_matches"] > 0).sum()))
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
