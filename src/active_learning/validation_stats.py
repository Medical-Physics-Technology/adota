"""Statistics of the frozen validation set, before any Monte Carlo is paid for.

Three readings of the Hounsfield units, each restricted to the **patient**: the
largest connected component of :func:`src.mc_generation.geometry.body_mask`, so
external air and a couch separated from the patient by air are left out (lung,
which ``body_mask`` keeps inside the patient, stays in):

- ``ct_body``: every voxel of the patient in a validation (or pool) CT, at the
  1 mm grid the generator resamples to;
- ``crop_body``: the voxels of each selected beamlet's 60 x 60 x 320 mm crop that
  lie inside the patient, the region the model sees;
- ``beam_path``: the same voxels weighted by the beamlet's flux and cut at the
  distal edge of its analytic Bragg-peak zone, the tissue the beam crosses before
  it stops. Each beamlet contributes a total weight of one.

Histograms are counts in fixed HU bins, so they add across beamlets, CTs and
anatomies; summaries (percentiles, tissue-class fractions of
:data:`src.figures.ct_visualizations.HU_LUT`) are read off them.
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.acquisition import BeamletCandidate, prepare_ct
from src.acquisition.candidates import candidate_crops
from src.active_learning.candidates import pin_worker_threads
from src.active_learning.pool import PoolEntry, RecordResolver
from src.beamlets.bdl import BeamDataLibrary
from src.figures.ct_visualizations import HU_LUT
from src.mc_generation.geometry import body_mask
from src.mc_generation.robustness import field_geometry
from src.mc_generation.sweep import RobustnessConfig

logger = logging.getLogger(__name__)

HU_MIN, HU_MAX = -1024.0, 3072.0
KINDS = ("ct_body", "crop_body", "beam_path")
PROVENANCE_KEYS = ("manufacturer", "model", "kvp", "convolution_kernel", "slice_thickness",
                   "pixel_spacing_x", "pixel_spacing_y", "n_slices", "study_date")


def hu_edges(bin_width: float = 10.0) -> np.ndarray:
    """Bin edges covering -1024 to 3072 HU (the scale's range) in ``bin_width``
    steps, on multiples of the width so the HU_LUT thresholds (-950, -500, -100,
    ...) fall on edges and tissue fractions are exact."""
    lo = np.floor(HU_MIN / bin_width) * bin_width
    hi = np.ceil(HU_MAX / bin_width) * bin_width
    return np.arange(lo, hi + bin_width / 2.0, bin_width)


def bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of every value; values outside the range go to the first or last bin."""
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def histogram_summary(counts: np.ndarray, edges: np.ndarray) -> Dict[str, float]:
    """Mean, spread, percentiles and HU_LUT tissue fractions of a binned distribution."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    out: Dict[str, float] = {"weight": float(total)}
    if total <= 0:
        return out
    centres = (edges[:-1] + edges[1:]) / 2.0
    p = counts / total
    mean = float((p * centres).sum())
    out.update(mean_hu=mean, std_hu=float(np.sqrt((p * (centres - mean) ** 2).sum())))
    cdf = np.cumsum(p)
    for q in (1, 5, 25, 50, 75, 95, 99):
        out[f"p{q:02d}_hu"] = float(centres[min(int(np.searchsorted(cdf, q / 100.0)), len(centres) - 1)])
    for name, lo, hi, _ in HU_LUT:
        inside = (centres >= lo) & (centres < hi)
        out[f"frac_{tissue_key(name)}"] = float(p[inside].sum())
    return out


def tissue_key(name: str) -> str:
    return name.lower().replace(" / ", "_").replace(" ", "_")


def _load(entry: PoolEntry, iso_spacing_mm: float) -> Tuple[object, sitk.Image, sitk.Image]:
    rec = RecordResolver([entry]).record(entry)
    image = rec.load_image()
    return rec, image, prepare_ct(image, iso_spacing_mm)


def ct_body_statistics(entry: PoolEntry, iso_spacing_mm: float, edges: np.ndarray,
                       body_hu_threshold: float = -500.0) -> Tuple[Dict, np.ndarray]:
    """One CT: the HU histogram of its patient and a row describing the scan."""
    rec, image, ct = _load(entry, iso_spacing_mm)
    arr = sitk.GetArrayFromImage(ct)
    every = body_mask(ct, body_hu_threshold) > 0.5
    patient = body_mask(ct, body_hu_threshold, largest_component=True) > 0.5
    counts = np.bincount(bin_index(arr[patient], edges), minlength=len(edges) - 1).astype(float)
    voxel_ml = float(np.prod(ct.GetSpacing())) / 1000.0
    prov = rec.provenance or {}
    row = {"patient_id": entry.patient_id, "anatomy": entry.anatomy, "role": entry.role,
           "dataset_name": entry.dataset_name,
           "original_spacing_mm": " x ".join(f"{s:.3g}" for s in image.GetSpacing()),
           "z_extent_mm": float(image.GetSize()[2] * image.GetSpacing()[2]),
           "grid_1mm": " x ".join(str(s) for s in ct.GetSize()),
           "patient_volume_l": float(patient.sum()) * voxel_ml / 1000.0,
           "non_air_volume_l": float(every.sum()) * voxel_ml / 1000.0,
           "removed_by_largest_component": float(1.0 - patient.sum() / max(every.sum(), 1)),
           **{k: prov.get(k) for k in PROVENANCE_KEYS if k in prov},
           **histogram_summary(counts, edges)}
    return row, counts


def _ct_task(args) -> Tuple[Dict, np.ndarray]:
    return ct_body_statistics(*args)


def beamlet_statistics(entry: PoolEntry, rows: pd.DataFrame, rob_cfg: RobustnessConfig, bdl_path: str,
                       edges: np.ndarray) -> Tuple[pd.DataFrame, Dict[Tuple[str, float], np.ndarray]]:
    """The selected beamlets of one CT: per-beamlet summaries and the
    ``crop_body`` / ``beam_path`` histograms summed per ``(kind, energy)``.

    ``rows`` needs the candidate columns and ``bp_range_max_mm`` (the distal edge
    of the analytic Bragg-peak zone, from the candidate scoring).
    """
    _, _, ct = _load(entry, rob_cfg.iso_spacing_mm)
    bdl = BeamDataLibrary.from_file(bdl_path)
    roi = tuple(rob_cfg.roi_size)
    n_bins = len(edges) - 1
    hists: Dict[Tuple[str, float], np.ndarray] = {}
    out: List[Dict] = []
    for gantry, part in rows.groupby("gantry_deg", sort=True):
        geom = field_geometry(ct, float(gantry), rob_cfg, bdl)
        patient = body_mask(geom.ct, rob_cfg.body_hu_threshold, largest_component=True)
        depth_mm = np.arange(roi[2]) + 0.5
        for row in part.itertuples():
            candidate = BeamletCandidate(float(row.gantry_deg), float(row.energy_mev), float(row.theta_x_deg),
                                         float(row.theta_y_deg), str(row.candidate_id))
            crops = candidate_crops(geom, candidate, bdl, roi, extra=[patient])
            record = {"candidate_id": row.candidate_id, "patient_id": entry.patient_id,
                      "anatomy": entry.anatomy, "energy_mev": float(row.energy_mev), "cropped": crops is not None}
            if crops is None:
                out.append(record)
                continue
            ct_crop, flux, (mask,) = crops
            inside = mask > 0.5
            idx = bin_index(ct_crop, edges)
            crop_counts = np.bincount(idx[inside], minlength=n_bins).astype(float)
            path = inside & (depth_mm <= float(row.bp_range_max_mm))[None, None, :]
            weights = np.where(path, flux, 0.0)
            total = float(weights.sum())
            path_counts = (np.bincount(idx.ravel(), weights=weights.ravel() / total, minlength=n_bins)
                           if total > 0 else np.zeros(n_bins))
            axis = inside[roi[0] // 2, roi[1] // 2, :]
            record.update(body_fraction_of_crop=float(inside.mean()),
                          entrance_depth_mm=float(np.argmax(axis)) if axis.any() else np.nan,
                          **{f"crop_{k}": v for k, v in histogram_summary(crop_counts, edges).items()},
                          **{f"path_{k}": v for k, v in histogram_summary(path_counts, edges).items()})
            out.append(record)
            for kind, counts in (("crop_body", crop_counts), ("beam_path", path_counts)):
                key = (kind, float(row.energy_mev))
                hists[key] = hists.get(key, 0.0) + counts
    return pd.DataFrame(out), hists


def _beamlet_task(args):
    return args[0].patient_id, beamlet_statistics(*args)


def run_pool(fn, payload: Sequence, n_workers: int, what: str) -> List:
    """``fn`` over ``payload`` in a process pool pinned to one thread per worker."""
    results: List = []
    if n_workers <= 1:
        pin_worker_threads()
        for i, args in enumerate(payload, 1):
            results.append(fn(args))
            logger.info("%s: %d/%d", what, i, len(payload))
        return results
    with ProcessPoolExecutor(max_workers=n_workers, initializer=pin_worker_threads) as pool:
        for i, result in enumerate(pool.map(fn, payload), 1):
            results.append(result)
            logger.info("%s: %d/%d", what, i, len(payload))
    return results


def ct_statistics(entries: Sequence[PoolEntry], iso_spacing_mm: float, edges: np.ndarray,
                  body_hu_threshold: float = -500.0, n_workers: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Every CT: the per-CT table and the long histogram frame (kind ``ct_body``)."""
    payload = [(e, iso_spacing_mm, edges, body_hu_threshold) for e in entries]
    results = run_pool(_ct_task, payload, n_workers, "CT histograms")
    table = pd.DataFrame([r for r, _ in results])
    hist = long_histogram([({"kind": "ct_body", "patient_id": r["patient_id"], "anatomy": r["anatomy"],
                             "role": r["role"]}, c) for r, c in results], edges)
    return table, hist


def selection_beamlet_statistics(entries: Sequence[PoolEntry], selected: pd.DataFrame,
                                 rob_cfg: RobustnessConfig, bdl_path: str, edges: np.ndarray,
                                 n_workers: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Every selected beamlet: the per-beamlet table and the long histogram frame
    (kinds ``crop_body`` and ``beam_path``, per CT and energy)."""
    by_patient = {e.patient_id: e for e in entries}
    payload = [(by_patient[p], part, rob_cfg, bdl_path, edges)
               for p, part in selected.groupby("patient_id", sort=True)]
    results = run_pool(_beamlet_task, payload, n_workers, "beamlet histograms")
    table = pd.concat([frame for _, (frame, _) in results], ignore_index=True)
    items = []
    for patient_id, (_, hists) in results:
        anatomy = by_patient[patient_id].anatomy
        for (kind, energy), counts in hists.items():
            items.append(({"kind": kind, "patient_id": patient_id, "anatomy": anatomy, "role": "validation",
                           "energy_mev": energy}, counts))
    return table, long_histogram(items, edges)


def long_histogram(items: Sequence[Tuple[Dict, np.ndarray]], edges: np.ndarray) -> pd.DataFrame:
    """One row per (label set, bin): ``bin_lo``, ``bin_hi`` and ``weight``."""
    frames = []
    for labels, counts in items:
        frame = pd.DataFrame({"bin_lo": edges[:-1], "bin_hi": edges[1:], "weight": np.asarray(counts, float)})
        for k, v in labels.items():
            frame.insert(0, k, v)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarise_histograms(hist: pd.DataFrame, by: Sequence[str], edges: np.ndarray) -> pd.DataFrame:
    """:func:`histogram_summary` of the summed histograms per group of ``by``."""
    rows = []
    for key, part in hist.groupby(list(by), sort=True):
        counts = part.groupby("bin_lo")["weight"].sum().reindex(edges[:-1], fill_value=0.0).to_numpy()
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     **histogram_summary(counts, edges)})
    return pd.DataFrame(rows)


def composition_tables(selected: pd.DataFrame, candidates: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Counts of the selection by anatomy, energy, patient, gantry and steering, and
    the candidates' validity by anatomy and energy."""
    out = {
        "by_anatomy_energy": selected.groupby(["anatomy", "energy_mev"]).size().unstack(fill_value=0),
        "by_patient_energy": selected.groupby(["anatomy", "patient_id", "energy_mev"]).size().unstack(fill_value=0),
        "by_patient_gantry": selected.groupby(["anatomy", "patient_id", "gantry_deg"]).size().rename("n")
        .reset_index(),
        "steering": selected.groupby(["anatomy", "theta_x_deg", "theta_y_deg"]).size().rename("n").reset_index(),
    }
    if candidates is not None:
        cand = candidates.assign(reason=candidates["reason"].fillna("").replace("", "valid"))
        out["candidate_validity"] = cand.groupby(["anatomy", "energy_mev", "reason"]).size() \
            .unstack(fill_value=0).assign(n=lambda f: f.sum(axis=1))
        out["candidate_validity"]["valid_fraction"] = out["candidate_validity"].get("valid", 0) \
            / out["candidate_validity"]["n"]
    return out


def metric_summary(selected: pd.DataFrame, metrics: Sequence[str], by: Sequence[str] = ("anatomy",)) -> pd.DataFrame:
    """Median, quartiles and 5/95th percentiles of ``metrics`` per group."""
    rows = []
    for key, part in selected.groupby(list(by), sort=True):
        for m in metrics:
            v = part[m].astype(float)
            labels = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            rows.append({**labels, "metric": m, "n": int(v.notna().sum()),
                         **{f"p{q:02d}": float(v.quantile(q / 100.0)) for q in (5, 25, 50, 75, 95)}})
    return pd.DataFrame(rows)
