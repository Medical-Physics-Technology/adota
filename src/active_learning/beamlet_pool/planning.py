"""Which beamlets to simulate: candidates per patient, the overshoot gate, batches.

Per patient of the registry: ``n_gantry`` field angles drawn at random (seeded
by the patient), steering pairs drawn on the generator's lattice, and energies
drawn **uniformly** in ``energy_range_mev``. A candidate is kept only if, on the
analytic surrogate and the model grid the network sees,

- its ray stays inside the CT (the generator's extraction succeeds),
- the analytic R80 (:func:`src.metrics.range_metrics.compute_range_metrics`, NaN
  when the depth dose never falls back) lies at least ``range_margin_mm`` before
  the end of the 320 mm crop, and
- the flux-weighted share of the patient at the analytic peak depth is at least
  ``min_body_fraction_at_peak`` (the beam stops inside the patient, not in the
  air or couch beyond it; patient = largest component of ``body_mask``).

Rejection keeps the accepted energies uniform over what each beam path allows:
a thorax rarely accepts more than about 180 MeV. The same R80 and patient checks
are repeated on the Monte Carlo dose after simulation (:mod:`.store`).

Each patient gets the same quota inside its split; the split's share of the total
is configured. Whole patients go to a batch, so a field's beamlets run in one
MCsquare call, and the batch order interleaves the splits, so a run stopped early
still holds every split in proportion.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.acquisition import BeamletCandidate, prepare_ct
from src.acquisition.candidates import candidate_crops, model_grid_inputs
from src.acquisition.surrogate import analytic_dose
from src.active_learning.candidates import candidate_id
from src.active_learning.pool import PoolEntry, RecordResolver
from src.beamlets.bdl import BeamDataLibrary
from src.mc_generation.geometry import body_mask
from src.mc_generation.robustness import field_geometry
from src.mc_generation.sweep import RobustnessConfig, angle_tag
from src.metrics.range_metrics import compute_range_metrics

logger = logging.getLogger(__name__)

MODEL_DZ_MM = 2.0
REFERENCE_SECONDS = 4.2
"""Measured Monte Carlo seconds per beamlet in beamlet mode at 1e6 primaries on
the full machine (energy-gantry run: 11,531 beamlets, 13.4 h, 102.6 and 135 MeV)."""
REFERENCE_ENERGY_MEV = 118.8
PACKED_MB, RAW_MB = 2.6, 16.1
"""Measured storage per beamlet: packed shard (CT int16 + dose float32, gzip),
and the generator's raw npy (with the float64 flux)."""


@dataclass
class PlanSettings:
    total_beamlets: int = 50000
    split_shares: Dict[str, float] = field(default_factory=lambda: {"train": 0.70, "validation": 0.15,
                                                                    "test": 0.15})
    energy_range_mev: Tuple[float, float] = (70.0, 250.0)
    energy_decimals: int = 1
    n_gantry: int = 2
    theta_half_range_deg: float = 1.5
    lattice_n: int = 18
    range_margin_mm: float = 10.0
    min_body_fraction_at_peak: float = 0.5
    oversample: float = 1.04
    max_draw_factor: float = 8.0
    batch_size: int = 2000
    seed: int = 20261027
    prefix: str = "pool"
    version: int = 1


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha1("|".join(map(str, parts)).encode()).digest()[:8], "big")


def patient_quotas(registry: pd.DataFrame, settings: PlanSettings) -> Dict[str, int]:
    """Beamlets per patient: the split's share of the total, evenly over its
    patients, times the oversampling for Monte Carlo QA losses."""
    quotas: Dict[str, int] = {}
    for split, share in settings.split_shares.items():
        ids = registry.loc[registry["split"] == split, "patient_id"].tolist()
        if not ids:
            continue
        per = settings.total_beamlets * share / len(ids) * settings.oversample
        for pid in ids:
            quotas[pid] = int(np.ceil(per))
    return quotas


def gate(geom, candidate: BeamletCandidate, bdl: BeamDataLibrary, roi: Tuple[int, int, int],
         patient: np.ndarray, settings: PlanSettings) -> Dict[str, object]:
    """The input-only overshoot gate of one candidate (see the module docstring)."""
    crops = candidate_crops(geom, candidate, bdl, roi, extra=[patient])
    if crops is None:
        return {"valid": False, "reason": "roi_out_of_bounds"}
    ct1, flux1, (body1,) = crops
    ct_hu, flux = model_grid_inputs(ct1, flux1, candidate.energy_mev)
    idd = analytic_dose(ct_hu, flux, candidate.energy_mev, MODEL_DZ_MM).sum(axis=(1, 2))
    rm = compute_range_metrics(idd, MODEL_DZ_MM)
    out: Dict[str, object] = {"analytic_r80_mm": float(rm.r80_mm)}
    if not np.isfinite(rm.r80_mm) or rm.r80_mm > roi[2] - settings.range_margin_mm:
        return {**out, "valid": False, "reason": "overshoot_crop"}
    k = min(int((int(np.argmax(idd)) + 0.5) * MODEL_DZ_MM), roi[2] - 1)
    weights = flux1[:, :, k]
    share = float((weights * (body1[:, :, k] > 0.5)).sum() / max(float(weights.sum()), 1e-30))
    out.update(analytic_peak_mm=float(k), body_fraction_at_peak=share)
    if share < settings.min_body_fraction_at_peak:
        return {**out, "valid": False, "reason": "peak_outside_patient"}
    return {**out, "valid": True, "reason": ""}


def plan_patient(entry: PoolEntry, split: str, quota: int, rob_cfg: RobustnessConfig, bdl_path: str,
                 settings: PlanSettings) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Accepted beamlets of one patient (up to ``quota``) and the draw counts."""
    rec = RecordResolver([entry]).record(entry)
    ct = prepare_ct(rec.load_image(), rob_cfg.iso_spacing_mm)
    bdl = BeamDataLibrary.from_file(bdl_path)
    roi = tuple(rob_cfg.roi_size)
    rng = np.random.default_rng(_seed(settings.seed, rec.uid, "pool"))
    gantries = rng.uniform(0.0, 360.0, settings.n_gantry)
    lattice = np.linspace(-settings.theta_half_range_deg, settings.theta_half_range_deg, settings.lattice_n)
    lo, hi = settings.energy_range_mev
    per_field = [quota // settings.n_gantry + (1 if g < quota % settings.n_gantry else 0)
                 for g in range(settings.n_gantry)]
    rows, counts = [], {"drawn": 0, "accepted": 0}
    for gantry, want in zip(gantries, per_field):
        geom = field_geometry(ct, float(gantry), rob_cfg, bdl)
        patient = body_mask(geom.ct, rob_cfg.body_hu_threshold, largest_component=True)
        accepted, drawn, limit = 0, 0, int(np.ceil(want * settings.max_draw_factor))
        seen = set()
        while accepted < want and drawn < limit:
            energy = round(float(rng.uniform(lo, hi)), settings.energy_decimals)
            tx, ty = float(rng.choice(lattice)), float(rng.choice(lattice))
            cid = candidate_id(rec.patient_id, float(gantry), energy, tx, ty)
            if cid in seen:
                continue
            seen.add(cid)
            drawn += 1
            cand = BeamletCandidate(float(gantry), energy, tx, ty, cid)
            verdict = gate(geom, cand, bdl, roi, patient, settings)
            row = {"candidate_id": cid, "patient_id": rec.patient_id, "anatomy": rec.anatomy,
                   "dataset_name": rec.dataset_name, "provenance_uid": rec.uid, "split": split,
                   "gantry_deg": float(gantry), "energy_mev": energy, "theta_x_deg": tx, "theta_y_deg": ty,
                   "field_dir": f"{settings.prefix}_{rec.anatomy}_{rec.patient_id}_g{angle_tag(float(gantry))}"
                                f"_v{settings.version}", **verdict}
            rows.append(row)
            accepted += bool(verdict["valid"])
        counts["drawn"] += drawn
        counts["accepted"] += accepted
        if accepted < want:
            logger.warning("%s g%.1f: %d of %d accepted after %d draws", rec.patient_id, gantry, accepted, want,
                           drawn)
    return pd.DataFrame(rows), counts


def _plan_task(args):
    entry = args[0]
    try:
        return entry.patient_id, plan_patient(*args), ""
    except Exception as exc:  # one unreadable CT must not end the plan
        return entry.patient_id, (pd.DataFrame(), {"drawn": 0, "accepted": 0}), repr(exc)


def assign_batches(accepted: pd.DataFrame, batch_size: int, seed: int) -> pd.Series:
    """Batch index per row: whole patients, splits interleaved by their progress
    (a patient's rank inside its split over the split's size), filled to
    ``batch_size`` beamlets."""
    patients = accepted.groupby("patient_id").agg(split=("split", "first"), n=("candidate_id", "size"))
    rng = np.random.default_rng(seed)
    patients = patients.loc[rng.permutation(patients.index.to_numpy())]
    patients["progress"] = patients.groupby("split").cumcount() / patients.groupby("split")["n"].transform("size")
    patients = patients.sort_values("progress", kind="stable")
    batch, filled, out = 0, 0, {}
    for pid, row in patients.iterrows():
        if filled >= batch_size:
            batch, filled = batch + 1, 0
        out[pid] = batch
        filled += int(row["n"])
    return accepted["patient_id"].map(out)


def estimate(plan: pd.DataFrame, num_primaries: float, n_processes: int = 1) -> Dict[str, float]:
    """Monte Carlo hours (full machine, scaled by primaries and by the CSDA-range
    power law 1.77 in energy from the measured reference) and storage."""
    seconds = REFERENCE_SECONDS * (num_primaries / 1e6) * (plan["energy_mev"] / REFERENCE_ENERGY_MEV) ** 1.77
    batches = plan.groupby("batch").size()
    return {"n_beamlets": int(len(plan)), "n_batches": int(len(batches)),
            "mc_hours_full_machine": float(seconds.sum() / 3600.0),
            "packed_storage_gb": float(len(plan) * PACKED_MB / 1000.0),
            "raw_batch_peak_gb": float(batches.max() * RAW_MB / 1000.0) * n_processes,
            "mean_energy_mev": float(plan["energy_mev"].mean())}


def acceptance_table(candidates: pd.DataFrame, bins: Sequence[float]) -> pd.DataFrame:
    """Share of drawn candidates accepted, per anatomy and energy bin, with reasons."""
    frame = candidates.assign(energy_bin=pd.cut(candidates["energy_mev"], bins=list(bins)).astype(str),
                              reason=candidates["reason"].replace("", "accepted"))
    table = frame.groupby(["anatomy", "energy_bin", "reason"]).size().unstack(fill_value=0)
    table["n"] = table.sum(axis=1)
    table["accepted_fraction"] = table.get("accepted", 0) / table["n"]
    return table


def entries_from_registry(registry: pd.DataFrame) -> List[PoolEntry]:
    """The registry's usable patients as :class:`PoolEntry` rows (role = split).

    Each entry loads exactly its screened series, re-checked slice by slice, so a
    registry that predates the slice-geometry screen is refused.
    """
    if "max_slice_gap_ratio" not in registry.columns:
        raise ValueError("the registry predates the slice-geometry screen: rerun `screen --overwrite`")
    keep = registry[registry["split"].isin(["train", "validation", "test"])]
    return [PoolEntry(role=r.split, anatomy=r.anatomy, dataset_name=r.collection, root=r.root,
                      collection=r.dataset_collection if isinstance(r.dataset_collection, str) else "",
                      patient_id=r.patient_id, series_uid=r.series_uid, n_slices=int(r.n_slices),
                      max_slice_gap_ratio=float(r.max_slice_gap_ratio))
            for r in keep.itertuples()]


def quotas_frame(quotas: Mapping[str, int]) -> pd.DataFrame:
    return pd.DataFrame({"patient_id": list(quotas), "quota": list(quotas.values())})
