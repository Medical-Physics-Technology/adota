"""Beamlet-angle robustness dataset generator (adota-native, config-driven).

For each (patient, energy, beamlet-angle) it runs one MCsquare beamlet, extracts
the ADoTA ROI crop of the CT and dose, builds the ADoTA flux channel, applies the
QA gates from the reference pipeline, and saves ``{stem}_ct/_ds/_flux.npy`` +
``{stem}_sim_res.json``. Resumable (skips angles already written) and optionally
emits a per-beamlet QC figure.
"""
from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from src.beamlets.bdl import BeamDataLibrary, angles_to_spot_position, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import flux_projection, flux_spatial_spread
from src.datasets.base import CTRecord
from src.figures.mc_beamlet_qc import mc_beamlet_qc_figure
from src.mc_generation.geometry import (
    extraction_isocenter_physical,
    mc_isocenter,
    reduce_vacuum_to_air,
    resample_to_isotropic,
)
from src.mc_generation.mcsquare_runner import MCSquareRunner

logger = logging.getLogger(__name__)


@dataclass
class RobustnessConfig:
    energies: List[float] = field(default_factory=lambda: [90.0, 140.0, 200.0])
    theta_x_range: Tuple[float, float] = (-2.0, 2.0)
    theta_y_range: Tuple[float, float] = (-2.0, 2.0)
    grid_n: int = 18
    gantry_mode: str = "fixed"           # "fixed" | "bimodal_random"
    gantry_value: float = 90.0
    gantry_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((30.0, 120.0), (240.0, 330.0))
    gantry_seed: int = 1234
    roi_size: Tuple[int, int, int] = (60, 60, 320)
    iso_spacing_mm: float = 1.0
    num_primaries: float = 1e7
    num_threads: int = 0
    rng_seed: int = 0
    min_deposition_ratio: float = 0.5
    output_root: str = "/scratch/mstryja/DoTA_dataset_v2"
    experiment_prefix: str = "beamlet_angle_robustness"
    experiment_version: Optional[int] = 2
    make_figures: bool = False
    overwrite: bool = False


def build_angle_grid(tx_range, ty_range, n) -> List[Tuple[int, int, float, float]]:
    """Return the (ix, iy, theta_x, theta_y) grid (row-major over theta_x)."""
    txs = np.linspace(tx_range[0], tx_range[1], n)
    tys = np.linspace(ty_range[0], ty_range[1], n)
    return [(ix, iy, float(tx), float(ty))
            for ix, tx in enumerate(txs) for iy, ty in enumerate(tys)]


def resolve_gantry(cfg: RobustnessConfig, patient_uid: str) -> float:
    """Gantry angle for a patient: fixed, or a seeded bimodal draw (reproducible)."""
    if cfg.gantry_mode == "fixed":
        return float(cfg.gantry_value)
    if cfg.gantry_mode == "bimodal_random":
        rng = random.Random(f"{cfg.gantry_seed}:{patient_uid}")
        lo1, hi1 = cfg.gantry_ranges[0]
        lo2, hi2 = cfg.gantry_ranges[1]
        return rng.uniform(lo1, hi1) if rng.random() < 0.5 else rng.uniform(lo2, hi2)
    raise ValueError(f"unknown gantry_mode {cfg.gantry_mode!r}")


def _experiment_dir(cfg: RobustnessConfig, rec: CTRecord, energy: float) -> Path:
    name = f"{cfg.experiment_prefix}_{rec.anatomy}_{rec.patient_id}_e{int(energy)}"
    if cfg.experiment_version is not None:
        name += f"_v{cfg.experiment_version}"
    return Path(cfg.output_root) / name


def generate_for_record(
    rec: CTRecord, runner: MCSquareRunner, bdl: BeamDataLibrary, cfg: RobustnessConfig,
) -> dict:
    """Generate all (energy x angle) beamlets for one patient CT."""
    d_nozzle, d_smx, d_smy = bdl.distances
    grid = build_angle_grid(cfg.theta_x_range, cfg.theta_y_range, cfg.grid_n)
    gantry = resolve_gantry(cfg, rec.uid)

    ct = resample_to_isotropic(rec.load_image(), cfg.iso_spacing_mm)
    ct = reduce_vacuum_to_air(ct)
    ct_arr = sitk.GetArrayFromImage(ct)
    iso_mc = mc_isocenter(ct)
    iso_ext = extraction_isocenter_physical(ct)

    stats = {"patient": rec.patient_id, "anatomy": rec.anatomy, "gantry": gantry,
             "saved": 0, "skipped_existing": 0, "skipped_qa": 0, "energies": {}}

    for energy in cfg.energies:
        out_dir = _experiment_dir(cfg, rec, energy)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = out_dir / "qc_figures"
        n_saved = n_qa = n_exist = 0
        for ix, iy, tx, ty in grid:
            stem = f"a{ix:02d}_{iy:02d}"
            sr_path = out_dir / f"{stem}_sim_res.json"
            if sr_path.exists() and not cfg.overwrite:
                n_exist += 1
                continue

            spot = angles_to_spot_position(tx, ty, d_smx, d_smy)
            dose_img, sim_res = runner.run_beamlet(
                ct, energy=energy, gantry_angle=gantry, spot_xy=spot,
                isocenter=iso_mc, num_primaries=cfg.num_primaries,
                num_threads=cfg.num_threads, rng_seed=cfg.rng_seed,
            )
            dose_arr = sitk.GetArrayFromImage(dose_img)

            cropped_ct, entrance, crp, oob = extract_beamlet_roi(
                ct, d_nozzle, d_smx, d_smy, spot, iso_ext, cfg.roi_size, ct_array=ct_arr)
            cropped_dose, _, _, _ = extract_beamlet_roi(
                ct, d_nozzle, d_smx, d_smy, spot, iso_ext, cfg.roi_size, ct_array=dose_arr)

            # --- QA gates (from the reference pipeline) ---
            total = float(dose_arr.sum())
            ratio = float(cropped_dose.sum()) / total if total > 0 else 0.0
            reason = None
            if oob:
                reason = "roi_out_of_bounds"
            elif cropped_dose.shape != tuple(cfg.roi_size):
                reason = f"bad_shape_{cropped_dose.shape}"
            elif cropped_dose.sum() <= 0:
                reason = "zero_dose_in_roi"
            elif ratio < cfg.min_deposition_ratio:
                reason = f"low_deposition_ratio_{ratio:.3f}"
            if reason is not None:
                logger.info("  QA skip %s e%d %s: %s", rec.patient_id, int(energy), stem, reason)
                n_qa += 1
                continue

            # --- ADoTA flux channel (same construction the model consumes) ---
            beamlet_angles = spot_position_to_angles(spot[0], spot[1], d_smx, d_smy)
            sigmas = flux_spatial_spread(bdl, energy)
            re_proj = [float(entrance[1]), float(entrance[2]), float(entrance[0])]
            flux = flux_projection(re_proj, beamlet_angles, sigmas, cropped_ct.shape,
                                   spacing=np.asarray([1, 1, 1], dtype=np.float32))

            record_id = str(uuid.uuid4())
            sim_res.update({
                "id": record_id,
                "provenance_uid": rec.uid,
                "dataset_name": rec.dataset_name,
                "anatomy": rec.anatomy,
                "patient_id": rec.patient_id,
                "series_uid": rec.series_uid,
                "grid_index": [ix, iy],
                "beamlet_angles": list(beamlet_angles),
                "spot_position": [float(spot[0]), float(spot[1])],
                "roi_size": list(cfg.roi_size),
                "image_origin": list(ct.GetOrigin()),
                "image_spacing": list(ct.GetSpacing()),
                "image_size": list(ct.GetSize()),
                "rays_entrence_point": [float(v) for v in entrance],
                "rays_entrence_point_proj": re_proj,
                "dose_deposition_ratio": ratio,
            })
            np.save(out_dir / f"{stem}_ct.npy", cropped_ct)
            np.save(out_dir / f"{stem}_ds.npy", cropped_dose)
            np.save(out_dir / f"{stem}_flux.npy", flux)
            sr_path.write_text(json.dumps(sim_res))
            n_saved += 1

            if cfg.make_figures:
                fig_dir.mkdir(parents=True, exist_ok=True)
                mc_beamlet_qc_figure(
                    cropped_ct, cropped_dose, flux, str(fig_dir / stem),
                    title=f"{rec.anatomy} {rec.patient_id}  E={energy:.0f} MeV  gantry={gantry:.1f}",
                    info={"theta": f"({tx:+.2f},{ty:+.2f})", "dep_ratio": f"{ratio:.3f}",
                          "stat_unc%": f"{sim_res.get('stat_uncertainty', float('nan')):.2f}"},
                )
        stats["energies"][int(energy)] = {"saved": n_saved, "qa_skipped": n_qa, "existing": n_exist}
        stats["saved"] += n_saved; stats["skipped_qa"] += n_qa; stats["skipped_existing"] += n_exist
        logger.info("  %s e%d: saved=%d qa_skip=%d existing=%d -> %s",
                    rec.patient_id, int(energy), n_saved, n_qa, n_exist, out_dir)
    return stats


def run_generation(dataset, runner: MCSquareRunner, bdl: BeamDataLibrary,
                   cfg: RobustnessConfig) -> dict:
    """Generate the whole sweep over a (multi-)dataset; return aggregate stats."""
    n_angles = cfg.grid_n * cfg.grid_n
    logger.info("Robustness generation: %d patients x %d energies x %d angles = %d beamlets",
                len(dataset), len(cfg.energies), n_angles,
                len(dataset) * len(cfg.energies) * n_angles)
    per_patient = []
    totals = {"saved": 0, "skipped_qa": 0, "skipped_existing": 0}
    for i in range(len(dataset)):
        rec = dataset.record(i)
        logger.info("[%d/%d] %s (%s)", i + 1, len(dataset), rec.patient_id, rec.anatomy)
        s = generate_for_record(rec, runner, bdl, cfg)
        per_patient.append(s)
        for k in totals:
            totals[k] += s[k]
    logger.info("DONE. saved=%d qa_skipped=%d existing=%d",
                totals["saved"], totals["skipped_qa"], totals["skipped_existing"])
    return {"totals": totals, "per_patient": per_patient}
