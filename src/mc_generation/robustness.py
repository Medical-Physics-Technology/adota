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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.beamlets.bdl import BeamDataLibrary, angles_to_spot_position, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import flux_projection, flux_spatial_spread
from src.beamlets.rotation import rotate_ct_around_isocenter
from src.datasets.base import CTRecord
from src.figures.mc_beamlet_qc import mc_beamlet_qc_figure
from src.mc_generation.geometry import (
    beam_entrance_index,
    body_mask,
    dose_in_body_fraction,
    extraction_isocenter_physical,
    isocenters_from_world,
    mc_isocenter,
    reduce_vacuum_to_air,
    resample_to_isotropic,
    trim_beam_axis,
)
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.sweep import (
    RobustnessConfig,
    angle_tag,
    build_angle_grid,
    energy_tag,
    resolve_gantries,
    sweep_fits_ct,
    sweep_lateral_half_extents,
    sweep_z_half_extent_mm,
)
from src.metrics.range_metrics import compute_range_metrics

logger = logging.getLogger(__name__)


def _experiment_dir(cfg: RobustnessConfig, rec: CTRecord, energy: float,
                    gantry: Optional[float] = None) -> Path:
    """Output dir for one (patient, energy[, gantry]) block.

    The ``_g{angle}`` segment only appears when several gantries are generated per
    patient -- single-gantry runs keep their historical dir names (resumable).
    """
    name = f"{cfg.experiment_prefix}_{rec.anatomy}_{rec.patient_id}_e{energy_tag(energy)}"
    if gantry is not None and cfg.n_gantry > 1:
        name += f"_g{angle_tag(gantry)}"
    if cfg.experiment_version is not None:
        name += f"_v{cfg.experiment_version}"
    return Path(cfg.output_root) / name


@dataclass(frozen=True)
class FieldGeometry:
    """The CT (and derived isocenters) for one field angle of one patient.

    Public because the active-learning candidate scorer
    (:mod:`src.acquisition.candidates`) puts a CT into the beam's-eye frame with
    exactly this code, so a candidate is extracted the way its label would be.
    """
    ct: "sitk.Image"
    ct_array: np.ndarray
    body: np.ndarray                     # (z,y,x) body contour, 1.0 inside the patient
    iso_mc: Sequence[float]
    iso_ext: Sequence[float]
    field_gantry: float                  # physical field angle (model metadata)
    mc_gantry: float                     # angle actually simulated (90 when rotated)
    ct_rotation_deg: float


def _isocenters(ct: "sitk.Image", body: np.ndarray, cfg: RobustnessConfig):
    """``(iso_mc, iso_ext)`` for one CT under ``cfg.isocenter_mode``.

    ``grid_center`` is the historical convention (and what every generated dataset
    so far used). ``body_com`` aims the beam at the centre of mass of the body
    contour instead, which was tried to keep extreme-steering beamlets inside thin
    thoracic anatomy; it did not measurably help (the corner beamlets over-range
    through lung either way), so it stays available but off by default.
    """
    if cfg.isocenter_mode != "body_com":
        return mc_isocenter(ct), extraction_isocenter_physical(ct)
    com_zyx = np.argwhere(body > 0).mean(axis=0)
    com = np.asarray(ct.TransformContinuousIndexToPhysicalPoint(
        [float(com_zyx[2]), float(com_zyx[1]), float(com_zyx[0])]))
    return isocenters_from_world(ct, com)


def field_geometry(ct: "sitk.Image", field_gantry: float, cfg: RobustnessConfig,
                   bdl: BeamDataLibrary) -> FieldGeometry:
    """Put the CT into the beam's-eye frame for ``field_gantry``.

    Random / non-90 gantry: rotate the CT into the gantry-aligned beam's-eye frame
    (A = -(gantry - 90) about the isocenter, grid-expanded so no anatomy is
    clipped) and run MC at the canonical 90 deg. The model consumes this canonical
    frame (gantry is metadata, not a geometric input), so extraction stays
    axis-aligned exactly as at gantry 90.

    The expanded grid is then trimmed back along the **beam axis** to the unrotated
    extent, positioned so the entrance face sits
    ``cfg.beam_entrance_standoff_mm`` before the patient. Both steps are needed:
    ``extract_beamlet_roi`` measures the ROI's 320 mm depth from the grid's x = 0
    face, and an oblique beam crosses a square FOV diagonally, so the expanded grid
    left 150-190 mm of air in front of the patient -- against 0-70 mm in the
    gantry-90 data the model was trained on -- which pushed the Bragg peak out of
    the crop (deposition ratios 0.5-0.9, WET of 42-125 mm where the paper's crops
    have 155-288 mm). Keeping the grid longer than the ROI (the trim restores the
    original extent, not the 320 mm crop) leaves room beyond the crop, so the
    ``min_deposition_ratio`` gate still measures escaped dose rather than reading 1
    by construction. The lateral (y) expansion is kept.
    """
    if not (cfg.rotate_to_canonical and abs(field_gantry - 90.0) > 1e-6):
        body = body_mask(ct, cfg.body_hu_threshold)
        iso_mc, iso_ext = _isocenters(ct, body, cfg)
        return FieldGeometry(
            ct=ct, ct_array=sitk.GetArrayFromImage(ct), body=body, iso_mc=iso_mc,
            iso_ext=iso_ext, field_gantry=float(field_gantry),
            mc_gantry=float(field_gantry), ct_rotation_deg=0.0)

    ct_rotation_deg = -(field_gantry - 90.0)
    x_size = ct.GetSize()[0]
    rotated = rotate_ct_around_isocenter(
        ct, ct_rotation_deg, extraction_isocenter_physical(ct), expand=True)
    arr = sitk.GetArrayFromImage(rotated)

    half_z, half_y = sweep_lateral_half_extents(cfg, bdl.d_smx, bdl.d_smy)
    nz, ny, _ = arr.shape
    zc, yc = nz // 2, ny // 2                     # the isocenter is the grid centre
    window = (slice(max(0, int(zc - half_z)), int(zc + half_z) + 1),
              slice(max(0, int(yc - half_y)), int(yc + half_y) + 1))
    x0 = beam_entrance_index(arr, window) - int(round(cfg.beam_entrance_standoff_mm))
    trimmed = trim_beam_axis(rotated, x_size, x0)
    body = body_mask(trimmed, cfg.body_hu_threshold)
    iso_mc, iso_ext = _isocenters(trimmed, body, cfg)
    return FieldGeometry(
        ct=trimmed, ct_array=sitk.GetArrayFromImage(trimmed), body=body, iso_mc=iso_mc,
        iso_ext=iso_ext, field_gantry=float(field_gantry),
        mc_gantry=90.0, ct_rotation_deg=float(ct_rotation_deg))


# Former private names, kept for existing imports.
_FieldGeometry = FieldGeometry
_field_geometry = field_geometry


def _process_beamlet(
    rec: CTRecord, bdl: BeamDataLibrary, cfg: RobustnessConfig, geom: FieldGeometry,
    energy: float, cell: Tuple[int, int, float, float], spot, dose_img, sim_res: dict,
    out_dir: Path, fig_dir: Path,
) -> str:
    """Crop, QA-gate and save one beamlet; return ``"saved"`` or ``"qa"``.

    Shared by both MC paths, so a beamlet is written identically whether its dose
    came from a per-spot call or from a beamlet-mode field.
    """
    ix, iy, tx, ty = cell
    ct = geom.ct
    d_nozzle, d_smx, d_smy = bdl.distances
    stem = f"a{ix:02d}_{iy:02d}"
    dose_arr = sitk.GetArrayFromImage(dose_img)

    cropped_ct, entrance, _, oob = extract_beamlet_roi(
        ct, d_nozzle, d_smx, d_smy, spot, geom.iso_ext, cfg.roi_size, ct_array=geom.ct_array)
    cropped_dose, _, _, _ = extract_beamlet_roi(
        ct, d_nozzle, d_smx, d_smy, spot, geom.iso_ext, cfg.roi_size, ct_array=dose_arr)

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
        logger.info("  QA skip %s e%s %s: %s", rec.patient_id, energy_tag(energy), stem, reason)
        return "qa"

    # --- ADoTA flux channel (same construction the model consumes) ---
    beamlet_angles = spot_position_to_angles(spot[0], spot[1], d_smx, d_smy)
    sigmas = flux_spatial_spread(bdl, energy)
    re_proj = [float(entrance[1]), float(entrance[2]), float(entrance[0])]
    flux = flux_projection(re_proj, beamlet_angles, sigmas, cropped_ct.shape,
                           spacing=np.asarray([1, 1, 1], dtype=np.float32))

    # --- numerical support: dose-in-patient + clean-Bragg-peak metrics ---
    body_crop, _, _, _ = extract_beamlet_roi(
        ct, d_nozzle, d_smx, d_smy, spot, geom.iso_ext, cfg.roi_size, ct_array=geom.body)
    dib = dose_in_body_fraction(cropped_dose, body_crop)
    idd = np.asarray(cropped_dose.sum(axis=(0, 1)), dtype=float)  # depth = axis 2
    rm = compute_range_metrics(idd, dz_mm=1.0)

    sim_res.update({
        "id": str(uuid.uuid4()),
        "dose_in_body_fraction": float(dib),
        "r100_mm": float(rm.r100_mm), "r80_mm": float(rm.r80_mm),
        "dfw_mm": float(rm.dfw_mm), "peak_dose": float(rm.peak_dose),
        "provenance_uid": rec.uid,
        "dataset_name": rec.dataset_name,
        "anatomy": rec.anatomy,
        "patient_id": rec.patient_id,
        "series_uid": rec.series_uid,
        "ct_provenance": getattr(rec, "provenance", None),  # acq params + QC
        "grid_index": [ix, iy],
        "gantry_angle": float(geom.field_gantry),   # physical field angle (model metadata)
        "mc_gantry_angle": float(geom.mc_gantry),   # angle actually simulated (90 when rotated)
        "ct_rotation_deg": float(geom.ct_rotation_deg),
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
    (out_dir / f"{stem}_sim_res.json").write_text(json.dumps(sim_res))

    if cfg.make_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
        mc_beamlet_qc_figure(
            cropped_ct, cropped_dose, flux, str(fig_dir / stem),
            title=(f"{rec.anatomy} {rec.patient_id}  E={energy:g} MeV  "
                   f"gantry={geom.field_gantry:.1f}"),
            info={"theta": f"({tx:+.2f},{ty:+.2f})", "dep_ratio": f"{ratio:.3f}",
                  "stat_unc%": f"{sim_res.get('stat_uncertainty', float('nan')):.2f}"},
        )
    return "saved"


def _generate_energy_block(
    rec: CTRecord, runner: MCSquareRunner, bdl: BeamDataLibrary, cfg: RobustnessConfig,
    geom: FieldGeometry, energy: float, grid: List[Tuple[int, int, float, float]],
) -> dict:
    """Generate every beamlet of one (patient, field angle, energy) block.

    With ``cfg.beamlet_mode`` the block's outstanding spots go through MCsquare in a
    single dose-influence call (~3x faster: the CT and scoring setup is paid once
    and each spot gets its own thread). Resume still works -- the plan is built from
    the spots that are missing, so a restart re-simulates only those -- but the
    granularity is the block: MCsquare writes every spot's dense dose grid before
    Python reads any of them, so a block needs ``len(spots) x grid`` of scratch
    (tens of GB) and an interrupted block is redone from its remaining spots.
    """
    d_smx, d_smy = bdl.d_smx, bdl.d_smy
    out_dir = _experiment_dir(cfg, rec, energy, geom.field_gantry)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "qc_figures"

    pending = [cell for cell in grid
               if cfg.overwrite or not (out_dir / f"a{cell[0]:02d}_{cell[1]:02d}_sim_res.json").exists()]
    n_exist = len(grid) - len(pending)
    n_saved = n_qa = 0
    spots = [angles_to_spot_position(tx, ty, d_smx, d_smy) for _, _, tx, ty in pending]

    if pending and cfg.beamlet_mode:
        for index, dose_img, sim_res in runner.run_beamlet_field(
                geom.ct, energy=energy, gantry_angle=geom.mc_gantry, spots_xy=spots,
                isocenter=geom.iso_mc, num_primaries=cfg.num_primaries,
                num_threads=cfg.num_threads, rng_seed=cfg.rng_seed):
            outcome = _process_beamlet(rec, bdl, cfg, geom, energy, pending[index],
                                       spots[index], dose_img, sim_res, out_dir, fig_dir)
            n_saved += outcome == "saved"
            n_qa += outcome == "qa"
    else:
        for index, cell in enumerate(pending):
            dose_img, sim_res = runner.run_beamlet(
                geom.ct, energy=energy, gantry_angle=geom.mc_gantry, spot_xy=spots[index],
                isocenter=geom.iso_mc, num_primaries=cfg.num_primaries,
                num_threads=cfg.num_threads, rng_seed=cfg.rng_seed,
            )
            outcome = _process_beamlet(rec, bdl, cfg, geom, energy, cell, spots[index],
                                       dose_img, sim_res, out_dir, fig_dir)
            n_saved += outcome == "saved"
            n_qa += outcome == "qa"

    logger.info("  %s e%s g%.1f: saved=%d qa_skip=%d existing=%d -> %s",
                rec.patient_id, energy_tag(energy), geom.field_gantry,
                n_saved, n_qa, n_exist, out_dir)
    return {"saved": n_saved, "qa_skipped": n_qa, "existing": n_exist, "dir": out_dir.name}


def generate_for_record(
    rec: CTRecord, runner: MCSquareRunner, bdl: BeamDataLibrary, cfg: RobustnessConfig,
) -> dict:
    """Generate all (field angle x energy x beamlet angle) beamlets for one patient CT.

    The gantry loop is outermost: the CT is rotated into the beam's-eye frame once
    per field angle and reused across the energies (the angles are drawn per
    patient, not per energy).
    """
    grid = build_angle_grid(cfg.theta_x_range, cfg.theta_y_range, cfg.grid_n, cfg.angles)
    if cfg.border_only:  # only the corner/edge beamlets (fast, extreme-angle test)
        n = cfg.grid_n
        grid = [g for g in grid if g[0] in (0, n - 1) or g[1] in (0, n - 1)]
    gantries = resolve_gantries(cfg, rec.uid)

    ct = resample_to_isotropic(rec.load_image(), cfg.iso_spacing_mm)
    ct = reduce_vacuum_to_air(ct)

    fits, z_extent, max_tx = sweep_fits_ct(ct, cfg, bdl.d_smy)
    if not fits:
        logger.warning(
            "  %s: CT is %.0f mm long in z, too short for the requested theta_x sweep "
            "(needs %.0f mm); |theta_x| > %.2f deg will be dropped as roi_out_of_bounds",
            rec.patient_id, z_extent, 2 * sweep_z_half_extent_mm(cfg, bdl.d_smy), max_tx)

    stats = {"patient": rec.patient_id, "anatomy": rec.anatomy,
             "z_extent_mm": round(z_extent, 1), "max_theta_x_deg": round(max_tx, 2),
             "gantries": [round(g, 2) for g in gantries],
             "saved": 0, "skipped_existing": 0, "skipped_qa": 0, "fields": []}

    for field_gantry in gantries:
        geom = field_geometry(ct, field_gantry, cfg, bdl)
        field_stats = {"gantry": float(field_gantry),
                       "ct_rotation_deg": geom.ct_rotation_deg,
                       "grid_size": list(geom.ct.GetSize()), "energies": {}}
        for energy in cfg.energies:
            block = _generate_energy_block(rec, runner, bdl, cfg, geom, energy, grid)
            field_stats["energies"][energy_tag(energy)] = block
            stats["saved"] += block["saved"]
            stats["skipped_qa"] += block["qa_skipped"]
            stats["skipped_existing"] += block["existing"]
        stats["fields"].append(field_stats)
    return stats


def run_generation(dataset, runner: MCSquareRunner, bdl: BeamDataLibrary,
                   cfg: RobustnessConfig) -> dict:
    """Generate the whole sweep over a (multi-)dataset; return aggregate stats."""
    n_angles = len(build_angle_grid(cfg.theta_x_range, cfg.theta_y_range, cfg.grid_n, cfg.angles))
    n_gantry = 1 if cfg.gantry_mode == "fixed" else max(1, cfg.n_gantry)
    logger.info(
        "Robustness generation: %d patients x %d gantries x %d energies x %d angles = %d beamlets",
        len(dataset), n_gantry, len(cfg.energies), n_angles,
        len(dataset) * n_gantry * len(cfg.energies) * n_angles)
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
