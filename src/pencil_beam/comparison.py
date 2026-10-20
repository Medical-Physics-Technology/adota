"""MC vs DoseCUDA pencil beam vs ADoTA for the same beamlets on the same CT grid.

For every (patient, field gantry, energy, beamlet angle) the three engines see
one CT: the field's canonical beam's-eye CT from
:func:`src.mc_generation.robustness.field_geometry` (rotated so the beam runs
along +x, 1 mm isotropic), i.e. exactly the grid the ADoTA training beamlets were
simulated on.

* **MC** -- :class:`~src.mc_generation.mcsquare_runner.MCSquareRunner` on that CT.
* **PBA** -- :class:`~src.pencil_beam.engine.PencilBeamEngine` on the same CT,
  same MC-frame isocenter and spot.
* **ADoTA** -- the dataset's own crop / flux / QA path
  (:func:`~src.mc_generation.robustness.process_beamlet`), batched inference
  (:func:`~src.beamlets.inference.run_inference`), and the predicted crop put back
  on the CT grid with :func:`~src.beamlets.accumulation.deposit_crop`.

Doses are then compared on one window of that grid (the MC/PBA dose bounding box
unioned with ADoTA's ROI), which is saved with its index offset so every number
traces back to grid voxels.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.beamlets.accumulation import deposit_crop
from src.beamlets.bdl import BeamDataLibrary, angles_to_spot_position
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.inference import InferenceConfig, run_inference
from src.mc_generation.robustness import FieldGeometry, field_geometry, process_beamlet
from src.mc_generation.sweep import RobustnessConfig, angle_tag, energy_tag
from src.pencil_beam.metrics import GammaCriterion, dose_agreement, dose_bbox, gamma_pass_rates

logger = logging.getLogger(__name__)

__all__ = ["ComparisonSettings", "block_dir_name", "compare_block", "summarize"]


@dataclass
class ComparisonSettings:
    """Run-level settings of the engine comparison (YAML ``comparison:``)."""

    output_dir: Path
    criteria: Tuple[GammaCriterion, ...]
    gamma_local: bool = False
    gamma_backend: str = "pymedphys"
    gamma_backend_options: dict = field(default_factory=dict)
    lateral_margin_mm: int = 30        # window = ADoTA's ROI widened by this much on each lateral side
    make_figures: bool = True
    adota_batch_size: int = 56


def block_dir_name(rec, field_gantry: float, energy: float) -> str:
    return f"{rec.anatomy}_{rec.patient_id}_g{angle_tag(field_gantry)}_e{energy_tag(energy)}"


def _region(grid_shape, crp, roi_size, margin: int):
    """Grid window of the comparison: ADoTA's ROI widened laterally, the full beam axis.

    Geometric rather than dose-thresholded: MC's scattered single-event voxels and
    air dose put voxels above 1% of the maximum across the whole grid.
    """
    h, w, _ = roi_size
    iz, iy, _ = crp
    lo = (max(iz - h // 2 - margin, 0), max(iy - w // 2 - margin, 0), 0)
    hi = (min(iz - h // 2 + h + margin, grid_shape[0]), min(iy - w // 2 + w + margin, grid_shape[1]), grid_shape[2])
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _simulate_cell(rec, runner, bdl: BeamDataLibrary, rcfg: RobustnessConfig, geom: FieldGeometry, pba,
                   energy: float, cell, block_dir: Path, settings: ComparisonSettings) -> dict:
    ix, iy, tx, ty = cell
    stem = f"a{ix:02d}_{iy:02d}"
    d_nozzle, d_smx, d_smy = bdl.distances
    spot = angles_to_spot_position(tx, ty, d_smx, d_smy)    # crossed names: see CLAUDE.md
    mc_img, sim_res = runner.run_beamlet(
        geom.ct, energy=energy, gantry_angle=geom.mc_gantry, spot_xy=spot, isocenter=geom.iso_mc,
        num_primaries=rcfg.num_primaries, num_threads=rcfg.num_threads, rng_seed=rcfg.rng_seed)
    mc = sitk.GetArrayFromImage(mc_img).astype(np.float32)
    pb = pba.dose(geom.ct, energy, [spot], geom.iso_mc)
    outcome = process_beamlet(rec, bdl, rcfg, geom, energy, cell, spot, mc_img, dict(sim_res),
                              block_dir, block_dir / "qc_figures", stem=stem)
    _, _, crp, oob = extract_beamlet_roi(geom.ct, d_nozzle, d_smx, d_smy, spot, geom.iso_ext,
                                         rcfg.roi_size, ct_array=geom.ct_array)
    region = _region(mc.shape, crp, rcfg.roi_size, int(settings.lateral_margin_mm / rcfg.iso_spacing_mm))
    np.savez_compressed(
        block_dir / f"{stem}_engines.npz", ct=geom.ct_array[region], mc=mc[region], pba=pb.dose[region],
        region_start=np.array([s.start for s in region]), crp=np.asarray(crp),
        mc_total=float(mc.sum()), pba_total=float(pb.dose.sum()))
    return {"stem": stem, "grid_index": [ix, iy], "theta_x": tx, "theta_y": ty,
            "spot_xy_mm": [float(spot[0]), float(spot[1])], "adota_input": outcome,
            "roi_out_of_bounds": bool(oob), "mc_seconds": float(sim_res["mc_seconds"]),
            "mc_stat_uncertainty_pct": float(sim_res["stat_uncertainty"]),
            "pba_seconds": float(pb.seconds), "pba_model_energy": pb.model_energy,
            "pba_interpolated_from": pb.interpolated_from, "num_primaries": float(rcfg.num_primaries)}


def _adota_on_region(block_dir: Path, stem: str, grid_shape, region, crp, roi_size) -> Optional[np.ndarray]:
    pred_path = block_dir / f"{stem}_ds_pred.npy"
    if not pred_path.exists():
        return None
    crop = np.moveaxis(np.squeeze(np.load(pred_path)).astype(np.float32), 0, -1)   # (D,H,W) -> (H,W,D)
    grid = np.zeros(grid_shape, dtype=np.float32)
    deposit_crop(grid, np.clip(crop, 0.0, None), crp, 1.0, roi_size)
    return grid[region]


def _score_cell(block_dir: Path, info: dict, grid_shape, roi_size, settings: ComparisonSettings,
                title: str) -> dict:
    data = np.load(block_dir / f"{info['stem']}_engines.npz")
    region = tuple(slice(int(s), int(s) + n) for s, n in zip(data["region_start"], data["mc"].shape))
    doses = {"MC": data["mc"], "PBA": data["pba"]}
    adota = _adota_on_region(block_dir, info["stem"], grid_shape, region, data["crp"], roi_size)
    if adota is not None:
        doses["ADoTA"] = adota
        np.save(block_dir / f"{info['stem']}_adota_region.npy", adota)
    row = dict(info)
    row["mc_dose_in_region"] = float(data["mc"].sum()) / float(data["mc_total"]) if data["mc_total"] > 0 else 0.0
    row["pba_dose_in_region"] = float(data["pba"].sum()) / float(data["pba_total"]) if data["pba_total"] > 0 else 0.0
    max_dist = max(c.dist for c in settings.criteria)
    min_cut = min(c.cutoff for c in settings.criteria) / 100.0
    gbox = dose_bbox(list(doses.values()), low_frac=0.5 * min_cut, margin=int(np.ceil(max_dist)) + 2)
    for label, dose in doses.items():
        if label == "MC":
            continue
        tag = label.lower()
        for k, v in dose_agreement(doses["MC"], dose).items():
            row[f"{tag}_{k}"] = v
        for k, v in gamma_pass_rates(doses["MC"][gbox], dose[gbox], settings.criteria, local=settings.gamma_local,
                                     backend=settings.gamma_backend,
                                     backend_options=settings.gamma_backend_options).items():
            row[f"{tag}_{k}"] = v
    if settings.make_figures:
        from src.figures.pencil_beam import engine_comparison_figure

        head = settings.criteria[0]
        text = {lab: f"{head.key} = {row.get(f'{lab.lower()}_{head.key}', float('nan')):.1f}%  "
                     f"dR80 = {row.get(f'{lab.lower()}_r80_delta_mm', float('nan')):+.2f} mm"
                for lab in doses if lab != "MC"}
        kind = "local" if settings.gamma_local else "global"
        engine_comparison_figure(
            data["ct"], doses, str(block_dir / "figures" / f"{info['stem']}_engines_{head.key}_{kind}"),
            title=f"{title}  {info['stem']} (theta {info['theta_x']:+.2f}, {info['theta_y']:+.2f})  "
                  f"gamma {head.dose:g}%/{head.dist:g}mm, {head.cutoff:g}% cutoff ({kind})",
            metrics_text=text)
    return row


def compare_block(rec, ct: sitk.Image, field_gantry: float, energy: float, cells: Sequence, runner,
                  bdl: BeamDataLibrary, rcfg: RobustnessConfig, pba, adota_model, device,
                  settings: ComparisonSettings) -> List[dict]:
    """Run all engines for one (patient, field gantry, energy) block and score them.

    Resumable per beamlet: a cell whose ``{stem}_engines.npz`` exists is not
    re-simulated (its timing is read from ``{stem}_cell.json``).
    """
    geom = field_geometry(ct, field_gantry, rcfg, bdl)
    block_dir = Path(settings.output_dir) / block_dir_name(rec, field_gantry, energy)
    (block_dir / "figures").mkdir(parents=True, exist_ok=True)
    infos = []
    for cell in cells:
        stem = f"a{cell[0]:02d}_{cell[1]:02d}"
        meta = block_dir / f"{stem}_cell.json"
        if meta.exists() and (block_dir / f"{stem}_engines.npz").exists():
            infos.append(json.loads(meta.read_text()))
            continue
        info = _simulate_cell(rec, runner, bdl, rcfg, geom, pba, energy, cell, block_dir, settings)
        meta.write_text(json.dumps(info, indent=2))
        infos.append(info)
        logger.info("  %s: MC %.0f s, PBA %.2f s, ADoTA input %s", stem, info["mc_seconds"],
                    info["pba_seconds"], info["adota_input"])

    adota_seconds = float("nan")
    if adota_model is not None and any(i["adota_input"] == "saved" for i in infos):
        t0 = perf_counter()
        run_inference(block_dir, adota_model, device,
                      InferenceConfig(batch_size=settings.adota_batch_size, grid_factor=1))
        n = sum(i["adota_input"] == "saved" for i in infos)
        adota_seconds = (perf_counter() - t0) / n

    title = f"{rec.anatomy} {rec.patient_id}  E={energy:g} MeV  gantry={field_gantry:.1f}"
    rows = []
    grid_shape = tuple(geom.ct_array.shape)
    for info in infos:
        row = _score_cell(block_dir, info, grid_shape, rcfg.roi_size, settings, title)
        row.update({"anatomy": rec.anatomy, "patient_id": rec.patient_id, "provenance_uid": rec.uid,
                    "field_gantry": float(field_gantry), "energy": float(energy),
                    "adota_seconds_per_beamlet": adota_seconds, "block_dir": block_dir.name})
        rows.append(row)
    (block_dir / "metrics.json").write_text(json.dumps(rows, indent=2, default=float))
    return rows


def summarize(rows: Sequence[Dict], criteria: Sequence[GammaCriterion]) -> Dict[str, Dict[str, float]]:
    """Mean / median / min of the headline metrics per engine across beamlets."""
    keys = [c.key for c in criteria] + ["r80_delta_mm", "mean_abs_diff_pct", "integral_dose_ratio"]
    out = {}
    for engine in ("pba", "adota"):
        for k in keys:
            vals = np.array([r[f"{engine}_{k}"] for r in rows if f"{engine}_{k}" in r], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[f"{engine}_{k}"] = {"n": int(vals.size), "mean": float(vals.mean()),
                                        "median": float(np.median(vals)), "min": float(vals.min()),
                                        "max": float(vals.max())}
    return out
