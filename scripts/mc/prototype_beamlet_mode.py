"""Beamlet-mode validation: MC vs MC, and ADoTA scored against each ground truth.

Runs one field's angle sweep twice -- once as a single MCsquare **beamlet-mode**
call (``Num_Primaries`` per spot, one OpenMP thread per spot) and once the
current way, **one call per spot** -- saves both as ordinary beamlet dirs, runs
ADoTA on each, and reports three gamma comparisons over the angle lattice:

* **MC vs MC**        -- the two ground truths against each other;
* **ADoTA vs base**   -- the model against the baseline ground truth;
* **ADoTA vs new**    -- the model against this run's ground truth.

The first says the two ground truths agree; the last two say the switch does not
move the number we actually report. The RNG differs between arms, so MC vs MC is a
statistical check, not equality. ADoTA's prediction is identical in both arms (it
sees only CT, flux and energy), so any difference between the last two rows comes
from the ground truth alone.

  # engine check: beamlet mode vs the sequential path, same primaries
  uv run python scripts/mc/prototype_beamlet_mode.py --grid-n 7

  # statistics check: 1e7 against an existing 1e6 beamlet run of the same field
  uv run python scripts/mc/prototype_beamlet_mode.py --grid-n 7 --num-primaries 1e7 \
      --baseline-dir <existing 1e6 dir> --label-a "MC beamlet 1e7" \
      --label-b "MC beamlet 1e6" --mc-reference a
"""
from __future__ import annotations

import json
import logging
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import numpy as np
import SimpleITK as sitk
import typer

from src.adota.config import load_yaml_config
from src.adota.utils import load_model
from src.beamlets.bdl import BeamDataLibrary, angles_to_spot_position, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import flux_projection, flux_spatial_spread
from src.datasets.registry import build_dataset_from_config
from src.evaluation.cli import resolve_device
from src.figures.angle_robustness_grid import angle_robustness_panel
from src.figures.beamlet_mode_comparison import beamlet_mode_comparison_figure
from src.mc_generation.angle_robustness_analysis import GammaCriterion, infer_dir, score_dir_grids
from src.mc_generation.geometry import reduce_vacuum_to_air, resample_to_isotropic
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.robustness import _field_geometry
from src.mc_generation.sweep import (
    build_angle_grid,
    energy_tag,
    resolve_gantries,
    robustness_config_from_dict,
)
from src.metrics.gamma_pass_rate import gamma_index

ROOT = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pymedphys").setLevel(logging.WARNING)
logger = logging.getLogger("beamlet_prototype")
app = typer.Typer(help="Beamlet mode vs the sequential path, MC and ADoTA.")

CRITERIA = (GammaCriterion(2, 2, 10), GammaCriterion(1, 3, 0.1))


def _save_record(out_dir: Path, stem: str, ct, geom, bdl, rcfg, spot, energy,
                 dose_array, ix, iy, rec, sim_res) -> Optional[float]:
    """Crop + flux + save one beamlet exactly as the generation pipeline does."""
    cropped_ct, entrance, _, oob = extract_beamlet_roi(
        ct, bdl.d_nozzle, bdl.d_smx, bdl.d_smy, spot, geom.iso_ext, rcfg.roi_size,
        ct_array=geom.ct_array)
    cropped_dose, _, _, _ = extract_beamlet_roi(
        ct, bdl.d_nozzle, bdl.d_smx, bdl.d_smy, spot, geom.iso_ext, rcfg.roi_size,
        ct_array=dose_array)
    if oob or cropped_dose.sum() <= 0:
        return None
    total = float(dose_array.sum())
    ratio = float(cropped_dose.sum()) / total if total > 0 else 0.0
    angles = spot_position_to_angles(spot[0], spot[1], bdl.d_smx, bdl.d_smy)
    re_proj = [float(entrance[1]), float(entrance[2]), float(entrance[0])]
    flux = flux_projection(re_proj, angles, flux_spatial_spread(bdl, energy),
                           cropped_ct.shape, spacing=np.asarray([1, 1, 1], dtype=np.float32))
    sim_res.update({
        "id": str(uuid.uuid4()), "provenance_uid": rec.uid, "dataset_name": rec.dataset_name,
        "anatomy": rec.anatomy, "patient_id": rec.patient_id, "series_uid": rec.series_uid,
        "grid_index": [ix, iy], "gantry_angle": float(geom.field_gantry),
        "mc_gantry_angle": float(geom.mc_gantry), "ct_rotation_deg": float(geom.ct_rotation_deg),
        "beamlet_angles": list(angles), "spot_position": [float(spot[0]), float(spot[1])],
        "roi_size": list(rcfg.roi_size), "image_origin": list(ct.GetOrigin()),
        "image_spacing": list(ct.GetSpacing()), "image_size": list(ct.GetSize()),
        "rays_entrence_point": [float(v) for v in entrance],
        "rays_entrence_point_proj": re_proj, "dose_deposition_ratio": ratio,
    })
    np.save(out_dir / f"{stem}_ct.npy", cropped_ct)
    np.save(out_dir / f"{stem}_ds.npy", cropped_dose)
    np.save(out_dir / f"{stem}_flux.npy", flux)
    (out_dir / f"{stem}_sim_res.json").write_text(json.dumps(sim_res))
    return ratio


def _mc_vs_mc(task):
    """Worker: local gamma of a beamlet-mode crop against the sequential one."""
    seq_path, beam_path, crit = task
    logging.getLogger("pymedphys").setLevel(logging.WARNING)
    ref = np.load(seq_path).astype(np.float64)
    other = np.load(beam_path).astype(np.float64)
    pos = ref > 0
    peak = float(np.percentile(ref[pos], 99.5)) if pos.any() else 0.0
    if peak <= 0:
        return float("nan")
    params = {"dose_percent_threshold": crit.dose, "distance_mm_threshold": crit.dist,
              "interp_fraction": 5, "max_gamma": 2, "lower_percent_dose_cutoff": crit.cutoff,
              "random_subset": None, "local_gamma": True, "quiet": True}
    try:
        _, gpr = gamma_index(ref, other, {"y_max": peak, "y_min": 0.0}, params, (1.0, 1.0, 1.0))
        return float(gpr[0]) * 100.0
    except Exception:
        return float("nan")


def _stats(grid: np.ndarray) -> dict:
    v = grid[np.isfinite(grid)]
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(v.size), "mean": float(v.mean()), "min": float(v.min()),
            "max": float(v.max())}


@app.command()
def main(
    config: Annotated[Path, typer.Option()] = Path("scripts/mc/config_energy_gantry_smoke.yaml"),
    plot_config: Annotated[Path, typer.Option(help="Supplies the ADoTA model to run.")] = Path(
        "scripts/mc/config_plot_energy_gantry_smoke.yaml"),
    grid_n: Annotated[int, typer.Option(help="Angle lattice for the prototype field.")] = 7,
    patient_index: Annotated[int, typer.Option()] = 3,
    gantry_index: Annotated[int, typer.Option()] = 0,
    energy: Annotated[Optional[float], typer.Option()] = None,
    num_primaries: Annotated[float, typer.Option()] = 1.0e6,
    out_root: Annotated[Path, typer.Option()] = Path(
        "/scratch/mstryja/DoTA_dataset_v2/beamlet_mode_prototype"),
    baseline_dir: Annotated[Optional[Path], typer.Option(
        help="Existing beamlet dir to compare against; skips the sequential arm.")] = None,
    label_a: Annotated[str, typer.Option(help="Name of this run in the table/figure.")] = "MC beamlet",
    label_b: Annotated[str, typer.Option(help="Name of the baseline arm.")] = "MC sequential",
    mc_reference: Annotated[str, typer.Option(
        help="Which arm is the gamma reference for MC vs MC: 'a' (this run) or 'b'.")] = "b",
    skip_mc: Annotated[bool, typer.Option(help="Reuse crops from a previous run.")] = False,
) -> None:
    if mc_reference not in ("a", "b"):
        raise typer.BadParameter("--mc-reference must be 'a' or 'b'")
    cfg = load_yaml_config(config)
    rcfg = robustness_config_from_dict(cfg["robustness"])
    rcfg.grid_n, rcfg.angles = grid_n, None
    e = float(energy if energy is not None else rcfg.energies[0])

    dataset = build_dataset_from_config({"datasets": cfg["datasets"], "name": "proto"})
    rec = dataset.record(patient_index)
    gantry = resolve_gantries(rcfg, rec.uid)[gantry_index]
    engine = cfg["engine"]
    bdl_name = engine.get("bdl_file", "hptc_beam_model_rsnone.txt")
    bdl = BeamDataLibrary.from_file(str(Path(engine["mcsquare_install"]) / "BDL" / bdl_name))

    primaries_tag = f"{num_primaries:.0e}".replace("e+0", "e").replace("e+", "e")
    tag = (f"{rec.anatomy}_{rec.patient_id}_e{energy_tag(e)}_g{gantry:.0f}"
           f"_n{grid_n}_{primaries_tag}")
    dir_beam = Path(out_root) / f"{tag}_beamletmode"
    dir_beam.mkdir(parents=True, exist_ok=True)
    # The baseline is either an existing dir (statistics check) or a sequential run
    # of the same field made below (engine check).
    run_sequential = baseline_dir is None
    dir_seq = Path(baseline_dir) if baseline_dir is not None else Path(out_root) / f"{tag}_sequential"
    if not run_sequential and not dir_seq.is_dir():
        raise typer.BadParameter(f"--baseline-dir does not exist: {dir_seq}")
    dir_seq.mkdir(parents=True, exist_ok=True)

    grid = build_angle_grid(rcfg.theta_x_range, rcfg.theta_y_range, grid_n)
    spots = [angles_to_spot_position(tx, ty, bdl.d_smx, bdl.d_smy) for _, _, tx, ty in grid]
    timings = {}

    if not skip_mc:
        ct = reduce_vacuum_to_air(resample_to_isotropic(rec.load_image(), rcfg.iso_spacing_mm))
        geom = _field_geometry(ct, gantry, rcfg, bdl)
        runner = MCSquareRunner(engine["mcsquare_install"], engine["mc_work_dir"],
                                bdl_file=bdl_name)
        logger.info("field %s gantry=%.1f E=%g MeV | %d spots | grid %s",
                    rec.patient_id, gantry, e, len(spots), geom.ct.GetSize())

        # --- beamlet mode: one call for the whole field ----------------------
        t0 = perf_counter()
        for index, dose_img, sim_res in runner.run_beamlet_field(
                geom.ct, energy=e, gantry_angle=geom.mc_gantry, spots_xy=spots,
                isocenter=geom.iso_mc, num_primaries=num_primaries,
                num_threads=rcfg.num_threads, rng_seed=rcfg.rng_seed):
            ix, iy, _, _ = grid[index]
            timings["beamlet_field_seconds"] = sim_res["mc_field_seconds"]
            _save_record(dir_beam, f"a{ix:02d}_{iy:02d}", geom.ct, geom, bdl, rcfg,
                         spots[index], e, sitk.GetArrayFromImage(dose_img), ix, iy, rec, sim_res)
        timings["beamlet_wall_seconds"] = perf_counter() - t0
        logger.info("beamlet mode: %d spots, %.1f s MC (%.2f s/spot)", len(spots),
                    timings["beamlet_field_seconds"],
                    timings["beamlet_field_seconds"] / len(spots))

        # --- sequential: the current path, one call per spot ------------------
        t0, seq_mc = perf_counter(), []
        if not run_sequential:
            logger.info("baseline: reusing %s (%d beamlets)", dir_seq.name,
                        len(list(dir_seq.glob("*_sim_res.json"))))
        for index, (ix, iy, _, _) in enumerate(grid if run_sequential else []):
            dose_img, sim_res = runner.run_beamlet(
                geom.ct, energy=e, gantry_angle=geom.mc_gantry, spot_xy=spots[index],
                isocenter=geom.iso_mc, num_primaries=num_primaries,
                num_threads=rcfg.num_threads, rng_seed=rcfg.rng_seed)
            seq_mc.append(sim_res["mc_seconds"])
            _save_record(dir_seq, f"a{ix:02d}_{iy:02d}", geom.ct, geom, bdl, rcfg,
                         spots[index], e, sitk.GetArrayFromImage(dose_img), ix, iy, rec, sim_res)
        if seq_mc:
            timings["sequential_mc_seconds"] = float(np.sum(seq_mc))
            timings["sequential_wall_seconds"] = perf_counter() - t0
            timings["speedup_mc"] = (timings["sequential_mc_seconds"]
                                     / timings["beamlet_field_seconds"])
            logger.info("sequential: %d spots, %.1f s MC (%.2f s/spot) -> speedup %.2fx",
                        len(spots), timings["sequential_mc_seconds"],
                        timings["sequential_mc_seconds"] / len(spots), timings["speedup_mc"])

    # --- ADoTA on both ground truths -----------------------------------------
    import torch
    m = load_yaml_config(plot_config)["model"]
    device = resolve_device(int(m.get("device_index", 0)))
    model = load_model(ROOT / "models" / m["name"] / m.get("fname", "best_model.pth"),
                       ROOT / "models" / m["name"] / "hyperparams.json", device)
    for d in (dir_seq, dir_beam):
        logger.info("inference %s", d.name)
        infer_dir(d, model, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    panel_seq = score_dir_grids(dir_seq, CRITERIA, grid_n)
    panel_beam = score_dir_grids(dir_beam, CRITERIA, grid_n)

    # --- MC vs MC over the same lattice --------------------------------------
    mc_grids = {c.key: np.full((grid_n, grid_n), np.nan) for c in CRITERIA}
    tasks, cells = [], []
    # gamma is asymmetric: the reference sets the evaluated points and the local
    # normalization, so the arm with the better statistics should usually be it.
    ref_dir, eval_dir = (dir_beam, dir_seq) if mc_reference == "a" else (dir_seq, dir_beam)
    for ix, iy, _, _ in grid:
        stem = f"a{ix:02d}_{iy:02d}"
        a, b = ref_dir / f"{stem}_ds.npy", eval_dir / f"{stem}_ds.npy"
        if a.exists() and b.exists():
            for c in CRITERIA:
                tasks.append((str(a), str(b), c))
                cells.append((c.key, ix, iy))
    workers = max(1, min((os.cpu_count() or 8) - 2, 46))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (key, ix, iy), gpr in zip(cells, ex.map(_mc_vs_mc, tasks, chunksize=1)):
            mc_grids[key][ix, iy] = gpr

    # --- table + figures ------------------------------------------------------
    out_root = Path(out_root)
    mc_label = (f"{label_b} vs {label_a} (ref)" if mc_reference == "a"
                else f"{label_a} vs {label_b} (ref)")
    labels = (mc_label, f"ADoTA vs {label_b}", f"ADoTA vs {label_a}")
    table = []
    for c in CRITERIA:
        for label, g in ((labels[0], mc_grids[c.key]),
                         (labels[1], panel_seq.grids[c.key]),
                         (labels[2], panel_beam.grids[c.key])):
            table.append({"criterion": c.key, "comparison": label, **_stats(g)})
    width = max(len(r["comparison"]) for r in table)
    logger.info("")
    logger.info("%-*s  %-10s %5s %8s %8s %8s", width, "comparison", "criterion", "n",
                "mean", "min", "max")
    for r in table:
        logger.info("%-*s  %-10s %5d %7.2f%% %7.2f%% %7.2f%%", width, r["comparison"],
                    r["criterion"], r["n"], r["mean"], r["min"], r["max"])

    thetas = np.linspace(rcfg.theta_x_range[0], rcfg.theta_x_range[1], grid_n)
    written = []
    for c in CRITERIA:
        panels = {labels[0]: mc_grids[c.key], labels[1]: panel_seq.grids[c.key],
                  labels[2]: panel_beam.grids[c.key]}
        vals = np.concatenate([g[np.isfinite(g)].ravel() for g in panels.values()])
        vmin, vmax = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 100.0)
        written += beamlet_mode_comparison_figure(
            panels, thetas, str(out_root / f"{tag}_comparison_{c.key}"),
            vmin, vmax, cbar_label=c.cbar_label)
        for name, g in panels.items():
            stem = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
            written += angle_robustness_panel(
                g, thetas, thetas, str(out_root / f"{tag}_panel_{stem}_{c.key}"), vmin, vmax,
                with_colorbar=True, cbar_label=c.cbar_label)
        np.savez(out_root / f"{tag}_grids_{c.key}.npz",
                 **{stem: g for stem, g in zip(("mc_vs_mc", "adota_vs_base", "adota_vs_new"),
                                               panels.values())})
    (out_root / f"{tag}_prototype.json").write_text(json.dumps(
        {"patient": rec.patient_id, "anatomy": rec.anatomy, "gantry": gantry, "energy": e,
         "n_spots": len(spots), "num_primaries": num_primaries, "grid_n": grid_n,
         "timings": timings, "table": table, "arm_a": str(dir_beam), "arm_b": str(dir_seq),
         "label_a": label_a, "label_b": label_b, "mc_reference": mc_reference}, indent=2))
    logger.info("wrote %d figure files + beamlet_mode_prototype.json -> %s",
                len(written), out_root)


if __name__ == "__main__":
    app()
