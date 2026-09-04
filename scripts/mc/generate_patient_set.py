"""General multi-patient MC dataset generation with random gantry (YAML-driven).

Training-set expansion: generate the same beamlet grid (angles x energies) as the
robustness runs for N + M patients across anatomies, each patient rotated by a
random gantry angle. Random gantry is realised by rotating the CT into the
canonical beam's-eye frame (A = -(gantry - 90) about the isocenter) and simulating
at 90 deg, so the ADoTA-frame extraction is unchanged and the field angle is kept
as metadata (see src/mc_generation/robustness.py :: generate_for_record).

Per-anatomy patient counts (the "N and M") are the ``n_patients`` knob on each
dataset entry; ``--n-patients`` overrides all of them. Everything else (grid,
energies, ROI, primaries, QA) is shared with the robustness pipeline via
``robustness_config_from_dict`` -- no logic is duplicated.

  uv run python scripts/mc/generate_patient_set.py --config scripts/mc/config_patient_set.yaml
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from src.adota.config import load_yaml_config
from src.beamlets.bdl import BeamDataLibrary
from src.datasets.registry import build_dataset_from_config
from src.mc_generation.geometry import resample_to_isotropic
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.robustness import run_generation
from src.mc_generation.sweep import (
    build_angle_grid,
    resolve_gantries,
    robustness_config_from_dict,
    sweep_fits_ct,
    sweep_z_half_extent_mm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("patient_set")
app = typer.Typer(help="General multi-patient MC generation with random gantry.")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config file.")] = Path(
        "scripts/mc/config_patient_set.yaml"),
    grid_n: Annotated[Optional[int], typer.Option(help="Override angle-grid resolution (e.g. 3 for QC).")] = None,
    make_figures: Annotated[Optional[bool], typer.Option(help="Emit per-beamlet QC figures.")] = None,
    num_primaries: Annotated[Optional[float], typer.Option()] = None,
    n_patients: Annotated[Optional[int], typer.Option(help="Override n_patients for every dataset.")] = None,
    n_gantry: Annotated[Optional[int], typer.Option(help="Override gantry draws per patient.")] = None,
    beamlet_mode: Annotated[Optional[bool], typer.Option(
        help="Run each block as one MCsquare beamlet-mode call (~3x faster).")] = None,
    overwrite: Annotated[Optional[bool], typer.Option()] = None,
    dry_run: Annotated[bool, typer.Option(help="List patients + resolved gantry, then exit (no MC).")] = False,
    check_geometry: Annotated[bool, typer.Option(
        help="Load each CT and report whether its z extent covers the theta_x sweep.")] = False,
) -> None:
    cfg = load_yaml_config(config)

    ds_cfg = {"datasets": cfg["datasets"], "name": cfg.get("name", "patient_set")}
    if n_patients is not None:
        for d in ds_cfg["datasets"]:
            d["n_patients"] = n_patients
    dataset = build_dataset_from_config(ds_cfg)

    rcfg = robustness_config_from_dict(
        cfg.get("robustness", {}), grid_n=grid_n, num_primaries=num_primaries,
        make_figures=make_figures, overwrite=overwrite, n_gantry=n_gantry,
        beamlet_mode=beamlet_mode, default_prefix="patient_set")

    # Preview the per-patient random gantries (seeded, reproducible) before running.
    n_angles = len(build_angle_grid(rcfg.theta_x_range, rcfg.theta_y_range,
                                    rcfg.grid_n, rcfg.angles))
    logger.info("Patient set (%d patients), gantry_mode=%s, energies=%s, %d beamlet angles, "
                "%s MC:", len(dataset), rcfg.gantry_mode, [f"{e:g}" for e in rcfg.energies],
                n_angles, "beamlet-mode" if rcfg.beamlet_mode else "one call per beamlet")
    for i in range(len(dataset)):
        rec = dataset.record(i)
        gantries = resolve_gantries(rcfg, rec.uid)
        logger.info("  %-14s %-10s gantry=%s deg", rec.patient_id, rec.anatomy,
                    ", ".join(f"{g:.1f}" for g in gantries))
    engine = cfg["engine"]
    bdl_name = engine.get("bdl_file", "hptc_beam_model_rsnone.txt")
    bdl = BeamDataLibrary.from_file(str(Path(engine["mcsquare_install"]) / "BDL" / bdl_name))

    # theta_x steers along the slice axis, so a short CT silently loses the outer
    # beamlets to the roi_out_of_bounds gate -- after paying for their MC. Screen
    # the selection first (loads each CT; no simulation).
    if check_geometry:
        need = 2 * sweep_z_half_extent_mm(rcfg, bdl.d_smy)
        logger.info("Geometry check: the sweep needs a CT >= %.0f mm long in z", need)
        n_short = 0
        for i in range(len(dataset)):
            rec = dataset.record(i)
            ct = resample_to_isotropic(rec.load_image(), rcfg.iso_spacing_mm)
            fits, z_extent, max_tx = sweep_fits_ct(ct, rcfg, bdl.d_smy)
            n_short += not fits
            logger.info("  %-28s %-10s z=%6.1f mm  max|theta_x|=%.2f deg  %s",
                        rec.patient_id, rec.anatomy, z_extent, max_tx,
                        "OK" if fits else "TOO SHORT")
        logger.info("Geometry check: %d/%d patients cover the sweep",
                    len(dataset) - n_short, len(dataset))

    if dry_run or check_geometry:
        logger.info("no simulation performed (dry-run/check-geometry).")
        raise typer.Exit()

    runner = MCSquareRunner(
        install_dir=engine["mcsquare_install"], work_root=engine["mc_work_dir"],
        bdl_file=bdl_name, scanner=engine.get("scanner", "default"),
    )
    result = run_generation(dataset, runner, bdl, rcfg)
    out = Path(rcfg.output_root) / f"{rcfg.experiment_prefix}_summary.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("Summary -> %s", out)


if __name__ == "__main__":
    app()
