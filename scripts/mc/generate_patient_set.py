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
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.robustness import (
    resolve_gantry,
    robustness_config_from_dict,
    run_generation,
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
    overwrite: Annotated[Optional[bool], typer.Option()] = None,
    dry_run: Annotated[bool, typer.Option(help="List patients + resolved gantry, then exit (no MC).")] = False,
) -> None:
    cfg = load_yaml_config(config)

    ds_cfg = {"datasets": cfg["datasets"], "name": cfg.get("name", "patient_set")}
    if n_patients is not None:
        for d in ds_cfg["datasets"]:
            d["n_patients"] = n_patients
    dataset = build_dataset_from_config(ds_cfg)

    rcfg = robustness_config_from_dict(
        cfg.get("robustness", {}), grid_n=grid_n, num_primaries=num_primaries,
        make_figures=make_figures, overwrite=overwrite, default_prefix="patient_set")

    # Preview the per-patient random gantry (seeded, reproducible) before running.
    logger.info("Patient set (%d patients), gantry_mode=%s:", len(dataset), rcfg.gantry_mode)
    for i in range(len(dataset)):
        rec = dataset.record(i)
        logger.info("  %-14s %-10s gantry=%6.1f deg", rec.patient_id, rec.anatomy,
                    resolve_gantry(rcfg, rec.uid))
    if dry_run:
        logger.info("dry-run: no simulation performed.")
        raise typer.Exit()

    engine = cfg["engine"]
    runner = MCSquareRunner(
        install_dir=engine["mcsquare_install"], work_root=engine["mc_work_dir"],
        bdl_file=engine.get("bdl_file", "hptc_beam_model_rsnone.txt"),
        scanner=engine.get("scanner", "default"),
    )
    bdl = BeamDataLibrary.from_file(
        str(Path(engine["mcsquare_install"]) / "BDL" / engine.get("bdl_file", "hptc_beam_model_rsnone.txt")))

    result = run_generation(dataset, runner, bdl, rcfg)
    out = Path(rcfg.output_root) / f"{rcfg.experiment_prefix}_summary.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("Summary -> %s", out)


if __name__ == "__main__":
    app()
