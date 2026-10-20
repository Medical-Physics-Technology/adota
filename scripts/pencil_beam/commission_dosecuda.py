"""Commission DoseCUDA's proton pencil beam against MCsquare (YAML-driven).

Stages (``--stages``, comma-separated, run in this order):

* ``rlsp``     -- HU -> RLSP from MCsquare's scanner + Geant4 stopping powers.
* ``simulate`` -- central-spot MC beamlets in a water box, every energy node x
                  source-to-surface distance (resumable, one ``.npz`` per run).
* ``fit``      -- IDD / sigma_mcs / sigma_air / R80 tables -> a model directory in
                  DoseCUDA's lookup-table layout.
* ``validate`` -- fresh MC at held-out SSDs and between-node energies, more
                  primaries; R80, IDD, width, gamma; CSV + figures.

  uv run python scripts/pencil_beam/commission_dosecuda.py \\
      --config scripts/pencil_beam/config_commission_dosecuda.yaml --stages rlsp,simulate,fit,validate

Guide: scripts/docs/commission_dosecuda.md.
"""
from __future__ import annotations

import os

# Pin BLAS to 1 thread so the sigma-fit process pool scales across cores
# (must be set before numpy loads).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

from src.beamlets.bdl import BeamDataLibrary
from src.evaluation.cli import SET_OVERRIDE_HELP, apply_set_overrides, load_yaml_config
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.pencil_beam.commissioning import CommissioningConfig, run_commissioning_beamlets
from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.model_card import write_model_card
from src.pencil_beam.pipeline import fit_from_runs, provenance, validate_water, write_rows_csv
from src.pencil_beam.rlsp import hu_to_rlsp_table, write_rlsp_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("commission_dosecuda")
app = typer.Typer(help="Commission the DoseCUDA pencil beam against MCsquare.")

STAGES = ("rlsp", "simulate", "fit", "validate")


def _commissioning_config(block: dict) -> CommissioningConfig:
    """Typed config from YAML; numbers are coerced (YAML reads ``2e5`` without a dot as a string)."""
    fields = CommissioningConfig.__dataclass_fields__
    unknown = set(block) - set(fields)
    if unknown:
        raise typer.BadParameter(f"unknown commissioning keys: {sorted(unknown)}")
    typed = {}
    for key, value in block.items():
        kind = str(fields[key].type)
        if kind.startswith("List"):
            typed[key] = [float(v) for v in value]
        elif kind in ("int", "float"):
            typed[key] = int(float(value)) if kind == "int" else float(value)
        else:
            typed[key] = value
    return CommissioningConfig(**typed)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config file.")] = Path(
        "scripts/pencil_beam/config_commission_dosecuda.yaml"),
    stages: Annotated[str, typer.Option(help=f"Comma-separated subset of {','.join(STAGES)}.")] = "rlsp,fit",
    set_: Annotated[Optional[List[str]], typer.Option("--set", help=SET_OVERRIDE_HELP)] = None,
    gpu_id: Annotated[Optional[int], typer.Option(help="CUDA device for the pencil beam (as visible).")] = None,
) -> None:
    cfg = apply_set_overrides(load_yaml_config(config), set_ or [])
    wanted = [s.strip() for s in stages.split(",") if s.strip()]
    bad = set(wanted) - set(STAGES)
    if bad:
        raise typer.BadParameter(f"unknown stages {sorted(bad)}; choose from {STAGES}")

    engine = cfg["engine"]
    install = Path(engine["mcsquare_install"])
    bdl_file = engine.get("bdl_file", "hptc_beam_model_rsnone.txt")
    scanner = engine.get("scanner", "default")
    out = Path(cfg["output_dir"])
    model_dir = Path(cfg.get("model_dir", out / "model"))
    runs_dir = out / "runs"
    ccfg = _commissioning_config(cfg.get("commissioning", {}))
    bdl = BeamDataLibrary.from_file(install / "BDL" / bdl_file)
    if abs(bdl.d_nozzle - ccfg.nozzle_distance_mm) > 1e-6:
        raise typer.BadParameter(f"commissioning.nozzle_distance_mm must equal the BDL's {bdl.d_nozzle}")
    e_ref = float(cfg.get("rlsp_reference_energy_mev", 100.0))
    hu, rlsp = hu_to_rlsp_table(install, scanner, reference_energy_mev=e_ref)
    rlsp_water = float(np.interp(ccfg.water_hu, hu, rlsp))
    rlsp_air = float(np.interp(ccfg.air_hu, hu, rlsp))
    kernels = Path(cfg["pencil_beam"]["dosecuda_kernels"])
    gpu = int(gpu_id if gpu_id is not None else cfg["pencil_beam"].get("gpu_id", 0))
    runner = MCSquareRunner(str(install), engine["mc_work_dir"], bdl_file=bdl_file, scanner=scanner)

    if "rlsp" in wanted:
        model_dir.mkdir(parents=True, exist_ok=True)
        write_rlsp_csv(model_dir / "HU_RLSP.csv", hu, rlsp)
        logger.info("rlsp: %d rows at %.0f MeV (HU 0 -> %.4f, air -> %.5f) -> %s", hu.size, e_ref,
                    rlsp_water, rlsp_air, model_dir / "HU_RLSP.csv")

    if "simulate" in wanted:
        logger.info("simulate: %d energies x %d SSDs at %.1e primaries -> %s", len(ccfg.energies),
                    len(ccfg.surface_to_iso_mm), ccfg.num_primaries, runs_dir)
        run_commissioning_beamlets(runner, ccfg, runs_dir)

    if "fit" in wanted:
        meta = {"provenance": provenance(install, bdl_file, scanner, kernels),
                "rlsp_reference_energy_mev": e_ref, "sigma_fit": "primary + DoseCUDA halo (kernel form)"}
        model = fit_from_runs(runs_dir, ccfg, rlsp_water, rlsp_air, bdl.d_smx, bdl.d_smy, meta,
                              n_workers=int(cfg.get("fit_workers", 16)), bdl=bdl)
        model.write(model_dir)
        if not (model_dir / "HU_RLSP.csv").exists():
            write_rlsp_csv(model_dir / "HU_RLSP.csv", hu, rlsp)
        write_model_card(model_dir)
        logger.info("fit: %d energy nodes %.1f-%.1f MeV -> %s", len(model.energies), model.energies[0],
                    model.energies[-1], model_dir)

    if "validate" in wanted:
        from src.pencil_beam.engine import PencilBeamEngine

        v = cfg.get("validation", {})
        criteria = criteria_from_config(v.get("gamma_criteria"))
        eng = PencilBeamEngine(kernels, model_dir, gpu_id=gpu)
        vdir = out / v.get("results_subdir", "validation")   # the cached validation MC stays in validation/runs
        rows = validate_water(
            runner, eng, ccfg, [float(e) for e in v["energies"]], [float(s) for s in v["surface_to_iso_mm"]],
            vdir, float(v.get("num_primaries", 1e7)), criteria, rlsp_water, rlsp_air,
            gamma_backend=v.get("gamma_backend", "pymedphys"),
            gamma_backend_options=v.get("gamma_backend_options"),
            make_figures=bool(v.get("make_figures", True)), runs_dir=out / "validation" / "runs")
        csv_path = write_rows_csv(rows, vdir / "water_validation.csv")
        (vdir / "water_validation.json").write_text(json.dumps(
            {"model_dir": str(model_dir), "criteria": [c.key for c in criteria], "rows": rows}, indent=2,
            default=float))
        from src.figures.pencil_beam import commissioning_summary_figure

        commissioning_summary_figure(rows, str(vdir / f"water_validation_summary_{criteria[0].key}"),
                                     gamma_key=f"gpr_{criteria[0].key}")
        # the model carries its own measured accuracy wherever it is copied
        import shutil

        keep = model_dir / "validation"
        (keep / "figures").mkdir(parents=True, exist_ok=True)
        for f in vdir.glob("water_validation*"):
            shutil.copy2(f, keep / f.name)
        for f in (vdir / "figures").glob("water_e*.png"):    # dose maps stay on /scratch
            shutil.copy2(f, keep / "figures" / f.name)
        write_model_card(model_dir, rows, gamma_key=criteria[0].key)
        logger.info("validate: %d runs -> %s (copied into %s)", len(rows), csv_path, keep)


if __name__ == "__main__":
    app()
