"""MC (MCsquare) vs pencil beam (DoseCUDA) vs ADoTA, same beamlets, same CT grid.

Patients, field gantries, energies and beamlet angles are selected with the same
``datasets`` + ``robustness`` blocks as ``scripts/mc/generate_patient_set.py``, so
a comparison beamlet is generated exactly like a training beamlet. Per beamlet:
the three doses on one window of the field's CT grid (``{stem}_engines.npz`` +
``{stem}_adota_region.npy``), gamma / range / dose-difference metrics, and a
figure; per run: ``engine_comparison.csv`` / ``.json`` and a summary.

  uv run python scripts/pencil_beam/compare_engines.py \\
      --config scripts/pencil_beam/config_compare_engines.yaml

Guide: scripts/docs/compare_engines.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, List, Optional

import typer

from src.adota.utils import load_model
from src.beamlets.bdl import BeamDataLibrary
from src.datasets.registry import build_dataset_from_config
from src.evaluation.cli import SET_OVERRIDE_HELP, apply_set_overrides, load_yaml_config, resolve_device
from src.mc_generation.geometry import reduce_vacuum_to_air, resample_to_isotropic
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.sweep import build_angle_grid, resolve_gantries, robustness_config_from_dict
from src.pencil_beam.comparison import ComparisonSettings, compare_block, summarize
from src.pencil_beam.engine import PencilBeamEngine
from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.pipeline import provenance, write_rows_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pymedphys").setLevel(logging.WARNING)
logger = logging.getLogger("compare_engines")
app = typer.Typer(help="Compare MCsquare, the DoseCUDA pencil beam and ADoTA on the same CT grid.")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config file.")] = Path(
        "scripts/pencil_beam/config_compare_engines.yaml"),
    set_: Annotated[Optional[List[str]], typer.Option("--set", help=SET_OVERRIDE_HELP)] = None,
    num_primaries: Annotated[Optional[float], typer.Option(help="MC primaries per beamlet.")] = None,
    no_adota: Annotated[bool, typer.Option(help="Skip ADoTA (MC vs pencil beam only).")] = False,
    dry_run: Annotated[bool, typer.Option(help="List the beamlets that would run, then exit.")] = False,
) -> None:
    cfg = apply_set_overrides(load_yaml_config(config), set_ or [])
    dataset = build_dataset_from_config({"datasets": cfg["datasets"], "name": cfg.get("name", "engine_comparison")})
    rcfg = robustness_config_from_dict(cfg.get("robustness", {}), num_primaries=num_primaries,
                                       default_prefix="engine_comparison")
    grid = build_angle_grid(rcfg.theta_x_range, rcfg.theta_y_range, rcfg.grid_n, rcfg.angles)
    plan = [(i, g) for i in range(len(dataset)) for g in resolve_gantries(rcfg, dataset.record(i).uid)]
    logger.info("%d patient-fields x %d energies x %d beamlets = %d beamlets (%.0e primaries each)",
                len(plan), len(rcfg.energies), len(grid), len(plan) * len(rcfg.energies) * len(grid),
                rcfg.num_primaries)
    for i, g in plan:
        rec = dataset.record(i)
        logger.info("  %-14s %-10s gantry %.1f deg", rec.patient_id, rec.anatomy, g)
    if dry_run:
        raise typer.Exit()

    engine = cfg["engine"]
    install = Path(engine["mcsquare_install"])
    bdl_file = engine.get("bdl_file", "hptc_beam_model_rsnone.txt")
    scanner = engine.get("scanner", "default")
    bdl = BeamDataLibrary.from_file(install / "BDL" / bdl_file)
    runner = MCSquareRunner(str(install), engine["mc_work_dir"], bdl_file=bdl_file, scanner=scanner)
    pb_cfg = cfg["pencil_beam"]
    pba = PencilBeamEngine(Path(pb_cfg["dosecuda_kernels"]), Path(pb_cfg["model_dir"]),
                           gpu_id=int(pb_cfg.get("gpu_id", 0)))

    model, device = None, None
    ad = cfg.get("adota", {})
    if not no_adota:
        device = resolve_device(int(ad.get("device_index", 0)))
        model_dir = Path(ad["model_dir"])
        model = load_model(model_dir / ad.get("fname", "best_model.pth"), model_dir / "hyperparams.json", device)

    comp = cfg.get("comparison", {})
    out_dir = Path(cfg["output_dir"])
    settings = ComparisonSettings(
        output_dir=out_dir, criteria=criteria_from_config(comp.get("gamma_criteria")),
        gamma_local=bool(comp.get("gamma_local", False)), gamma_backend=comp.get("gamma_backend", "pymedphys"),
        gamma_backend_options=comp.get("gamma_backend_options") or {},
        lateral_margin_mm=int(comp.get("lateral_margin_mm", 30)),
        make_figures=bool(comp.get("make_figures", True)), adota_batch_size=int(ad.get("batch_size", 56)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps({
        "config": cfg, "provenance": provenance(install, bdl_file, scanner, Path(pb_cfg["dosecuda_kernels"])),
        "pencil_beam_model": str(pb_cfg["model_dir"]),
        "adota_model": None if no_adota else str(Path(ad["model_dir"]) / ad.get("fname", "best_model.pth")),
    }, indent=2, default=str))

    rows = []
    current = None
    for i, field_gantry in plan:
        rec = dataset.record(i)
        if current != i:
            ct = reduce_vacuum_to_air(resample_to_isotropic(rec.load_image(), rcfg.iso_spacing_mm))
            current = i
        for energy in rcfg.energies:
            logger.info("[%s g%.1f E%.1f]", rec.patient_id, field_gantry, energy)
            rows += compare_block(rec, ct, field_gantry, energy, grid, runner, bdl, rcfg, pba, model, device,
                                  settings)
            write_rows_csv(rows, out_dir / "engine_comparison.csv")

    summary = summarize(rows, settings.criteria)
    (out_dir / "engine_comparison.json").write_text(json.dumps(
        {"criteria": [c.key for c in settings.criteria], "gamma_local": settings.gamma_local,
         "summary": summary, "rows": rows}, indent=2, default=float))
    for k, v in summary.items():
        logger.info("  %-32s n=%d mean %.3f median %.3f min %.3f", k, v["n"], v["mean"], v["median"], v["min"])
    logger.info("%d beamlets -> %s", len(rows), out_dir / "engine_comparison.csv")


if __name__ == "__main__":
    app()
