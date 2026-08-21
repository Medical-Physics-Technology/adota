"""Water-phantom MC dataset generation (adota-native, YAML-driven).

Phantom counterpart of ``beamlet_angle_robustness.py``: instead of real patient
CTs it sweeps synthetic phantoms (a homogeneous water box, optionally with an air
shell) over the *same* beamlet-angle grid, energies, ROI, flux construction and QA
gates. This gives a homogeneous-medium control for the angular robustness study.

The whole per-beamlet pipeline (MC run -> ROI crop -> flux -> QA -> save) is reused
unchanged from :mod:`src.mc_generation.robustness`; only the CT source differs
(:mod:`src.datasets.phantom`). Phantom geometry and the sweep are fully controlled
by the YAML config. Extensible to high-density slab phantoms via new ``kind``s in
``src/datasets/phantom.py`` -- no change needed here.

  uv run python scripts/mc/generate_phantom_set.py --config scripts/mc/config_phantom.yaml
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adota.config import load_yaml_config
from src.beamlets.bdl import BeamDataLibrary
from src.datasets.phantom import build_phantom_dataset
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.robustness import RobustnessConfig, run_generation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phantom")
app = typer.Typer(help="Water-phantom MC dataset generation (same sweep as real CTs).")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config file.")] = Path(
        "scripts/mc/config_phantom.yaml"),
    grid_n: Annotated[Optional[int], typer.Option(help="Override angle-grid resolution (e.g. 3 for QC).")] = None,
    make_figures: Annotated[Optional[bool], typer.Option(help="Emit per-beamlet QC figures.")] = None,
    num_primaries: Annotated[Optional[float], typer.Option()] = None,
    overwrite: Annotated[Optional[bool], typer.Option()] = None,
) -> None:
    cfg = load_yaml_config(config)

    dataset = build_phantom_dataset({"name": cfg.get("name", "phantom"),
                                     "phantoms": cfg["phantoms"]})
    logger.info("Phantom set: %d phantom(s) -> %s", len(dataset),
                [dataset.record(i).patient_id for i in range(len(dataset))])

    engine = cfg["engine"]
    runner = MCSquareRunner(
        install_dir=engine["mcsquare_install"], work_root=engine["mc_work_dir"],
        bdl_file=engine.get("bdl_file", "hptc_beam_model_rsnone.txt"),
        scanner=engine.get("scanner", "default"),
    )
    bdl = BeamDataLibrary.from_file(
        str(Path(engine["mcsquare_install"]) / "BDL" / engine.get("bdl_file", "hptc_beam_model_rsnone.txt")))

    r = cfg.get("robustness", {})
    rcfg = RobustnessConfig(
        energies=[float(e) for e in r.get("energies", [90.0, 140.0, 200.0])],
        theta_x_range=tuple(r.get("theta_x_range", (-2.0, 2.0))),
        theta_y_range=tuple(r.get("theta_y_range", (-2.0, 2.0))),
        grid_n=int(grid_n if grid_n is not None else r.get("grid_n", 18)),
        gantry_mode=r.get("gantry_mode", "fixed"),
        gantry_value=float(r.get("gantry_value", 90.0)),
        gantry_seed=int(r.get("gantry_seed", 1234)),
        roi_size=tuple(r.get("roi_size", (60, 60, 320))),
        iso_spacing_mm=float(r.get("iso_spacing_mm", 1.0)),
        num_primaries=float(num_primaries if num_primaries is not None else r.get("num_primaries", 1e7)),
        num_threads=int(r.get("num_threads", 0)),
        rng_seed=int(r.get("rng_seed", 0)),
        min_deposition_ratio=float(r.get("min_deposition_ratio", 0.5)),
        output_root=r.get("output_root", "/scratch/mstryja/DoTA_dataset_v2"),
        experiment_prefix=r.get("experiment_prefix", "water_phantom"),
        experiment_version=r.get("experiment_version", 2),
        make_figures=bool(make_figures if make_figures is not None else r.get("make_figures", False)),
        overwrite=bool(overwrite if overwrite is not None else r.get("overwrite", False)),
    )
    result = run_generation(dataset, runner, bdl, rcfg)
    out = Path(rcfg.output_root) / f"{rcfg.experiment_prefix}_summary.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("Summary -> %s", out)


if __name__ == "__main__":
    app()
