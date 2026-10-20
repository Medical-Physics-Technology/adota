"""MC vs pencil beam vs ADoTA on saved MC beamlet directories, ranked by gamma pass rate.

Stages (``--stages``, comma-separated, run in this order):

* ``mirror`` -- select the full-lattice directories and mirror each one into the
                output root as symlinks (the source datasets are never written).
* ``pba``    -- rebuild each field's canonical CT from the source DICOM (checked
                bit for bit against the saved crops), ray-trace it once, and write
                ``{id}_ds_pba.npy`` per beamlet in the ADoTA crop layout.
* ``adota``  -- fresh ADoTA inference into the mirror (``{id}_ds_pred.npy``).
* ``score``  -- gamma and dose agreement of PBA and ADoTA against MC, per beamlet.
* ``report`` -- the best and worst ``n`` beamlets per engine, with a three-engine
                figure each, and the tables behind them.

  uv run python scripts/pencil_beam/score_saved_beamlets.py \\
      --config scripts/pencil_beam/config_score_energy_gantry_full.yaml

Guide: scripts/docs/score_saved_beamlets.md.
"""
from __future__ import annotations

import os

# Pin BLAS to 1 thread so the gamma process pool scales across cores (before numpy loads).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

from src.evaluation.cli import SET_OVERRIDE_HELP, apply_set_overrides, load_yaml_config
from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.pipeline import provenance, write_rows_csv
from src.pencil_beam.saved_beamlets import (
    compute_field_pba,
    load_triplet,
    mirror_inputs,
    rank_beamlets,
    score_beamlet,
    survey_dirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pymedphys").setLevel(logging.WARNING)
logger = logging.getLogger("score_saved_beamlets")
app = typer.Typer(help="MC vs pencil beam vs ADoTA on saved MC beamlet directories.")

STAGES = ("mirror", "pba", "adota", "score", "report")


def _stage_pba(blocks, cfg, out_root: Path, overwrite: bool) -> None:
    from src.beamlets.bdl import BeamDataLibrary
    from src.datasets.registry import build_dataset_from_config
    from src.mc_generation.geometry import reduce_vacuum_to_air, resample_to_isotropic
    from src.mc_generation.robustness import field_geometry
    from src.mc_generation.sweep import robustness_config_from_dict
    from src.pencil_beam.engine import PencilBeamEngine

    gen = load_yaml_config(Path(cfg["generation_config"]))
    rcfg = robustness_config_from_dict(gen["robustness"])
    engine_cfg = gen["engine"]
    bdl = BeamDataLibrary.from_file(Path(engine_cfg["mcsquare_install"]) / "BDL" / engine_cfg["bdl_file"])
    dataset = build_dataset_from_config({"datasets": gen["datasets"], "name": gen.get("name", "generation")})
    records = {dataset.record(i).uid: dataset.record(i) for i in range(len(dataset))}
    pb = cfg["pencil_beam"]
    engine = PencilBeamEngine(Path(pb["dosecuda_kernels"]), Path(pb["model_dir"]), gpu_id=int(pb.get("gpu_id", 0)))

    by_patient = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        by_patient[b.provenance_uid][round(b.gantry, 6)].append(b)
    for uid, fields in by_patient.items():
        if uid not in records:
            raise typer.BadParameter(f"{uid} is not in the generation config's datasets")
        ct = reduce_vacuum_to_air(resample_to_isotropic(records[uid].load_image(), rcfg.iso_spacing_mm))
        for field_blocks in fields.values():
            # rotate by the exact saved angle: the rounded grouping key changes the interpolated CT
            geom = field_geometry(ct, field_blocks[0].gantry, rcfg, bdl)
            for b in field_blocks:
                stats = compute_field_pba(engine, geom, bdl, rcfg.roi_size, b, out_root / b.name, overwrite)
                logger.info("pba %s: %d beamlets (ray trace %.1f s, spots %.1f s)", b.name, stats["computed"],
                            stats["field_seconds"], stats["spot_seconds"])
                (out_root / b.name / "pba_run.json").write_text(json.dumps(stats, indent=2, default=str))


def _stage_adota(blocks, cfg, out_root: Path) -> None:
    from src.adota.utils import load_model
    from src.beamlets.inference import InferenceConfig, run_inference
    from src.evaluation.cli import resolve_device

    ad = cfg["adota"]
    device = resolve_device(int(ad.get("device_index", 0)))
    model_dir = Path(ad["model_dir"])
    model = load_model(model_dir / ad.get("fname", "best_model.pth"), model_dir / "hyperparams.json", device)
    for b in blocks:
        out = out_root / b.name
        if all((out / f"{s}_ds_pred.npy").exists() for s in b.stems):
            continue
        run_inference(out, model, device, InferenceConfig(batch_size=int(ad.get("batch_size", 56)), grid_factor=1))
        # the paper's predictions, if the source has them, are read (never written) as a consistency check
        src_pred = b.path / f"{b.stems[0]}_ds_pred.npy"
        if src_pred.exists():
            diff = float(np.max(np.abs(np.load(src_pred) - np.load(out / f"{b.stems[0]}_ds_pred.npy"))))
            logger.info("adota %s: fresh vs source prediction max|diff| %.3g (%s)", b.name, diff, b.stems[0])


def _stage_score(blocks, cfg, out_root: Path, criteria, overwrite: bool) -> List[dict]:
    workers = int(cfg.get("scoring", {}).get("workers", 32))
    rows = []
    for b in blocks:
        path = out_root / b.name / "metrics.json"
        if path.exists() and not overwrite:
            rows += json.loads(path.read_text())
            continue
        tasks = [(str(out_root / b.name), s, tuple(criteria), b.name) for s in b.stems]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            block_rows = list(ex.map(score_beamlet, tasks, chunksize=2))
        meta = {"anatomy": b.anatomy, "patient_id": b.patient_id, "energy": b.energy, "gantry": b.gantry}
        block_rows = [dict(r, **meta) for r in block_rows]
        path.write_text(json.dumps(block_rows, indent=1, default=float))
        head = criteria[0].key
        logger.info("score %s: median %s PBA %.1f%%, ADoTA %.1f%%", b.name, head,
                    np.nanmedian([r[f"pba_{head}"] for r in block_rows]),
                    np.nanmedian([r[f"adota_{head}"] for r in block_rows]))
        rows += block_rows
    write_rows_csv(rows, out_root / "all_beamlets.csv")
    return rows


def _summary(rows, criteria) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["anatomy"]].append(r)
        groups[f"{r['anatomy']}_e{r['energy']:g}"].append(r)
    for name, group in sorted(groups.items()):
        out[name] = {"n": len(group)}
        for engine in ("pba", "adota"):
            for key in [c.key for c in criteria] + ["r80_delta_mm"]:
                vals = np.array([g[f"{engine}_{key}"] for g in group], dtype=float)
                vals = vals[np.isfinite(vals)]
                out[name][f"{engine}_{key}"] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)),
                                                "p5": float(np.percentile(vals, 5)), "min": float(vals.min())}
    return out


def _stage_report(rows, cfg, out_root: Path, criteria) -> None:
    from src.figures.pencil_beam import engine_comparison_figure

    rep = cfg.get("report", {})
    n = int(rep.get("n", 10))
    head = criteria[0]
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ranking = []
    for engine in ("PBA", "ADoTA"):
        best, worst = rank_beamlets(rows, engine, criteria, n)
        for kind, chosen in (("best", best), ("worst", worst)):
            for rank, r in enumerate(chosen, start=1):
                ranking.append(dict(r, ranked_engine=engine, ranked_as=kind, rank=rank))
                if not rep.get("figures", True):
                    continue
                d = load_triplet(out_root / r["block"], r["stem"])
                text = {lab: f"{head.key} = {r[f'{lab.lower()}_{head.key}']:.1f}%  "
                             f"dR80 = {r[f'{lab.lower()}_r80_delta_mm']:+.2f} mm" for lab in ("PBA", "ADoTA")}
                engine_comparison_figure(
                    d["CT"], {"MC": d["MC"], "PBA": d["PBA"], "ADoTA": d["ADoTA"]},
                    str(fig_dir / f"{engine.lower()}_{kind}{rank:02d}_{r['block']}_{r['stem']}_{head.key}_global"),
                    title=(f"{engine} {kind} #{rank}: {r['anatomy']} {r['patient_id']}  E={r['energy']:g} MeV  "
                           f"gantry={r['gantry']:.1f}  {r['stem']}  gamma {head.dose:g}%/{head.dist:g}mm, "
                           f"{head.cutoff:g}% cutoff (global)"),
                    metrics_text=text)
    write_rows_csv(ranking, out_root / f"ranking_best_worst_{n}_{head.key}_global.csv")
    (out_root / "summary.json").write_text(json.dumps(
        {"criteria": [c.key for c in criteria], "gamma": "global, normalised to the MC maximum",
         "n_beamlets": len(rows), "groups": _summary(rows, criteria)}, indent=2))
    logger.info("report: %d ranked beamlets, figures in %s", len(ranking), fig_dir)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config file.")] = Path(
        "scripts/pencil_beam/config_score_energy_gantry_full.yaml"),
    stages: Annotated[str, typer.Option(help=f"Comma-separated subset of {','.join(STAGES)}.")] = ",".join(STAGES),
    set_: Annotated[Optional[List[str]], typer.Option("--set", help=SET_OVERRIDE_HELP)] = None,
    limit_blocks: Annotated[Optional[int], typer.Option(help="Only the first N selected blocks (smoke).")] = None,
    limit_beamlets: Annotated[Optional[int], typer.Option(help="Only the first N beamlets per block (smoke).")] = None,
    overwrite: Annotated[bool, typer.Option(help="Recompute PBA doses and scores that exist.")] = False,
) -> None:
    cfg = apply_set_overrides(load_yaml_config(config), set_ or [])
    wanted = [s.strip() for s in stages.split(",") if s.strip()]
    if set(wanted) - set(STAGES):
        raise typer.BadParameter(f"unknown stages {sorted(set(wanted) - set(STAGES))}; choose from {STAGES}")
    out_root = Path(cfg["output_root"])
    sel = cfg.get("selection", {})
    candidates = sorted({Path(p) for pattern in cfg["sources"] for p in glob(pattern) if Path(p).is_dir()})
    blocks, excluded = survey_dirs(candidates, int(sel.get("grid_n", 18)), int(sel.get("min_beamlets", 200)))
    for path, why in excluded:
        logger.info("excluded %s: %s", path.name, why)
    if limit_blocks:
        blocks = blocks[:limit_blocks]
    if limit_beamlets:
        from dataclasses import replace
        blocks = [replace(b, stems=b.stems[:limit_beamlets]) for b in blocks]
    logger.info("%d blocks, %d beamlets selected", len(blocks), sum(len(b.stems) for b in blocks))
    criteria = criteria_from_config(cfg.get("scoring", {}).get("gamma_criteria"))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config.json").write_text(json.dumps({
        "config": cfg, "stages": wanted, "blocks": [str(b.path) for b in blocks],
        "excluded": {str(p): why for p, why in excluded},
        "provenance": provenance(Path(load_yaml_config(Path(cfg["generation_config"]))["engine"]["mcsquare_install"]),
                                 "hptc_beam_model_rsnone.txt", "default", Path(cfg["pencil_beam"]["dosecuda_kernels"])),
    }, indent=2, default=str))

    if "mirror" in wanted:
        for b in blocks:
            mirror_inputs(b, out_root / b.name)
        logger.info("mirror: %d blocks -> %s (symlinks; sources untouched)", len(blocks), out_root)
    if "pba" in wanted:
        _stage_pba(blocks, cfg, out_root, overwrite)
    if "adota" in wanted:
        _stage_adota(blocks, cfg, out_root)
    rows = None
    if "score" in wanted:
        rows = _stage_score(blocks, cfg, out_root, criteria, overwrite)
    if "report" in wanted:
        if rows is None:
            rows = [r for b in blocks for r in json.loads((out_root / b.name / "metrics.json").read_text())]
        _stage_report(rows, cfg, out_root, criteria)


if __name__ == "__main__":
    app()
