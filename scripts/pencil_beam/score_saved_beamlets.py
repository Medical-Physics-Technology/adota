"""Pencil beam on saved MC beamlet directories; PBA and saved ADoTA ranked against the saved MC.

Stages (``--stages``, comma-separated, run in this order):

* ``pba``    -- rebuild each field's canonical CT from the source DICOM (every crop
                checked bit for bit against the saved ones before anything is
                written), ray-trace it once, and add ``{id}_ds_pba.npy`` per beamlet
                plus a ``pba_provenance.json`` to the source directory.
* ``adota``  -- only for beamlets without one: add ``{id}_ds_pred.npy`` (and an
                ``adota_provenance.json``). Existing predictions are used as they are.
* ``score``  -- gamma and dose agreement against the saved MC (``_ds.npy``) of the
                pencil beam and of the ADoTA prediction (``_ds_pred.npy``).
* ``report`` -- the best and worst ``n`` beamlets per engine, with a three-engine
                figure each, and the tables behind them.

Files are only ever added to a source directory, each created exclusively: no
existing file there is modified or replaced. Metrics, rankings and figures are
written to ``output_root`` only. ``--dry-run`` lists the selection and runs the
geometry check without writing anything.

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
from src.pencil_beam.beam_model import model_fingerprint
from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.pipeline import provenance, write_rows_csv
from src.pencil_beam.saved_beamlets import (
    compute_field_pba,
    field_windows,
    load_triplet,
    rank_beamlets,
    score_beamlet,
    survey_dirs,
    write_missing_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pymedphys").setLevel(logging.WARNING)
logger = logging.getLogger("score_saved_beamlets")
app = typer.Typer(help="MC vs pencil beam vs ADoTA on saved MC beamlet directories.")

STAGES = ("pba", "adota", "score", "report")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_pba(blocks, cfg, out_root: Path, run_provenance: dict, dry_run: bool = False) -> None:
    """Rebuild each field, verify every crop, then (unless ``dry_run``) add the PBA doses."""
    from src.beamlets.bdl import BeamDataLibrary
    from src.datasets.registry import build_dataset_from_config
    from src.mc_generation.geometry import reduce_vacuum_to_air, resample_to_isotropic
    from src.mc_generation.robustness import field_geometry
    from src.mc_generation.sweep import robustness_config_from_dict

    gen = load_yaml_config(Path(cfg["generation_config"]))
    rcfg = robustness_config_from_dict(gen["robustness"])
    engine_cfg = gen["engine"]
    bdl = BeamDataLibrary.from_file(Path(engine_cfg["mcsquare_install"]) / "BDL" / engine_cfg["bdl_file"])
    dataset = build_dataset_from_config({"datasets": gen["datasets"], "name": gen.get("name", "generation")})
    records = {dataset.record(i).uid: dataset.record(i) for i in range(len(dataset))}
    missing = sorted({b.provenance_uid for b in blocks} - set(records))
    if missing:
        raise typer.BadParameter(f"CTs {missing} are not in the datasets of {cfg['generation_config']}; "
                                 "generation_config must be the config that produced these directories")
    pb = cfg["pencil_beam"]
    engine = None
    if not dry_run:
        from src.pencil_beam.engine import PencilBeamEngine

        engine = PencilBeamEngine(Path(pb["dosecuda_kernels"]), Path(pb["model_dir"]),
                                  gpu_id=int(pb.get("gpu_id", 0)))
    record = {"model_fingerprint": run_provenance["pencil_beam_model_fingerprint"],
              "model_dir": str(pb["model_dir"]), "generation_config": str(cfg["generation_config"]),
              "layout": "(H, W, D) = (z, y, x), the _ds.npy crop of the field's canonical CT; dose per spot "
                        "weight 1000 on the MCsquare scale", **run_provenance}

    by_patient = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        by_patient[b.provenance_uid][round(b.gantry, 6)].append(b)
    for uid, fields in by_patient.items():
        ct = reduce_vacuum_to_air(resample_to_isotropic(records[uid].load_image(), rcfg.iso_spacing_mm))
        for field_blocks in fields.values():
            # rotate by the exact saved angle: the rounded grouping key changes the interpolated CT
            geom = field_geometry(ct, field_blocks[0].gantry, rcfg, bdl)
            for b in field_blocks:
                if dry_run:
                    field_windows(geom, bdl, rcfg.roi_size, b)
                    logger.info("dry-run %s: all %d rebuilt CT crops match the saved ones", b.name, len(b.stems))
                    continue
                stats = compute_field_pba(engine, geom, bdl, rcfg.roi_size, b, record)
                logger.info("pba %s: %d beamlets (ray trace %.1f s, spots %.1f s)", b.name, stats["computed"],
                            stats["field_seconds"], stats["spot_seconds"])
                (out_root / b.name).mkdir(parents=True, exist_ok=True)
                (out_root / b.name / "pba_run.json").write_text(json.dumps(stats, indent=2, default=str))


def _stage_adota(blocks, cfg, run_provenance: dict) -> None:
    """Add ADoTA predictions only where a beamlet has none (existing ones are used as they are)."""
    todo = [b for b in blocks if b.n_missing_predictions]
    if not todo:
        logger.info("adota: every beamlet already has a _ds_pred.npy; nothing to add")
        return
    from src.adota.utils import load_model
    from src.evaluation.cli import resolve_device

    ad = cfg.get("adota")
    if not ad:
        raise typer.BadParameter("blocks lack predictions; add an `adota:` block (model_dir, fname) to the config")
    model_dir = Path(ad["model_dir"])
    ckpt = model_dir / ad.get("fname", "best_model.pth")
    device = resolve_device(int(ad.get("device_index", 0)))
    model = load_model(ckpt, model_dir / "hyperparams.json", device)
    record = {"checkpoint": str(ckpt), "checkpoint_sha256": _sha256(ckpt),
              "hyperparams_sha256": _sha256(model_dir / "hyperparams.json"), **run_provenance}
    for b in todo:
        n = write_missing_predictions(b, model, device, record, int(ad.get("batch_size", 56)))
        logger.info("adota %s: added %d missing predictions", b.name, n)


def _stage_score(blocks, cfg, out_root: Path, criteria, overwrite: bool) -> List[dict]:
    workers = int(cfg.get("scoring", {}).get("workers", 32))
    rows = []
    for b in blocks:
        path = out_root / b.name / "metrics.json"
        if path.exists() and not overwrite:
            rows += json.loads(path.read_text())
            continue
        for suffix, stage in (("_ds_pba.npy", "pba"), ("_ds_pred.npy", "adota")):
            missing = [s for s in b.stems if not (b.path / f"{s}{suffix}").exists()]
            if missing:
                raise typer.BadParameter(f"{b.name}: {len(missing)} beamlets have no {suffix}; run the {stage} stage")
        tasks = [(str(b.path), s, tuple(criteria), b.name) for s in b.stems]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            block_rows = list(ex.map(score_beamlet, tasks, chunksize=2))
        meta = {"anatomy": b.anatomy, "patient_id": b.patient_id, "energy": b.energy, "gantry": b.gantry,
                "block_dir": str(b.path)}
        block_rows = [dict(r, **meta) for r in block_rows]
        path.parent.mkdir(parents=True, exist_ok=True)
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
                d = load_triplet(Path(r["block_dir"]), r["stem"])
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
    rescore: Annotated[bool, typer.Option(help="Recompute scores that exist (PBA doses are never replaced).")] = False,
    dry_run: Annotated[bool, typer.Option(help="List the selection, verify the geometry, write nothing.")] = False,
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
    for b in blocks:
        logger.info("selected %s: %d beamlets, %s %s, E=%g MeV, gantry %.2f%s", b.name, len(b.stems), b.anatomy,
                    b.patient_id, b.energy, b.gantry,
                    f", {b.n_missing_predictions} without ADoTA prediction" if b.n_missing_predictions else "")
    if limit_blocks:
        blocks = blocks[:limit_blocks]
    if limit_beamlets:
        from dataclasses import replace
        blocks = [replace(b, stems=b.stems[:limit_beamlets]) for b in blocks]
    logger.info("%d blocks, %d beamlets selected", len(blocks), sum(len(b.stems) for b in blocks))
    criteria = criteria_from_config(cfg.get("scoring", {}).get("gamma_criteria"))
    gen_engine = load_yaml_config(Path(cfg["generation_config"]))["engine"]
    prov = provenance(Path(gen_engine["mcsquare_install"]), gen_engine.get("bdl_file", "hptc_beam_model_rsnone.txt"),
                      gen_engine.get("scanner", "default"), Path(cfg["pencil_beam"]["dosecuda_kernels"]))
    prov["pencil_beam_model_fingerprint"] = model_fingerprint(Path(cfg["pencil_beam"]["model_dir"]))
    prov["created"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    if prov["adota_uncommitted_changes"]:
        logger.warning("the adota checkout has uncommitted changes: results will not be traceable to a commit")
    if dry_run:
        _stage_pba(blocks, cfg, out_root, prov, dry_run=True)
        logger.info("dry-run: %d blocks selected and verified; nothing written", len(blocks))
        raise typer.Exit()

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_config.json").write_text(json.dumps({
        "config": cfg, "stages": wanted, "blocks": [str(b.path) for b in blocks],
        "excluded": {str(p): why for p, why in excluded}, "provenance": prov,
    }, indent=2, default=str))

    if "pba" in wanted:
        _stage_pba(blocks, cfg, out_root, prov)
    if "adota" in wanted:
        _stage_adota(blocks, cfg, prov)
    rows = None
    if "score" in wanted:
        rows = _stage_score(blocks, cfg, out_root, criteria, rescore)
    if "report" in wanted:
        if rows is None:
            rows = [r for b in blocks for r in json.loads((out_root / b.name / "metrics.json").read_text())]
        _stage_report(rows, cfg, out_root, criteria)


if __name__ == "__main__":
    app()
