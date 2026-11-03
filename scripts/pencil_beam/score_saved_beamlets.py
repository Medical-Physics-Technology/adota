"""Dose engines on saved MC beamlet directories, each ranked against the saved MC.

The engines are a config list (``engines:``): pencil-beam models and ADoTA models,
each with its own file next to ``{id}_ds.npy`` (``_ds_pba.npy``, ``_ds_pred.npy``,
``_ds_pred_<model>.npy``). A config with ``pencil_beam:`` + ``adota:`` blocks
instead is read as the two engines ``PBA`` and ``ADoTA``.

Stages (``--stages``, comma-separated, run in this order):

* ``pba``    -- rebuild each field's canonical CT from the source DICOM (every crop
                checked bit for bit against the saved ones before anything is
                written), ray-trace it once, and add each pencil-beam engine's file
                per beamlet plus its provenance file to the source directory.
* ``adota``  -- for each ADoTA engine, only for beamlets without its file: add the
                prediction (and its provenance file). Existing files are used as they are.
* ``score``  -- gamma and dose agreement of every engine against the saved MC
                (``_ds.npy``). Scores are cached per block; an engine already scored
                there is not scored again (``--rescore`` recomputes all).
* ``report`` -- the best and worst ``n`` beamlets per engine, with a figure of all
                engines each, the summary and the tables behind them.

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
    engines_from_config,
    engines_to_score,
    field_windows,
    load_doses,
    rank_beamlets,
    score_beamlet,
    survey_dirs,
    write_missing_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pymedphys").setLevel(logging.WARNING)
logger = logging.getLogger("score_saved_beamlets")
app = typer.Typer(help="Dose engines (pencil beam, ADoTA models) against the saved MC on beamlet directories.")

STAGES = ("pba", "adota", "score", "report")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stage_pba(blocks, cfg, engines, out_root: Path, run_provenance: dict, dry_run: bool = False) -> None:
    """Rebuild each field, verify every crop, then (unless ``dry_run``) add each pencil-beam engine's doses."""
    engines = [e for e in engines if e.kind == "pencil_beam"]
    if not engines and not dry_run:
        logger.info("pba: no pencil-beam engine in the config; nothing to add")
        return
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
    runners = []
    if not dry_run:
        from src.pencil_beam.engine import PencilBeamEngine

        for e in engines:
            pb = e.settings
            runner = PencilBeamEngine(Path(pb["dosecuda_kernels"]), Path(pb["model_dir"]),
                                      gpu_id=int(pb.get("gpu_id", 0)))
            record = {"model_fingerprint": run_provenance["pencil_beam_model_fingerprints"][e.name],
                      "model_dir": str(pb["model_dir"]), "generation_config": str(cfg["generation_config"]),
                      "layout": "(H, W, D) = (z, y, x), the _ds.npy crop of the field's canonical CT; dose per "
                                "spot weight 1000 on the MCsquare scale", **run_provenance}
            runners.append((e, runner, record))

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
                for e, runner, record in runners:
                    stats = compute_field_pba(runner, geom, bdl, rcfg.roi_size, b, record, suffix=e.suffix,
                                              provenance_file=e.provenance_file)
                    logger.info("%s %s: %d beamlets (ray trace %.1f s, spots %.1f s)", e.name, b.name,
                                stats["computed"], stats["field_seconds"], stats["spot_seconds"])
                    (out_root / b.name).mkdir(parents=True, exist_ok=True)
                    (out_root / b.name / f"{e.tag}_run.json").write_text(json.dumps(stats, indent=2, default=str))


def _stage_adota(blocks, engines, run_provenance: dict) -> None:
    """For each ADoTA engine, add its prediction only where a beamlet has none."""
    from src.adota.utils import load_model
    from src.evaluation.cli import resolve_device

    for e in (e for e in engines if e.kind == "adota"):
        todo = [b for b in blocks if any(not (b.path / f"{s}{e.suffix}").exists() for s in b.stems)]
        if not todo:
            logger.info("%s: every beamlet already has a %s; nothing to add", e.name, e.suffix)
            continue
        if not e.settings.get("model_dir"):
            raise typer.BadParameter(f"{len(todo)} blocks lack {e.suffix}; give engine {e.name!r} a model_dir")
        model_dir = Path(e.settings["model_dir"])
        ckpt = model_dir / e.settings.get("fname", "best_model.pth")
        device = resolve_device(int(e.settings.get("device_index", 0)))
        model = load_model(ckpt, model_dir / "hyperparams.json", device)
        record = {"engine": e.name, "suffix": e.suffix, "checkpoint": str(ckpt), "checkpoint_sha256": _sha256(ckpt),
                  "hyperparams_sha256": _sha256(model_dir / "hyperparams.json"), **run_provenance}
        for b in todo:
            n = write_missing_predictions(b, model, device, record, int(e.settings.get("batch_size", 56)),
                                          suffix=e.suffix, provenance_file=e.provenance_file)
            logger.info("%s %s: added %d missing predictions (%s)", e.name, b.name, n, e.suffix)


def _stage_score(blocks, cfg, engines, out_root: Path, criteria, overwrite: bool) -> List[dict]:
    workers = int(cfg.get("scoring", {}).get("workers", 32))
    keys = [c.key for c in criteria] + ["r80_delta_mm"]
    rows = []
    for b in blocks:
        path = out_root / b.name / "metrics.json"
        cached = [] if overwrite or not path.exists() else json.loads(path.read_text())
        if cached and {r["stem"] for r in cached} != set(b.stems):
            cached = []                                   # another beamlet selection: score again
        todo = engines_to_score(cached, engines, keys)
        if not todo:
            rows += cached
            continue
        for e in todo:
            missing = [s for s in b.stems if not (b.path / f"{s}{e.suffix}").exists()]
            if missing:
                stage = "pba" if e.kind == "pencil_beam" else "adota"
                raise typer.BadParameter(f"{b.name}: {len(missing)} beamlets have no {e.suffix} ({e.name}); "
                                         f"run the {stage} stage")
        tasks = [(str(b.path), s, tuple(criteria), b.name, tuple(todo)) for s in b.stems]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            scored = {r["stem"]: r for r in ex.map(score_beamlet, tasks, chunksize=2)}
        meta = {"anatomy": b.anatomy, "patient_id": b.patient_id, "energy": b.energy, "gantry": b.gantry,
                "block_dir": str(b.path)}
        by_stem = {r["stem"]: r for r in cached}
        block_rows = [dict(by_stem.get(s, {}), **scored[s], **meta) for s in b.stems]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(block_rows, indent=1, default=float))
        head = criteria[0].key
        logger.info("score %s (%s): median %s %s", b.name, ", ".join(e.name for e in todo), head, ", ".join(
            f"{e.name} {np.nanmedian([r[f'{e.tag}_{head}'] for r in block_rows]):.1f}%" for e in engines))
        rows += block_rows
    write_rows_csv(rows, out_root / "all_beamlets.csv")
    return rows


def _groups(rows, summary_groups) -> dict:
    """``all``, per anatomy, per anatomy and energy, per patient, and the config's
    ``summary_groups`` (``name: [patient ids]``)."""
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["anatomy"]].append(r)
        groups[f"{r['anatomy']}_e{r['energy']:g}"].append(r)
        groups[f"patient_{r['patient_id']}"].append(r)
        for name, patients in (summary_groups or {}).items():
            if r["patient_id"] in patients:
                groups[name].append(r)
    return groups


def _summary(rows, criteria, engines, summary_groups=None) -> dict:
    out = {}
    for name, group in sorted(_groups(rows, summary_groups).items()):
        out[name] = {"n": len(group)}
        for e in engines:
            for key in [c.key for c in criteria] + ["r80_delta_mm"]:
                vals = np.array([g[f"{e.tag}_{key}"] for g in group], dtype=float)
                vals = vals[np.isfinite(vals)]
                out[name][f"{e.tag}_{key}"] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)),
                                               "p5": float(np.percentile(vals, 5)), "min": float(vals.min())}
    return out


def _summary_table(rows, criteria, engines, summary_groups=None) -> List[dict]:
    """One row per (group, engine): gamma mean / median / 5th percentile / minimum and
    the share below 95% per criterion, |dR80| median and 95th percentile, and the
    mean absolute dose difference (% of the MC maximum)."""
    table = []
    for name, group in sorted(_groups(rows, summary_groups).items()):
        for e in engines:
            row = {"group": name, "engine": e.name, "n": len(group)}
            for c in criteria:
                v = np.array([g[f"{e.tag}_{c.key}"] for g in group], dtype=float)
                v = v[np.isfinite(v)]
                row.update({f"{c.key}_mean": v.mean(), f"{c.key}_median": np.median(v),
                            f"{c.key}_p5": np.percentile(v, 5), f"{c.key}_min": v.min(),
                            f"{c.key}_below95_pct": 100.0 * float(np.mean(v < 95.0))})
            dr = np.abs(np.array([g[f"{e.tag}_r80_delta_mm"] for g in group], dtype=float))
            dr = dr[np.isfinite(dr)]
            mad = np.array([g[f"{e.tag}_mean_abs_diff_pct"] for g in group], dtype=float)
            row.update({"abs_r80_delta_mm_median": np.median(dr), "abs_r80_delta_mm_p95": np.percentile(dr, 95),
                        "mean_abs_diff_pct_mean": float(np.nanmean(mad))})
            table.append(row)
    return table


def _stage_report(rows, cfg, engines, out_root: Path, criteria) -> None:
    from src.figures.pencil_beam import engine_comparison_figure

    rep = cfg.get("report", {})
    n = int(rep.get("n", 10))
    head = criteria[0]
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    from src.utils.unit_conversions import mcsquare_to_gy_per_mu

    gen_engine = load_yaml_config(Path(cfg["generation_config"]))["engine"]
    bdl_path = Path(gen_engine["mcsquare_install"]) / "BDL" / gen_engine.get("bdl_file", "hptc_beam_model_rsnone.txt")
    ranking = []
    for engine in engines:
        best, worst = rank_beamlets(rows, engine.name, criteria, n)
        for kind, chosen in (("best", best), ("worst", worst)):
            for rank, r in enumerate(chosen, start=1):
                ranking.append(dict(r, ranked_engine=engine.name, ranked_as=kind, rank=rank))
                if not rep.get("figures", True):
                    continue
                d = load_doses(Path(r["block_dir"]), r["stem"], engines)
                text = {e.name: f"{head.key} = {r[f'{e.tag}_{head.key}']:.1f}%  "
                                f"dR80 = {r[f'{e.tag}_r80_delta_mm']:+.2f} mm" for e in engines}
                engine_comparison_figure(
                    d["CT"], {"MC": d["MC"], **{e.name: d[e.name] for e in engines}},
                    str(fig_dir / f"{engine.tag}_{kind}{rank:02d}_{r['block']}_{r['stem']}_{head.key}_global"),
                    title=(f"{engine.name} {kind} #{rank}: {r['anatomy']} {r['patient_id']}  E={r['energy']:g} MeV  "
                           f"gantry={r['gantry']:.1f}  {r['stem']}  gamma {head.dose:g}%/{head.dist:g}mm, "
                           f"{head.cutoff:g}% cutoff (global)"),
                    metrics_text=text, dose_unit="Gy per MU",
                    dose_scale=float(mcsquare_to_gy_per_mu(1.0, float(r["energy"]), bdl_path)))
    write_rows_csv(ranking, out_root / f"ranking_best_worst_{n}_{head.key}_global.csv")
    groups = cfg.get("summary_groups")
    write_rows_csv(_summary_table(rows, criteria, engines, groups), out_root / "summary_table.csv")
    (out_root / "summary.json").write_text(json.dumps(
        {"criteria": [c.key for c in criteria], "gamma": "global, normalised to the MC maximum",
         "engines": [{"name": e.name, "kind": e.kind, "suffix": e.suffix, "column_prefix": e.tag} for e in engines],
         "n_beamlets": len(rows), "groups": _summary(rows, criteria, engines, groups)}, indent=2))
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
    engines = engines_from_config(cfg)
    logger.info("engines: %s", ", ".join(f"{e.name} ({e.kind}, {e.suffix})" for e in engines))
    for b in blocks:
        missing = {e.name: sum(not (b.path / f"{s}{e.suffix}").exists() for s in b.stems) for e in engines}
        logger.info("selected %s: %d beamlets, %s %s, E=%g MeV, gantry %.2f%s", b.name, len(b.stems), b.anatomy,
                    b.patient_id, b.energy, b.gantry,
                    "".join(f", {k}: {v} to add" for k, v in missing.items() if v))
    if limit_blocks:
        blocks = blocks[:limit_blocks]
    if limit_beamlets:
        from dataclasses import replace
        blocks = [replace(b, stems=b.stems[:limit_beamlets]) for b in blocks]
    logger.info("%d blocks, %d beamlets selected", len(blocks), sum(len(b.stems) for b in blocks))
    criteria = criteria_from_config(cfg.get("scoring", {}).get("gamma_criteria"))
    gen_engine = load_yaml_config(Path(cfg["generation_config"]))["engine"]
    pencil = [e for e in engines if e.kind == "pencil_beam"]
    kernels = Path(pencil[0].settings["dosecuda_kernels"]) if pencil and pencil[0].settings.get("dosecuda_kernels") \
        else None
    prov = provenance(Path(gen_engine["mcsquare_install"]), gen_engine.get("bdl_file", "hptc_beam_model_rsnone.txt"),
                      gen_engine.get("scanner", "default"), kernels)
    prov["pencil_beam_model_fingerprints"] = {e.name: model_fingerprint(Path(e.settings["model_dir"]))
                                              for e in pencil if e.settings.get("model_dir")}
    prov["created"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    if prov["adota_uncommitted_changes"]:
        logger.warning("the adota checkout has uncommitted changes: results will not be traceable to a commit")
    if dry_run:
        _stage_pba(blocks, cfg, engines, out_root, prov, dry_run=True)
        logger.info("dry-run: %d blocks selected and verified; nothing written", len(blocks))
        raise typer.Exit()

    out_root.mkdir(parents=True, exist_ok=True)
    invocation = {"config": cfg, "stages": wanted, "blocks": [str(b.path) for b in blocks],
                  "engines": [{"name": e.name, "kind": e.kind, "suffix": e.suffix} for e in engines],
                  "excluded": {str(p): why for p, why in excluded}, "provenance": prov}
    # run_config.json describes the run that created output_root and is never replaced; every
    # invocation (a later report or rescore) is appended to invocations.jsonl instead
    if not (out_root / "run_config.json").exists():
        (out_root / "run_config.json").write_text(json.dumps(invocation, indent=2, default=str))
    with open(out_root / "invocations.jsonl", "a") as fh:
        fh.write(json.dumps(invocation, default=str) + "\n")

    if "pba" in wanted:
        _stage_pba(blocks, cfg, engines, out_root, prov)
    if "adota" in wanted:
        _stage_adota(blocks, engines, prov)
    rows = None
    if "score" in wanted:
        rows = _stage_score(blocks, cfg, engines, out_root, criteria, rescore)
    if "report" in wanted:
        if rows is None:
            rows = [r for b in blocks for r in json.loads((out_root / b.name / "metrics.json").read_text())]
        _stage_report(rows, cfg, engines, out_root, criteria)


if __name__ == "__main__":
    app()
