"""Does an input-only difficulty score predict the error of dose engines on saved beamlet grids?

Stages (``--stages``, comma-separated, run in this order):

* ``adota``      -- add each ADoTA engine's prediction where a beamlet has none.
* ``score``      -- gamma of each engine against the saved MC (cached per block; an optional
                    ``seed_metrics_from`` supplies what is already computed) and its MAPE.
* ``difficulty`` -- the score from each beamlet's CT, flux and energy, and the example beamlets,
                    chosen from the scores alone before any error is read.
* ``analyse``    -- within-field and pooled Spearman, per-decile tables, per-engine summary.
* ``figure``     -- the story figure and the trend figure, with the numbers beside them.

Files are only ever added to a source directory, each created exclusively. Everything else
goes to ``output_root``. Guide: scripts/docs/difficulty_engine_grids.md.

  TMPDIR=/scratch/mstryja/tmp uv run python scripts/active_learning/difficulty_engine_grids.py \\
      --config scripts/active_learning/config_difficulty_engine_grids.yaml
"""
from __future__ import annotations

import os

# Pin BLAS to 1 thread so the gamma and feature process pools scale across cores (before numpy loads).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import datetime
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from glob import glob
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import typer

from src.acquisition.scorer import DifficultyScorer
from src.active_learning.ensemble.engine_grids import (
    add_within_field_decile,
    beamlet_features,
    beamlet_mape,
    choose_examples,
    decile_table,
    metric_columns,
    per_engine_summary,
    reference_percentile,
    spearman_summary,
    within_field_spearman,
    write_json,
)
from src.active_learning.ensemble.independence import inside_crop_features
from src.evaluation.cli import SET_OVERRIDE_HELP, apply_set_overrides, load_yaml_config
from src.figures.difficulty_engine_grids import StoryCase, difficulty_story_figure, difficulty_trend_figure
from src.pencil_beam.metrics import criteria_from_config
from src.pencil_beam.pipeline import provenance
from src.pencil_beam.saved_beamlets import Engine, engines_from_config, load_doses, survey_dirs
from src.pencil_beam.saved_stages import add_adota_predictions, score_blocks
from src.utils.unit_conversions import mcsquare_to_gy_per_mu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("pymedphys").setLevel(logging.WARNING)
logger = logging.getLogger("difficulty_engine_grids")
app = typer.Typer(help="An input-only difficulty score against dose engines on saved beamlet grids.")

STAGES = ("adota", "score", "difficulty", "analyse", "figure")


def _label(engine: Engine) -> str:
    return str(engine.settings.get("label", engine.name))


def _metrics(criteria) -> Dict[str, str]:
    """``{metric key: column suffix}``: a pass rate per gamma criterion, MAPE on both grids."""
    keys = {f"gpr_{c.dose:g}pct_{c.dist:g}mm_{c.cutoff:g}pct": c.key for c in criteria}
    return {**keys, "mape_1mm": "mape_1mm", "mape_2mm": "mape_2mm"}


def _seed_metrics(blocks, seed_root: Optional[str], out_root: Path) -> None:
    """Copy each block's existing ``metrics.json`` from ``seed_root``, never over a file."""
    if not seed_root:
        return
    n = 0
    for b in blocks:
        source, target = Path(seed_root) / b.name / "metrics.json", out_root / b.name / "metrics.json"
        if source.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            n += 1
    logger.info("seeded %d of %d blocks' metrics from %s", n, len(blocks), seed_root)


def _stage_mape(blocks, engines: Sequence[Engine], out_root: Path, cutoff: float, workers: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    columns = {f"{e.tag}_mape_{g}" for e in engines for g in ("1mm", "2mm")}
    for b in blocks:
        path = out_root / b.name / "mape.csv"
        cached = pd.read_csv(path) if path.exists() else None
        if cached is not None and set(cached["stem"]) == set(b.stems) and columns <= set(cached.columns):
            frames.append(cached)
            continue
        tasks = [(str(b.path), s, b.name, tuple(engines), cutoff) for s in b.stems]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            frame = pd.DataFrame(list(ex.map(beamlet_mape, tasks, chunksize=8)))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _stage_difficulty(blocks, cfg, out_root: Path) -> pd.DataFrame:
    """Features and score from the input alone, the reference percentile, then the examples."""
    block_cfg = cfg["difficulty"]
    path = out_root / "features_analytic.csv"
    wanted = {(b.name, s) for b in blocks for s in b.stems}
    features = pd.read_csv(path) if path.exists() else None
    if features is None or set(zip(features["block"], features["stem"])) != wanted:
        tasks = [(str(b.path), s, b.name, float(b.energy)) for b in blocks for s in b.stems]
        with ProcessPoolExecutor(max_workers=int(block_cfg.get("workers", 32))) as ex:
            features = pd.DataFrame(list(ex.map(beamlet_features, tasks, chunksize=8)))
        features.to_csv(path, index=False)
    scorer = DifficultyScorer.load(block_cfg["scorer"], variant=block_cfg.get("variant", "full (ridge)"))
    inside = features["peak_inside_crop"].astype(bool)
    frame = features[["block", "stem", "peak_inside_crop"]].copy()
    frame["d_score_raw"] = np.nan
    frame.loc[inside, "d_score_raw"] = scorer.score(features[inside])
    reference = scorer.score(inside_crop_features(block_cfg["percentile_reference_features"]))
    # The raw score is a linear fit to a standardised error, so it is unbounded. d is its rank among
    # the reference beamlets, in [0, 1]: the order is unchanged and it does not depend on this set.
    frame["d_score"] = reference_percentile(frame["d_score_raw"].to_numpy(), reference) / 100.0
    frame.loc[~inside, "d_score"] = np.nan
    frame.to_csv(out_root / "difficulty.csv", index=False)
    write_json(out_root / "difficulty_reference.json", {"n_reference": len(reference), "scorer": scorer.source,
                                                        "features": block_cfg["percentile_reference_features"]})
    choose_examples(frame, dict(cfg["figure"]["examples"])).to_csv(out_root / "examples.csv", index=False)
    logger.info("difficulty: %d of %d beamlets have their peak inside the crop; d median %.3f", int(inside.sum()),
                len(frame), frame["d_score"].median())
    return frame


def _per_beamlet(out_root: Path, blocks) -> pd.DataFrame:
    rows = [r for b in blocks for r in json.loads((out_root / b.name / "metrics.json").read_text())]
    mape = pd.concat([pd.read_csv(out_root / b.name / "mape.csv") for b in blocks], ignore_index=True)
    difficulty = pd.read_csv(out_root / "difficulty.csv")
    return pd.DataFrame(rows).merge(mape, on=["block", "stem"], validate="one_to_one").merge(
        difficulty, on=["block", "stem"], validate="one_to_one")


def _stage_analyse(blocks, cfg, engines, criteria, out_root: Path) -> None:
    analysis = cfg.get("analysis", {})
    out = out_root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    frame = _per_beamlet(out_root, blocks)
    usable = add_within_field_decile(frame.dropna(subset=["d_score"]))
    columns = metric_columns(engines, _metrics(criteria))
    per_field = within_field_spearman(usable, columns)
    keep = ["block", "stem", "patient_id", "anatomy", "energy", "gantry", "grid_ix", "grid_iy", "peak_inside_crop",
            "d_score", "d_score_raw"]
    frame = frame.merge(usable[["block", "stem", "field_decile"]], on=["block", "stem"], how="left")
    frame[keep + ["field_decile"] + sorted(set(columns.values()))].to_csv(out / "per_beamlet.csv", index=False)
    per_field.to_csv(out / "spearman_per_field.csv", index=False)
    summary = spearman_summary(usable, per_field, columns, int(analysis.get("n_bootstrap", 2000)),
                               int(analysis.get("seed", 0)))
    summary.to_csv(out / "spearman_summary.csv", index=False)
    decile_table(usable, columns).to_csv(out / "deciles.csv", index=False)
    per_engine_summary(usable, columns).to_csv(out / "per_engine.csv", index=False)
    write_json(out / "population.json", {
        "n_beamlets": len(frame), "n_with_score": len(usable), "n_fields": int(frame["block"].nunique()),
        "patients": sorted(frame["patient_id"].unique()),
        "energies": sorted(float(e) for e in frame["energy"].unique()),
        "reference": "saved MC at 1e6 primaries, 1 mm crop", "gamma": "global, normalised to the MC maximum"})
    logger.info("analysis: %d beamlets with a score, tables in %s", len(usable), out)


def _stage_figure(blocks, cfg, engines, criteria, out_root: Path) -> None:
    analysis, figures = out_root / "analysis", out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    head = criteria[0]
    gpr_metric = f"gpr_{head.dose:g}pct_{head.dist:g}mm_{head.cutoff:g}pct"
    criterion_text = f"{head.dose:g}%/{head.dist:g}mm, {head.cutoff:g}% cutoff"
    frame = pd.read_csv(analysis / "per_beamlet.csv")
    examples = pd.read_csv(out_root / "examples.csv")
    labels = {e.name: _label(e) for e in engines}
    gen_engine = load_yaml_config(Path(cfg["generation_config"]))["engine"]
    bdl = Path(gen_engine["mcsquare_install"]) / "BDL" / gen_engine.get("bdl_file", "hptc_beam_model_rsnone.txt")
    block_dirs = {b.name: b.path for b in blocks}
    cases, numbers = [], []
    for ex in examples.itertuples(index=False):
        row = frame[(frame["block"] == ex.block) & (frame["stem"] == ex.stem)].iloc[0]
        doses = load_doses(block_dirs[ex.block], ex.stem, engines)
        scale = float(mcsquare_to_gy_per_mu(1.0, float(row["energy"]), bdl))
        notes = {}
        for e in engines:
            gpr, mape = float(row[f"{e.tag}_{head.key}"]), float(row[f"{e.tag}_mape_1mm"])
            notes[labels[e.name]] = f"GPR {gpr:.1f}%  MAPE {mape:.1f}%"
            numbers.append({"case": ex.case, "block": ex.block, "stem": ex.stem, "d_score": row["d_score"],
                            "d_score_raw": row["d_score_raw"], "energy_mev": row["energy"],
                            "anatomy": row["anatomy"], "patient_id": row["patient_id"], "engine": labels[e.name],
                            gpr_metric: gpr, "mape_1mm": mape, "mape_2mm": float(row[f"{e.tag}_mape_2mm"])})
        cases.append(StoryCase(
            title=f"{ex.case} difficulty",
            subtitle=f"d = {row['d_score']:.2f}\n{row['anatomy'].capitalize()}, E = {row['energy']:g} MeV",
            ct=doses["CT"], flux=np.load(block_dirs[ex.block] / f"{ex.stem}_flux.npy"),
            reference=doses["MC"] * scale, doses={labels[e.name]: doses[e.name] * scale for e in engines},
            notes=notes))
    pd.DataFrame(numbers).to_csv(figures / "story_numbers.csv", index=False)
    trend = pd.read_csv(analysis / "deciles.csv")
    trend["engine"] = trend["engine"].map(labels)
    order = [labels[e.name] for e in engines]
    trend.to_csv(figures / "trend_numbers.csv", index=False)
    population = json.loads((analysis / "population.json").read_text())
    n_patients = len(population["patients"])
    pop_text = (f"{population['n_with_score']:,} beamlets, {population['n_fields']} complete fields, "
                f"{n_patients} patients no neural engine trained on; reference: MC at 1e6 primaries on the 1 mm crop.")
    n_reference = json.loads((out_root / "difficulty_reference.json").read_text())["n_reference"]
    formats = tuple(cfg["figure"].get("formats", ("png",)))
    stem = f"gamma{head.dose:g}pct_{head.dist:g}mm_cutoff{head.cutoff:g}pct"
    difficulty_story_figure(
        cases, order, figure_path=str(figures / f"difficulty_story_{stem}"), formats=formats,
        title="CT + flux + energy give an estimated difficulty d: what each dose engine then does with that beamlet",
        caption=(f"d: input-only difficulty score, no dose engine run; 0 to 1, the share of {n_reference:,} unseen "
                 f"pool beamlets that score lower. Examples are the beamlets nearest the 10th, 50th and 90th score "
                 f"percentiles of this set, chosen before any error was read. GPR: gamma pass rate, {criterion_text}, "
                 f"global, against MC. MAPE: voxels above 10% of the MC maximum. White contour: MC 50% isodose; the "
                 f"dose scale tops at the MC maximum. Flux: fraction of its maximum in the crop. Strips are stretched "
                 f"laterally to one height. {pop_text}"))
    trend_caption = f"Median per score decile, within each field. {pop_text}"
    difficulty_trend_figure(
        trend, gpr_metric, order, figure_path=str(figures / f"difficulty_vs_gpr_{stem}"), formats=formats,
        ylabel=f"Median gamma pass rate, {criterion_text} [%]", title="Gamma pass rate against estimated difficulty",
        caption=f"Gamma: global, against MC. {trend_caption}")
    difficulty_trend_figure(
        trend, "mape_1mm", order, figure_path=str(figures / "difficulty_vs_mape_cutoff10pct"), formats=formats,
        ylabel="Median MAPE, voxels above 10% of the MC maximum [%]", title="MAPE against estimated difficulty",
        caption=trend_caption)
    logger.info("figures in %s", figures)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config file.")] = Path(
        "scripts/active_learning/config_difficulty_engine_grids.yaml"),
    stages: Annotated[str, typer.Option(help=f"Comma-separated subset of {','.join(STAGES)}.")] = ",".join(STAGES),
    set_: Annotated[Optional[List[str]], typer.Option("--set", help=SET_OVERRIDE_HELP)] = None,
    limit_blocks: Annotated[Optional[int], typer.Option(help="Only the first N selected blocks (smoke).")] = None,
    limit_beamlets: Annotated[Optional[int], typer.Option(help="Only the first N beamlets per block (smoke).")] = None,
    rescore: Annotated[bool, typer.Option(help="Recompute gamma scores that exist.")] = False,
    dry_run: Annotated[bool, typer.Option(help="List the selection and what is to add; write nothing.")] = False,
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
    for b in blocks:
        missing = {e.name: sum(not (b.path / f"{s}{e.suffix}").exists() for s in b.stems) for e in engines}
        logger.info("selected %s: %d beamlets, E=%g MeV, gantry %.2f%s", b.name, len(b.stems), b.energy, b.gantry,
                    "".join(f", {k}: {v} to add" for k, v in missing.items() if v))
    logger.info("%d blocks, %d beamlets, %d patients", len(blocks), sum(len(b.stems) for b in blocks),
                len({b.patient_id for b in blocks}))
    if dry_run:
        raise typer.Exit()
    if limit_blocks:
        blocks = blocks[:limit_blocks]
    if limit_beamlets:
        blocks = [replace(b, stems=b.stems[:limit_beamlets]) for b in blocks]
    criteria = criteria_from_config(cfg.get("scoring", {}).get("gamma_criteria"))
    gen_engine = load_yaml_config(Path(cfg["generation_config"]))["engine"]
    prov = provenance(Path(gen_engine["mcsquare_install"]), gen_engine.get("bdl_file", "hptc_beam_model_rsnone.txt"),
                      gen_engine.get("scanner", "default"), None)
    prov["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    if prov["adota_uncommitted_changes"]:
        logger.warning("the adota checkout has uncommitted changes: results will not be traceable to a commit")
    out_root.mkdir(parents=True, exist_ok=True)
    invocation = {"config": cfg, "stages": wanted, "blocks": [str(b.path) for b in blocks],
                  "excluded": {str(p): why for p, why in excluded}, "provenance": prov}
    with open(out_root / "invocations.jsonl", "a") as fh:
        fh.write(json.dumps(invocation, default=str) + "\n")
    scoring = cfg.get("scoring", {})
    workers = int(scoring.get("workers", 32))
    if "adota" in wanted:
        add_adota_predictions(blocks, engines, prov)
    if "score" in wanted:
        _seed_metrics(blocks, cfg.get("seed_metrics_from"), out_root)
        score_blocks(blocks, engines, out_root, criteria, rescore, workers=workers)
        _stage_mape(blocks, engines, out_root, float(scoring.get("mape_cutoff_fraction", 0.1)), workers)
    if "difficulty" in wanted:
        _stage_difficulty(blocks, cfg, out_root)
    if "analyse" in wanted:
        _stage_analyse(blocks, cfg, engines, criteria, out_root)
    if "figure" in wanted:
        _stage_figure(blocks, cfg, engines, criteria, out_root)


if __name__ == "__main__":
    app()
