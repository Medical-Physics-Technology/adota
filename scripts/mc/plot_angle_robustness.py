"""Beamlet-angle robustness GPR grid panels (reviewer R3), YAML-driven.

Per generated experiment dir: ADoTA inference + gamma vs MC -> an 18x18 GPR grid
over (Beamlet angle X, Y). Panels are saved separately (PNG/PDF/SVG), share a
color scale per gamma criterion across all comparable panels, carry no titles,
and are A4-legible. ``mode: aggregate`` averages a site's patients per cell.

  uv run python scripts/mc/plot_angle_robustness.py --config scripts/mc/config_plot_angle_robustness.yaml
"""
from __future__ import annotations

import os

# Pin BLAS to 1 thread so the per-beamlet gamma process pool scales across cores
# (must be set before numpy/torch import).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

from src.adota.config import DEFAULT_SCALE, load_yaml_config
from src.adota.utils import load_model
from src.evaluation.cli import resolve_device
from src.figures.angle_robustness_grid import angle_robustness_panel, standalone_colorbar
from src.mc_generation.angle_robustness_analysis import (
    GammaCriterion,
    Panel,
    aggregate_panels,
    infer_dir,
    load_panel_grids,
    panel_name,
    render_beamlet_examples,
    save_panel_grids,
    score_dir_grids,
    shared_scale_per_criterion,
)

ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("robustness_plot")
app = typer.Typer(help="Beamlet-angle robustness GPR grid panels.")


def _panel_stem(p: Panel, crit: GammaCriterion) -> str:
    return f"{panel_name(p)}_{crit.key}"


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config.")] = Path(
        "scripts/mc/config_plot_angle_robustness.yaml"),
    mode: Annotated[Optional[str], typer.Option(help="per_patient | aggregate.")] = None,
    grid_n: Annotated[Optional[int], typer.Option()] = None,
    no_inference: Annotated[bool, typer.Option(help="Reuse existing *_ds_pred.npy.")] = False,
) -> None:
    cfg = load_yaml_config(config)

    criteria = [GammaCriterion(float(c["dose"]), float(c["dist"]), float(c["cutoff"]))
                for c in cfg["gamma_criteria"]]
    gn = int(grid_n if grid_n is not None else cfg.get("grid_n", 18))
    lo, hi = cfg.get("theta_range", [-2.0, 2.0])
    thetas = np.linspace(float(lo), float(hi), gn)
    run_mode = mode or cfg.get("mode", "per_patient")
    cmap = cfg.get("cmap", "viridis")
    emit = cfg.get("emit", {"bare": True, "with_colorbar": True, "standalone_colorbar": True})
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ``grids_glob`` re-renders from grids a previous run saved: the per-cell GPRs
    # are the whole result of inference + gamma, so a different scale, criterion
    # subset or `mode` costs no GPU and no gamma time.
    grids_glob = cfg.get("grids_glob")
    inputs: List[str] = list(cfg.get("inputs", []))
    if cfg.get("input_glob"):
        inputs += sorted(glob(cfg["input_glob"]))
    if not inputs and not grids_glob:
        raise ValueError("config provides no 'inputs', 'input_glob' or 'grids_glob'")
    logger.info("mode=%s | %s | criteria=%s", run_mode,
                f"{len(inputs)} experiment dirs" if not grids_glob else f"grids {grids_glob}",
                [c.key for c in criteria])

    # Stage 1a: inference (GPU) for all dirs, then release the model + CUDA so the
    # gamma process pool forks a CUDA-idle parent.
    if not no_inference and not grids_glob:
        import torch
        m = cfg["model"]
        device = resolve_device(int(m.get("device_index", 0)))
        model = load_model(ROOT / "models" / m["name"] / m.get("fname", "best_model.pth"),
                           ROOT / "models" / m["name"] / "hyperparams.json", device)
        for d in inputs:
            logger.info("inference %s", Path(d).name)
            try:
                infer_dir(Path(d), model, device)
            except FileNotFoundError:
                # every beamlet of this (patient, energy, gantry) block was dropped
                # by generation QA -- nothing to infer, and nothing to score later.
                logger.warning("  no complete beamlets in %s -- skipped", Path(d).name)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Organized run-directory layout (categories, then per-anatomy subfolders).
    CBAR_DIR = out_dir / "gamma_pass_rates_with_colorbars"
    BARE_DIR = out_dir / "gamma_pass_rates_without_colorbars"
    EX_DIR = out_dir / "worst_cases"
    GRIDS_DIR = out_dir / "grids"
    for _d in (CBAR_DIR, BARE_DIR, EX_DIR, GRIDS_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    def _sub(base: Path, anatomy: str) -> Path:
        p = base / anatomy
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Stage 1b: parallel gamma scoring (CPU pool) for all dirs.
    # One panel per input dir. ``distinguish_gantry`` puts the field angle in every
    # panel filename -- required when a run holds several gantries per (patient,
    # energy), and off by default so single-gantry runs keep their historical names.
    distinguish_gantry = bool(cfg.get("distinguish_gantry", False))
    scored: List[tuple] = []            # (dir, panel), skipping dirs QA emptied
    if grids_glob:
        paths = sorted(glob(grids_glob))
        if not paths:
            raise FileNotFoundError(f"grids_glob matched nothing: {grids_glob}")
        dir_panels = [load_panel_grids(Path(g)) for g in paths]
        logger.info("loaded %d saved grids (no inference, no gamma)", len(dir_panels))
    else:
        for d in inputs:
            logger.info("scoring %s", Path(d).name)
            try:
                panel = score_dir_grids(Path(d), criteria, gn)
            except FileNotFoundError:
                logger.warning("  no scorable beamlets in %s -- skipped", Path(d).name)
                continue
            scored.append((d, panel))
        if not scored:
            raise FileNotFoundError(f"no scorable beamlets in any of the {len(inputs)} input dirs")
        dir_panels = [p for _, p in scored]
    if not distinguish_gantry:
        for p in dir_panels:
            p.gantry = None
    names = [panel_name(p) for p in dir_panels]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"panel name collision for {dupes}: several input dirs share "
            "(site, patient, energy). Set 'distinguish_gantry: true' if they differ "
            "by field angle, otherwise narrow 'input_glob'.")
    if not grids_glob:                  # don't rewrite the grids we just loaded
        for panel in dir_panels:
            save_panel_grids(panel, GRIDS_DIR)  # per-cell grids (cheap re-render later)

    # Stage 1c: best/worst beamlet publication figures (reload the model; GPU).
    if cfg.get("render_examples", True) and not grids_glob:
        import torch
        n_ex = int(cfg.get("n_examples", 3))
        ex_key = cfg.get("example_criterion", criteria[0].key)
        crit = next((c for c in criteria if c.key == ex_key), criteria[0])
        mm = cfg["model"]
        device2 = resolve_device(int(mm.get("device_index", 0)))
        model2 = load_model(ROOT / "models" / mm["name"] / mm.get("fname", "best_model.pth"),
                            ROOT / "models" / mm["name"] / "hyperparams.json", device2)
        for d, panel in scored:
            logger.info("examples (%d worst + %d best) for %s", n_ex, n_ex, Path(d).name)
            render_beamlet_examples(Path(d), panel.grids[crit.key], crit, model2, device2,
                                    dict(DEFAULT_SCALE), _sub(EX_DIR, panel.anatomy), n=n_ex,
                                    beamlet_shape=bool(cfg.get("example_beamlet_shape", True)),
                                    gantry=panel.gantry)
        del model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stage 2: choose panels (per-patient, or per-site+energy aggregate).
    if run_mode == "aggregate":
        groups = defaultdict(list)
        for p in dir_panels:
            groups[(p.anatomy, round(p.energy))].append(p)
        panels = [aggregate_panels(ps) for ps in groups.values()]
        logger.info("aggregated -> %d panels (per anatomy+energy)", len(panels))
    else:
        panels = dir_panels

    # Stage 3: shared color scale per criterion across ALL comparable panels.
    # A robust low percentile keeps the scale stable (a few low-outlier beamlets
    # don't wash out the high-GPR bulk); vmax capped at 100.
    low_pct = float(cfg.get("scale_low_percentile", 0.0))
    scales = shared_scale_per_criterion(panels, criteria, low_percentile=low_pct, vmax_cap=100.0)
    logger.info("shared scales (low_pct=%.1f): %s", low_pct,
                {k: (round(v[0], 2), round(v[1], 2)) for k, v in scales.items()})

    # Stage 4: render into the categorized layout.
    manifest = []
    for crit in criteria:
        vmin, vmax = scales[crit.key]
        if emit.get("standalone_colorbar", True):
            # shared colorbar lives once in the 'without_colorbars' category
            standalone_colorbar(str(BARE_DIR / f"colorbar_{crit.key}"), vmin, vmax,
                                cmap=cmap, label=crit.cbar_label)
        for p in panels:
            stem = _panel_stem(p, crit)
            if emit.get("bare", True):
                angle_robustness_panel(p.grids[crit.key], thetas, thetas,
                                       str(_sub(BARE_DIR, p.anatomy) / stem), vmin, vmax,
                                       cmap=cmap, with_colorbar=False)
            if emit.get("with_colorbar", True):
                angle_robustness_panel(p.grids[crit.key], thetas, thetas,
                                       str(_sub(CBAR_DIR, p.anatomy) / f"{stem}_cbar"), vmin, vmax,
                                       cmap=cmap, with_colorbar=True, cbar_label=crit.cbar_label)
            manifest.append({"panel": stem, "anatomy": p.anatomy, "energy": p.energy,
                             "patient": p.patient, "n_patients": p.n_patients,
                             "criterion": crit.key, "vmin": vmin, "vmax": vmax,
                             "mean_gpr": float(np.nanmean(p.grids[crit.key]))})
    (out_dir / "panels_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Wrote %d panel entries -> %s\n  layout: gamma_pass_rates_with_colorbars/<site>/, "
        "gamma_pass_rates_without_colorbars/<site>/ (+ colorbar_*), worst_cases/<site>/, grids/",
        len(manifest), out_dir,
    )


if __name__ == "__main__":
    app()
