"""Statistics and figures of the frozen validation set, before its Monte Carlo is run.

    uv run python scripts/al_validation_set_stats.py --config scripts/config_al.yaml

Reads what ``scripts/al_build_validation_set.py`` wrote under
``validation_set.tables_dir`` (the scored candidates and the selection with its
thirty metrics) and the pool CSV, and writes into a timestamped directory under
``validation_stats.output_dir``:

- ``ct_table.csv``: per CT (validation and, with ``include_pool_cts``, pool): grid,
  scanner, patient volume, how much the largest-component rule removed (couch),
  HU percentiles and tissue fractions inside the patient;
- ``beamlet_table.csv``: per selected beamlet: share of the crop inside the
  patient, entrance depth, and HU summaries of the crop and of the beam path;
- ``hu_histograms.csv`` (long, per level, CT and energy) and
  ``hu_summary_<level>.csv`` per anatomy (and energy);
- ``composition_*.csv``: counts by anatomy, energy, CT, field and steering, and
  the candidates' validity per energy; ``metric_summary.csv``;
- ``figures/``: HU histograms, tissue fractions, composition, metric
  distributions, each with its numbers beside it;
- ``summary.json``, ``manifest.json``, ``config.yaml``.

"Inside the patient" is the largest connected component of ``body_mask``: lung is
in, external air and a couch separated by air are out. Any config key can be
overridden with ``--set KEY=VALUE``.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
import typer

from src.acquisition.features import FEATURE_NAMES
from src.active_learning.config import mc_from_config
from src.active_learning.pool import read_pool
from src.active_learning.validation_stats import (
    composition_tables,
    ct_statistics,
    hu_edges,
    metric_summary,
    selection_beamlet_statistics,
    summarise_histograms,
)
from src.adota.config import setup_logging, setup_run_directory
from src.evaluation.cli import SET_OVERRIDE_HELP, apply_set_overrides, load_yaml_config, on_root_filesystem
from src.figures.validation_set import (
    composition_figure,
    hu_histograms_figure,
    metric_distributions_figure,
    tissue_fractions_figure,
)
from src.training.logging_utils import silence_pymedphys
from src.training.run_dir import save_resolved_config, write_manifest

app = typer.Typer(help="Statistics and figures of the frozen validation set.", add_completion=False)
logger = logging.getLogger("al_validation_set_stats")

DEFAULTS: Dict[str, Any] = {"output_dir": "/scratch/mstryja/adota_runs/al_validation_set", "hu_bin_width": 10.0,
                            "hu_plot_max": 2000.0, "include_pool_cts": True, "n_workers": 10}
FIGURE_METRICS = {
    "wepl_mean": "water-equivalent depth of the peak [mm]", "bp_range_max_mm": "distal edge of the peak zone [mm]",
    "pflugfelder_hi": "lateral path heterogeneity", "total_hu_change": "HU change in the peak zone [HU]",
    "sobel_th_edge_energy": "CT edge energy near the peak", "isi_sum": "interface severity (sum)",
    "ct_max_hu": "densest voxel in the crop [HU]", "body_fraction_of_crop": "crop share inside the patient",
    "entrance_depth_mm": "patient entrance depth on the axis [mm]", "path_mean_hu": "beam-path mean HU",
    "path_frac_lung": "beam-path lung fraction", "path_frac_cortical_bone": "beam-path cortical-bone fraction",
}


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Active-learning YAML config.")] = Path("scripts/config_al.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Parent of the run directory.")] = None,
    n_workers: Annotated[Optional[int], typer.Option(help="Worker processes, one CT each.")] = None,
    set_: Annotated[Optional[List[str]], typer.Option("--set", help=SET_OVERRIDE_HELP)] = None,
) -> None:
    try:
        cfg = apply_set_overrides(load_yaml_config(config), set_ or [])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    stats = {**DEFAULTS, **(cfg.get("validation_stats") or {})}
    if output_dir is not None:
        stats["output_dir"] = str(output_dir)
    if n_workers is not None:
        stats["n_workers"] = int(n_workers)
    val = cfg.get("validation_set", {})
    manifest = Path(val.get("manifest", "registry/al_validation_set.csv"))
    tables_dir = Path(val.get("tables_dir") or manifest.parent)
    full_path, cand_path = tables_dir / f"{manifest.stem}_full.csv", tables_dir / f"{manifest.stem}_candidates.csv"
    for what, path in (("output_dir", stats["output_dir"]), ("TMPDIR", tempfile.gettempdir())):
        if on_root_filesystem(Path(path)):
            raise typer.BadParameter(f"{what} {path} is on the root filesystem; use /scratch")
    if not full_path.exists():
        raise typer.BadParameter(f"{full_path} not found: run al_build_validation_set.py (--dry-run is enough)")

    out = setup_run_directory(Path(stats["output_dir"]), prefix="stats_")
    (out / "figures").mkdir(exist_ok=True)
    setup_logging(out, log_filename="al_validation_set_stats.log")
    silence_pymedphys()
    save_resolved_config(cfg, out / "config.yaml")

    selected = pd.read_csv(full_path)
    frozen = pd.read_csv(manifest) if manifest.exists() else None
    if frozen is not None and set(frozen["candidate_id"]) != set(selected["candidate_id"]):
        raise RuntimeError(f"{full_path} and the frozen manifest {manifest} hold different beamlets")
    candidates = pd.read_csv(cand_path) if cand_path.exists() else None
    pool_csv = Path(cfg["pool"]["pool_csv"])
    entries = read_pool(pool_csv)
    validation = [e for e in entries if e.role == "validation"]
    rob_cfg, _, _, bdl_path = mc_from_config(cfg)
    edges = hu_edges(float(stats["hu_bin_width"]))
    logger.info("%d selected beamlets on %d validation CTs", len(selected), len(validation))

    cts = entries if stats["include_pool_cts"] else validation
    ct_table, ct_hist = ct_statistics(cts, rob_cfg.iso_spacing_mm, edges, rob_cfg.body_hu_threshold,
                                      int(stats["n_workers"]))
    beamlets, beam_hist = selection_beamlet_statistics(validation, selected, rob_cfg, bdl_path, edges,
                                                       int(stats["n_workers"]))
    ct_table.to_csv(out / "ct_table.csv", index=False)
    beamlets.to_csv(out / "beamlet_table.csv", index=False)
    hist = pd.concat([ct_hist, beam_hist], ignore_index=True)
    hist.to_csv(out / "hu_histograms.csv", index=False)
    summaries = {
        "ct_body": summarise_histograms(ct_hist, ["anatomy", "role"], edges),
        "crop_body": summarise_histograms(beam_hist[beam_hist["kind"] == "crop_body"], ["anatomy"], edges),
        "beam_path": summarise_histograms(beam_hist[beam_hist["kind"] == "beam_path"], ["anatomy"], edges),
        "crop_body_by_energy": summarise_histograms(beam_hist[beam_hist["kind"] == "crop_body"],
                                                    ["anatomy", "energy_mev"], edges),
        "beam_path_by_energy": summarise_histograms(beam_hist[beam_hist["kind"] == "beam_path"],
                                                    ["anatomy", "energy_mev"], edges),
    }
    for name, frame in summaries.items():
        frame.to_csv(out / f"hu_summary_{name}.csv", index=False)
    comp = composition_tables(selected, candidates)
    for name, frame in comp.items():
        frame.to_csv(out / f"composition_{name}.csv")
    joined = selected.merge(beamlets.drop(columns=["patient_id", "anatomy", "energy_mev"]), on="candidate_id",
                            how="left")
    metrics = [*FEATURE_NAMES, "score_full", "body_fraction_of_crop", "entrance_depth_mm", "path_mean_hu",
               "path_frac_lung", "path_frac_cortical_bone", "crop_mean_hu"]
    metric_summary(joined, [m for m in metrics if m in joined.columns]).to_csv(out / "metric_summary.csv",
                                                                               index=False)

    fig = out / "figures"
    n_val = len(validation)
    _, curves = hu_histograms_figure(
        hist, edges, figure_path=str(fig / "hu_histograms"), x_max=float(stats["hu_plot_max"]),
        title=f"Validation set: Hounsfield units inside the patient ({len(selected):,} beamlets, {n_val} CTs)",
        caption=("Patient = largest connected component of the body mask (lung inside; external air and a couch "
                 "separated by air excluded). 1 mm grid, 10 HU bins; densities per anatomy (thick) and per CT "
                 "(thin); dashed: pool CTs, for comparison. Beam path: flux-weighted, cut at the distal edge of the "
                 "analytic Bragg-peak zone, each beamlet weighted equally. Bands: HU_LUT tissue classes."))
    curves.to_csv(fig / "hu_histograms.csv", index=False)
    per_anatomy = pd.concat([summaries["ct_body"].query("role == 'validation'").assign(kind="ct_body"),
                             summaries["crop_body"].assign(kind="crop_body"),
                             summaries["beam_path"].assign(kind="beam_path")], ignore_index=True)
    per_anatomy.to_csv(fig / "tissue_fractions_per_anatomy.csv", index=False)
    tissue_fractions_figure(ct_table, per_anatomy, figure_path=str(fig / "tissue_fractions"),
                            title="Tissue classes inside the patient: per CT (v: validation, p: pool), then per "
                                  "anatomy and level",
                            caption="HU_LUT classes of src/figures/ct_visualizations.py. Beam path fractions are "
                                    "shares of flux weight.")
    validity = comp.get("candidate_validity")
    composition_figure(selected, validity, figure_path=str(fig / "composition"),
                       title=f"Validation set composition ({len(selected):,} beamlets)",
                       caption=("Stratified draw: equal counts per anatomy, energy, CT and field where the valid "
                                "candidates allow, shortfalls passed to the other cells of the same level. Validity: "
                                "ray inside the CT and analytic Bragg peak inside the 320 mm crop."))
    metric_distributions_figure(joined, FIGURE_METRICS, figure_path=str(fig / "metric_distributions"),
                                title="Validation set: input-only physics of the beamlets per anatomy",
                                caption="Analytic (input-only) metrics of the candidate scoring and the crop "
                                        "statistics of this script; 0.5 to 99.5 percentile range shown.")
    summary = {"n_beamlets": int(len(selected)), "n_validation_cts": n_val, "n_cts_histogrammed": len(cts),
               "beamlets_cropped": int(beamlets["cropped"].sum()),
               "by_anatomy_energy": comp["by_anatomy_energy"].to_dict(),
               "removed_by_largest_component": ct_table.set_index("patient_id")["removed_by_largest_component"]
               .round(4).to_dict(),
               "full_table": str(full_path), "manifest": str(manifest)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_manifest(out, config=cfg, dataset_path=manifest if manifest.exists() else full_path,
                   extra={"kind": "al_validation_set_stats", "inputs": {"full_table": str(full_path),
                                                                        "candidates": str(cand_path),
                                                                        "pool_csv": str(pool_csv)}})
    logger.info("outputs in %s", out)


if __name__ == "__main__":
    app()
