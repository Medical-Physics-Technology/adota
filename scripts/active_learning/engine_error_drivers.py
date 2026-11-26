"""What drives each dose engine's error? EXP-0023, the dose and scoring stages.

    TMPDIR=/scratch/mstryja/tmp uv run python scripts/active_learning/engine_error_drivers.py \\
        --config scripts/active_learning/config_engine_error_drivers.yaml --stages pba,adota,score

Stages (``--stages``, comma-separated, run in this order; ``--run-dir`` resumes a run):

* ``pba``   -- the commissioned pencil beam on every stored beamlet of the pool split:
               each field is rebuilt from DICOM exactly as the pool generator built it,
               every crop is checked against the stored CT voxel for voxel, and the dose is
               saved on the 2 mm model grid (``doses/pba/<id>.npy``).
* ``adota`` -- ADoTA predictions on the split's export, by the loader and de-normalisation
               the stored EXP-0014 pool errors came from (``doses/adota/<id>.npy``).
* ``score`` -- every engine against the export's Monte Carlo through ``record_metrics``:
               gamma 2%/2mm and 3%/3mm (10% cutoff, global), MAPE, RDE, signed dR80
               (``tables/errors_<engine>.csv``).
* ``panels`` -- one slide panel per beamlet of ``panels.ids_csv``: Monte Carlo, ADoTA and
               the pencil beam over the CT, their differences, the errors, depth dose and
               lateral profile (``figures/panel_<label>.png`` and the numbers beside them).

* ``analyse`` -- what drives each engine's error (:mod:`src.engine_drivers.analysis`):
               per error metric and site, Spearman of every feature with patient-bootstrap
               intervals, grouped permutation importance of a gradient-boosted model fitted
               with folds grouped by patient, partial dependence, and the difficulty score
               against the error (``analysis/*.csv``, ``analysis/summary.md``).

* ``ct_vs_error`` -- the per-beamlet table of the 32 input-only features
               (``analysis/beamlet_features.csv``, joined to both error tables with no lost
               rows) and, per engine, each CT feature against the GPR 2%/2mm and MAPE in both
               directions, with pooled, per-site and energy-controlled correlations
               (``analysis/ct_vs_error/``).

* ``dataset_overview`` -- what the split holds: per-beamlet metadata (site, field, gantry,
               energy, spot position, joined to the error table with no lost rows), HU
               histograms of the model-grid crops per site, energy, spot positions and gantry
               per site, and a summary (``analysis/dataset_overview/``,
               :mod:`src.engine_drivers.overview_stage`).

``--ids-file`` (a CSV with ``sample_id``) restricts every stage to those beamlets; the
pencil beam still rebuilds their whole field, but computes only them.

A dose file is never replaced, so an interrupted stage resumes where it stopped.
Guide: scripts/docs/engine_error_drivers.md
"""
from __future__ import annotations

import os

# Pin BLAS to 1 thread (the gamma and feature code runs many small NumPy calls).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Annotated, List, Optional

import h5py
import numpy as np
import pandas as pd
import typer

from src.active_learning.beamlet_pool.planning import entries_from_registry
from src.active_learning.config import mc_from_config
from src.active_learning.pool import RecordResolver
from src.datasets.beamlet_h5 import sha256_of
from src.engine_drivers.adota import predict_and_save
from src.engine_drivers.analysis import run_analysis, run_ct_vs_error
from src.engine_drivers.overview_stage import run_dataset_overview
from src.engine_drivers.pba import field_doses, patient_cts, pool_fields, save_dose
from src.engine_drivers.scoring import CRITERIA, export_scale, mc_model_grid, score_doses
from src.evaluation.cli import (
    load_yaml_config,
    merge_config,
    resolve_device,
    setup_logging,
    setup_run_directory,
)
from src.figures.engine_drivers import engine_panel_figure
from src.pencil_beam.beam_model import model_fingerprint
from src.training.run_dir import _git_info, save_resolved_config, write_manifest
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import mcsquare_to_gy_per_mu

logger = logging.getLogger("engine_error_drivers")
STAGES = ("pba", "adota", "score", "panels", "analyse", "ct_vs_error", "dataset_overview")
DEFAULTS = {"output_dir": None, "device_index": 0, "split": "test"}


def _provenance(path: Path, record: dict) -> None:
    """Write the engine's provenance once; a later run must match it."""
    if path.exists():
        old = json.loads(path.read_text())
        if old != record:
            raise typer.BadParameter(f"{path} records another engine: {old} vs {record}")
        return
    path.write_text(json.dumps(record, indent=1))


def stage_pba(cfg: dict, run: Path, ids: Optional[List[str]]) -> None:
    pool_cfg = load_yaml_config(Path(cfg["pool_config"]))
    rob, _, bdl, bdl_path = mc_from_config(pool_cfg)
    pba = cfg["engines"]["pba"]
    from src.pencil_beam.engine import PencilBeamEngine  # loads the CUDA kernels

    engine = PencilBeamEngine(Path(pba["dosecuda_kernels"]), Path(pba["model_dir"]), gpu_id=int(pba["gpu_id"]))
    out = run / "doses" / "pba"
    out.mkdir(parents=True, exist_ok=True)
    _provenance(out / "provenance.json", {"model_dir": pba["model_dir"], "bdl": bdl_path,
                                          "model_fingerprint": model_fingerprint(Path(pba["model_dir"])),
                                          "grid": "2 mm model grid (D, H, W), average of the 1 mm window"})
    fields = pool_fields(Path(cfg["pool_store"]), cfg["split"], ids)
    registry = pd.read_csv(pool_cfg["registry_csv"])
    entries = {e.patient_id: e for e in entries_from_registry(registry)}
    resolver = RecordResolver(list(entries.values()))
    patients = sorted({p for p, _ in fields})
    timings, done, started = [], 0, perf_counter()
    for pid, ct in patient_cts(resolver, entries, patients, rob.iso_spacing_mm):
        for (p, gantry), beamlets in sorted(fields.items()):
            if p != pid:
                continue
            todo = [b for b in beamlets if not (out / f"{b.sample_id}.npy").exists()]
            if not todo:
                continue
            doses, t = field_doses(engine, ct, gantry, rob, bdl, todo)
            for sid, dose in doses.items():
                save_dose(out, sid, dose)
            done += 1
            timings.append({"patient_id": pid, "gantry_deg": gantry, "n": len(todo), **t})
            logger.info("pba field %d/%d %s g%.1f: %d beamlets, %.0f s total", done, len(fields), pid, gantry,
                        len(todo), perf_counter() - started)
    pd.DataFrame(timings).to_csv(run / "tables" / "pba_field_timings.csv", index=False)


def stage_adota(cfg: dict, run: Path, ids: List[str]) -> None:
    spec = cfg["engines"]["adota"]
    out = run / "doses" / "adota"
    out.mkdir(parents=True, exist_ok=True)
    _provenance(out / "provenance.json", {"checkpoint": spec["checkpoint"], "model_config": spec["model_config"],
                                          "checkpoint_sha256": sha256_of(Path(spec["checkpoint"])),
                                          "dataset": cfg["dataset_path"]})
    info = predict_and_save(Path(spec["checkpoint"]), Path(spec["model_config"]), Path(cfg["dataset_path"]), ids,
                            out, resolve_device(cfg["device_index"]), int(spec.get("batch_size", 16)),
                            int(spec.get("num_workers", 8)))
    logger.info("adota: %s", info)


def stage_score(cfg: dict, run: Path, ids: List[str]) -> None:
    device = resolve_device(cfg["device_index"])
    options = {"device": device} if device.type == "cuda" else None
    for name in cfg["engines"]:
        frame = score_doses(Path(cfg["dataset_path"]), run / "doses" / name, ids, name,
                            backend="torch" if options else "pymedphys", backend_options=options)
        frame.to_csv(run / "tables" / f"errors_{name}.csv", index=False)
        logger.info("score %s: %d beamlets, median 2%%/2mm %.2f %%, 3%%/3mm %.2f %%", name, len(frame),
                    frame["gpr_2pct_2mm_10pct"].median(), frame["gpr_3pct_3mm_10pct"].median())


ENGINE_NAMES = {"adota": "ADoTA", "pba": "PBA"}


def stage_panels(cfg: dict, run: Path) -> None:
    spec = cfg["panels"]
    chosen = pd.read_csv(spec["ids_csv"], dtype={"sample_id": str})
    label_col = spec.get("label_column", "label")
    _, _, bdl, _ = mc_from_config(load_yaml_config(Path(cfg["pool_config"])))
    meta = pd.read_csv(Path(cfg["dataset_path"]).with_name(Path(cfg["dataset_path"]).stem + "_index.csv"),
                       dtype={"sample_id": str}).set_index("sample_id")
    errors = pd.concat([pd.read_csv(run / "tables" / f"errors_{e}.csv", dtype={"sample_id": str})
                        for e in cfg["engines"]]).assign(engine=lambda f: f["engine"].map(ENGINE_NAMES))
    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    scale = export_scale(Path(cfg["dataset_path"]))
    formats = tuple(cfg.get("figure_formats", ["png"]))
    shown = errors[errors["sample_id"].isin(set(chosen["sample_id"]))]
    shown.merge(chosen[["sample_id", label_col]], on="sample_id").to_csv(fig_dir / "panels_numbers.csv", index=False)
    with h5py.File(cfg["dataset_path"], "r") as handle:
        for row in chosen.itertuples(index=False):
            sid, label = row.sample_id, str(getattr(row, label_col))
            m = meta.loc[sid]
            ct = inverse_minmax(np.asarray(handle[sid]["ct"][...], dtype=np.float32), scale["min_ct"],
                                scale["max_ct"]).transpose(2, 0, 1)
            doses = {"MC": mc_model_grid(handle, sid, scale)}
            doses.update({ENGINE_NAMES[e]: np.load(run / "doses" / e / f"{sid}.npy") for e in cfg["engines"]})
            energy = float(m["energy_mev"])
            title = (f"{label}: {m['pool_anatomy']}, {m['pool_patient_id']}, {energy:g} MeV, "
                     f"gantry {float(m['gantry_angle']):.0f} deg  [{sid}]")
            engine_panel_figure(ct, doses, errors[errors["sample_id"] == sid].set_index("engine"), title=title,
                                figure_path=str(fig_dir / f"panel_{label}"),
                                dose_scale=float(mcsquare_to_gy_per_mu(1.0, energy, bdl)), formats=formats)
            logger.info("panel %s (%s)", label, sid)


def main(
    config: Annotated[Path, typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_engine_error_drivers.yaml"),
    stages: Annotated[str, typer.Option(help=f"Comma-separated, from {','.join(STAGES)}.")] = "pba,adota,score",
    run_dir: Annotated[Optional[Path], typer.Option(help="Resume this run directory.")] = None,
    limit_fields: Annotated[Optional[int], typer.Option(help="Smoke test: the first N fields only.")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="GPU for ADoTA and gamma.")] = None,
    ids_file: Annotated[Optional[Path], typer.Option(help="CSV with sample_id: only these beamlets.")] = None,
) -> None:
    chosen = [s.strip() for s in stages.split(",") if s.strip()]
    unknown = sorted(set(chosen) - set(STAGES))
    if unknown:
        raise typer.BadParameter(f"unknown stages {unknown}; known: {STAGES}")
    cfg = merge_config({"device_index": device_index}, load_yaml_config(config), DEFAULTS)
    if run_dir is None:
        run = setup_run_directory(Path(cfg["output_dir"]), prefix="engine_error_drivers_", subdirs=("tables",))
        save_resolved_config(cfg, run / "config.yaml")
        write_manifest(run, config=cfg, dataset_path=Path(cfg["dataset_path"]),
                       extra={"kind": "engine_error_drivers", "criteria": [list(c) for c in CRITERIA]})
    else:
        run = Path(run_dir)
        (run / "tables").mkdir(parents=True, exist_ok=True)
    # dataset_overview alone adds files to a finished run, so it logs inside its own folder.
    log_dir = run / "analysis" / "dataset_overview" if chosen == ["dataset_overview"] else run
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, log_filename="engine_error_drivers.log")

    only = (list(pd.read_csv(ids_file, dtype={"sample_id": str})["sample_id"]) if ids_file else None)
    fields = pool_fields(Path(cfg["pool_store"]), cfg["split"], only)
    keys = sorted(fields)[:limit_fields] if limit_fields else sorted(fields)
    ids = sorted(b.sample_id for k in keys for b in fields[k])
    logger.info("run %s: %d fields, %d beamlets, stages %s", run, len(keys), len(ids), chosen)
    if "pba" in chosen:
        stage_pba(cfg, run, ids)
    if "adota" in chosen:
        stage_adota(cfg, run, ids)
    if "score" in chosen:
        stage_score(cfg, run, ids)
    if "panels" in chosen:
        stage_panels(cfg, run)
    if "analyse" in chosen:
        lines = run_analysis(cfg, run)
        (run / "analysis" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if "ct_vs_error" in chosen:
        lines = run_ct_vs_error(cfg, run)
        (run / "analysis" / "ct_vs_error" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        (run / "analysis" / "ct_vs_error" / "provenance.json").write_text(
            json.dumps({"stage": "ct_vs_error", "git": _git_info(), "config": cfg.get("analysis")}, indent=1,
                       default=str))
    if "dataset_overview" in chosen:
        out = run / "analysis" / "dataset_overview"
        lines = run_dataset_overview(cfg, run)
        (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        (out / "provenance.json").write_text(json.dumps(
            {"stage": "dataset_overview", "git": _git_info(), "dataset_path": cfg["dataset_path"],
             "config": cfg.get("dataset_overview"), "features_csv": cfg["analysis"]["features_csv"]},
            indent=1, default=str))
    logger.info("done: %s", run)


if __name__ == "__main__":
    typer.run(main)
