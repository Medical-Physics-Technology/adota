"""Pairwise agreement of an ensemble of trained ADoTA models on which records are
hard (EXP-0014; stage 0b is the nine cycle-5 models of EXP-0012 on its 12
held-out scans).

    uv run python scripts/active_learning/ensemble_agreement.py \
        --config scripts/active_learning/config_ensemble_agreement.yaml
    uv run python scripts/active_learning/ensemble_agreement.py --config <yaml> --source infer \
        --device-index 0 --output-dir /scratch/mstryja/adota_runs/exp0014/stage0b_infer

Any members and any evaluation set: every member is a retrospective run and
cycle, a checkpoint with its training YAML, or a per-record errors CSV (see
:mod:`src.active_learning.ensemble.members`); the evaluation set is a CSV of
sample ids in the HDF5 of ``dataset_path``. Errors are reused from the CSV a
run already wrote when it covers the set with the configured gamma criteria,
and are otherwise recomputed from the checkpoint (``source: infer`` forces
that, and needs a GPU to be quick).

Before any number is computed the script checks, per member, that none of the
evaluation scans (``group_column``) is among its training scans, and stops
otherwise unless ``allow_training_overlap`` is set.

Writes into a timestamped directory under ``output_dir``:

- ``members.csv`` (source, checkpoint and CSV hashes) and ``held_out.csv``;
- per error metric: ``pairs_<metric>.csv`` (every pair with its level, its
  training-id overlap, the median within-cell Spearman with its scan-bootstrap
  interval, the pooled Spearman and the top-fraction Jaccard overlaps),
  ``levels_<metric>.csv`` (mean per level with an interval, and the contrast
  ``data - seed_only``), ``cells_<metric>.csv`` (every cell's Spearman per
  pair), and ``figures/agreement_matrix_<metric>`` with its matrix as CSV;
- ``summary.json``, ``manifest.json`` and the resolved ``config.yaml``.

Figures come from :mod:`src.figures.ensemble_agreement`.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import typer

from src.active_learning.ensemble.agreement import (
    LEVEL_SEED,
    agreement_matrix,
    assign_cells,
    cell_correlations,
    error_matrix,
    pair_levels,
    pair_table,
    scan_bootstrap,
)
from src.active_learning.ensemble.members import (
    EvaluationSettings,
    held_out_table,
    resolve_member,
    unique_member_labels,
)
from src.active_learning.retrospective.dataset import record_metadata
from src.adota.config import DEFAULT_GAMMA_PARAMS
from src.datasets.beamlet_h5 import sha256_of
from src.evaluation.cli import (
    load_yaml_config,
    merge_config,
    on_root_filesystem,
    resolve_device,
    setup_logging,
    setup_run_directory,
)
from src.figures.ensemble_agreement import agreement_matrix_figure
from src.training.run_dir import save_resolved_config, write_manifest

app = typer.Typer(help="Pairwise agreement of ensemble members on which records are hard.",
                  add_completion=False)
logger = logging.getLogger("ensemble_agreement")

DEFAULTS: Dict[str, Any] = {
    "experiment": "EXP-0014", "stage": "0b",
    "output_dir": "/scratch/mstryja/adota_runs/exp0014/stage0b",
    "dataset_path": None, "record_ids": None, "metadata_csv": None,
    "group_column": "patient_key", "energy_column": "energy_mev",
    "members": [], "cycle": None, "checkpoint": "last", "source": "auto",
    "device_index": None, "batch_size": None, "num_workers": None,
    "gamma_params": dict(DEFAULT_GAMMA_PARAMS), "gamma_resolution_mm": [2.0, 2.0, 2.0],
    "gamma_cutoff_percent": 10.0, "gamma_backend": "torch",
    "metrics": ["rde_pct", "gamma_fail", "mape_pct", "abs_dr80_mm"],
    "min_cell_size": 10, "top_fraction": 0.10,
    "n_bootstrap": 2000, "bootstrap_seed": 20260929, "ci": 0.95,
    "allow_training_overlap": False,
}

METRIC_LABELS = {"rde_pct": "relative dose error", "gamma_fail": "1 - gamma pass rate",
                 "mape_pct": "MAPE", "abs_dr80_mm": "|dR80|"}


def assert_off_root_filesystem(path: Path, what: str) -> None:
    """Raise if ``path`` lives on the root filesystem (see :func:`on_root_filesystem`)."""
    if on_root_filesystem(path):
        raise typer.BadParameter(f"{what} {path} is on the root filesystem; use /scratch")


def _read_ids(path: Path) -> List[str]:
    frame = pd.read_csv(path, dtype=str)
    column = "sample_id" if "sample_id" in frame.columns else frame.columns[0]
    return frame[column].astype(str).tolist()


def _metric_tag(metric: str, gamma_label: str) -> str:
    return f"{metric}_{gamma_label}" if metric == "gamma_fail" else metric


@app.command()
def main(
    config: Annotated[Optional[Path], typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_ensemble_agreement.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Parent of the run directory.")] = None,
    source: Annotated[Optional[str], typer.Option(help="auto | csv | infer.")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA index for inference.")] = None,
    n_bootstrap: Annotated[Optional[int], typer.Option(help="Scan-bootstrap replicates.")] = None,
) -> None:
    raw = load_yaml_config(config) if config and Path(config).exists() else {}
    cfg = merge_config({"output_dir": str(output_dir) if output_dir else None, "source": source,
                        "device_index": device_index, "n_bootstrap": n_bootstrap}, raw, DEFAULTS)
    for key in ("dataset_path", "record_ids"):
        if not cfg[key]:
            raise typer.BadParameter(f"`{key}` is required")
    if not cfg["members"]:
        raise typer.BadParameter("no members: list them under `members` in the config")
    assert_off_root_filesystem(Path(cfg["output_dir"]), "output_dir")
    assert_off_root_filesystem(Path(tempfile.gettempdir()), "the temporary directory (TMPDIR)")

    out = setup_run_directory(Path(cfg["output_dir"]), prefix="ensemble_agreement_",
                              subdirs=("figures", "members"))
    setup_logging(out, log_filename="ensemble_agreement.log")
    save_resolved_config(cfg, out / "config.yaml")

    ids = _read_ids(Path(cfg["record_ids"]))
    settings = EvaluationSettings(
        dataset_path=str(cfg["dataset_path"]), gamma_params=dict(cfg["gamma_params"]),
        gamma_resolution_mm=tuple(cfg["gamma_resolution_mm"]),
        gamma_cutoff_percent=float(cfg["gamma_cutoff_percent"]),
        gamma_backend=str(cfg["gamma_backend"]),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    device = resolve_device(cfg["device_index"]) if cfg["source"] != "csv" else None
    members = []
    for spec in cfg["members"]:
        spec = dict(spec)
        if "run_dir" in spec and "cycle" not in spec:
            if cfg["cycle"] is None:
                raise typer.BadParameter(f"member {spec['run_dir']} has no cycle and no default")
            spec["cycle"] = int(cfg["cycle"])
        members.append(resolve_member(spec, ids, settings, source=str(cfg["source"]),
                                      checkpoint_name=str(cfg["checkpoint"]), device=device,
                                      cache_dir=out / "members"))
    unique_member_labels(members)
    pd.DataFrame([{"label": m.label, "source": m.source, **m.provenance} for m in members]) \
        .to_csv(out / "members.csv", index=False)
    logger.info("%d members on %d records: %s", len(members), len(ids),
                ", ".join(f"{m.label} ({m.source})" for m in members))

    needed = set(ids).union(*(set(m.training_ids or []) for m in members))
    metadata = (pd.read_csv(cfg["metadata_csv"], dtype={"sample_id": str, cfg["group_column"]: str})
                if cfg["metadata_csv"] else
                record_metadata(Path(cfg["dataset_path"]), sorted(needed), provenance_csv=None))
    group_of = metadata.drop_duplicates("sample_id").set_index("sample_id")[cfg["group_column"]]
    held_out = held_out_table(members, ids, group_of)
    held_out.to_csv(out / "held_out.csv", index=False)
    leaking = held_out[held_out["n_shared_groups"].fillna(0) > 0]
    if len(leaking):
        message = (f"{len(leaking)} members trained on evaluation scans: "
                   f"{leaking[['label', 'n_shared_groups', 'n_shared_records']].to_dict('records')}")
        if not cfg["allow_training_overlap"]:
            raise RuntimeError(message + "; set allow_training_overlap to proceed anyway")
        logger.warning(message)
    unknown = held_out[held_out["n_training"].isna()]["label"].tolist()
    if unknown:
        logger.warning("no training ids for %s: held-out status and pair level unknown", unknown)

    cells = assign_cells(metadata, ids, cfg["group_column"], cfg["energy_column"])
    pairs = pair_levels(members)
    labels = [m.label for m in members]
    seed_pairs = [(a, b) for a, b, lv in zip(pairs["a"], pairs["b"], pairs["level"])
                  if lv == LEVEL_SEED]
    logger.info("pair levels: %s", pairs["level"].value_counts().to_dict())
    summary: Dict[str, Any] = {"n_members": len(members), "n_records": len(ids),
                               "n_scans": int(cells["scan"].nunique()),
                               "pair_levels": pairs["level"].value_counts().to_dict(),
                               "gamma_label": settings.gamma_label, "metrics": {}}
    for metric in cfg["metrics"]:
        tag = _metric_tag(metric, settings.gamma_label)
        matrix = error_matrix(members, metric)
        corr, cell_frame = cell_correlations(matrix, cells, int(cfg["min_cell_size"]))
        table = pair_table(matrix, cells, pairs, corr, float(cfg["top_fraction"]))
        pair_ci, levels = scan_bootstrap(corr, cell_frame["scan"], table["level"],
                                         int(cfg["n_bootstrap"]), int(cfg["bootstrap_seed"]),
                                         float(cfg["ci"]))
        table = pd.concat([table, pair_ci], axis=1)
        table.to_csv(out / f"pairs_{tag}.csv", index=False)
        levels.to_csv(out / f"levels_{tag}.csv", index=False)
        pair_names = [f"{a}__{b}" for a, b in zip(pairs["a"], pairs["b"])]
        pd.concat([cell_frame, pd.DataFrame(corr, columns=pair_names)], axis=1) \
            .to_csv(out / f"cells_{tag}.csv", index=False)

        grid = agreement_matrix(table, labels, "median_cell_spearman")
        grid.to_csv(out / "figures" / f"agreement_matrix_{tag}.csv")
        caption = (f"Spearman of {METRIC_LABELS.get(metric, metric)} within scan x 20 MeV cells "
                   f"(n >= {cfg['min_cell_size']}), median over {len(cell_frame)} cells, "
                   f"{len(matrix)} records, {cells['scan'].nunique()} scans"
                   + (f"; gamma {settings.gamma_label.replace('_', ' ')}" if metric == "gamma_fail"
                      else "") + ". Framed: pairs with identical training ids (seed only).")
        agreement_matrix_figure(grid, figure_path=str(out / "figures" / f"agreement_matrix_{tag}"),
                                value_label="median within-cell Spearman",
                                title=f"{cfg['experiment']} stage {cfg['stage']}: {tag}",
                                caption=caption, outlined=seed_pairs,
                                vmin=float(np.floor(np.nanmin(grid.to_numpy()) * 20.0) / 20.0))
        summary["metrics"][tag] = {"n_records": len(matrix), "n_cells": len(cell_frame),
                                   "levels": levels.to_dict("records")}
        for row in levels.itertuples():
            logger.info("%s  %-20s mean median-cell Spearman %.3f  [%.3f, %.3f]", tag, row.level,
                        row.mean_median_cell_spearman, row.ci_lo, row.ci_hi)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_manifest(out, config=cfg, dataset_path=Path(cfg["dataset_path"]),
                   extra={"experiment": cfg["experiment"], "stage": cfg["stage"],
                          "kind": "ensemble_agreement",
                          "record_ids": {"path": str(cfg["record_ids"]), "n": len(ids),
                                         "sha256": sha256_of(Path(cfg["record_ids"]))},
                          "members": [{"label": m.label, "source": m.source, **m.provenance}
                                      for m in members]})
    logger.info("outputs in %s", out)


if __name__ == "__main__":
    app()
