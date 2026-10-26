"""Write the scan-level partition of EXP-0014 once: F / V / R, the cross-fitting
folds and the HPO split of R.

    uv run python scripts/active_learning/cross_fit_partition.py \
        --config scripts/active_learning/config_cross_fit_partition.yaml

D is every record of ``dataset_path`` not on the exclusion list. F is every scan
with a record in ``frozen_test_ids`` (EXP-0004), V every scan in
``held_out_patients`` (EXP-0012), R every other scan. All of D is split into
``n_folds`` scan folds stratified by anatomy, independently of F / V / R; the
HPO split holds ``hpo_val_scans_per_stratum`` R scans out for HPO validation.
See :mod:`src.active_learning.ensemble.partition` for the rules.

The partition is written once and referenced by EXP-0014 to EXP-0017, so the
script refuses an existing ``output_dir`` unless ``--overwrite`` is given.
Writes into ``output_dir``:

- ``partition.csv`` (one row per scan: anatomy, records, role, fold, HPO role)
  and ``records.csv`` (one row per record, the same columns);
- ``folds/fold_<k>_train_ids.csv`` and ``folds/fold_<k>_eval_ids.csv``;
- ``hpo/hpo_train_ids.csv`` and ``hpo/hpo_val_ids.csv``;
- ``fold_summary.csv``, ``summary.json``, ``manifest.json`` (with the sha256 of
  every input file) and the resolved ``config.yaml``.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
import typer

from src.active_learning.ensemble.partition import (
    HPO_TRAIN,
    HPO_VAL,
    assign_folds,
    check_partition,
    fold_ids,
    fold_summary,
    hpo_split,
    record_table,
    scan_table,
    scans_from_frozen_ids,
    scans_from_patients,
)
from src.active_learning.retrospective.config import PROVENANCE_CSV
from src.active_learning.retrospective.dataset import (
    DEFAULT_EXCLUDE_PATH,
    apply_exclusions,
    read_exclusion_list,
    read_record_ids,
    record_metadata,
)
from src.datasets.beamlet_h5 import sha256_of
from src.evaluation.cli import load_yaml_config, merge_config, on_root_filesystem, setup_logging
from src.training.run_dir import save_resolved_config, write_manifest

app = typer.Typer(help="Write the EXP-0014 scan partition, cross-fitting folds and HPO split.",
                  add_completion=False)
logger = logging.getLogger("cross_fit_partition")

DEFAULTS: Dict[str, Any] = {
    "experiment": "EXP-0014",
    "output_dir": "/scratch/mstryja/adota_runs/exp0014/partition",
    "dataset_path": None, "exclude_indexes_path": DEFAULT_EXCLUDE_PATH,
    "record_provenance_csv": PROVENANCE_CSV,
    "frozen_test_ids": None, "held_out_patients": None,
    "group_column": "patient_key", "stratify_column": "anatomy",
    "n_folds": 4, "fold_seed": 20261019,
    "hpo_val_scans_per_stratum": {}, "hpo_seed": 20261019,
}


def _write_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": ids}).to_csv(path, index=False)


@app.command()
def main(
    config: Annotated[Optional[Path], typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_cross_fit_partition.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Where the partition is written.")] = None,
    n_folds: Annotated[Optional[int], typer.Option(help="Number of scan folds.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing partition.")] = False,
) -> None:
    raw = load_yaml_config(config) if config and Path(config).exists() else {}
    cfg = merge_config({"output_dir": str(output_dir) if output_dir else None, "n_folds": n_folds},
                       raw, DEFAULTS)
    for key in ("dataset_path", "frozen_test_ids", "held_out_patients"):
        if not cfg[key]:
            raise typer.BadParameter(f"`{key}` is required")
    if not cfg["hpo_val_scans_per_stratum"]:
        raise typer.BadParameter("`hpo_val_scans_per_stratum` is required, e.g. {thorax: 6}")
    out = Path(cfg["output_dir"])
    if on_root_filesystem(out):
        raise typer.BadParameter(f"output_dir {out} is on the root filesystem; use /scratch")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise typer.BadParameter(f"{out} already holds a partition; pass --overwrite to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, log_filename="cross_fit_partition.log")
    save_resolved_config(cfg, out / "config.yaml")

    dataset = Path(cfg["dataset_path"])
    ids = apply_exclusions(read_record_ids(dataset), read_exclusion_list(Path(cfg["exclude_indexes_path"])))
    provenance = Path(cfg["record_provenance_csv"]) if cfg["record_provenance_csv"] else None
    metadata = record_metadata(dataset, ids, provenance)
    unknown = int((metadata[cfg["stratify_column"]] == "unknown").sum())
    if unknown:
        raise RuntimeError(f"{unknown} records have no {cfg['stratify_column']!r}; check the provenance map")

    frozen = pd.read_csv(cfg["frozen_test_ids"], dtype=str)["sample_id"].tolist()
    patients = [line.strip() for line in Path(cfg["held_out_patients"]).read_text().splitlines()
                if line.strip()]
    f_scans = scans_from_frozen_ids(frozen, metadata, cfg["group_column"])
    v_scans = scans_from_patients(patients, metadata, group_column=cfg["group_column"])
    scans = scan_table(metadata, f_scans, v_scans, cfg["group_column"], cfg["stratify_column"])
    scans["fold"] = assign_folds(scans, int(cfg["n_folds"]), int(cfg["fold_seed"]))
    scans["hpo"] = hpo_split(scans, cfg["hpo_val_scans_per_stratum"], int(cfg["hpo_seed"]))
    records = record_table(metadata, scans, cfg["group_column"])
    check_partition(records, int(cfg["n_folds"]))

    scans.to_csv(out / "partition.csv", index=False)
    records.to_csv(out / "records.csv", index=False)
    for k in range(int(cfg["n_folds"])):
        split = fold_ids(records, k)
        _write_ids(out / "folds" / f"fold_{k}_train_ids.csv", split["train"])
        _write_ids(out / "folds" / f"fold_{k}_eval_ids.csv", split["eval"])
    for role, name in ((HPO_TRAIN, "hpo_train_ids.csv"), (HPO_VAL, "hpo_val_ids.csv")):
        _write_ids(out / "hpo" / name, records.loc[records["hpo"] == role, "sample_id"].tolist())
    folds = fold_summary(records)
    folds.to_csv(out / "fold_summary.csv", index=False)

    overlap = sorted(set(f_scans) & set(v_scans))
    summary = {
        "n_records": len(records), "n_scans": len(scans),
        "scans_per_role": scans["role"].value_counts().to_dict(),
        "records_per_role": records["role"].value_counts().to_dict(),
        "n_f_scans": len(f_scans), "n_v_scans": len(v_scans), "f_and_v_scans": overlap,
        "n_frozen_test_ids": len(frozen),
        "n_f_records": int(records["scan"].isin(set(f_scans)).sum()),
        "hpo": {role: {"n_scans": int((scans["hpo"] == role).sum()),
                       "n_records": int((records["hpo"] == role).sum()),
                       "per_anatomy": scans[scans["hpo"] == role]["anatomy"].value_counts().to_dict()}
                for role in (HPO_TRAIN, HPO_VAL)},
        "folds": folds.to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    inputs = {key: {"path": str(cfg[key]), "sha256": sha256_of(Path(cfg[key]))}
              for key in ("frozen_test_ids", "held_out_patients", "exclude_indexes_path",
                          "record_provenance_csv") if cfg[key] and Path(cfg[key]).exists()}
    write_manifest(out, config=cfg, dataset_path=dataset,
                   excluded_indexes_path=Path(cfg["exclude_indexes_path"]),
                   extra={"experiment": cfg["experiment"], "kind": "cross_fit_partition",
                          "inputs": inputs})
    logger.info("%d records in %d scans; roles %s; F and V share %s", len(records), len(scans),
                summary["scans_per_role"], overlap)
    for row in folds.itertuples():
        logger.info("fold %d %-17s %2d scans %6d records (F %d, V %d)", row.fold, row.anatomy,
                    row.n_scans, row.n_records, row.n_f_scans, row.n_v_scans)
    logger.info("HPO: %s", {k: (v["n_scans"], v["n_records"]) for k, v in summary["hpo"].items()})
    logger.info("outputs in %s", out)


if __name__ == "__main__":
    app()
