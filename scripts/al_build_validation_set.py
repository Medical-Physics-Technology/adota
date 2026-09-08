"""Build the frozen, difficulty-balanced validation set the loop is judged on.

    # what it would cost, without simulating anything:
    uv run python scripts/al_build_validation_set.py --config scripts/config_al.yaml --dry-run
    # generate it:
    uv run python scripts/al_build_validation_set.py --config scripts/config_al.yaml

Candidates are generated on the validation CTs, scored with the input-only difficulty
score, and drawn with equal counts per score decile, stratified by anatomy and energy
layer. The selection is written first and the Monte Carlo runs against it, so the set
is defined before it is paid for and an interrupted run resumes into the same set.

Run this once. The loop refuses to start without it, and every arm is measured on the
same beamlets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from src.active_learning.candidates import score_pool
from src.active_learning.config import candidate_config_from_dict, mc_from_config
from src.active_learning.oracle import batch_cost_estimate, label_batch, labelled_records
from src.active_learning.pool import RecordResolver, read_pool
from src.active_learning.validation import select_balanced
from src.adota.config import load_yaml_config
from src.training.logging_utils import silence_pymedphys

# force=True: importing pymedphys configures the root logger, which would make a
# plain basicConfig a no-op and leak per-record DEBUG lines into the run log.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
silence_pymedphys()
logger = logging.getLogger("al_validation")
app = typer.Typer(help="Generate the frozen difficulty-balanced validation set.")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Active-learning YAML config.")] = Path(
        "scripts/config_al.yaml"),
    n_beamlets: Annotated[Optional[int], typer.Option(help="Override the set size.")] = None,
    n_workers: Annotated[int, typer.Option(help="Scoring worker processes.")] = 12,
    n_cts: Annotated[Optional[int], typer.Option(
        help="Use only the first N validation CTs. For the smoke run; the real set "
             "spans every validation CT so each anatomy is represented.")] = None,
    num_threads: Annotated[Optional[int], typer.Option(
        help="MCsquare threads (0 = all). The config reserves half the cores so two "
             "loop arms can run side by side; this build runs alone.")] = None,
    dry_run: Annotated[bool, typer.Option(
        help="Score, select and report the cost; simulate nothing.")] = False,
) -> None:
    cfg = load_yaml_config(config)
    val_cfg = cfg.get("validation_set", {})
    n = int(n_beamlets or val_cfg.get("n_beamlets", 4000))
    manifest_path = Path(val_cfg.get("manifest", "registry/al_validation_set.csv"))
    prefix = val_cfg.get("beamlet_prefix", "al_val")
    version = val_cfg.get("beamlet_version", 1)

    rob_cfg, runner, bdl, bdl_path = mc_from_config(cfg)
    if num_threads is not None:
        rob_cfg.num_threads = int(num_threads)
    cand_cfg = candidate_config_from_dict(cfg.get("candidates", {}))
    cand_cfg.seed = int(val_cfg.get("seed", cand_cfg.seed + 1))
    entries = read_pool(Path(cfg["pool"]["pool_csv"]), role="validation")
    if not entries:
        raise typer.BadParameter("the pool CSV has no validation CTs; run al_build_pool.py")
    if n_cts is not None:
        entries = entries[:n_cts]
    logger.info("validation CTs: %s", ", ".join(f"{e.patient_id} ({e.anatomy})"
                                                for e in entries))

    if manifest_path.exists():
        logger.info("selection already exists at %s; reusing it (Monte Carlo resumes)",
                    manifest_path)
        import pandas as pd

        chosen = pd.read_csv(manifest_path)
    else:
        table = score_pool(entries, cand_cfg, rob_cfg, bdl_path, n_workers=n_workers,
                           prefix=prefix, version=version)
        table.to_csv(manifest_path.with_name(manifest_path.stem + "_candidates.csv"),
                     index=False)
        chosen = select_balanced(table, n, rng=np.random.default_rng(cand_cfg.seed))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        chosen.to_csv(manifest_path, index=False)
        logger.info("selection -> %s", manifest_path)

    cost = batch_cost_estimate(chosen)
    logger.info("validation set: %d beamlets in %d groups (%.1f per group, min %d) | "
                "estimated %.1f h of Monte Carlo", cost["n_beamlets"], cost["n_groups"],
                cost["beamlets_per_group_mean"], cost["beamlets_per_group_min"],
                cost["estimated_total_hours"])
    if dry_run:
        logger.info("dry run: nothing simulated.")
        raise typer.Exit()

    output_root = Path(rob_cfg.output_root)
    stats = label_batch(chosen, entries, runner, bdl, rob_cfg, output_root,
                        resolver=RecordResolver(entries))
    on_disk = labelled_records(chosen, output_root)
    summary = {"n_selected": int(len(chosen)), "n_on_disk": len(on_disk),
               "manifest": str(manifest_path), "cost_estimate": cost,
               **{k: v for k, v in stats.items() if k != "groups"}}
    out = manifest_path.with_name(manifest_path.stem + "_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    logger.info("validation set built: %d of %d beamlets on disk (%.0f MC seconds) -> %s",
                len(on_disk), len(chosen), stats["mc_seconds"], out)


if __name__ == "__main__":
    app()
