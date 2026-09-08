"""Assign pool and validation roles to held-out CTs, and record the split.

    uv run python scripts/al_build_pool.py --config scripts/config_al.yaml

Reads the ``datasets`` block of the active-learning config, takes the tail of each
collection (the part training never used), drops every patient claimed by another
experiment, and splits what remains into validation CTs and pool CTs. The result is a
CSV under ``registry/``; the loop reads that file rather than re-deriving the rule, so
a selection cannot drift between runs.

Run it once. Rerunning overwrites the CSV, which changes the split -- pass
``--out`` to write a new file instead if that is what you meant.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from src.active_learning.pool import build_pool, write_pool
from src.adota.config import load_yaml_config
from src.training.logging_utils import silence_pymedphys

# force=True: importing pymedphys configures the root logger, which would make a
# plain basicConfig a no-op and leak per-record DEBUG lines into the run log.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
silence_pymedphys()
logger = logging.getLogger("al_pool")
app = typer.Typer(help="Build the active-learning CT pool and validation split.")


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Active-learning YAML config.")] = Path(
        "scripts/config_al.yaml"),
    out: Annotated[Optional[Path], typer.Option(help="Output CSV; defaults to the "
                                                "config's pool_csv.")] = None,
    n_holdout: Annotated[Optional[int], typer.Option(
        help="Patients from the tail of each collection to consider safe.")] = None,
    n_validation: Annotated[Optional[int], typer.Option(
        help="Validation CTs per anatomy; the rest of the tail becomes the pool.")] = None,
    overwrite: Annotated[bool, typer.Option(help="Overwrite an existing CSV.")] = False,
) -> None:
    cfg = load_yaml_config(config)
    pool_cfg = cfg.get("pool", {})
    target = Path(out or pool_cfg.get("pool_csv", "registry/al_pool_selection.csv"))
    if target.exists() and not overwrite:
        raise typer.BadParameter(
            f"{target} already exists; the split it records is what the loop uses. "
            "Pass --overwrite to redraw it, or --out to write elsewhere.")

    entries = build_pool(
        cfg["datasets"],
        n_holdout=int(n_holdout if n_holdout is not None else pool_cfg.get("n_holdout", 30)),
        n_validation=int(n_validation if n_validation is not None
                         else pool_cfg.get("n_validation", 5)),
        exclude_files=pool_cfg.get("exclude_files", []),
    )
    write_pool(entries, target)
    n_val = sum(e.role == "validation" for e in entries)
    logger.info("wrote %s: %d CTs (%d validation, %d pool)",
                target, len(entries), n_val, len(entries) - n_val)
    for anatomy in sorted({e.anatomy for e in entries}):
        rows = [e for e in entries if e.anatomy == anatomy]
        logger.info("  %-10s %d validation: %s", anatomy,
                    sum(r.role == "validation" for r in rows),
                    ", ".join(r.patient_id for r in rows if r.role == "validation"))


if __name__ == "__main__":
    app()
