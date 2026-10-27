"""Build the frozen validation set the loop is judged on.

    # what it would cost, without simulating anything:
    uv run python scripts/al_build_validation_set.py --config scripts/config_al.yaml --dry-run
    # generate it:
    uv run python scripts/al_build_validation_set.py --config scripts/config_al.yaml

Candidates are generated on the validation CTs and passed through the validity gate
(ray inside the CT, analytic Bragg peak inside the crop). With ``balance: stratified``
(the default) they are drawn with equal counts per level of ``strata`` in order (anatomy,
energy layer, patient, field), **without the difficulty score**, so the yardstick does
not change when the score does; the score is still computed and its decile within the
drawn set is recorded as ``score_decile`` for per-decile curves. ``balance:
score_deciles`` is the earlier recipe (equal counts per score decile). The selection is
written first and the Monte Carlo runs against it, so the set is defined before it is
paid for and an interrupted run resumes into the same set.

Outputs: the frozen manifest (``validation_set.manifest``, the loop's columns only, in
the repository's registry) and, under ``validation_set.tables_dir``, the scored
candidates and the selection with all thirty metrics, for
``scripts/al_validation_set_stats.py``.

Run this once. The loop refuses to start without it, and every arm is measured on the
same beamlets. Any config key can be overridden with the repeatable ``--set KEY=VALUE``
(``--set validation_set.n_beamlets=8``); a per-field option beats ``--set`` for the
same key, and ``--set`` beats the YAML. The smoke test is the main config plus
``--set`` entries; see ``scripts/docs/al_loop.md``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import pandas as pd
import typer

from src.active_learning.candidates import score_pool
from src.active_learning.config import candidate_config_from_dict, mc_from_config
from src.active_learning.oracle import batch_cost_estimate, label_batch, labelled_records
from src.active_learning.pool import RecordResolver, read_pool
from src.active_learning.validation import select_balanced, select_stratified
from src.evaluation.cli import SET_OVERRIDE_HELP, apply_set_overrides, load_yaml_config
from src.training.logging_utils import silence_pymedphys

# force=True: importing pymedphys configures the root logger, which would make a
# plain basicConfig a no-op and leak per-record DEBUG lines into the run log.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
silence_pymedphys()
logger = logging.getLogger("al_validation")
app = typer.Typer(help="Generate the frozen validation set.")

STRATA = ["anatomy", "energy_mev", "patient_id", "gantry_deg"]
SIZING_KEYS = {"n_gantry_per_ct", "n_per_field"}
MANIFEST_COLUMNS = ["candidate_id", "patient_id", "anatomy", "dataset_name", "provenance_uid",
                    "gantry_deg", "energy_mev", "theta_x_deg", "theta_y_deg", "mc_gantry_deg",
                    "ct_rotation_deg", "field_dir", "valid", "reason", "score_full", "score_decile"]
"""What the loop reads from the frozen manifest; the thirty metrics stay in the full table."""


def with_score_decile(chosen: pd.DataFrame, column: str = "score_full", n: int = 10) -> pd.DataFrame:
    """The decile of the difficulty score inside the drawn set, for per-decile curves.

    Computed after the draw, so the score never decides what the set contains; the
    score_deciles recipe already carries its own decile, which is kept.
    """
    if "score_decile" in chosen.columns or column not in chosen.columns:
        return chosen
    return chosen.assign(score_decile=pd.qcut(chosen[column], n, labels=False, duplicates="drop"))


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
    set_: Annotated[Optional[List[str]], typer.Option("--set", help=SET_OVERRIDE_HELP)] = None,
) -> None:
    try:
        cfg = apply_set_overrides(load_yaml_config(config), set_ or [])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    val_cfg = cfg.get("validation_set", {})
    n = int(n_beamlets or val_cfg.get("n_beamlets", 4000))
    manifest_path = Path(val_cfg.get("manifest", "registry/al_validation_set.csv"))
    prefix = val_cfg.get("beamlet_prefix", "al_val")
    version = val_cfg.get("beamlet_version", 1)

    tables_dir = Path(val_cfg.get("tables_dir") or manifest_path.parent)
    balance = str(val_cfg.get("balance", "stratified"))
    strata = list(val_cfg.get("strata", STRATA))
    if balance not in ("stratified", "score_deciles"):
        raise typer.BadParameter(f"validation_set.balance must be stratified or score_deciles, got {balance!r}")

    rob_cfg, runner, bdl, bdl_path = mc_from_config(cfg)
    if num_threads is not None:
        rob_cfg.num_threads = int(num_threads)
    # The candidate space is shared with the loop; only its size may differ here.
    sizing = {k: v for k, v in (val_cfg.get("candidates") or {}).items() if k in SIZING_KEYS}
    cand_cfg = candidate_config_from_dict({**cfg.get("candidates", {}), **sizing})
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
        chosen = pd.read_csv(manifest_path)
    else:
        table = score_pool(entries, cand_cfg, rob_cfg, bdl_path, n_workers=n_workers,
                           prefix=prefix, version=version)
        tables_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(tables_dir / f"{manifest_path.stem}_candidates.csv", index=False)
        rng = np.random.default_rng(cand_cfg.seed)
        if balance == "stratified":
            chosen = select_stratified(table, n, strata, rng=rng)
        else:
            chosen = select_balanced(table, n, rng=rng)
        chosen = with_score_decile(chosen)
        chosen.to_csv(tables_dir / f"{manifest_path.stem}_full.csv", index=False)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        chosen[[c for c in MANIFEST_COLUMNS if c in chosen.columns]].to_csv(manifest_path, index=False)
        (tables_dir / f"{manifest_path.stem}_selection.json").write_text(json.dumps({
            "n_requested": n, "n_selected": int(len(chosen)), "balance": balance, "strata": strata,
            "candidates": {"n_gantry_per_ct": cand_cfg.n_gantry_per_ct, "n_per_field": cand_cfg.n_per_field,
                           "energies": cand_cfg.energies, "seed": cand_cfg.seed},
            "n_candidates": int(len(table)), "n_valid": int(table["valid"].sum()),
            "validation_cts": [e.patient_id for e in entries], "manifest": str(manifest_path)}, indent=2))
        logger.info("selection -> %s (%d beamlets); tables -> %s", manifest_path, len(chosen), tables_dir)

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
