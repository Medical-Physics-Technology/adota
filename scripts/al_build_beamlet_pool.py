"""Build the labelled beamlet pool: new patients, a patient-level split, batched Monte Carlo.

    uv run python scripts/al_build_beamlet_pool.py screen   --config scripts/config_beamlet_pool.yaml
    uv run python scripts/al_build_beamlet_pool.py plan     --config scripts/config_beamlet_pool.yaml
    uv run python scripts/al_build_beamlet_pool.py check-energy-mixing --config scripts/config_beamlet_pool.yaml
    uv run python scripts/al_build_beamlet_pool.py generate --config scripts/config_beamlet_pool.yaml \
        --shard 0 --n-shards 2
    uv run python scripts/al_build_beamlet_pool.py status   --config scripts/config_beamlet_pool.yaml

- ``screen`` (DICOM headers only, minutes): every CT of the configured collections
  is checked for image quality, matched against the training HDF5's scans by its
  1 mm grid, claimed by other experiments or split per collection. Writes the
  patient registry (``registry_csv``), which every run of every machine shares.
- ``plan`` (CPU, no Monte Carlo): candidates per patient with energies uniform in
  ``plan.energy_range_mev``, the input-only overshoot gate, quotas per split and
  batches. Writes ``<store_root>/plan/`` (plan, all candidates, acceptance per
  energy, estimate) and refuses to overwrite an existing plan.
- ``check-energy-mixing`` (a few minutes of Monte Carlo): one field simulated as
  one mixed-energy beamlet-mode call and as single-energy calls; must pass
  before ``generate``.
- ``generate`` (the long part): batch after batch, Monte Carlo on ``work_root``,
  shards moved to ``store_root``; resumable, ``--shard i --n-shards n`` splits
  the batches over processes (``--shard slurm`` reads the SLURM array index).
- ``status``: batches done, beamlets stored per split, hours per batch.

The data format and the checks are in :mod:`src.active_learning.beamlet_pool`;
guide: ``scripts/docs/beamlet_pool.md``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import typer

from src.active_learning.beamlet_pool.batches import BatchSettings, progress, run_batches
from src.active_learning.beamlet_pool.mc_check import PASS, compare_mixed_energy
from src.active_learning.beamlet_pool.planning import (
    PlanSettings,
    _plan_task,
    acceptance_table,
    assign_batches,
    entries_from_registry,
    estimate,
    patient_quotas,
    quotas_frame,
)
from src.active_learning.beamlet_pool.registry import (
    Claim,
    CollectionSpec,
    assign_splits,
    screen_collection,
    split_summary,
    training_scans,
)
from src.active_learning.beamlet_pool.store import StoreChecks, free_gb
from src.active_learning.candidates import pin_worker_threads
from src.active_learning.config import mc_from_config
from src.evaluation.cli import SET_OVERRIDE_HELP, apply_set_overrides, load_yaml_config, on_root_filesystem
from src.training.logging_utils import silence_pymedphys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
silence_pymedphys()
logger = logging.getLogger("beamlet_pool")
app = typer.Typer(help="Build the labelled beamlet pool.", add_completion=False)

ConfigOpt = Annotated[Path, typer.Option(help="Pool YAML config.")]
SetOpt = Annotated[Optional[List[str]], typer.Option("--set", help=SET_OVERRIDE_HELP)]


def _config(path: Path, sets: Optional[List[str]]) -> Dict[str, Any]:
    try:
        cfg = apply_set_overrides(load_yaml_config(path), sets or [])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if on_root_filesystem(Path(tempfile.gettempdir())):
        raise typer.BadParameter(f"TMPDIR {tempfile.gettempdir()} is on the root filesystem; "
                                 "set TMPDIR to a scratch directory")
    return cfg


def _plan_settings(cfg: Dict[str, Any]) -> PlanSettings:
    raw = dict(cfg.get("plan") or {})
    if "energy_range_mev" in raw:
        raw["energy_range_mev"] = tuple(float(e) for e in raw["energy_range_mev"])
    return PlanSettings(**raw)


def _store(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["batches"]["store_root"])


def _registry(cfg: Dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["registry_csv"])
    if not path.exists():
        raise typer.BadParameter(f"{path} not found: run `screen` first")
    return pd.read_csv(path, dtype={"patient_id": str, "series_uid": str})


@app.command()
def screen(config: ConfigOpt = Path("scripts/config_beamlet_pool.yaml"), set_: SetOpt = None,
           overwrite: Annotated[bool, typer.Option(help="Replace an existing registry.")] = False) -> None:
    """Screen the collections and split the usable patients."""
    cfg = _config(config, set_)
    out = Path(cfg["registry_csv"])
    if out.exists() and not overwrite:
        raise typer.BadParameter(f"{out} exists; it defines the splits of every run. Pass --overwrite to redraw.")
    scans = training_scans(Path(cfg["training_index_csv"]))
    frames = [screen_collection(CollectionSpec(**spec), scans) for spec in cfg["collections"]]
    claims = [Claim(**c) for c in cfg.get("claims", [])]
    split = cfg["split"]
    registry = assign_splits(pd.concat(frames, ignore_index=True), claims, split["fractions"], int(split["seed"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(out, index=False)
    n_matched = int((registry["n_training_matches"].fillna(0) > 0).sum())
    logger.info("training scans matched: %d of %d in the index", n_matched, len(scans))
    logger.info("registry -> %s\n%s", out, split_summary(registry).to_string())


@app.command()
def plan(config: ConfigOpt = Path("scripts/config_beamlet_pool.yaml"), set_: SetOpt = None,
         n_workers: Annotated[int, typer.Option(help="Worker processes, one patient each.")] = 10,
         overwrite: Annotated[bool, typer.Option(help="Replace an existing plan.")] = False) -> None:
    """Draw and gate candidates, set quotas and batches; estimate time and storage."""
    cfg = _config(config, set_)
    settings = _plan_settings(cfg)
    plan_dir = _store(cfg) / "plan"
    if (plan_dir / "plan.csv").exists() and not overwrite:
        raise typer.BadParameter(f"{plan_dir / 'plan.csv'} exists; batches may already refer to it")
    registry = _registry(cfg)
    entries = entries_from_registry(registry)
    split_of = dict(zip(registry["patient_id"], registry["split"]))
    quotas = patient_quotas(registry, settings)
    rob_cfg, _, _, bdl_path = mc_from_config(cfg)
    payload = [(e, split_of[e.patient_id], quotas[e.patient_id], rob_cfg, bdl_path, settings) for e in entries]
    frames, counts, failures = [], [], {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=pin_worker_threads) as pool:
        for i, (pid, (frame, count), error) in enumerate(pool.map(_plan_task, payload), 1):
            if error:
                failures[pid] = error
                logger.error("%s failed: %s", pid, error)
            frames.append(frame)
            counts.append({"patient_id": pid, **count})
            logger.info("planned %s (%d/%d): %d of %d accepted", pid, i, len(payload), count["accepted"],
                        count["drawn"])
    candidates = pd.concat(frames, ignore_index=True)
    accepted = candidates[candidates["valid"].astype(bool)].reset_index(drop=True)
    accepted["batch"] = assign_batches(accepted, settings.batch_size, settings.seed).to_numpy()
    plan_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(plan_dir / "candidates.csv", index=False)
    accepted.to_csv(plan_dir / "plan.csv", index=False)
    quotas_frame(quotas).merge(pd.DataFrame(counts), on="patient_id").to_csv(plan_dir / "quotas.csv", index=False)
    acceptance_table(candidates, np.arange(70, 271, 20)).to_csv(plan_dir / "acceptance.csv")
    rob_primaries = float(rob_cfg.num_primaries)
    est = {**estimate(accepted, rob_primaries, int(cfg.get("n_processes", 1))),
           "by_split": accepted.groupby("split").size().to_dict(),
           "by_anatomy": accepted.groupby("anatomy").size().to_dict(),
           "patients_short_of_quota": int((pd.DataFrame(counts)["accepted"] <
                                           pd.Series([quotas[c["patient_id"]] for c in counts])).sum()),
           "failures": failures, "num_primaries": rob_primaries, "settings": settings.__dict__}
    (plan_dir / "estimate.json").write_text(json.dumps(est, indent=2, default=str))
    logger.info("plan -> %s\n%s", plan_dir, json.dumps({k: v for k, v in est.items() if k != "settings"},
                                                       indent=2, default=str))


@app.command("check-energy-mixing")
def check_energy_mixing(config: ConfigOpt = Path("scripts/config_beamlet_pool.yaml"), set_: SetOpt = None,
                        n_beamlets: Annotated[int, typer.Option(help="Beamlets of one field to compare.")] = 6
                        ) -> None:
    """Mixed-energy beamlet-mode call against single-energy calls on one planned field."""
    cfg = _config(config, set_)
    plan_df = pd.read_csv(_store(cfg) / "plan" / "plan.csv")
    first = plan_df.groupby(["patient_id", "gantry_deg"]).size().idxmax()
    rows = plan_df[(plan_df["patient_id"] == first[0]) & (plan_df["gantry_deg"] == first[1])]
    rows = rows.sort_values("energy_mev").iloc[np.linspace(0, len(rows) - 1, n_beamlets).astype(int)]
    entry = next(e for e in entries_from_registry(_registry(cfg)) if e.patient_id == first[0])
    rob_cfg, runner, bdl, _ = mc_from_config(cfg)
    frame = compare_mixed_energy(rows, entry, runner, bdl, rob_cfg)
    out = _store(cfg) / "plan" / "energy_mixing_check.csv"
    frame.to_csv(out, index=False)
    verdict = "PASS" if frame["passes"].all() else "FAIL"
    logger.info("energy mixing check %s (thresholds %s) -> %s\n%s", verdict, PASS, out, frame.to_string())
    if verdict == "FAIL":
        raise typer.Exit(code=1)


@app.command()
def generate(config: ConfigOpt = Path("scripts/config_beamlet_pool.yaml"), set_: SetOpt = None,
             shard: Annotated[str, typer.Option(help="This process's shard index, or 'slurm'.")] = "0",
             n_shards: Annotated[int, typer.Option(help="Number of processes sharing the plan.")] = 1,
             max_batches: Annotated[Optional[int], typer.Option(help="Stop after this many batches.")] = None,
             num_threads: Annotated[Optional[int], typer.Option(help="MCsquare threads of this process.")] = None,
             ) -> None:
    """Run this process's batches: Monte Carlo, pack, move, clean up."""
    cfg = _config(config, set_)
    if shard == "slurm":
        index, n_shards = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    else:
        index = int(shard)
    store = _store(cfg)
    check = store / "plan" / "energy_mixing_check.csv"
    if not check.exists() or not pd.read_csv(check)["passes"].all():
        raise typer.BadParameter(f"{check} missing or failing: run check-energy-mixing first")
    plan_df = pd.read_csv(store / "plan" / "plan.csv")
    entries = entries_from_registry(_registry(cfg))
    rob_cfg, runner, bdl, _ = mc_from_config(cfg)
    if num_threads is not None:
        rob_cfg.num_threads = int(num_threads)
    settings = BatchSettings(**cfg["batches"])
    checks = StoreChecks(**(cfg.get("checks") or {}))
    logger.info("shard %d/%d, %g primaries, %d threads, work %s (%.0f GB free), store %s (%.0f GB free)", index,
                n_shards, rob_cfg.num_primaries, rob_cfg.num_threads, settings.work_root,
                free_gb(Path(settings.work_root)), store, free_gb(store))
    done = run_batches(plan_df, entries, runner, bdl, rob_cfg, settings, checks, index, n_shards, max_batches)
    logger.info("shard %d finished %d batches this session", index, len(done))


@app.command()
def status(config: ConfigOpt = Path("scripts/config_beamlet_pool.yaml"), set_: SetOpt = None) -> None:
    """Batches done, beamlets stored per split, hours per batch."""
    cfg = _config(config, set_)
    store = _store(cfg)
    plan_df = pd.read_csv(store / "plan" / "plan.csv")
    print(json.dumps({**progress(plan_df, store), "store_free_gb": round(free_gb(store), 1)}, indent=2,
                     default=str))


if __name__ == "__main__":
    app()
