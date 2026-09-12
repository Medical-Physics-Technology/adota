"""Run one arm of the active-learning loop: sample, label, retrain, validate, repeat.

    # what the first cycle would select and what it would cost, no simulation:
    uv run python scripts/al_loop.py --config scripts/config_al.yaml --strategy score --dry-run
    # the real thing:
    uv run python scripts/al_loop.py --config scripts/config_al.yaml --strategy score \
        --device-index 1 --run-name score_arm

One process is one arm. Two arms at equal Monte Carlo budget -- ``--strategy random``
and ``--strategy score`` -- are the comparison; run them side by side on separate GPUs,
with ``num_threads`` in the config set so the two Monte Carlo phases share the cores.

Everything is resumable: a rerun with the same run directory skips the cycles whose
manifest is already written, and the Monte Carlo underneath skips the beamlets already
simulated.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer

from src.active_learning.candidates import score_pool
from src.active_learning.config import (
    candidate_config_from_dict,
    gamma_params_from_config,
    mc_from_config,
    scale_from_config,
)
from src.active_learning.loop import LoopConfig, run_loop
from src.active_learning.oracle import batch_cost_estimate
from src.active_learning.pool import read_pool
from src.active_learning.sampling import STRATEGIES, select, selection_fingerprint
from src.adota.config import load_yaml_config
from src.training.logging_utils import silence_pymedphys

# force=True: importing pymedphys configures the root logger, which would make a
# plain basicConfig a no-op and leak per-record DEBUG lines into the run log.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
silence_pymedphys()
logger = logging.getLogger("al_loop")
app = typer.Typer(help="Run one arm of the active-learning loop.")


def _loop_config(cfg: dict, **overrides) -> LoopConfig:
    raw = {**cfg.get("loop", {}), **{k: v for k, v in overrides.items() if v is not None}}
    raw["pool_csv"] = cfg["pool"]["pool_csv"]
    raw["validation_manifest"] = cfg["validation_set"]["manifest"]
    raw["output_root"] = cfg["robustness"]["output_root"]
    raw["scale"] = scale_from_config(cfg)
    raw["gamma_params"] = gamma_params_from_config(cfg)
    known = set(LoopConfig.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown loop config keys {sorted(unknown)}")
    return LoopConfig(**raw)


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="Active-learning YAML config.")] = Path(
        "scripts/config_al.yaml"),
    strategy: Annotated[str, typer.Option(help=f"One of {', '.join(STRATEGIES)}.")] = "score",
    run_name: Annotated[Optional[str], typer.Option(help="Run directory name.")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA device.")] = None,
    n_cycles: Annotated[Optional[int], typer.Option()] = None,
    beamlets_per_cycle: Annotated[Optional[int], typer.Option()] = None,
    runs_dir: Annotated[Optional[Path], typer.Option(
        help="Base directory for the run.")] = None,
    dry_run: Annotated[bool, typer.Option(
        help="Score and select one cycle, report the cost, simulate nothing.")] = False,
) -> None:
    cfg = load_yaml_config(config)
    loop_cfg = _loop_config(cfg, strategy=strategy, device_index=device_index,
                            n_cycles=n_cycles, beamlets_per_cycle=beamlets_per_cycle,
                            name=run_name)
    if loop_cfg.strategy not in STRATEGIES:
        raise typer.BadParameter(f"unknown strategy {loop_cfg.strategy!r}")
    rob_cfg, runner, bdl, bdl_path = mc_from_config(cfg)
    cand_cfg = candidate_config_from_dict(cfg.get("candidates", {}))
    base = Path(runs_dir or cfg.get("runs_dir", "/scratch/mstryja/adota_runs"))
    run_dir = base / f"al_{loop_cfg.name}_{loop_cfg.strategy}"

    if dry_run:
        pool = read_pool(Path(loop_cfg.pool_csv), role="pool")
        rng = np.random.default_rng([loop_cfg.seed, 0])
        n_cts = min(loop_cfg.n_cts_per_cycle, len(pool))
        picked = sorted(rng.choice(len(pool), size=n_cts, replace=False))
        entries = [pool[int(i)] for i in picked]
        table = score_pool(entries, cand_cfg, rob_cfg, bdl_path,
                           n_workers=loop_cfg.n_score_workers,
                           prefix=loop_cfg.beamlet_prefix, version=loop_cfg.beamlet_version)
        chosen = select(table, loop_cfg.strategy, loop_cfg.beamlets_per_cycle, rng=rng,
                        score_column=loop_cfg.score_column, alpha=loop_cfg.score_alpha,
                        max_per_patient_frac=loop_cfg.max_per_patient_frac)
        logger.info("dry run: %s", selection_fingerprint(chosen, loop_cfg.score_column))
        logger.info("dry run cost: %s", batch_cost_estimate(chosen))
        logger.info("nothing simulated, nothing trained.")
        raise typer.Exit()

    logger.info("run dir: %s", run_dir)
    manifests = run_loop(loop_cfg, run_dir, cand_cfg, rob_cfg, runner, bdl, bdl_path)
    total_mc = sum(m["mc_seconds"] for m in manifests)
    logger.info("arm %s finished: %d cycles, %.1f h of Monte Carlo -> %s",
                loop_cfg.strategy, len(manifests), total_mc / 3600.0,
                run_dir / "metrics.jsonl")


if __name__ == "__main__":
    app()
