"""The retrospective active-learning benchmark on the training HDF5 (EXP-0009).

Three stages, run in this order::

    uv run python scripts/al_retro_loop.py splits --config scripts/config_al_retro_loop.yaml
    uv run python scripts/al_retro_loop.py cycle0 --config scripts/config_al_retro_loop.yaml --device-index 0
    uv run python scripts/al_retro_loop.py run --config scripts/config_al_retro_loop.yaml \
        --strategy random --cycle0-run /scratch/.../train_<ts>_al_EXP-0009_cycle0 --device-index 0

``splits`` applies the exclusion list and writes the validation set, the training
split and the cycle-0 set to CSV once. ``cycle0`` trains the shared baseline from
random weights. ``run`` is one strategy: it resumes from the cycle-0 checkpoint,
and at every cycle scores the remaining pool, selects ``N`` records, adds them,
trains ``epochs_per_cycle`` epochs and validates. One process is one strategy;
the three strategies are three independent runs, compared afterwards by
``scripts/al_compare.py``.

Precedence is CLI > YAML > built-in defaults through ``merge_config``. A run
directory passed as ``--resume-dir`` continues where it stopped: completed cycles
are skipped and an interrupted one restarts from its ``last.pth``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Optional

import typer

from src.active_learning.retrospective.loop import RetroConfig, prepare_splits, run_cycle0, run_strategy
from src.active_learning.retrospective.sampling import available_strategies
from src.evaluation.cli import load_yaml_config, merge_config
from src.training.logging_utils import log_banner, setup_training_logging, silence_pymedphys
from src.training.run_dir import setup_training_run_directory

app = typer.Typer(help="Retrospective active-learning benchmark: splits, cycle 0, one strategy.",
                  no_args_is_help=True)
logger = logging.getLogger("al_retro_loop")

ConfigOption = Annotated[Path, typer.Option(help="Retrospective benchmark YAML config.")]
DEFAULT_CONFIG = Path("scripts/config_al_retro_loop.yaml")


def _config(config: Path, **overrides) -> RetroConfig:
    raw = load_yaml_config(config)
    merged = merge_config(overrides, raw, defaults=asdict(RetroConfig()))
    return RetroConfig.from_dict(merged)


def _start_run(cfg: RetroConfig, name: str, resume_dir: Optional[Path]) -> Path:
    run_dir = Path(resume_dir) if resume_dir else setup_training_run_directory(
        Path(cfg.runs_dir), name)
    setup_training_logging(run_dir, start_time=time.time())
    silence_pymedphys()
    log_banner(f"ADoTA RETROSPECTIVE ACTIVE LEARNING ({cfg.experiment})")
    logger.info("run dir: %s", run_dir)
    return run_dir


@app.command()
def splits(
    config: ConfigOption = DEFAULT_CONFIG,
    max_records: Annotated[Optional[int], typer.Option(
        help="Subsample D before splitting. Smoke tests only.")] = None,
    splits_dir: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    """Apply the exclusion list; write V, T and the cycle-0 set to CSV."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    cfg = _config(config, max_records=max_records,
                  splits_dir=str(splits_dir) if splits_dir else None)
    summary = prepare_splits(cfg)
    typer.echo(f"|D| after exclusion: {summary['n_after_exclusion']} "
               f"(file {summary['n_records_file']}, listed {summary['n_excluded_listed']})")
    typer.echo(f"|V| = {summary['n_validation']}  |T| = {summary['n_training']}  "
               f"cycle-0 = {summary['n_initial']}  pool = {summary['n_pool']}")
    typer.echo(f"splits: {cfg.splits_dir}  fingerprint {summary['fingerprint'][:12]}")


@app.command()
def cycle0(
    config: ConfigOption = DEFAULT_CONFIG,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA device; -1 for CPU.")] = None,
    epochs_per_cycle: Annotated[Optional[int], typer.Option()] = None,
    seed: Annotated[Optional[int], typer.Option()] = None,
    runs_dir: Annotated[Optional[Path], typer.Option()] = None,
    resume_dir: Annotated[Optional[Path], typer.Option(
        help="An existing cycle-0 run directory to continue.")] = None,
) -> None:
    """Train the shared cycle-0 baseline from random weights."""
    cfg = _config(config, device_index=device_index, epochs_per_cycle=epochs_per_cycle,
                  seed=seed, runs_dir=str(runs_dir) if runs_dir else None)
    run_dir = _start_run(cfg, f"al_{cfg.experiment}_cycle0_seed{cfg.seed}", resume_dir)
    checkpoint = run_cycle0(cfg, run_dir)
    typer.echo(f"cycle-0 run: {run_dir}\ncheckpoint: {checkpoint}")


@app.command()
def run(
    config: ConfigOption = DEFAULT_CONFIG,
    cycle0_run: Annotated[Path, typer.Option(help="The cycle-0 run directory.")] = ...,
    strategy: Annotated[Optional[str], typer.Option(
        help=f"One of {', '.join(available_strategies())}.")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA device; -1 for CPU.")] = None,
    n_cycles: Annotated[Optional[int], typer.Option()] = None,
    epochs_per_cycle: Annotated[Optional[int], typer.Option()] = None,
    seed: Annotated[Optional[int], typer.Option()] = None,
    runs_dir: Annotated[Optional[Path], typer.Option()] = None,
    resume_dir: Annotated[Optional[Path], typer.Option(
        help="An existing strategy run directory to continue.")] = None,
) -> None:
    """Run one strategy from the cycle-0 checkpoint."""
    cfg = _config(config, strategy=strategy, device_index=device_index, n_cycles=n_cycles,
                  epochs_per_cycle=epochs_per_cycle, seed=seed,
                  runs_dir=str(runs_dir) if runs_dir else None)
    if cfg.strategy not in available_strategies():
        raise typer.BadParameter(f"unknown strategy {cfg.strategy!r}; "
                                 f"registered: {available_strategies()}")
    run_dir = _start_run(cfg, f"al_{cfg.experiment}_{cfg.strategy}_seed{cfg.seed}", resume_dir)
    cycles = run_strategy(cfg, run_dir, Path(cycle0_run))
    last = cycles[-1].get("metrics_full") or {}
    typer.echo(f"strategy {cfg.strategy}: {len(cycles) - 1} cycles -> {run_dir}\n"
               f"final GPR {last.get('gpr_mean', float('nan')):.4f} "
               f"(p05 {last.get('gpr_p05', float('nan')):.4f}), "
               f"MAPE {last.get('mape_pct_mean', float('nan')):.2f}%")


if __name__ == "__main__":
    app()
