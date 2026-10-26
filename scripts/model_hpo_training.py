"""Hyperparameter optimisation, confirmation, final training and the one test
evaluation of a single ADoTA model, on patient-level record sets.

    # 1. search: Optuna TPE with median pruning, one worker per GPU (long)
    uv run python scripts/model_hpo_training.py search --config scripts/config_model_hpo_training.yaml
    # 2. confirm: the top-k trials trained at full length, the winner chosen by a fixed rule
    uv run python scripts/model_hpo_training.py confirm --run-dir <output_dir>/model_hpo_<ts>
    # 3. final: the winner trained on the final training records
    uv run python scripts/model_hpo_training.py final --run-dir <run>
    # 4. test: the final model evaluated once on the test records
    uv run python scripts/model_hpo_training.py test --run-dir <run>

Every stage writes into one run directory; the configuration is frozen there by
``search`` and read back by the later stages, so a run cannot change its inputs
half way. ``search --run-dir <run>`` resumes an interrupted search.

Every run trains from scratch with a linear warmup and a cosine decay of the
learning rate to zero over its own epochs, and keeps its last checkpoint
(:mod:`src.model_search.runs`). Guide: ``scripts/docs/model_hpo_training.md``.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
import typer
import yaml

from src.active_learning.retrospective.trainer import validation_loader
from src.active_learning.retrospective.validation import (
    MetricSettings,
    draw_eval_subsample,
    evaluate_metrics,
    summarise_metrics,
)
from src.evaluation.cli import (
    apply_set_overrides,
    load_yaml_config,
    resolve_device,
    setup_logging,
    setup_run_directory,
)
from src.model_search.runs import RunSpec, prepare_ids, resolve_ids, scan_overlap, select_winner, train_run
from src.model_search.space import build_candidate, rejection, validate_space
from src.model_search.study import (
    SearchContext,
    fail_stale_trials,
    make_objective,
    open_study,
    stop_at_budget,
    top_trials,
)
from src.schemas.configs import TrainingConfig
from src.training.checkpoints import CheckpointManager
from src.training.factory import build_adota_model
from src.training.run_dir import write_manifest

app = typer.Typer(help="HPO, confirmation, final training and test of one ADoTA model.", add_completion=False)
logger = logging.getLogger("model_hpo_training")
SETS = ("search_train", "search_val", "final_train", "test")
# A run directory frozen before the schedule became a config key trained with
# "cosine_per_cycle" (its first warmup epoch and its last epoch at a learning rate
# of 0); without the key, that run keeps its schedule so a resume reproduces it.
LEGACY_LR_SCHEDULE = "cosine_per_cycle"
STAGES = ("ids", "search", "confirm", "final", "test")


# --------------------------------------------------------------------------- configuration

def _read_run(run: Path) -> Dict[str, Any]:
    return yaml.safe_load((run / "config.yaml").read_text())


def base_config(cfg: Dict[str, Any]) -> TrainingConfig:
    """The fixed part of every run: the ``training`` block on the dataset."""
    known = {f.name for f in fields(TrainingConfig)}
    unknown = sorted(set(cfg["training"]) - known)
    if unknown:
        raise typer.BadParameter(f"training: unknown TrainingConfig fields {unknown}")
    config = TrainingConfig(**cfg["training"], dataset_path=cfg["dataset_path"],
                            excluded_indexes_file=cfg.get("exclude_indexes_path") or "",
                            max_energy_mev=cfg.get("max_energy_mev"))
    if config.adaptive_after_epoch < 10 ** 4:
        raise typer.BadParameter("training.adaptive_after_epoch must keep the loss weights static, "
                                 "or validation losses are not comparable between runs")
    return config


def metric_settings(cfg: Dict[str, Any], gamma_params: Optional[Dict] = None) -> MetricSettings:
    t = cfg["training"]
    return MetricSettings(scale=t["scale"], gamma_params=dict(gamma_params or cfg["gamma_params"]),
                          resolution_mm=tuple(cfg["gamma_resolution_mm"]),
                          gamma_cutoff_percent=float(cfg["gamma_cutoff_percent"]),
                          gamma_backend=str(cfg["gamma_backend"]),
                          lps_dx_mm=float(t.get("lps_dx_mm", 2.0)), lps_dy_mm=float(t.get("lps_dy_mm", 2.0)))


def lr_schedule(cfg: Dict[str, Any], stage: str) -> str:
    """The schedule of ``stage`` (search, confirm, final): the stage's own
    ``lr_schedule`` if set, else the run's, else the legacy one."""
    return str(cfg.get(stage, {}).get("lr_schedule") or cfg.get("lr_schedule") or LEGACY_LR_SCHEDULE)


STAGE_OVERRIDES = ("max_grad_norm", "warmup_epochs")


def stage_overrides(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    """The run-level settings a stage replaces in the confirmed configuration
    (``max_grad_norm``, ``warmup_epochs`` under ``cfg[stage]``), only those set."""
    section = cfg.get(stage) or {}
    return {k: section[k] for k in STAGE_OVERRIDES if section.get(k) is not None}


def _ids(run: Path, name: str) -> List[str]:
    return pd.read_csv(run / "ids" / f"{name}.csv", dtype=str)["sample_id"].tolist()


def _on_root_disk(path: Path) -> bool:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return os.stat(probe).st_dev == os.stat("/").st_dev


def _launch(args: List[str], device: int, log_path: Path) -> subprocess.Popen:
    """One worker process on one GPU, its output in ``log_path``."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}
    log = open(log_path, "a")
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *args], env=env,
                            stdout=log, stderr=subprocess.STDOUT, cwd=str(Path(__file__).resolve().parents[1]))


def _prepare(run: Path, cfg: Dict[str, Any]) -> None:
    """Resolve the record sets, apply the exclusions and the energy cut, check that
    no scan is shared between sets that must be disjoint, and write ``ids/``."""
    base = Path(cfg["partition_dir"])
    listed = {name: resolve_ids(cfg["id_sets"][name], base) for name in SETS}
    sets, counts = prepare_ids(listed, Path(cfg["dataset_path"]), Path(cfg["exclude_indexes_path"]),
                               cfg.get("max_energy_mev"), cfg["training"]["scale"])
    table = cfg["scan_table"]
    scans = pd.read_csv(base / table["table"], dtype=str).set_index("sample_id")[table["column"]].to_dict()
    overlap = scan_overlap(sets, scans, [("search_train", "search_val"), ("search_train", "test"),
                                         ("search_val", "test"), ("final_train", "test")])
    shared = {k: v for k, v in overlap.items() if v}
    if shared:
        raise RuntimeError(f"record sets share scans: {shared}")
    if cfg.get("max_records"):  # smoke tests only: a seeded subsample of every set
        sets = {n: draw_eval_subsample(ids, int(cfg["max_records"]), int(cfg["subsample_seed"]))
                for n, ids in sets.items()}
        counts["kept"] = [len(sets[n]) for n in counts["set"]]
    counts["scans"] = [len({scans.get(i) for i in sets[n]}) for n in counts["set"]]
    for name, ids in sets.items():
        pd.DataFrame({"sample_id": ids}).to_csv(run / "ids" / f"{name}.csv", index=False)
    subsample = draw_eval_subsample(sets["search_val"], int(cfg["epoch_val_size"]), int(cfg["subsample_seed"]))
    monitor = draw_eval_subsample(sets["final_train"], int(cfg["epoch_val_size"]), int(cfg["subsample_seed"]))
    pd.DataFrame({"sample_id": subsample}).to_csv(run / "ids" / "search_val_subsample.csv", index=False)
    pd.DataFrame({"sample_id": monitor}).to_csv(run / "ids" / "final_train_monitor.csv", index=False)
    counts.to_csv(run / "ids" / "counts.csv", index=False)
    logger.info("record sets (no scan shared between train, validation and test):\n%s", counts.to_string(index=False))


def _study(run: Path, cfg: Dict[str, Any], worker: int = 0):
    s = cfg["search"]
    return open_study(run / "search" / "study.journal", seed=int(s["sampler_seed"]) + worker,
                      n_startup_trials=int(s["n_startup_trials"]),
                      pruner_startup_trials=int(s["pruner_startup_trials"]),
                      pruner_warmup_steps=int(s["pruner_warmup_steps"]))


def _enqueue_fixed_trials(study, cfg: Dict[str, Any]) -> None:
    """Queue the configurations of ``search.fixed_trials`` ahead of the sampler's
    own, each checked against the space's parameters and the cap first."""
    for entry in cfg["search"].get("fixed_trials") or []:
        params, label = dict(entry["params"]), str(entry["label"])
        missing = sorted(set(cfg["space"]) - set(params) - {"weight_standardization"})
        extra = sorted(set(params) - set(cfg["space"]))
        if missing or extra:
            raise typer.BadParameter(f"fixed trial {label}: no value for {missing}; not in the space {extra}")
        reason = rejection(build_candidate(base_config(cfg), params), cfg["search"].get("max_params"))
        if reason:
            raise typer.BadParameter(f"fixed trial {label}: {reason}")
        study.enqueue_trial(params, user_attrs={"fixed": label})
        logger.info("fixed trial queued: %s", label)


def write_trials_table(run: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    frame = _study(run, cfg).trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs",
                                                     "duration"))
    frame.to_csv(run / "search" / "trials.csv", index=False)
    return frame


# --------------------------------------------------------------------------- stages

@app.command()
def search(config: Annotated[Optional[Path], typer.Option(help="YAML config (new run).")] = None,
           run_dir: Annotated[Optional[Path], typer.Option(help="Resume this run's search.")] = None,
           set_: Annotated[Optional[List[str]], typer.Option("--set", help="KEY=VALUE override (new run).")] = None,
           ) -> None:
    """Create the run (or resume it) and run one search worker per GPU until the
    budget of trained trials is reached."""
    if run_dir is None:
        if config is None:
            raise typer.BadParameter("give --config for a new run or --run-dir to resume one")
        cfg = apply_set_overrides(load_yaml_config(config), set_ or [])
        validate_space(cfg["space"])
        parent = Path(cfg["output_dir"])
        if _on_root_disk(parent):
            raise typer.BadParameter(f"output_dir {parent} is on the root disk; use /scratch")
        run = setup_run_directory(parent, prefix="model_hpo_", subdirs=STAGES)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        setup_logging(run, log_filename="model_hpo_training.log")
        write_manifest(run, config=base_config(cfg), dataset_path=Path(cfg["dataset_path"]),
                       excluded_indexes_path=Path(cfg["exclude_indexes_path"]), extra={"run_config": cfg})
        _prepare(run, cfg)
        _enqueue_fixed_trials(_study(run, cfg), cfg)
    else:
        run = Path(run_dir)
        cfg = _read_run(run)
        setup_logging(run, log_filename="model_hpo_training.log")
    study = _study(run, cfg)
    stale = fail_stale_trials(study)
    if stale:
        logger.warning("trials left RUNNING by a stopped worker, marked FAIL: %s", stale)
    logger.info("run directory %s; search workers on GPUs %s", run, cfg["devices"])
    workers = [_launch(["worker", "--run-dir", str(run), "--worker", str(k)], device,
                       run / "search" / f"worker_{k}.log") for k, device in enumerate(cfg["devices"])]
    codes = [w.wait() for w in workers]
    table = write_trials_table(run, cfg)
    logger.info("search finished, worker exit codes %s; %d trials, table in %s", codes, len(table),
                run / "search" / "trials.csv")


@app.command(hidden=True)
def worker(run_dir: Annotated[Path, typer.Option()], worker: Annotated[int, typer.Option()]) -> None:
    """One search worker on the GPU it was given."""
    run = Path(run_dir)
    cfg = _read_run(run)
    setup_logging(run / "search", log_filename=f"worker_{worker}_trials.log")
    s = cfg["search"]
    ctx = SearchContext(base=base_config(cfg), space=cfg["space"], max_params=s.get("max_params"),
                        epochs=int(s["epochs"]), train_ids=_ids(run, "search_train"),
                        epoch_val_ids=_ids(run, "search_val_subsample"), final_val_ids=_ids(run, "search_val"),
                        trials_dir=run / "search" / "trials", device=resolve_device(0),
                        settings=metric_settings(cfg), target_trained=int(s["n_trials"]),
                        lr_schedule=lr_schedule(cfg, "search"))
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    _study(run, cfg, worker).optimize(make_objective(ctx), callbacks=[stop_at_budget(ctx.target_trained)],
                                      catch=(RuntimeError,), gc_after_trial=True)


@app.command()
def confirm(run_dir: Annotated[Path, typer.Option(help="The run directory.")]) -> None:
    """Train the top-k trials at full length and choose the winner."""
    run = Path(run_dir)
    cfg = _read_run(run)
    setup_logging(run, log_filename="model_hpo_training.log")
    c = cfg["confirm"]
    top = top_trials(_study(run, cfg), int(c["top_k"]))
    if not top:
        raise RuntimeError("no completed search trial")
    logger.info("confirming trials %s on GPUs %s", [t.number for t in top], cfg["devices"])
    queue, rows = list(top), []
    while queue:
        batch = [queue.pop(0) for _ in cfg["devices"] if queue]
        procs = [_launch(["candidate", "--run-dir", str(run), "--trial", str(t.number), "--stage", "confirm"],
                         device, run / "confirm" / f"trial_{t.number:04d}.log")
                 for t, device in zip(batch, cfg["devices"])]
        for t, p in zip(batch, procs):
            if p.wait() != 0:
                raise RuntimeError(f"confirmation of trial {t.number} failed; see its log in {run / 'confirm'}")
    for t in top:
        result = json.loads((run / "confirm" / f"trial_{t.number:04d}" / "result.json").read_text())
        rows.append({"label": f"trial_{t.number:04d}", "trial": t.number, "search_value": t.value,
                     "final_loss": result["final_val_loss"]["loss_combined_mean"],
                     "gpr_mean": result["final_val_metrics"]["gpr_mean"],
                     "gpr_p05": result["final_val_metrics"]["gpr_p05"],
                     "mape_pct_mean": result["final_val_metrics"]["mape_pct_mean"],
                     "n_params": t.user_attrs.get("n_params"), "params": json.dumps(t.params)})
    pd.DataFrame(rows).to_csv(run / "confirm" / "results.csv", index=False)
    winner = select_winner(rows, float(c["gpr_guard_pp"]))
    (run / "confirm" / "selection.json").write_text(json.dumps(winner, indent=2, default=str))
    logger.info("winner: %s (%s)", winner["label"], winner["rule"])


@app.command(hidden=True)
def candidate(run_dir: Annotated[Path, typer.Option()], trial: Annotated[int, typer.Option()],
              stage: Annotated[str, typer.Option(help="confirm | final")]) -> None:
    """Train one search trial's configuration for a later stage."""
    run = Path(run_dir)
    cfg = _read_run(run)
    frozen = next(t for t in _study(run, cfg).get_trials(deepcopy=False) if t.number == trial)
    built = build_candidate(base_config(cfg), frozen.params)
    if stage == "confirm":
        out = run / "confirm" / f"trial_{trial:04d}"
        spec = RunSpec(label=f"confirm_trial_{trial:04d}", config=built.config, warmup_epochs=built.warmup_epochs,
                       epochs=int(cfg["confirm"]["epochs"]), train_ids=_ids(run, "search_train"),
                       epoch_val_ids=_ids(run, "search_val_subsample"), final_val_ids=_ids(run, "search_val"),
                       final_metrics=True, lr_schedule=lr_schedule(cfg, "confirm"))
    elif stage == "final":
        out = run / "final"
        # No held-out records: the per-epoch loss is on a subsample of the training
        # records (in-sample, for monitoring only); nothing is selected on it.
        over = stage_overrides(cfg, "final")
        config = replace(built.config, max_grad_norm=over.get("max_grad_norm", built.config.max_grad_norm))
        spec = RunSpec(label=f"final_trial_{trial:04d}", config=config,
                       warmup_epochs=int(over.get("warmup_epochs", built.warmup_epochs)),
                       epochs=int(cfg["final"]["epochs"]), train_ids=_ids(run, "final_train"),
                       epoch_val_ids=_ids(run, "final_train_monitor"), lr_schedule=lr_schedule(cfg, "final"))
    else:
        raise typer.BadParameter(f"unknown stage {stage!r}")
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, log_filename="run.log")
    train_run(spec, out, resolve_device(0), metric_settings(cfg))


@app.command()
def final(run_dir: Annotated[Path, typer.Option(help="The run directory.")],
          device: Annotated[int, typer.Option(help="GPU index.")] = 0,
          max_grad_norm: Annotated[Optional[float], typer.Option(
              help="Clip the global gradient norm to this value (a deviation from the confirmed run).")] = None,
          warmup_epochs: Annotated[Optional[int], typer.Option(
              help="Replace the winner's warmup (a deviation from the confirmed run).")] = None,
          restart: Annotated[bool, typer.Option(
              help="Move an existing final/ aside to final_aborted_<ts>/ and train again.")] = False,
          ) -> None:
    """Train the confirmed winner on the final training records.

    ``--max-grad-norm`` and ``--warmup-epochs`` change the confirmed configuration;
    they are written into the run's frozen ``config.yaml`` under ``final:`` and
    logged, so the deviation is on record. A ``final/`` with checkpoints is never
    overwritten: ``--restart`` moves it to ``final_aborted_<ts>/`` first.
    """
    run = Path(run_dir)
    setup_logging(run, log_filename="model_hpo_training.log")
    winner = json.loads((run / "confirm" / "selection.json").read_text())
    out = run / "final"
    if (out / "checkpoints").exists() and any((out / "checkpoints").iterdir()):
        if not restart:
            raise typer.BadParameter(f"{out} already holds a final training; pass --restart to move it aside")
        aside = run / f"final_aborted_{datetime.now():%Y%m%d_%H%M%S}"
        out.rename(aside)
        out.mkdir()
        logger.warning("previous final training moved to %s", aside)
    changes = {k: v for k, v in (("max_grad_norm", max_grad_norm), ("warmup_epochs", warmup_epochs))
               if v is not None}
    if changes:
        cfg = _read_run(run)
        cfg.setdefault("final", {}).update(changes)
        (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        logger.warning("final stage deviates from the confirmed configuration: %s (written to config.yaml)", changes)
    logger.info("final training of trial %s on %s records", winner["trial"], len(_ids(run, "final_train")))
    code = _launch(["candidate", "--run-dir", str(run), "--trial", str(winner["trial"]), "--stage", "final"],
                   device, run / "final" / "launch.log").wait()
    if code != 0:
        raise RuntimeError(f"final training failed; see {run / 'final'}")
    logger.info("final model: %s", run / "final" / "checkpoints" / "last.pth")


@app.command()
def test(run_dir: Annotated[Path, typer.Option(help="The run directory.")],
         device: Annotated[int, typer.Option(help="GPU index.")] = 0,
         force: Annotated[bool, typer.Option(help="Evaluate again although the test set was used.")] = False,
         ) -> None:
    """Evaluate the final model once on the test records, at every gamma criterion
    of ``test_gamma``."""
    run = Path(run_dir)
    cfg = _read_run(run)
    setup_logging(run, log_filename="model_hpo_training.log")
    marker = run / "test" / "summary.json"
    if marker.exists() and not force:
        raise typer.BadParameter(f"the test set was already evaluated ({marker}); it is used once")
    winner = json.loads((run / "confirm" / "selection.json").read_text())
    frozen = next(t for t in _study(run, cfg).get_trials(deepcopy=False) if t.number == int(winner["trial"]))
    config = build_candidate(base_config(cfg), frozen.params).config
    torch_device = resolve_device(device)
    model = build_adota_model(config, torch_device)
    CheckpointManager.load_weights_only(run / "final" / "checkpoints" / "last.pth", model=model,
                                        device=torch_device)
    ids = _ids(run, "test")
    summary: Dict[str, Any] = {"trial": winner["trial"], "n_test": len(ids), "criteria": {}}
    for gamma in cfg["test_gamma"]:
        params = {**cfg["gamma_params"], "dose_percent_threshold": gamma[0], "distance_mm_threshold": gamma[1]}
        settings = metric_settings(cfg, params)
        frame = evaluate_metrics(model, validation_loader(config, ids), ids, device=torch_device,
                                 settings=settings, desc=f"test {settings.gamma_label}")
        frame.to_csv(run / "test" / f"test_{settings.gamma_label}.csv", index=False)
        summary["criteria"][settings.gamma_label] = summarise_metrics(frame)
        logger.info("test %s: %s", settings.gamma_label, summary["criteria"][settings.gamma_label])
    marker.write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    app()
