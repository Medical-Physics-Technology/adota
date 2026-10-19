"""Hyperparameter optimisation of the EXP-0014 members: one search per sequence
mixer (transformer, mlstm, none), the same budget for each, on R only.

    uv run python scripts/active_learning/ensemble_hpo.py run \
        --config scripts/active_learning/config_ensemble_hpo.yaml
    # resume an interrupted run (finished segments are kept):
    uv run python scripts/active_learning/ensemble_hpo.py run --resume <hpo_dir>

``run`` is the driver. It draws ``n_trials`` scrambled-Sobol configurations per
mixer under the parameter cap (``param_cap_factor`` x the reference model, v11),
checks that every trial fits in GPU memory, then runs synchronous successive
halving: every trial trains ``min_epochs`` on the HPO training scans, the best
``1/eta`` continue from their checkpoint to ``min_epochs * eta``, and so on for
``n_rungs`` rungs. Trials are ranked by the best validation loss (static loss
weights) on a fixed, seeded subsample of the HPO validation scans. The trials
of the top rung, and the transformer trial chosen as C1, then get the full
metric set (gamma, MAPE, RDE, dR80) on the same subsample.

Each training segment, memory check and finalist evaluation runs as its own
worker process (``segment``, ``memcheck``, ``finalist``) on one GPU of
``devices``, so a crash or out-of-memory error ends one trial, not the search.
The driver rebuilds its state from the trial directories every few seconds,
which makes ``--resume`` exact. Everything is written under ``output_dir`` (on
``/scratch``): the trained checkpoints of every trial, ``results.csv`` (one row
per trial: parameters, parameter count, objective per rung), ``selection.json``
(C0, C1, X0, U0 with their configurations, checkpoints and finalist metrics),
``summary.json``, ``manifest.json`` and the resolved ``config.yaml``.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import typer
import yaml

from src.active_learning.retrospective.validation import MetricSettings, draw_eval_subsample
from src.datasets.beamlet_h5 import sha256_of
from src.evaluation.cli import (
    apply_set_overrides,
    load_yaml_config,
    on_root_filesystem,
    setup_logging,
)
from src.hpo.halving import promote, rung_budgets, schedule, total_epochs
from src.hpo.selection import finalists, member_roles, results_table
from src.hpo.space import draw_trials
from src.hpo.trial import (
    count_parameters,
    evaluate_finalist,
    read_segment,
    run_segment,
    trial_config,
    write_segment,
)
from src.training.factory import build_adota_model
from src.training.run_dir import save_resolved_config, write_manifest

app = typer.Typer(help="Per-mixer HPO for the EXP-0014 members.", add_completion=False)
logger = logging.getLogger("ensemble_hpo")
POLL_SECONDS = 10


# ── Setup ───────────────────────────────────────────────────────────────────


def _ids(path: Path) -> List[str]:
    return pd.read_csv(path, dtype=str)["sample_id"].tolist()


def _hpo_config(hpo_dir: Path) -> Dict[str, Any]:
    return yaml.safe_load((hpo_dir / "config.yaml").read_text())


def _trials(hpo_dir: Path) -> List[Dict[str, Any]]:
    return json.loads((hpo_dir / "trials.json").read_text())


def _config_for(cfg: Dict[str, Any], trial: Dict[str, Any]):
    return trial_config(cfg["training"], trial["params"], dataset_path=cfg["dataset_path"],
                        excluded_indexes_file=cfg["exclude_indexes_path"], mixer=trial["mixer"])


def _settings(cfg: Dict[str, Any]) -> MetricSettings:
    t = cfg["training"]
    return MetricSettings(scale=t["scale"], gamma_params=dict(cfg["gamma_params"]),
                          resolution_mm=tuple(cfg["gamma_resolution_mm"]),
                          gamma_cutoff_percent=float(cfg["gamma_cutoff_percent"]),
                          gamma_backend=str(cfg["gamma_backend"]),
                          lps_dx_mm=float(t.get("lps_dx_mm", 2.0)), lps_dy_mm=float(t.get("lps_dy_mm", 2.0)))


def prepare(cfg: Dict[str, Any], out: Path) -> None:
    """Write the id lists and draw the trials of a new run."""
    partition = Path(cfg["partition_dir"])
    train_ids = _ids(partition / "hpo" / "hpo_train_ids.csv")
    if cfg.get("max_train_records"):
        rng = np.random.RandomState(int(cfg["subsample_seed"]))
        keep = sorted(rng.choice(len(train_ids), int(cfg["max_train_records"]), replace=False))
        train_ids = [train_ids[i] for i in keep]
    val_ids = draw_eval_subsample(_ids(partition / "hpo" / "hpo_val_ids.csv"),
                                  int(cfg["val_subsample_size"]), int(cfg["subsample_seed"]))
    (out / "ids").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sample_id": train_ids}).to_csv(out / "ids" / "train_ids.csv", index=False)
    pd.DataFrame({"sample_id": val_ids}).to_csv(out / "ids" / "val_ids.csv", index=False)

    reference = json.loads(Path(cfg["reference_hyperparams"]).read_text())
    reference.pop("output_shape", None)
    ref_config = trial_config(cfg["training"], {}, dataset_path=cfg["dataset_path"],
                              excluded_indexes_file=cfg["exclude_indexes_path"], mixer="transformer")
    ref_config = replace(ref_config, **{k: v for k, v in reference.items()
                                        if hasattr(ref_config, k) and k != "input_shape"})
    reference_params = count_parameters(ref_config)
    cap = int(float(cfg["param_cap_factor"]) * reference_params)
    logger.info("reference model %s: %d parameters; cap %d", cfg["reference_hyperparams"],
                reference_params, cap)

    trials = []
    for k, mixer in enumerate(cfg["mixers"]):
        space = {**cfg["space"]["shared"], **(cfg["space"]["mixers"].get(mixer) or {})}

        def count(params, mixer=mixer):
            return count_parameters(trial_config(cfg["training"], params, dataset_path=cfg["dataset_path"],
                                                 excluded_indexes_file=cfg["exclude_indexes_path"],
                                                 mixer=mixer))

        fixed = (cfg.get("fixed_trials") or {}).get(mixer) or []
        drawn = draw_trials(mixer, space, int(cfg["n_trials"]), int(cfg["sobol_seed"]) + k, count, cap,
                            fixed=fixed)
        trials += [t.__dict__ for t in drawn]
        logger.info("%s: %d trials, parameters %d to %d", mixer, len(drawn),
                    min(t.n_params for t in drawn), max(t.n_params for t in drawn))
    (out / "trials.json").write_text(json.dumps(trials, indent=2, default=str))
    for t in trials:
        trial_dir = out / "trials" / t["trial_id"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "trial.json").write_text(json.dumps(t, indent=2, default=str))
        save_resolved_config(_config_for(cfg, t), trial_dir / "config.yaml")
    (out / "reference.json").write_text(json.dumps(
        {"path": cfg["reference_hyperparams"], "n_params": reference_params, "cap": cap}, indent=2))


# ── The driver ──────────────────────────────────────────────────────────────


class Workers:
    """One worker process per GPU at a time."""

    def __init__(self, devices: List[int], hpo_dir: Path):
        self.devices, self.hpo_dir = list(devices), hpo_dir
        self.running: Dict[int, Dict[str, Any]] = {}

    def free(self) -> List[int]:
        return [d for d in self.devices if d not in self.running]

    def busy(self, key: str) -> bool:
        return any(job["key"] == key for job in self.running.values())

    def launch(self, device: int, key: str, args: List[str], log_path: Path) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}
        handle = open(log_path, "a")
        command = [sys.executable, str(Path(__file__).resolve())] + args + ["--hpo-dir", str(self.hpo_dir)]
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=str(Path(__file__).resolve().parents[2]))
        self.running[device] = {"key": key, "process": process, "handle": handle, "args": args,
                                "log": log_path, "started": time.time()}
        logger.info("GPU %d: %s (pid %d)", device, key, process.pid)

    def reap(self) -> List[Dict[str, Any]]:
        done = []
        for device, job in list(self.running.items()):
            code = job["process"].poll()
            if code is None:
                continue
            job["handle"].close()
            job.update(device=device, code=code, seconds=time.time() - job["started"])
            done.append(job)
            del self.running[device]
            logger.info("GPU %d: %s finished, exit %d, %.0f min", device, job["key"], code,
                        job["seconds"] / 60)
        return done


def rung_state(trials: List[Dict[str, Any]], hpo_dir: Path, budgets: List[int], eta: int,
               mixers: List[str]) -> Dict[str, Any]:
    """From the segment files on disk: per mixer, the rung in progress and the
    trials still owing a segment there (``None`` once the mixer is finished)."""
    out = {}
    for mixer in mixers:
        active = [t["trial_id"] for t in trials if t["mixer"] == mixer]
        out[mixer] = None
        for r, budget in enumerate(budgets):
            segs = {t: read_segment(hpo_dir / "trials" / t, budget) for t in active}
            missing = [t for t, s in segs.items() if s is None]
            if missing:
                out[mixer] = {"rung": r, "budget": budget, "missing": missing}
                break
            objectives = {t: float("nan") if s.get("failed") else float(s["objective"])
                          for t, s in segs.items()}
            if r + 1 < len(budgets):
                active = promote(objectives, eta)
    return out


def _run_jobs(workers: Workers, jobs: List[Dict[str, Any]], on_done) -> None:
    """Dispatch ``jobs`` over the GPUs and wait for all of them."""
    queue = list(jobs)
    while queue or workers.running:
        for job in workers.reap():
            on_done(job)
        for device in workers.free():
            if not queue:
                break
            job = queue.pop(0)
            workers.launch(device, job["key"], job["args"], job["log"])
        time.sleep(POLL_SECONDS)


def drive(hpo_dir: Path, cfg: Dict[str, Any], devices: List[int]) -> None:
    trials = _trials(hpo_dir)
    budgets = rung_budgets(int(cfg["min_epochs"]), int(cfg["eta"]), int(cfg["n_rungs"]))
    eta, mixers = int(cfg["eta"]), list(cfg["mixers"])
    workers = Workers(devices, hpo_dir)
    by_id = {t["trial_id"]: t for t in trials}

    # 1. Every trial must fit in memory before any epoch is spent.
    pending = [t["trial_id"] for t in trials
               if not (hpo_dir / "trials" / t["trial_id"] / "memcheck.json").exists()]
    if pending and cfg.get("memcheck", True):
        chunks = [pending[i::len(devices)] for i in range(len(devices))]
        _run_jobs(workers, [{"key": f"memcheck x{len(c)}", "log": hpo_dir / "memcheck.log",
                             "args": ["memcheck"] + sum((["--trial", t] for t in c), [])}
                            for c in chunks if c], lambda job: None)
        failed = [t for t in pending
                  if not json.loads((hpo_dir / "trials" / t / "memcheck.json").read_text())["ok"]]
        if failed:
            raise RuntimeError(f"{len(failed)} trials do not fit in GPU memory: {failed}; "
                               "lower the batch size or the parameter cap")

    # 2. Successive halving, all mixers interleaved, state rebuilt from disk.
    def on_segment_done(job):
        trial, end = job["trial"], job["end_epoch"]
        if job["code"] != 0 and read_segment(hpo_dir / "trials" / trial, end) is None:
            write_segment(hpo_dir / "trials" / trial,
                          {"trial": trial, "end_epoch": end, "objective": float("nan"),
                           "failed": f"worker exit {job['code']}; see {job['log']}"})
            logger.warning("%s failed at the %d-epoch rung; it ranks last", trial, end)

    while True:
        for job in workers.reap():  # before reading the state, so a finished trial is not relaunched
            on_segment_done(job)
        state = rung_state(trials, hpo_dir, budgets, eta, mixers)
        if all(s is None for s in state.values()) and not workers.running:
            break
        queue = []
        for k in range(max((len(s["missing"]) for s in state.values() if s), default=0)):
            for mixer in mixers:  # round robin over mixers keeps their progress even
                s = state[mixer]
                if s and k < len(s["missing"]) and not workers.busy(s["missing"][k]):
                    queue.append((s["missing"][k], s["budget"]))
        for device in workers.free():
            if not queue:
                break
            trial, budget = queue.pop(0)
            job_args = ["segment", "--trial", trial, "--end-epoch", str(budget)]
            workers.launch(device, trial, job_args, hpo_dir / "trials" / trial / "worker.log")
            workers.running[device].update(trial=trial, end_epoch=budget)
        time.sleep(POLL_SECONDS)

    # 3. Results, roles, finalists.
    segments = {t["trial_id"]: {b: read_segment(hpo_dir / "trials" / t["trial_id"], b) for b in budgets}
                for t in trials}
    table = results_table(trials, {k: {b: s for b, s in v.items() if s} for k, v in segments.items()},
                          budgets)
    roles = member_roles(table, {t["trial_id"]: t["unit_point"] for t in trials})
    final = finalists(table, roles)
    label = _settings(cfg).gamma_label
    todo = [t for t in final if not (hpo_dir / "trials" / t / f"finalist_{label}.json").exists()]
    _run_jobs(workers, [{"key": f"finalist {t}", "log": hpo_dir / "trials" / t / "worker.log",
                         "args": ["finalist", "--trial", t]} for t in todo], lambda job: None)
    write_results(hpo_dir, cfg, trials, table, roles, final, budgets, by_id, label)


def write_results(hpo_dir, cfg, trials, table, roles, final, budgets, by_id, label) -> None:
    metrics = {}
    for t in final:
        path = hpo_dir / "trials" / t / f"finalist_{label}.json"
        metrics[t] = json.loads(path.read_text()) if path.exists() else None
    table["finalist"] = table["trial"].isin(final)
    for key in ("gpr_mean", "gpr_p05", "mape_pct_mean", "rde_pct_mean", "abs_dr80_p95_mm"):
        table[f"final_{key}"] = table["trial"].map(lambda t: (metrics.get(t) or {}).get(key))
    table.to_csv(hpo_dir / "results.csv", index=False)
    selection = {}
    for role in ("C0", "C1", "X0", "U0"):
        if role not in roles:
            continue
        t = by_id[roles[role]]
        trial_dir = hpo_dir / "trials" / t["trial_id"]
        selection[role] = {"trial": t["trial_id"], "mixer": t["mixer"], "params": t["params"],
                           "n_params": t["n_params"], "config": str(trial_dir / "config.yaml"),
                           "best_checkpoint": str(trial_dir / "checkpoints" / "best.pth"),
                           "finalist_metrics": metrics.get(t["trial_id"])}
    if "C1_candidates" in roles:
        selection["C1_candidates"] = roles["C1_candidates"]
    (hpo_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    summary = {"budgets": budgets, "eta": cfg["eta"], "n_trials_per_mixer": cfg["n_trials"],
               "rung_sizes": schedule(range(int(cfg["n_trials"])), int(cfg["eta"]), len(budgets)),
               "epochs_per_mixer": total_epochs(int(cfg["n_trials"]), budgets, int(cfg["eta"])),
               "gamma_label": label, "roles": {k: v for k, v in roles.items() if k != "C1_candidates"},
               "failed": table[[c for c in table.columns if c.startswith("failed_")]].notna().any(axis=1)
               .pipe(lambda s: table.loc[s, "trial"].tolist())}
    (hpo_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("selection: %s", summary["roles"])


# ── Commands ────────────────────────────────────────────────────────────────


@app.command()
def run(
    config: Annotated[Optional[Path], typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_ensemble_hpo.yaml"),
    resume: Annotated[Optional[Path], typer.Option(help="An existing HPO directory to continue.")] = None,
    devices: Annotated[Optional[str], typer.Option(help="Comma-separated CUDA indices.")] = None,
    set_: Annotated[Optional[List[str]], typer.Option("--set", help="KEY=VALUE override.")] = None,
) -> None:
    # torch.compile and the loader workers write under TMPDIR; refuse the root
    # disk before anything is created.
    if on_root_filesystem(Path(tempfile.gettempdir())):
        raise typer.BadParameter("TMPDIR is on the root filesystem; point it at /scratch "
                                 "(e.g. TMPDIR=/scratch/$USER/tmp)")
    if resume is not None:
        hpo_dir = Path(resume)
        cfg = _hpo_config(hpo_dir)
        if set_:
            raise typer.BadParameter("--set cannot change a run being resumed")
    else:
        cfg = apply_set_overrides(load_yaml_config(config), set_ or [])
        root = Path(cfg["output_dir"])
        if on_root_filesystem(root):
            raise typer.BadParameter(f"output_dir {root} is on the root filesystem; use /scratch")
        hpo_dir = root / f"hpo_{datetime.now():%Y%m%d_%H%M%S}"
        hpo_dir.mkdir(parents=True)
    setup_logging(hpo_dir, log_filename="hpo.log")
    device_list = [int(d) for d in devices.split(",")] if devices else list(cfg["devices"])
    if resume is None:
        save_resolved_config(cfg, hpo_dir / "config.yaml")
        prepare(cfg, hpo_dir)
        inputs = {k: {"path": str(cfg[k]), "sha256": sha256_of(Path(cfg[k]))}
                  for k in ("reference_hyperparams",)}
        partition = Path(cfg["partition_dir"])
        for name in ("hpo_train_ids.csv", "hpo_val_ids.csv"):
            inputs[name] = {"path": str(partition / "hpo" / name), "sha256": sha256_of(partition / "hpo" / name)}
        write_manifest(hpo_dir, config=cfg, dataset_path=Path(cfg["dataset_path"]),
                       excluded_indexes_path=Path(cfg["exclude_indexes_path"]),
                       extra={"experiment": cfg["experiment"], "kind": "ensemble_hpo", "inputs": inputs})
    logger.info("HPO directory %s on GPUs %s", hpo_dir, device_list)
    drive(hpo_dir, cfg, device_list)
    logger.info("done: %s", hpo_dir)


def _worker_setup(hpo_dir: Path, trial: str):
    cfg = _hpo_config(hpo_dir)
    t = json.loads((hpo_dir / "trials" / trial / "trial.json").read_text())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        force=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return cfg, t, _config_for(cfg, t), device


@app.command()
def segment(trial: Annotated[str, typer.Option()], end_epoch: Annotated[int, typer.Option()],
            hpo_dir: Annotated[Path, typer.Option()]) -> None:
    """Worker: train one trial to ``end_epoch`` epochs."""
    cfg, t, config, device = _worker_setup(hpo_dir, trial)
    run_segment(config, hpo_dir / "trials" / trial, _ids(hpo_dir / "ids" / "train_ids.csv"),
                _ids(hpo_dir / "ids" / "val_ids.csv"), end_epoch, device, _settings(cfg))


@app.command()
def finalist(trial: Annotated[str, typer.Option()], hpo_dir: Annotated[Path, typer.Option()]) -> None:
    """Worker: the full metric set of a trial's best checkpoint."""
    cfg, t, config, device = _worker_setup(hpo_dir, trial)
    evaluate_finalist(config, hpo_dir / "trials" / trial, _ids(hpo_dir / "ids" / "val_ids.csv"),
                      device, _settings(cfg))


@app.command()
def memcheck(trial: Annotated[List[str], typer.Option()], hpo_dir: Annotated[Path, typer.Option()]) -> None:
    """Worker: one training step at the full batch size for each trial."""
    for name in trial:
        cfg, t, config, device = _worker_setup(hpo_dir, name)
        result = {"ok": True, "peak_gb": None, "error": None}
        try:
            torch.cuda.reset_peak_memory_stats(device)
            model = build_adota_model(config, device)
            shape = (config.batch_size, *config.input_shape)
            out, _ = model(torch.rand(shape, device=device), torch.rand(config.batch_size, 1, device=device))
            out.mean().backward()
            result["peak_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 2)
        except torch.cuda.OutOfMemoryError as exc:
            result.update(ok=False, error=str(exc)[:500])
        finally:
            model = out = None
            torch.cuda.empty_cache()
        (hpo_dir / "trials" / name / "memcheck.json").write_text(json.dumps(result))
        logging.info("memcheck %s: %s", name, result)


if __name__ == "__main__":
    app()
