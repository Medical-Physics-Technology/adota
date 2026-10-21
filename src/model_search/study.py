"""The Optuna study of a model HPO: storage, sampler, pruner and the objective.

One study is shared by one worker process per GPU through a journal file, which
needs no database server. The sampler is TPE: the search wants one best model,
so concentrating trials around the optimum is what is wanted (unlike the
ensemble HPO of EXP-0014, which needed spread-out configurations). Every worker
gets its own sampler seed, so their random start-up trials differ, and
``constant_liar`` keeps workers from proposing the same point at once.

A trial is *trained* once it passes the configuration checks; a configuration
rejected by the parameter cap or an impossible head count is pruned before any
training and does not count towards the budget of trained trials.
"""
from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from src.active_learning.retrospective.validation import MetricSettings
from src.model_search.runs import RunSpec, train_run
from src.model_search.space import build_candidate, rejection, suggest
from src.schemas.configs import TrainingConfig

logger = logging.getLogger(__name__)

STUDY_NAME = "model_hpo"


def open_study(journal: Path, *, seed: int, n_startup_trials: int, pruner_startup_trials: int,
               pruner_warmup_steps: int) -> optuna.Study:
    """Create or load the study kept in ``journal``."""
    storage = JournalStorage(JournalFileBackend(str(journal)))
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials, multivariate=True,
                                         group=True, constant_liar=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=pruner_startup_trials,
                                         n_warmup_steps=pruner_warmup_steps)
    return optuna.create_study(study_name=STUDY_NAME, storage=storage, sampler=sampler, pruner=pruner,
                               direction="minimize", load_if_exists=True)


def fail_stale_trials(study: optuna.Study) -> List[int]:
    """Mark every trial still RUNNING as FAIL. Called before workers start, when
    no trial can legitimately be running: a killed worker leaves its trial
    RUNNING forever, since the journal storage keeps no heartbeat."""
    stale = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.RUNNING]
    for trial in stale:
        study._storage.set_trial_user_attr(trial._trial_id, "error", "stale: worker stopped")
        study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
    return [t.number for t in stale]


def n_trained(study: optuna.Study) -> int:
    """Trials that started training, in any final or running state."""
    return sum(1 for t in study.get_trials(deepcopy=False) if t.user_attrs.get("trained"))


def top_trials(study: optuna.Study, k: int) -> List[optuna.trial.FrozenTrial]:
    """The ``k`` completed trials with the lowest objective, ties to the lower number."""
    done = [t for t in study.get_trials(deepcopy=False)
            if t.state == TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)]
    return sorted(done, key=lambda t: (t.value, t.number))[:k]


@dataclass
class SearchContext:
    """Everything a worker's objective needs."""

    base: TrainingConfig
    space: Mapping[str, Mapping[str, Any]]
    max_params: Optional[int]
    epochs: int
    train_ids: List[str]
    epoch_val_ids: List[str]
    final_val_ids: List[str]
    trials_dir: Path
    device: torch.device
    settings: MetricSettings
    target_trained: int
    lr_schedule: str = "warmup_cosine"


def make_objective(ctx: SearchContext):
    """The objective: sample, check, train, prune on the per-epoch loss, and
    return the last checkpoint's loss on the full validation records."""

    def objective(trial: optuna.Trial) -> float:
        if n_trained(trial.study) >= ctx.target_trained:
            trial.study.stop()
            trial.set_user_attr("rejected", "budget of trained trials reached")
            raise optuna.TrialPruned("budget reached")
        params = suggest(trial, ctx.space)
        try:
            candidate = build_candidate(ctx.base, params)
        except ValueError as exc:
            trial.set_user_attr("rejected", str(exc))
            raise optuna.TrialPruned(str(exc))
        trial.set_user_attr("n_params", candidate.n_params)
        trial.set_user_attr("token_size", candidate.token_size)
        reason = rejection(candidate, ctx.max_params)
        if reason:
            trial.set_user_attr("rejected", reason)
            raise optuna.TrialPruned(reason)

        trial.set_user_attr("trained", True)
        out = ctx.trials_dir / f"trial_{trial.number:04d}"
        trial.set_user_attr("run_dir", str(out))
        logger.info("trial %d: %s, %s parameters", trial.number, params, f"{candidate.n_params:,}")

        def on_epoch(epoch: int, record: Dict[str, Any]) -> None:
            value = float(record["val_loss"]["loss_combined_mean"])
            trial.report(value, epoch)
            if not math.isfinite(value):
                raise optuna.TrialPruned(f"non-finite validation loss at epoch {epoch}")
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (loss {value:.4g})")

        spec = RunSpec(label=f"trial_{trial.number:04d}", config=candidate.config,
                       warmup_epochs=candidate.warmup_epochs, epochs=ctx.epochs, train_ids=ctx.train_ids,
                       epoch_val_ids=ctx.epoch_val_ids, final_val_ids=ctx.final_val_ids,
                       lr_schedule=ctx.lr_schedule)
        try:
            result = train_run(spec, out, ctx.device, ctx.settings, on_epoch=on_epoch)
        except optuna.TrialPruned:
            shutil.rmtree(out / "checkpoints", ignore_errors=True)
            raise
        except RuntimeError as exc:  # out of memory, non-finite training loss
            trial.set_user_attr("error", str(exc)[:500])
            shutil.rmtree(out / "checkpoints", ignore_errors=True)
            raise
        value = float(result["final_val_loss"]["loss_combined_mean"])
        metrics = result["summary"].get("metrics_full") or {}
        trial.set_user_attr("epoch_val_gpr_mean", metrics.get("gpr_mean"))
        trial.set_user_attr("final_val_loss", value)
        return value

    return objective


def stop_at_budget(target: int):
    """A study callback that stops the worker once ``target`` trials trained.
    With several workers the budget can be exceeded by at most the number of
    workers minus one, the trials already running when it is reached."""

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if n_trained(study) >= target:
            study.stop()

    return callback
