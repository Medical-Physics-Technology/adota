"""The record sets of a model HPO and one training run of a candidate.

A run is one cycle of the retrospective trainer
(:func:`src.active_learning.retrospective.trainer.train_cycle`): fresh weights,
a linear warmup and a cosine decay of the learning rate over the run's own
epochs (``warmup_cosine``, in which every epoch trains; see
:func:`src.active_learning.retrospective.lr_schedule.fixed_lr`), the loss on
``epoch_val_ids`` after every epoch, and the last checkpoint kept. Nothing here is a new training loop.
"""
from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

from src.active_learning.retrospective.trainer import (
    CycleSpec,
    build_train_state,
    build_validation_bundle,
    train_cycle,
    training_loader,
    validation_loader,
)
from src.active_learning.retrospective.validation import (
    MetricSettings,
    evaluate_loss,
    evaluate_metrics,
    summarise_metrics,
)
from src.datasets.energy_cut import drop_above_energy
from src.schemas.configs import TrainingConfig
from src.training.run_dir import MetricsLog, save_resolved_config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- record sets

def resolve_ids(spec: Any, base_dir: Path) -> List[str]:
    """One record set from the configuration: either a CSV path (absolute or
    relative to ``base_dir``) with a ``sample_id`` column, or a filter
    ``{table: <csv>, column: <name>, values: [...]}`` on such a table."""
    if isinstance(spec, str):
        path = Path(spec) if Path(spec).is_absolute() else base_dir / spec
        return pd.read_csv(path, dtype=str)["sample_id"].tolist()
    table = Path(spec["table"]) if Path(spec["table"]).is_absolute() else base_dir / spec["table"]
    frame = pd.read_csv(table, dtype=str)
    return frame.loc[frame[spec["column"]].isin([str(v) for v in spec["values"]]), "sample_id"].tolist()


def prepare_ids(sets: Mapping[str, Sequence[str]], dataset_path: Path, excluded_file: Optional[Path],
                max_energy_mev: Optional[float], scale: Optional[dict]) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Every set without the excluded records and without the beamlets above
    ``max_energy_mev`` (:func:`src.datasets.energy_cut.drop_above_energy`), and a
    table of what was removed from each."""
    excluded = set()
    if excluded_file is not None and Path(excluded_file).exists():
        excluded = {line.strip() for line in Path(excluded_file).read_text().splitlines() if line.strip()}
    out, rows = {}, []
    for name, ids in sets.items():
        ids = list(ids)
        kept = [i for i in ids if i not in excluded]
        kept, dropped = drop_above_energy(Path(dataset_path), kept, max_energy_mev, scale)
        out[name] = kept
        rows.append({"set": name, "listed": len(ids), "excluded": len(ids) - len(kept) - len(dropped),
                     "above_energy_cut": len(dropped), "kept": len(kept)})
    return out, pd.DataFrame(rows)


def scan_overlap(sets: Mapping[str, Sequence[str]], scan_of: Mapping[str, str],
                 pairs: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    """For each ``(a, b)`` of ``pairs``, the scans that hold records of both sets."""
    out = {}
    for a, b in pairs:
        scans_a = {scan_of[i] for i in sets[a] if i in scan_of}
        scans_b = {scan_of[i] for i in sets[b] if i in scan_of}
        out[f"{a}|{b}"] = sorted(scans_a & scans_b)
    return out


# --------------------------------------------------------------------------- one run

@dataclass
class RunSpec:
    """One training run: what it trains on, how long, and what it is judged on."""

    label: str
    config: TrainingConfig
    warmup_epochs: int
    epochs: int
    train_ids: List[str]
    epoch_val_ids: List[str]
    """Loss after every epoch (the pruning signal), and the metrics after the last."""
    final_val_ids: Optional[List[str]] = None
    """When set, the last checkpoint's loss on these records is the run's objective."""
    final_metrics: bool = False
    """Also the gamma pass rate and the other metrics on ``final_val_ids``."""
    lr_schedule: str = "warmup_cosine"
    """A fixed schedule of :mod:`src.active_learning.retrospective.lr_schedule`."""


def train_run(spec: RunSpec, out_dir: Path, device: torch.device, settings: MetricSettings,
              on_epoch: Optional[Callable[[int, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Train ``spec`` from scratch into ``out_dir``; returns (and writes
    ``result.json``) the summary of the run and the final evaluation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(spec.config, out_dir / "config.yaml")
    state = build_train_state(spec.config, device, lr_schedule=spec.lr_schedule, lr_min=0.0,
                              warmup_epochs=spec.warmup_epochs)
    loaders: List[Any] = []
    try:
        loader = training_loader(spec.config, spec.train_ids, spec.epoch_val_ids[:1], cycle=0)
        bundle = build_validation_bundle(spec.config, spec.epoch_val_ids, spec.epoch_val_ids, settings)
        loaders += [loader, bundle.full_loader, bundle.subsample_loader]
        cycle = CycleSpec(cycle=0, strategy=spec.label, n_train=len(spec.train_ids), epochs=spec.epochs,
                          cumulative_epoch_start=0, eval_every=10 ** 9)
        summary = train_cycle(state, loader, bundle, cycle, run_dir=out_dir,
                              metrics_log=MetricsLog(out_dir / "metrics.jsonl"),
                              checkpoint_dir=out_dir / "checkpoints", checkpoint_every=10 ** 9,
                              on_epoch=on_epoch)
        result: Dict[str, Any] = {"label": spec.label, "epochs": spec.epochs, "warmup_epochs": spec.warmup_epochs,
                                  "lr_schedule": spec.lr_schedule,
                                  "n_train": len(spec.train_ids), "n_epoch_val": len(spec.epoch_val_ids),
                                  "summary": summary}
        if spec.final_val_ids:
            # The objective uses the static loss weights, so runs stay comparable.
            final_loader = validation_loader(spec.config, spec.final_val_ids)
            loaders.append(final_loader)
            result["final_val_loss"] = evaluate_loss(
                state.model, final_loader, device=device, weight_mse=float(spec.config.initial_weight_mse),
                weight_ps=float(spec.config.initial_weight_ps), settings=settings)
            result["n_final_val"] = len(spec.final_val_ids)
            if spec.final_metrics:
                frame = evaluate_metrics(state.model, final_loader, spec.final_val_ids, device=device,
                                         settings=settings, desc=f"{spec.label} final val")
                frame.to_csv(out_dir / f"final_val_{settings.gamma_label}.csv", index=False)
                result["final_val_metrics"] = summarise_metrics(frame)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        return result
    finally:
        _shutdown_workers(loaders)
        del state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        torch._dynamo.reset()


def _shutdown_workers(loaders: Sequence[Any]) -> None:
    """Stop the persistent worker processes of ``loaders`` now. Left to the garbage
    collector, a run ended early (a pruned trial) tears them down mid-queue and
    floods the log with harmless but alarming tracebacks."""
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            loader._iterator = None


def select_winner(rows: Sequence[Mapping[str, Any]], gpr_guard_pp: float) -> Dict[str, Any]:
    """The candidate with the lowest final validation loss, unless another
    candidate's mean gamma pass rate exceeds its own by more than ``gpr_guard_pp``
    percentage points; then the candidate with the highest pass rate wins.

    Each row needs ``label``, ``final_loss`` and ``gpr_mean`` (a fraction).
    Ties go to the lower label, so the choice is deterministic.
    """
    if not rows:
        raise ValueError("no candidates to choose from")
    by_loss = sorted(rows, key=lambda r: (float(r["final_loss"]), str(r["label"])))[0]
    by_gpr = sorted(rows, key=lambda r: (-float(r["gpr_mean"]), str(r["label"])))[0]
    margin_pp = (float(by_gpr["gpr_mean"]) - float(by_loss["gpr_mean"])) * 100.0
    if margin_pp > gpr_guard_pp:
        return {**dict(by_gpr), "rule": f"gamma guard: +{margin_pp:.2f} pp over the lowest-loss "
                                         f"candidate {by_loss['label']}"}
    return {**dict(by_loss), "rule": "lowest final validation loss"}
