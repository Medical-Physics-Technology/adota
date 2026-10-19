"""The members of an ensemble: one trained model each, with its per-record errors
on an evaluation set and the ids it was trained on.

A member is declared by a small spec (a dict from the YAML config) in one of
three forms, so any trained model and any evaluation set can enter:

- ``run_dir`` + ``cycle``: a run of the retrospective benchmark. The checkpoint
  is ``checkpoints/cycle_<cc>/<checkpoint>.pth``, the model config is the run's
  ``config.yaml``, the training ids are ``cycles/cycle_<cc>/training_ids.csv``,
  and the errors the run already wrote at that cycle boundary are
  ``validation/cycle_<cc>_full_gamma_<label>.csv``.
- ``checkpoint`` + ``model_config``: any ADoTA checkpoint with the YAML it was
  trained from, either a retrospective run config (a ``training:`` block) or a
  plain :class:`TrainingConfig` YAML; ``training_ids`` is optional.
- ``errors_csv``: a per-record table computed elsewhere, with the columns of
  :func:`src.active_learning.retrospective.validation.evaluate_metrics`;
  ``training_ids`` is optional.

Errors are taken from an existing CSV when it covers every evaluation record
and was computed with the configured gamma criteria (``source: auto`` or
``csv``), and are otherwise recomputed from the checkpoint with the same
per-record metric code the benchmark uses (``source: infer`` forces this).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import torch

from src.active_learning.retrospective.config import RetroConfig
from src.active_learning.retrospective.trainer import validation_loader
from src.active_learning.retrospective.validation import MetricSettings, evaluate_metrics
from src.adota.config import load_yaml_config
from src.datasets.beamlet_h5 import sha256_of
from src.schemas.configs import TrainingConfig
from src.training.checkpoints import CheckpointManager
from src.training.factory import build_adota_model

logger = logging.getLogger(__name__)

SOURCES = ("auto", "csv", "infer")
SPEC_KEYS = {"label", "run_dir", "cycle", "checkpoint", "model_config", "errors_csv",
             "training_ids"}


@dataclass
class Member:
    """One trained model: its per-record errors and where they came from."""

    label: str
    errors: pd.DataFrame
    training_ids: Optional[List[str]]
    source: str
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSettings:
    """What every member is measured with, identical across members so that
    their errors are comparable."""

    dataset_path: str
    gamma_params: Dict[str, Any]
    gamma_resolution_mm: Sequence[float] = (2.0, 2.0, 2.0)
    gamma_cutoff_percent: float = 10.0
    gamma_backend: str = "torch"
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None

    def metric_settings(self, config: TrainingConfig) -> MetricSettings:
        return MetricSettings(scale=dict(config.scale), gamma_params=dict(self.gamma_params),
                              resolution_mm=tuple(self.gamma_resolution_mm),
                              gamma_cutoff_percent=self.gamma_cutoff_percent,
                              gamma_backend=self.gamma_backend,
                              lps_dx_mm=config.lps_dx_mm, lps_dy_mm=config.lps_dy_mm)

    @property
    def gamma_label(self) -> str:
        return MetricSettings(scale={}, gamma_params=dict(self.gamma_params),
                              gamma_cutoff_percent=self.gamma_cutoff_percent).gamma_label


def training_config_from_yaml(path: Path, dataset_path: str) -> TrainingConfig:
    """The model and data settings of a trained model, pointed at ``dataset_path``
    and with ``torch.compile`` off (inference needs no compiled graph and must not
    write a compile cache)."""
    raw = load_yaml_config(Path(path))
    if "training" in raw:
        config = RetroConfig.from_dict(raw).training_config()
    else:
        valid = {f.name for f in fields(TrainingConfig)}
        block = {k: v for k, v in raw.items() if k in valid}
        if "input_shape" in block:
            block["input_shape"] = tuple(block["input_shape"])
        config = TrainingConfig(**block)
    return replace(config, dataset_path=str(dataset_path), compile=False)


@torch.no_grad()
def infer_errors(checkpoint: Path, model_config: Path, ids: Sequence[str],
                 settings: EvaluationSettings, device: torch.device) -> pd.DataFrame:
    """Per-record errors of ``checkpoint`` on ``ids``, by the benchmark's own
    metric code (:func:`evaluate_metrics`)."""
    config = training_config_from_yaml(model_config, settings.dataset_path)
    if settings.batch_size is not None:
        config = replace(config, batch_size=int(settings.batch_size))
    if settings.num_workers is not None:
        config = replace(config, num_workers=int(settings.num_workers))
    model = build_adota_model(config, device)
    CheckpointManager.load_weights_only(Path(checkpoint), model=model, device=device)
    loader = validation_loader(config, ids)
    if len(loader.dataset) != len(ids):
        # A centerline flux mode drops ids missing from its sidecar, which would
        # shift every later record onto the wrong sample id.
        raise ValueError(f"the loader holds {len(loader.dataset)} of {len(ids)} evaluation "
                         f"records (flux_mode {config.flux_mode!r})")
    return evaluate_metrics(model, loader, list(ids), device=device,
                            settings=settings.metric_settings(config),
                            desc=Path(checkpoint).parent.name)


def _read_ids(path: Path) -> List[str]:
    frame = pd.read_csv(path, dtype=str)
    column = "sample_id" if "sample_id" in frame.columns else frame.columns[0]
    return frame[column].astype(str).tolist()


def _covers(errors: pd.DataFrame, ids: Sequence[str]) -> bool:
    return set(ids) <= set(errors["sample_id"].astype(str))


def _retro_run_paths(spec: Mapping[str, Any], checkpoint_name: str,
                     gamma_label: str) -> Dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    cycle = int(spec["cycle"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    seed = (manifest.get("config") or {}).get("seed")
    strategy = str(manifest.get("strategy", "unknown"))
    return {
        "label": spec.get("label") or f"{strategy}_seed{seed}",
        "run_dir": str(run_dir), "cycle": cycle, "strategy": strategy, "seed": seed,
        "checkpoint": str(run_dir / "checkpoints" / f"cycle_{cycle:02d}" / f"{checkpoint_name}.pth"),
        "model_config": str(run_dir / "config.yaml"),
        "training_ids": spec.get("training_ids")
        or str(run_dir / "cycles" / f"cycle_{cycle:02d}" / "training_ids.csv"),
        "errors_csv": spec.get("errors_csv")
        or str(run_dir / "validation" / f"cycle_{cycle:02d}_full_gamma_{gamma_label}.csv"),
    }


def resolve_member(spec: Mapping[str, Any], ids: Sequence[str], settings: EvaluationSettings, *,
                   source: str = "auto", checkpoint_name: str = "last",
                   device: Optional[torch.device] = None,
                   cache_dir: Optional[Path] = None) -> Member:
    """Build one :class:`Member` from its spec, restricted to ``ids``.

    Inferred errors are written to ``cache_dir/<label>_errors.csv`` so a rerun
    of the analysis can pass that file back as ``errors_csv``.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    unknown = sorted(set(spec) - SPEC_KEYS)
    if unknown:
        raise ValueError(f"unknown member keys {unknown}; allowed {sorted(SPEC_KEYS)}")
    if "run_dir" in spec:
        info = _retro_run_paths(spec, checkpoint_name, settings.gamma_label)
    elif "checkpoint" in spec or "errors_csv" in spec:
        info = {k: (str(v) if v is not None else None) for k, v in spec.items()}
        info.setdefault("label", Path(spec.get("checkpoint") or spec["errors_csv"]).stem)
    else:
        raise ValueError(f"a member needs run_dir, checkpoint or errors_csv: {dict(spec)}")
    label = str(info["label"])

    errors = None
    csv = Path(info["errors_csv"]) if info.get("errors_csv") else None
    if source != "infer" and csv is not None and csv.exists():
        frame = pd.read_csv(csv, dtype={"sample_id": str})
        if _covers(frame, ids):
            errors, used = frame, "csv"
            info["errors_csv_sha256"] = sha256_of(csv)
        else:
            logger.warning("%s: %s does not cover every evaluation record", label, csv)
    if errors is None:
        if source == "csv":
            raise FileNotFoundError(f"{label}: no errors CSV covering the evaluation set "
                                    f"({csv}) and source is 'csv'")
        if not info.get("checkpoint") or not info.get("model_config"):
            raise ValueError(f"{label}: inference needs checkpoint and model_config")
        if device is None:
            raise ValueError(f"{label}: inference needs a device")
        logger.info("%s: inferring %d records from %s", label, len(ids), info["checkpoint"])
        errors, used = infer_errors(Path(info["checkpoint"]), Path(info["model_config"]), ids,
                                    settings, device), "inference"
        if cache_dir is not None:
            info["errors_csv"] = str(Path(cache_dir) / f"{label}_errors.csv")
            errors.to_csv(info["errors_csv"], index=False)
            info["errors_csv_sha256"] = sha256_of(Path(info["errors_csv"]))
    if info.get("checkpoint") and Path(info["checkpoint"]).exists():
        info["checkpoint_sha256"] = sha256_of(Path(info["checkpoint"]))

    errors = errors.assign(sample_id=errors["sample_id"].astype(str))
    errors = errors.drop_duplicates("sample_id").set_index("sample_id").loc[list(ids)].reset_index()
    training = info.get("training_ids")
    training_ids = _read_ids(Path(training)) if training and Path(training).exists() else None
    if training and training_ids is None:
        logger.warning("%s: training ids %s not found; its pair level stays unknown", label,
                       training)
    info["n_training"] = len(training_ids) if training_ids is not None else None
    return Member(label=label, errors=errors, training_ids=training_ids, source=used,
                  provenance=info)


def unique_member_labels(members: Sequence[Member]) -> None:
    labels = [m.label for m in members]
    duplicates = sorted({x for x in labels if labels.count(x) > 1})
    if duplicates:
        raise ValueError(f"member labels must be unique; repeated {duplicates}: set `label`")


def held_out_table(members: Sequence[Member], eval_ids: Sequence[str],
                   group_of: pd.Series) -> pd.DataFrame:
    """Per member, how many evaluation records and groups (scans) it trained on.

    ``group_of`` maps ``sample_id`` to the group (``patient_key``) and must
    cover the evaluation and training ids.
    """
    eval_groups = set(group_of.loc[list(eval_ids)].astype(str))
    rows = []
    for m in members:
        row = {"label": m.label, "n_eval": len(eval_ids), "n_eval_groups": len(eval_groups),
               "n_training": None, "n_shared_records": None, "n_shared_groups": None}
        if m.training_ids is not None:
            missing = [i for i in m.training_ids if i not in group_of.index]
            if missing:
                raise KeyError(f"{m.label}: {len(missing)} training ids have no metadata, "
                               f"e.g. {missing[0]!r}")
            train_groups = set(group_of.loc[m.training_ids].astype(str))
            row.update(n_training=len(m.training_ids),
                       n_shared_records=len(set(m.training_ids) & set(eval_ids)),
                       n_shared_groups=len(train_groups & eval_groups))
        rows.append(row)
    return pd.DataFrame(rows)
