"""The dose engines of the hardness analysis, as config entries.

An engine is one way of producing a beamlet dose that is compared with Monte
Carlo. Three kinds enter:

- a trained ADoTA model given by an HPO ``trial`` name (resolved inside
  ``hpo_dir``) or by ``checkpoint`` + ``model_config``; on the HPO validation
  records its errors come from ``errors_csv`` (the trial's finalist CSV by
  default) or are inferred, through
  :func:`src.active_learning.ensemble.members.resolve_member`;
- a dose saved next to every beamlet of a block directory (``saved_dose``, the
  file suffix: ``_ds_pba`` for the DoseCUDA pencil beam, ``_ds_pred`` for the
  published ADoTA);
- on block directories, a trained model is run on the saved inputs.

Every engine on a block directory is scored by
:func:`src.active_learning.retrospective.validation.record_metrics` on the 2 mm
model grid, against the same downsampled Monte Carlo dose, so all of them share
one error definition with the HPO finalists.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.active_learning.ensemble.members import training_config_from_yaml
from src.active_learning.retrospective.validation import MetricSettings, record_metrics
from src.loaders.dir_based import get_single_record
from src.training.checkpoints import CheckpointManager
from src.training.factory import build_adota_model
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)

MODEL_GRID = (160, 30, 30)
ENGINE_KEYS = {"label", "family", "consensus", "trial", "checkpoint", "model_config", "errors_csv",
               "saved_dose", "tables", "note"}
TABLES = ("A", "B")


@dataclass(frozen=True)
class EngineSpec:
    """One dose engine; see the module docstring for the kinds."""

    label: str
    family: str = "transformer"            # transformer | mlstm | none | pencil_beam | published
    consensus: bool = False                # one of the engines whose mean rank is "hard for everyone"
    trial: Optional[str] = None
    checkpoint: Optional[str] = None
    model_config: Optional[str] = None
    errors_csv: Optional[str] = None
    saved_dose: Optional[str] = None
    tables: Tuple[str, ...] = TABLES
    note: str = ""

    @property
    def is_model(self) -> bool:
        return self.checkpoint is not None

    def member_spec(self) -> Dict[str, Any]:
        """The :func:`resolve_member` spec of this engine on the HPO records."""
        spec = {"label": self.label}
        for key in ("checkpoint", "model_config", "errors_csv"):
            if getattr(self, key):
                spec[key] = getattr(self, key)
        return spec


def parse_engines(entries: Sequence[Mapping[str, Any]], hpo_dir: Optional[Path], gamma_label: str
                  ) -> List[EngineSpec]:
    """Engine specs from the config, with HPO trials resolved to their files."""
    engines = []
    for raw in entries:
        unknown = sorted(set(raw) - ENGINE_KEYS)
        if unknown:
            raise ValueError(f"unknown engine keys {unknown}; allowed {sorted(ENGINE_KEYS)}")
        spec = dict(raw)
        if spec.get("trial"):
            if hpo_dir is None:
                raise ValueError(f"engine {spec['label']!r} names a trial but no hpo_dir is set")
            trial_dir = Path(hpo_dir) / "trials" / str(spec["trial"])
            spec.setdefault("checkpoint", str(trial_dir / "checkpoints" / "best.pth"))
            spec.setdefault("model_config", str(trial_dir / "config.yaml"))
            spec.setdefault("errors_csv", str(trial_dir / f"finalist_{gamma_label}.csv"))
        if not (spec.get("checkpoint") or spec.get("errors_csv") or spec.get("saved_dose")):
            raise ValueError(f"engine {spec.get('label')!r} needs a trial, checkpoint, errors_csv or saved_dose")
        if spec.get("saved_dose"):
            spec.setdefault("tables", ["B"])
        spec["tables"] = tuple(str(t) for t in spec.get("tables", TABLES))
        bad = sorted(set(spec["tables"]) - set(TABLES))
        if bad:
            raise ValueError(f"engine {spec['label']!r}: unknown tables {bad}")
        engines.append(EngineSpec(**spec))
    labels = [e.label for e in engines]
    if len(set(labels)) != len(labels):
        raise ValueError(f"engine labels must be unique: {labels}")
    return engines


def load_model(engine: EngineSpec, dataset_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """The engine's trained network in eval mode, and its MinMax scale."""
    config = training_config_from_yaml(Path(engine.model_config), dataset_path)
    model = build_adota_model(replace(config, compile=False), device)
    CheckpointManager.load_weights_only(Path(engine.checkpoint), model=model, device=device)
    model.eval()
    return model, dict(config.scale)


def to_model_grid(dose: np.ndarray, layout: str) -> torch.Tensor:
    """A saved 1 mm dose on the 2 mm model grid, exactly as the loader resamples
    the Monte Carlo dose: ``(D, H, W)`` then trilinear to (160, 30, 30), which at
    factor two is a 2x2x2 average.

    ``layout`` is ``"hwd"`` for the generator's ``(60, 60, 320)`` arrays
    (``_ds``, ``_ds_pba``) or ``"dhw"`` for an already depth-first array
    (``_ds_pred``, saved as ``(1, 1, 320, 60, 60)``).
    """
    t = torch.as_tensor(np.squeeze(dose), dtype=torch.float32)
    if layout == "hwd":
        t = t.permute(2, 0, 1)
    elif layout != "dhw":
        raise ValueError(f"layout must be 'hwd' or 'dhw', got {layout!r}")
    return F.interpolate(t[None, None], size=MODEL_GRID, mode="trilinear", align_corners=False)[0, 0]


def saved_layout(array: np.ndarray) -> str:
    shape = np.squeeze(array).shape
    return "dhw" if shape[0] > shape[-1] else "hwd"


class BlockDataset(Dataset):
    """The saved beamlets of block directories: the model input, the Monte Carlo
    dose and every saved engine dose, on the model grid."""

    def __init__(self, entries: Sequence[Tuple[str, str]], scale: Mapping[str, float],
                 saved: Mapping[str, str]):
        self.entries = [(str(d), str(s)) for d, s in entries]
        self.scale = dict(scale)
        self.saved = dict(saved)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        directory, stem = self.entries[i]
        x, energy, y = get_single_record(stem, directory, scale=self.scale)
        doses = {label: to_model_grid(arr, saved_layout(arr))
                 for label, suffix in self.saved.items()
                 for arr in [np.load(Path(directory) / f"{stem}{suffix}.npy")]}
        return x, energy, y, doses, i


def _collate(batch):
    xs, es, ys, doses, idx = zip(*batch)
    return (torch.stack(xs), torch.stack(es), torch.stack(ys),
            {k: torch.stack([d[k] for d in doses]) for k in doses[0]} if doses[0] else {}, list(idx))


@dataclass
class BlockScoring:
    """What :func:`score_blocks` needs besides the entries."""

    settings: MetricSettings
    device: torch.device
    batch_size: int = 16
    num_workers: int = 4
    models: Dict[str, torch.nn.Module] = field(default_factory=dict)
    saved: Dict[str, str] = field(default_factory=dict)


@torch.no_grad()
def score_blocks(entries: Sequence[Tuple[str, str]], scoring: BlockScoring) -> Dict[str, pd.DataFrame]:
    """Per engine, the per-record errors on every ``(directory, stem)`` entry.

    One pass over the files: each batch is loaded once and every model and
    every saved dose is scored on it. Returns ``{label: frame}`` with the columns
    of :func:`record_metrics` plus ``directory`` and ``stem``.
    """
    settings = scoring.settings
    scale = settings.scale
    backend_options = {"device": scoring.device} if settings.gamma_backend == "torch" else None
    loader = DataLoader(BlockDataset(entries, scale, scoring.saved), batch_size=scoring.batch_size,
                        shuffle=False, num_workers=scoring.num_workers, collate_fn=_collate)
    rows: Dict[str, List[Dict]] = {label: [] for label in [*scoring.models, *scoring.saved]}
    started, done = perf_counter(), 0
    heartbeat = max(1, len(entries) // 20)
    for x, energy, y, doses, idx in loader:
        x, energy = x.to(scoring.device), energy.to(scoring.device)
        y_np = inverse_minmax(y.numpy(), scale["min_ds"], scale["max_ds"])
        predictions = {label: inverse_minmax(model(x, energy)[0].cpu().numpy(), scale["min_ds"], scale["max_ds"])
                       for label, model in scoring.models.items()}
        predictions.update({label: d.numpy() for label, d in doses.items()})
        for b, i in enumerate(idx):
            directory, stem = entries[i]
            for label, pred in predictions.items():
                rows[label].append({"directory": directory, "stem": stem,
                                    **record_metrics(y_np[b], pred[b], settings, backend_options,
                                                     sample_id=f"{directory}/{stem}")})
        before, done = done, done + len(idx)
        if done // heartbeat > before // heartbeat or done == len(entries):
            logger.info("block scoring: %d/%d beamlets, %.0f s", done, len(entries), perf_counter() - started)
    return {label: pd.DataFrame(r) for label, r in rows.items()}


@torch.no_grad()
def predict_h5_records(engines: Sequence[EngineSpec], dataset_path: str, ids: Sequence[str],
                       device: torch.device) -> Dict[str, Dict[str, Any]]:
    """Ground truth, CT channel and every engine's prediction for a few HDF5
    records, loaded as the finalists were validated (for the example figures).

    Returns ``{sample_id: {"ct", "gt", "pred": {label: dose}}}`` in the dataset's
    dose unit, ``(depth, y, x)``.
    """
    from src.active_learning.retrospective.trainer import validation_loader

    out: Dict[str, Dict[str, Any]] = {}
    for engine in engines:
        model, scale = load_model(engine, dataset_path, device)
        config = replace(training_config_from_yaml(Path(engine.model_config), dataset_path), num_workers=0,
                         batch_size=1)
        for sid, (x, energy, y) in zip(ids, validation_loader(config, list(ids))):
            entry = out.setdefault(sid, {"pred": {}})
            pred = model(x.to(device), energy.to(device))[0]
            entry["pred"][engine.label] = np.squeeze(inverse_minmax(pred.cpu().numpy(), scale["min_ds"],
                                                                    scale["max_ds"]))
            entry.setdefault("gt", np.squeeze(inverse_minmax(y.numpy(), scale["min_ds"], scale["max_ds"])))
            entry.setdefault("ct", np.squeeze(x[0, 0].numpy()))
        del model
    return out


def _score_shard(engines: Sequence[EngineSpec], dataset_path: str, entries: Sequence[Tuple[str, str]],
                 settings: MetricSettings, device_index: int, batch_size: int,
                 num_workers: int) -> Dict[str, pd.DataFrame]:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s - cuda:{device_index} - %(message)s")
    device = torch.device(f"cuda:{device_index}")
    models = {e.label: load_model(e, dataset_path, device)[0] for e in engines if e.is_model}
    scoring = BlockScoring(settings=settings, device=device, batch_size=batch_size, num_workers=num_workers,
                           models=models, saved={e.label: e.saved_dose for e in engines if e.saved_dose})
    return score_blocks(entries, scoring)


def score_blocks_sharded(engines: Sequence[EngineSpec], dataset_path: str, entries: Sequence[Tuple[str, str]],
                         settings: MetricSettings, device_indices: Sequence[int], batch_size: int = 16,
                         num_workers: int = 4) -> Dict[str, pd.DataFrame]:
    """:func:`score_blocks` with the entries split into contiguous shards, one
    process per GPU (gamma, one call per record and engine, is the cost).
    Results come back in entry order."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    devices = list(device_indices)
    bounds = np.linspace(0, len(entries), len(devices) + 1).astype(int)
    shards = [list(entries[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    if len(devices) == 1:
        return _score_shard(engines, dataset_path, shards[0], settings, devices[0], batch_size, num_workers)
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_score_shard, list(engines), dataset_path, shard, settings, d, batch_size,
                               num_workers) for shard, d in zip(shards, devices)]
        parts = [f.result() for f in futures]
    return {label: pd.concat([p[label] for p in parts], ignore_index=True) for label in parts[0]}
