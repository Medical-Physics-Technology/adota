"""ADoTA predictions on a v3-layout HDF5 export, saved on the model grid.

The loader, model build, checkpoint load and de-normalisation are the ones the
EXP-0014 ensemble run used for its stored pool errors
(:func:`src.active_learning.ensemble.members.infer_errors`), so a prediction here is
the one those errors were computed from. Each prediction is saved de-normalised, in
the model layout ``(D, H, W)``, never replacing an existing file.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Dict, Sequence

import numpy as np
import torch

from src.active_learning.ensemble.members import training_config_from_yaml
from src.active_learning.retrospective.trainer import validation_loader
from src.engine_drivers.pba import save_dose
from src.training.checkpoints import CheckpointManager
from src.training.factory import build_adota_model
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_and_save(checkpoint: Path, model_config: Path, dataset_path: Path, ids: Sequence[str],
                     out_dir: Path, device: torch.device, batch_size: int = 16, num_workers: int = 8
                     ) -> Dict[str, float]:
    """Predict every id that has no saved dose yet and save it to ``out_dir/<id>.npy``.

    Returns the model's dose scale (``min_ds``, ``max_ds``) and the number written.

    Raises:
        ValueError: if the loader does not hold exactly the requested ids.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.npy").exists()]
    config = training_config_from_yaml(Path(model_config), str(dataset_path))
    config = replace(config, batch_size=int(batch_size), num_workers=int(num_workers), excluded_indexes_file=None)
    scale = dict(config.scale)
    if not todo:
        return {"min_ds": float(scale["min_ds"]), "max_ds": float(scale["max_ds"]), "written": 0}
    model = build_adota_model(config, device)
    CheckpointManager.load_weights_only(Path(checkpoint), model=model, device=device)
    model.eval()
    loader = validation_loader(config, todo)
    if len(loader.dataset) != len(todo):
        raise ValueError(f"the loader holds {len(loader.dataset)} of {len(todo)} records")
    written, flat, started = 0, 0, perf_counter()
    heartbeat = max(1, len(todo) // 10)
    for x, energy, _ in loader:
        pred = model(x.to(device, non_blocking=True), energy.to(device, non_blocking=True))[0]
        for b in range(int(pred.shape[0])):
            dose = inverse_minmax(pred[b].detach().cpu().numpy(), scale["min_ds"], scale["max_ds"])
            save_dose(out_dir, todo[flat], np.squeeze(dose))
            flat += 1
            written += 1
            if flat % heartbeat == 0:
                logger.info("ADoTA %d/%d  %.0f s", flat, len(todo), perf_counter() - started)
    return {"min_ds": float(scale["min_ds"]), "max_ds": float(scale["max_ds"]), "written": written}


__all__ = ["predict_and_save"]
