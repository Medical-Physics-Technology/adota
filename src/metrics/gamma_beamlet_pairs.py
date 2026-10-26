# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""The (Monte Carlo, ADoTA) beamlet dose pairs the gamma benchmark measures on.

Split from :mod:`src.metrics.gamma_beamlet_benchmark` by role: this module
produces the data, that one times the backends against it. The split also keeps
the heavy dependencies apart -- building a pair needs h5py, the dataset loader
and a checkpoint, while timing a backend needs none of them, so a sweep over a
cached pair file imports nothing of the training stack.

Doses are cached in the training normalisation rather than in physical units,
because that is the form both gamma entry points take: ``gamma_index`` is handed
the de-normalised arrays and ``gamma_index_torch`` the normalised tensors, and
one cache then feeds both without a round trip through the inverse scaling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "BEAMLET_RESOLUTION_MM",
    "BeamletPair",
    "beamlet_resolution_mm",
    "load_baseline_model",
    "open_beamlet_dataset",
    "pair_from_record",
    "build_pairs",
    "save_pairs",
    "load_pairs",
]

# Voxel spacing of a training beamlet, in millimetres. Matches
# ``gpr_resolution_mm`` in the training configs; the dataset carries no spacing
# of its own.
BEAMLET_RESOLUTION_MM: Tuple[float, float, float] = (2.0, 2.0, 2.0)


def beamlet_resolution_mm() -> Tuple[float, float, float]:
    """Voxel spacing assumed for a beamlet grid, in millimetres."""
    return BEAMLET_RESOLUTION_MM


@dataclass
class BeamletPair:
    """One (Monte Carlo, ADoTA) beamlet dose pair, normalised.

    Doses are kept in the training normalisation rather than in physical units,
    because that is the form both entry points take: ``gamma_index`` is handed
    the de-normalised arrays and ``gamma_index_torch`` the normalised tensors,
    and storing the normalised pair lets one cache feed both without a
    round-trip through the inverse scaling.

    Attributes:
        sample_id: Record id in the source HDF5 dataset.
        energy_mev: Initial proton energy of the beamlet, in MeV.
        reference: Monte Carlo dose, normalised, shape ``(D, H, W)``.
        evaluation: ADoTA prediction, normalised, same shape.
    """

    sample_id: str
    energy_mev: float
    reference: np.ndarray
    evaluation: np.ndarray

    @property
    def shape(self) -> Tuple[int, ...]:
        """Grid shape of the pair."""
        return tuple(self.reference.shape)

    @property
    def voxels(self) -> int:
        """Number of voxels in one volume of the pair."""
        return int(np.prod(self.reference.shape))



def load_baseline_model(run_dir: Path, device: torch.device, checkpoint_name: str = "best.pth"):
    """Load a training snapshot's weights into a fresh model, in eval mode.

    Loaded through the checkpoint manager rather than through
    ``src.adota.utils.load_model``: the file is a *training snapshot*, which
    keeps the weights under "model" alongside optimizer and RNG state, and only
    the manager unwraps that form.

    Raises:
        FileNotFoundError: If the hyperparameters or checkpoint are missing.
    """
    from src.adota.models import DoTA3D_v3
    from src.training.checkpoints import CheckpointManager

    hyperparams_path = run_dir / "hyperparams.json"
    model_path = run_dir / "checkpoints" / checkpoint_name
    for path in (hyperparams_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint artefact not found: {path}")
    model = DoTA3D_v3(**json.loads(hyperparams_path.read_text()))
    CheckpointManager.load_weights_only(model_path, model=model, device=device)
    model.eval()
    model.to(device)
    return model


def open_beamlet_dataset(h5_path: Path, record_ids: Sequence[str]):
    """The evaluation-mode generator over the given records, as the scripts use it."""
    from src.loaders.generator import H5PYGenerator

    if not h5_path.is_file():
        raise FileNotFoundError(f"Beamlet dataset not found: {h5_path}")
    return H5PYGenerator(
        file_path=str(h5_path),
        indexes=list(record_ids),
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )


def pair_from_record(dataset, index: int, model, device: torch.device, scale: Dict[str, float]) -> BeamletPair:
    """Run the model over one dataset record and return its dose pair."""
    from src.adota.config import denormalize_energy

    x, energy, y = dataset[index]
    with torch.no_grad():
        prediction = model(x.unsqueeze(0).to(device), energy.unsqueeze(0).to(device))[0]
    return BeamletPair(
        sample_id=dataset.record_ids[index],
        energy_mev=float(denormalize_energy(float(energy.item()), scale)),
        reference=np.ascontiguousarray(y.squeeze().numpy(), dtype=np.float32),
        evaluation=np.ascontiguousarray(prediction.squeeze().detach().cpu().numpy(), dtype=np.float32),
    )


def build_pairs(
    h5_path: Path,
    run_dir: Path,
    count: int,
    device: torch.device,
    scale: Dict[str, float],
    seed: int = 1234,
    checkpoint_name: str = "best.pth",
    record_ids: Optional[Sequence[str]] = None,
) -> List[BeamletPair]:
    """Run the model over ``count`` beamlets and keep the dose pairs.

    The evaluation dose has to come from a real prediction rather than from a
    perturbed copy of the reference: gamma cost depends on how far the search
    has to travel before it converges, so a synthetic disagreement would give a
    timing that is not the timing of the metric as it is actually used.

    Args:
        h5_path: HDF5 beamlet dataset (the test set).
        run_dir: Training run directory holding ``hyperparams.json`` and
            ``checkpoints/``.
        count: Number of beamlets to draw.
        device: Device to run inference on.
        scale: Training scale dict (``min_ds`` / ``max_ds`` / ...).
        seed: Seed for the record draw, so the same beamlets come back.
        checkpoint_name: File under ``run_dir/checkpoints`` to load.
        record_ids: Explicit records to use instead of a random draw; ``count``
            and ``seed`` are then ignored.

    Returns:
        The drawn pairs, in dataset order.

    Raises:
        FileNotFoundError: If the dataset, hyperparameters or checkpoint are
            missing.
    """
    import h5py

    if not h5_path.is_file():
        raise FileNotFoundError(f"Beamlet dataset not found: {h5_path}")
    if record_ids is None:
        with h5py.File(h5_path, "r") as handle:
            all_ids = sorted(handle.keys())
        rng = np.random.RandomState(seed)
        picked = sorted(
            all_ids[i] for i in rng.choice(len(all_ids), size=min(count, len(all_ids)), replace=False)
        )
        logger.info("Drew %d of %d records from %s", len(picked), len(all_ids), h5_path.name)
    else:
        picked = list(record_ids)
        logger.info("Using %d given records from %s", len(picked), h5_path.name)

    dataset = open_beamlet_dataset(h5_path, picked)
    model = load_baseline_model(run_dir, device, checkpoint_name)
    pairs: List[BeamletPair] = [pair_from_record(dataset, index, model, device, scale) for index in range(len(dataset))]
    logger.info("Built %d beamlet pairs of shape %s", len(pairs), pairs[0].shape)
    return pairs


def save_pairs(path: Path, pairs: Sequence[BeamletPair], scale: Dict[str, float]) -> None:
    """Cache the pairs so a sweep can be repeated without re-running inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.array([p.sample_id for p in pairs]),
        energies=np.array([p.energy_mev for p in pairs], dtype=np.float64),
        reference=np.stack([p.reference for p in pairs]),
        evaluation=np.stack([p.evaluation for p in pairs]),
        scale=json.dumps(scale),
    )
    logger.info("Wrote %d pairs to %s", len(pairs), path)


def load_pairs(path: Path) -> Tuple[List[BeamletPair], Dict[str, float]]:
    """Read back a cache written by :func:`save_pairs`."""
    with np.load(path, allow_pickle=False) as data:
        scale = json.loads(str(data["scale"]))
        pairs = [
            BeamletPair(
                sample_id=str(sample_id),
                energy_mev=float(energy),
                reference=reference,
                evaluation=evaluation,
            )
            for sample_id, energy, reference, evaluation in zip(
                data["sample_ids"], data["energies"], data["reference"], data["evaluation"]
            )
        ]
    logger.info("Loaded %d pairs from %s", len(pairs), path)
    return pairs, scale
