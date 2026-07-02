"""Data-pipeline helpers for ADoTA training.

Everything needed to go from an on-disk H5 dataset to the train / val
:class:`torch.utils.data.DataLoader` pair the training loop consumes:
record-id discovery + exclusion, a deterministic train/val split, the
H5 collate function, dataloader construction, and a thin loader wrapper
that caps iteration for smoke tests.

These helpers used to live inline in ``scripts/train_adota.py``; they are
factored out here so they can be unit-tested and reused.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.loaders.generator import H5PYGenerator
from src.schemas.configs import TrainingConfig

logger = logging.getLogger(__name__)


def train_val_split(
    record_ids: List[str], test_size: float, random_state: int
) -> Tuple[List[str], List[str]]:
    """Deterministic shuffle then split; matches sklearn semantics.

    Args:
        record_ids: Full list of usable record ids.
        test_size: Fraction of records assigned to the validation split.
        random_state: Seed for the shuffle RNG.

    Returns:
        ``(train_ids, val_ids)``. Each list is sorted by original index, and
        the two are disjoint and cover every input id exactly once.
    """
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(record_ids))
    rng.shuffle(indices)
    n_val = int(round(len(record_ids) * test_size))
    val_idx = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())
    return [record_ids[i] for i in train_idx], [record_ids[i] for i in val_idx]


def load_record_ids(
    dataset_path: Path, excluded_indexes_path: Optional[Path]
) -> List[str]:
    """Read the H5 keys and remove anything listed in the exclusion file.

    Args:
        dataset_path: Path to the H5 dataset.
        excluded_indexes_path: Optional path to a text file of record ids to
            drop, one per line. Missing / ``None`` means no exclusion.

    Returns:
        Record ids present in the dataset and not excluded, in file order.
    """
    with h5py.File(str(dataset_path), "r") as ds:
        record_ids = list(ds.keys())

    excluded: List[str] = []
    if excluded_indexes_path is not None and excluded_indexes_path.exists():
        with open(excluded_indexes_path, "r") as f:
            excluded = [line.strip() for line in f if line.strip()]
        logger.info(
            "Loaded %d excluded indexes from %s", len(excluded), excluded_indexes_path
        )

    excluded_set = set(excluded)
    return [rid for rid in record_ids if rid not in excluded_set]


def collate_h5(batch):
    """Stack H5PYGenerator samples into contiguous batched tensors.

    Args:
        batch: Sequence of ``(x, energy, y)`` samples from
            :class:`H5PYGenerator`.

    Returns:
        ``(X, E, Y)`` where ``X``/``Y`` are stacked to ``(B, C, D, H, W)`` and
        ``E`` is ``(B, 1)``.
    """
    xs, es, ys = zip(*batch)
    X = torch.stack([t.contiguous() if not t.is_contiguous() else t for t in xs], dim=0)
    E = torch.stack([e.view(1) for e in es], dim=0)
    Y = torch.stack([t.contiguous() if not t.is_contiguous() else t for t in ys], dim=0)
    return X, E, Y


def build_dataloaders(
    config: TrainingConfig,
    train_indexes: List[str],
    val_indexes: List[str],
) -> Tuple[DataLoader, DataLoader]:
    """Build train / val dataloaders from the resolved training config."""
    train_ds = H5PYGenerator(
        file_path=config.dataset_path,
        indexes=train_indexes,
        augmentation=config.augmentation,
        normalize=False,
        normalize_flux_only=config.normalize_flux_only,
        flux_mode=config.flux_mode,
        indexes_to_exclude_list=config.excluded_indexes_file,
    )
    val_ds = H5PYGenerator(
        file_path=config.dataset_path,
        indexes=val_indexes,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=config.normalize_flux_only,
        flux_mode=config.flux_mode,
        indexes_to_exclude_list=config.excluded_indexes_file,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        generator=generator,
        collate_fn=collate_h5,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_h5,
    )
    return train_loader, val_loader


class LimitedLoader:
    """Wrap a DataLoader and stop after ``max_batches`` iterations."""

    def __init__(self, loader: DataLoader, max_batches: Optional[int]):
        self._loader = loader
        self._max = max_batches
        self.dataset = loader.dataset

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if self._max is not None and i >= self._max:
                break
            yield batch

    def __len__(self) -> int:
        if self._max is None:
            return len(self._loader)
        return min(self._max, len(self._loader))


def limited_loader(loader: DataLoader, max_batches: Optional[int]):
    """Return ``loader`` unchanged, or wrapped to stop after ``max_batches``."""
    if max_batches is None:
        return loader
    return LimitedLoader(loader, max_batches)
