"""Training on what the loop bought, unioned with the designated training set.

Newly labelled beamlets are Monte Carlo output directories at 1 mm; the reference
training set is the HDF5 file. Rather than rewrite the HDF5 every cycle, the two are
concatenated and the training set becomes a list of sources in the cycle manifest.

The directory side goes through :func:`src.loaders.dir_based.get_single_record`, the
same trilinear resample to the model's 2 mm grid that inference uses, so a bought
beamlet and a reference beamlet reach the model in the same frame. The one difference
is augmentation: HDF5 records are stored as a 40x40x200 window and get a moving crop
around the Bragg peak, while a Monte Carlo directory holds exactly the 60x60x320 crop,
so directory records get the lateral rot90 and no moving window.

Sampling is the part that departs from a plain concatenation. A cycle buys a few
thousand beamlets against a training set of tens of thousands; under uniform sampling
a new beamlet would be seen once every few epochs and the cycle would measure nothing.
:func:`build_union_dataloaders` therefore draws each batch with a fixed share of new
beamlets (``al_oversample_fraction``) and defines a cycle as a fixed number of
optimizer steps. This is a deliberate departure from the design document's
"continue on the full union", recorded as such: the arms stay comparable to each
other because both use it, but neither is comparable to a plain-union baseline.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from src.loaders.dir_based import get_single_record
from src.loaders.generator import H5PYGenerator
from src.training.data import collate_h5

logger = logging.getLogger(__name__)


class DirBeamletDataset(Dataset):
    """Monte Carlo beamlet directories as a training dataset.

    Args:
        entries: ``(directory, stem)`` pairs, one per beamlet.
        scale: The run's MinMax scaling; must match the HDF5 side.
        augmentation: Apply the lateral rot90 the HDF5 pipeline applies.
        preload: Decode every record once into memory (about 1.7 MB each) instead of
            re-reading and re-interpolating 14 MB of numpy on every draw. With
            oversampling a record is drawn many times per cycle, so this is the
            difference between an I/O-bound and a GPU-bound loop.
    """

    def __init__(self, entries: Sequence[Tuple[str, str]], *, scale: dict,
                 augmentation: bool = True, preload: bool = True,
                 normalize_flux: bool = True):
        self.entries = [(str(d), str(s)) for d, s in entries]
        self.scale = dict(scale)
        self.augmentation = augmentation
        self.normalize_flux = normalize_flux
        self.record_ids = [s for _, s in self.entries]
        self._cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None
        if preload:
            self._cache = [self._load(i) for i in range(len(self.entries))]
            nbytes = sum(t.element_size() * t.nelement()
                         for record in self._cache for t in record)
            logger.info("DirBeamletDataset: preloaded %d records (%.1f GB)",
                        len(self._cache), nbytes / 1e9)

    def _load(self, idx: int):
        directory, stem = self.entries[idx]
        x, energy, y = get_single_record(
            stem, directory, scale=self.scale, normalize_flux=self.normalize_flux,
            downsampling_method="interpolation")
        return x.contiguous(), energy, y.contiguous()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        x, energy, y = self._cache[idx] if self._cache is not None else self._load(idx)
        if self.augmentation:
            k = int(np.random.choice(4))
            if k:
                x = torch.rot90(x, k, dims=(2, 3)).contiguous()
                y = torch.rot90(y, k, dims=(2, 3)).contiguous()
        return x, energy, y


def read_training_sources(path: Path) -> List[Tuple[str, str]]:
    """Read the loop's accumulated ``(dir, stem)`` training sources."""
    with Path(path).open() as handle:
        return [(row["dir"], row["stem"]) for row in csv.DictReader(handle)]


def write_training_sources(entries: Sequence[Tuple[str, str]], path: Path) -> Path:
    """Write the accumulated sources, de-duplicated, in first-seen order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for directory, stem in entries:
        if (directory, stem) in seen:
            continue
        seen.add((directory, stem))
        rows.append({"dir": directory, "stem": stem})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dir", "stem"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _h5_datasets(config, train_indexes, val_indexes):
    """The HDF5 train and val generators, exactly as ``build_dataloaders`` builds them."""
    common = dict(file_path=config.dataset_path, normalize=False,
                  normalize_flux_only=config.normalize_flux_only,
                  flux_mode=config.flux_mode, centerline_sidecar=config.centerline_sidecar,
                  indexes_to_exclude_list=config.excluded_indexes_file)
    train_ds = H5PYGenerator(indexes=train_indexes, augmentation=config.augmentation, **common)
    val_ds = H5PYGenerator(indexes=val_indexes, augmentation=False, cropp=True, **common)
    return train_ds, val_ds


def union_sampler(n_h5: int, n_dir: int, fraction: float, steps: int,
                  batch_size: int, generator: torch.Generator) -> WeightedRandomSampler:
    """Draw batches holding ``fraction`` new beamlets on average, for ``steps`` steps.

    Weights are per record: every new beamlet carries ``fraction / n_dir`` of the
    probability mass and every reference record ``(1 - fraction) / n_h5``, so the
    share is what was asked for whatever the two set sizes are. Sampling is with
    replacement, which is the point: a new beamlet is meant to be seen many times.
    """
    if n_dir == 0:
        raise ValueError("union sampler needs at least one new beamlet")
    weights = np.concatenate([
        np.full(n_h5, (1.0 - fraction) / max(n_h5, 1), dtype=np.float64),
        np.full(n_dir, fraction / n_dir, dtype=np.float64)])
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=steps * batch_size, replacement=True, generator=generator)


def build_union_dataloaders(config, train_indexes: List[str], val_indexes: List[str]
                            ) -> Tuple[DataLoader, DataLoader]:
    """Train on the HDF5 set unioned with the loop's beamlets; validate on the HDF5.

    The validation loader is the HDF5 split untouched, so it keeps meaning what it
    meant for every earlier run and stays free of the beamlets a cycle just bought.
    The active-learning yardstick is a separate frozen set, evaluated by
    :mod:`src.active_learning.validation` between cycles.
    """
    train_h5, val_ds = _h5_datasets(config, train_indexes, val_indexes)
    entries: List[Tuple[str, str]] = []
    for source in config.al_dir_sources:
        entries.extend(read_training_sources(Path(source)))
    dir_ds = DirBeamletDataset(entries, scale=config.scale,
                               augmentation=config.augmentation,
                               preload=config.al_preload_dir_records)
    union = ConcatDataset([train_h5, dir_ds])

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    steps = config.al_steps_per_epoch or max(1, len(train_h5) // config.batch_size)
    sampler = union_sampler(len(train_h5), len(dir_ds), config.al_oversample_fraction,
                            steps, config.batch_size, generator)
    logger.info("union training set: %d HDF5 + %d new beamlets | %d steps/epoch, "
                "%.0f%% new per batch", len(train_h5), len(dir_ds), steps,
                100.0 * config.al_oversample_fraction)

    train_loader = DataLoader(
        union, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0, collate_fn=collate_h5)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
        persistent_workers=config.num_workers > 0, collate_fn=collate_h5)
    return train_loader, val_loader
