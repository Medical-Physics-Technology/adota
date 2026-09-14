"""Uniform sample sources for the evaluation engine.

A :class:`SampleSource` yields :class:`Sample` records with identical per-sample
semantics to the current scripts: tensors are exactly what the underlying loader
returns (same normalization, same shapes, on CPU), and no batching or persistent
file handle is introduced here (those are Part 2 of the refactor).

* :class:`DirSource` wraps :func:`src.loaders.dir_based.get_single_record`
  (the directory-of-``.npy`` layout used by ``run_model.py``).
* :class:`H5Source` wraps an :class:`src.loaders.generator.H5PYGenerator`
  (the HDF5 layout used by ``run_model_h5py.py`` and the training-set analyses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, Sequence, Union, runtime_checkable

import torch

from src.loaders.dir_based import get_single_record
from src.loaders.generator import H5PYGenerator


@dataclass
class Sample:
    """One loaded record, before any device move or metric computation.

    Attributes:
        sample_id: Stable identifier (directory sample id or H5 record id).
        x: Normalized model input, shape ``(C, D, H, W)`` (CT + flux channels).
        energy: Normalized initial-energy tensor as the loader returns it.
        y: Normalized ground-truth dose, shape ``(1, D, H, W)``.
        extra: Source-specific extras (e.g. ``{"beamlet_angles": (...)}``).
    """

    sample_id: str
    x: torch.Tensor
    energy: torch.Tensor
    y: torch.Tensor
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SampleSource(Protocol):
    """Iterable of :class:`Sample` with a known length."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Sample]: ...


class DirSource:
    """Directory-backed source wrapping :func:`get_single_record`.

    Yields samples for the given ``sample_ids`` in order, with the same loader
    arguments ``run_model.py`` uses. ``extra`` carries ``beamlet_angles`` (or
    ``None`` when the record has no angles).
    """

    def __init__(
        self,
        test_data_path: Union[str, Path],
        sample_ids: Sequence[str],
        *,
        scale: dict,
        normalize_flux: bool = True,
        downsampling_method: str = "interpolation",
        beamlet_angle: bool = True,
    ):
        self.test_data_path = str(test_data_path)
        self.sample_ids = list(sample_ids)
        self.scale = scale
        self.normalize_flux = normalize_flux
        self.downsampling_method = downsampling_method
        self.beamlet_angle = beamlet_angle

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __iter__(self) -> Iterator[Sample]:
        for sid in self.sample_ids:
            yield self.load(sid)

    def load(self, sample_id: str) -> Sample:
        """Load a single record by id (mirrors get_single_record exactly)."""
        record = get_single_record(
            sample_id,
            self.test_data_path,
            scale=self.scale,
            normalize_flux=self.normalize_flux,
            downsampling_method=self.downsampling_method,
            beamlet_angle=self.beamlet_angle,
        )
        if len(record) == 4:
            x, energy, y, beamlet_angles = record
        else:
            x, energy, y = record
            beamlet_angles = None
        return Sample(
            sample_id=sample_id,
            x=x,
            energy=energy,
            y=y,
            extra={"beamlet_angles": beamlet_angles},
        )


class MultiDirSource:
    """A :class:`DirSource` spanning several directories.

    The Monte Carlo generators write one directory per ``(patient, gantry, energy)``,
    so any set assembled across patients -- the active-learning validation set, a
    labelled batch -- is a list of ``(directory, sample_id)`` pairs rather than one
    directory with many ids. Loading is otherwise identical to :class:`DirSource`;
    ``sample_id`` stays the stem, and ``extra`` also carries the directory so a
    result row can be traced back to the field it came from.
    """

    def __init__(
        self,
        entries: Sequence[tuple],
        *,
        scale: dict,
        normalize_flux: bool = True,
        downsampling_method: str = "interpolation",
        beamlet_angle: bool = True,
    ):
        self.entries = [(str(d), str(s)) for d, s in entries]
        self._sources = {
            directory: DirSource(directory, [], scale=scale, normalize_flux=normalize_flux,
                                 downsampling_method=downsampling_method,
                                 beamlet_angle=beamlet_angle)
            for directory in {d for d, _ in self.entries}
        }

    @property
    def sample_ids(self) -> list:
        return [s for _, s in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Sample]:
        for directory, sample_id in self.entries:
            yield self.load(directory, sample_id)

    def load(self, directory: str, sample_id: str) -> Sample:
        sample = self._sources[directory].load(sample_id)
        sample.extra["directory"] = directory
        return sample


class H5Source:
    """HDF5-backed source wrapping an existing :class:`H5PYGenerator`.

    Iterates the generator by index, pairing each sample with its record id.
    Per-sample semantics (per-``__getitem__`` file open, crop, normalization)
    are unchanged; the persistent-handle optimization is deferred to Part 2.
    """

    def __init__(
        self,
        dataset: H5PYGenerator,
        record_ids: Optional[Sequence[str]] = None,
    ):
        self.dataset = dataset
        self.record_ids = (
            list(record_ids) if record_ids is not None else list(dataset.record_ids)
        )
        if len(self.record_ids) != len(dataset):
            raise ValueError(
                f"record_ids length ({len(self.record_ids)}) does not match "
                f"dataset length ({len(dataset)})"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[Sample]:
        for idx in range(len(self.dataset)):
            yield self.load(idx)

    def load(self, idx: int) -> Sample:
        """Load the record at dataset index ``idx``."""
        x, energy, y = self.dataset[idx]
        return Sample(
            sample_id=self.record_ids[idx],
            x=x,
            energy=energy,
            y=y,
            extra={},
        )
