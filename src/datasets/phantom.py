"""Synthetic phantom CT sources for MC generation (water box; extensible to slabs).

Produces ``CTRecord``-compatible records whose ``load_image()`` builds a synthetic
SimpleITK phantom, so the existing generation spine
(:func:`src.mc_generation.robustness.generate_for_record` /
:func:`~src.mc_generation.robustness.run_generation`) runs on phantoms **exactly**
as on real CTs -- no pipeline duplication. A phantom record is swept over the same
beamlet-angle grid, energies, ROI and QA gates as a real CT.

Currently supported: a homogeneous water box with an optional air shell. The
:class:`PhantomSpec` / :func:`build_phantom_image` split is deliberately open for
future high-density slab inserts (horizontal / vertical), which will add new
``kind`` branches without touching the generation code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.datasets.base import CTDataset


@dataclass(frozen=True)
class PhantomSpec:
    """Parameters that fully determine a synthetic phantom (content-addressable).

    Args:
        kind: Phantom type. ``"water"`` = homogeneous water box (with an optional
            air shell). Future: slab variants.
        size: Grid size in voxels ``(x, y, z)`` (SimpleITK order).
        spacing: Voxel spacing in mm.
        origin: World origin in mm.
        water_hu / air_hu: HU values for the water body and the surrounding air.
        air_layer_depth: Thickness (voxels ~= mm at 1 mm spacing) of an air shell
            on every face; ``0`` = plain water box (no air).
        name: Short variant label used as the record's ``patient_id`` (dir naming).
    """

    kind: str = "water"
    size: Tuple[int, int, int] = (512, 512, 512)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    water_hu: int = 0
    air_hu: int = -1024
    air_layer_depth: int = 0     # air shell on ALL faces (legacy; causes lateral grazing)
    air_front_mm: int = 0        # air slab ONLY in front of the beam (proximal x), full width
    name: str = "water"

    @property
    def content_hash(self) -> str:
        """Deterministic 16-hex id of the geometry (stable provenance key)."""
        payload = json.dumps(
            {k: getattr(self, k) for k in
             ("kind", "size", "spacing", "origin", "water_hu", "air_hu",
              "air_layer_depth", "air_front_mm")},
            sort_keys=True, default=list,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


def build_phantom_image(spec: PhantomSpec) -> sitk.Image:
    """Build the synthetic phantom as a SimpleITK image.

    The numpy array is ``(z, y, x)`` (SimpleITK order); an air shell, if requested,
    is a border of ``air_layer_depth`` voxels on all six faces around a water core.
    """
    sx, sy, sz = int(spec.size[0]), int(spec.size[1]), int(spec.size[2])
    d = int(spec.air_layer_depth)
    af = int(spec.air_front_mm)
    if spec.kind == "water":
        arr = np.full((sz, sy, sx), spec.water_hu, dtype=np.float32)
        if d > 0:  # legacy all-faces air shell
            if 2 * d >= min(sx, sy, sz):
                raise ValueError(f"air_layer_depth {d} too large for size {spec.size}")
            arr[:] = spec.air_hu
            arr[d:sz - d, d:sy - d, d:sx - d] = spec.water_hu
        if af > 0:  # full-width air slab in front of the beam (proximal x = last axis)
            if af >= sx:
                raise ValueError(f"air_front_mm {af} too large for x-size {sx}")
            arr[:, :, 0:af] = spec.air_hu
    else:
        raise ValueError(f"unknown phantom kind {spec.kind!r}")

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(s) for s in spec.spacing])
    img.SetOrigin([float(o) for o in spec.origin])
    img.SetDirection([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    return img


@dataclass(frozen=True)
class PhantomRecord:
    """A ``CTRecord``-compatible handle for a synthetic phantom (provenance-first).

    Duck-typed to what the generation spine needs (``dataset_name``, ``anatomy``,
    ``patient_id``, ``series_uid``, ``uid``, ``load_image``); provenance is derived
    from the deterministic :attr:`PhantomSpec.content_hash`, so a record is fully
    regenerable and traceable.
    """

    dataset_name: str
    anatomy: str
    patient_id: str
    spec: PhantomSpec

    @property
    def series_uid(self) -> str:
        return self.spec.content_hash

    @property
    def n_slices(self) -> int:
        return int(self.spec.size[2])

    @property
    def uid(self) -> str:
        return f"{self.dataset_name}/{self.patient_id}/{self.series_uid}"

    def load_image(self) -> sitk.Image:
        return build_phantom_image(self.spec)


class PhantomDataset(CTDataset):
    """A flat, provenance-first dataset of :class:`PhantomRecord`s."""

    def __init__(self, records: Sequence[PhantomRecord], name: str = "phantom"):
        self._records: List[PhantomRecord] = list(records)
        self.name = name
        self.anatomy = "phantom"

    def __len__(self) -> int:
        return len(self._records)

    def record(self, idx: int) -> PhantomRecord:
        return self._records[idx]


def build_phantom_dataset(cfg: dict) -> PhantomDataset:
    """Build a :class:`PhantomDataset` from a config's ``phantoms`` list of specs."""
    specs = cfg.get("phantoms")
    if not specs:
        raise ValueError("phantom config has no 'phantoms' list")
    name = cfg.get("name", "phantom")
    records: List[PhantomRecord] = []
    for p in specs:
        spec = PhantomSpec(
            kind=p.get("kind", "water"),
            size=tuple(p.get("size", (512, 512, 512))),
            spacing=tuple(p.get("spacing", (1.0, 1.0, 1.0))),
            origin=tuple(p.get("origin", (0.0, 0.0, 0.0))),
            water_hu=int(p.get("water_hu", 0)),
            air_hu=int(p.get("air_hu", -1024)),
            air_layer_depth=int(p.get("air_layer_depth", 0)),
            air_front_mm=int(p.get("air_front_mm", 0)),
            name=str(p.get("name", p.get("kind", "water"))),
        )
        records.append(PhantomRecord(
            dataset_name=name, anatomy=str(p.get("anatomy", "phantom")),
            patient_id=spec.name, spec=spec,
        ))
    return PhantomDataset(records, name=name)
