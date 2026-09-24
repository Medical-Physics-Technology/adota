"""Pure per-record recipe for the v3 beamlet HDF5 build (spec section 6.1).

Loading, flux projection, normalisation and downsampling for a single raw Monte
Carlo record, kept free of any HDF5/CLI/process-pool concerns so the recipe can be
pinned and tested in isolation. ``list_record_ids`` and ``load_raw_record`` are the
only functions that touch the filesystem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.flux import flux_projection_gpu_batched, flux_spatial_spread

RECORD_SUFFIXES: Dict[str, str] = {"ct": "_ct.npy", "ds": "_ds.npy", "metadata": "_metadata.json"}
DOWNSAMPLE_METHODS: Tuple[str, ...] = ("average", "linear", "trilinear")
# float64 on purpose (spec 2.7): NEP 50 lets a Python-float sigma divided by a
# float32 spacing demote silently, which is the exact bug the recipe pins around.
FLUX_SPACING_MM: np.ndarray = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
EXPECTED_LATERAL: Tuple[int, int] = (40, 40)
MIN_DEPTH: int = 160
REQUIRED_METADATA_KEYS: Tuple[str, ...] = (
    "simulation_log.energy",
    "simulation_log.beamlet_angles",
    "rays_entrence_point_proj",
    "id",
    "gantry_angle",
    "dose_deposition_ratio",
    "stat_uncertainty",
    "simulation_log.n_spots",
    "simulation_log.bixelgrid_shifts_xy",
)


@dataclass
class RawRecord:
    sample_id: str
    ct: np.ndarray
    ds: np.ndarray
    metadata: dict
    metadata_text: str


@dataclass
class FluxInputs:
    sample_id: str
    entrance_proj: Tuple[float, float, float]
    beamlet_angles: Tuple[float, float]
    sigmas_xy: Tuple[float, float]
    shape: Tuple[int, int, int]
    energy_mev: float


@dataclass
class PreprocessedRecord:
    sample_id: str
    ct: np.ndarray
    dose: np.ndarray
    flux: np.ndarray
    initial_energy_norm: float
    energy_mev: float
    sigmas_xy: Tuple[float, float]


def _get_path(metadata: dict, dotted_path: str) -> object:
    """Walk ``dotted_path`` in ``metadata``, raising ``KeyError(dotted_path)`` if missing."""
    node = metadata
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted_path)
        node = node[part]
    return node


def list_record_ids(record_dir: Path) -> List[str]:
    """Sorted ids of every ``*_metadata.json`` in ``record_dir`` (ignores other files)."""
    record_dir = Path(record_dir)
    suffix = RECORD_SUFFIXES["metadata"]
    return sorted(p.name[: -len(suffix)] for p in record_dir.glob(f"*{suffix}"))


def load_raw_record(record_dir: Path, sample_id: str) -> RawRecord:
    """Load the raw ``(ct, ds, metadata)`` triple for ``sample_id``.

    Raises whatever ``np.load``/``json.loads`` raise for a missing or corrupt file,
    and ``KeyError`` naming the dotted path of the first missing required metadata
    key. Nothing here is swallowed.
    """
    record_dir = Path(record_dir)
    ct = np.load(record_dir / f"{sample_id}{RECORD_SUFFIXES['ct']}")
    ds = np.load(record_dir / f"{sample_id}{RECORD_SUFFIXES['ds']}")
    metadata_text = (record_dir / f"{sample_id}{RECORD_SUFFIXES['metadata']}").read_text()
    metadata = json.loads(metadata_text)
    for key in REQUIRED_METADATA_KEYS:
        _get_path(metadata, key)
    return RawRecord(sample_id=sample_id, ct=ct, ds=ds, metadata=metadata, metadata_text=metadata_text)


def flux_inputs(raw: RawRecord, bdl: BeamDataLibrary) -> FluxInputs:
    """Build the :class:`FluxInputs` for ``raw`` (energy from ``simulation_log.energy[0]``)."""
    energy = float(raw.metadata["simulation_log"]["energy"][0])
    sigmas = flux_spatial_spread(bdl, energy)
    entrance = tuple(float(v) for v in raw.metadata["rays_entrence_point_proj"])
    angles = tuple(float(v) for v in raw.metadata["simulation_log"]["beamlet_angles"])
    return FluxInputs(
        sample_id=raw.sample_id,
        entrance_proj=entrance,
        beamlet_angles=angles,
        sigmas_xy=sigmas,
        shape=tuple(int(v) for v in raw.ds.shape),
        energy_mev=energy,
    )


def flux_batch(inputs: Sequence[FluxInputs], device: str = "cuda") -> List[np.ndarray]:
    """Compute float64 flux projections for ``inputs``, batched per raw shape.

    One ``flux_projection_gpu_batched`` call per distinct shape group (not per
    input), with ``initial_energies=None``, ``spacing=FLUX_SPACING_MM``,
    ``dtype=torch.float64``, ``return_numpy=True``. Results are returned in input
    order. ``torch.cuda.OutOfMemoryError`` propagates uncaught.
    """
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for idx, item in enumerate(inputs):
        groups.setdefault(item.shape, []).append(idx)

    results: List[Optional[np.ndarray]] = [None] * len(inputs)
    for shape, indices in groups.items():
        entrances = [inputs[i].entrance_proj for i in indices]
        angles = [inputs[i].beamlet_angles for i in indices]
        sigmas = [inputs[i].sigmas_xy for i in indices]
        batch = flux_projection_gpu_batched(
            entrances,
            angles,
            sigmas,
            shape,
            initial_energies=None,
            spacing=FLUX_SPACING_MM,
            device=device,
            dtype=torch.float64,
            return_numpy=True,
        )
        for pos, i in enumerate(indices):
            results[i] = batch[pos]
    return results  # type: ignore[return-value]


def downsample_grid(grid: np.ndarray, method: str = "average") -> np.ndarray:
    """Downsample ``grid`` by a factor of 2 per axis, float32 out (spec section 3)."""
    if method == "average":
        tensor = torch.tensor(grid, dtype=torch.float32)[None, None]
        pooled = F.avg_pool3d(tensor, kernel_size=2, stride=2)
        return pooled.squeeze().numpy()
    if method == "linear":
        return ndimage.zoom(grid, zoom=(0.5, 0.5, 0.5), order=1).astype(np.float32)
    if method == "trilinear":
        tensor = torch.tensor(grid, dtype=torch.float32)[None, None]
        size = [s // 2 for s in grid.shape]
        pooled = F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
        return pooled.squeeze().numpy()
    raise ValueError(f"Unknown downsample method: {method!r}")


def normalise_ct(ct: np.ndarray, scale: Mapping) -> np.ndarray:
    return (ct - scale["min_ct"]) / (scale["max_ct"] - scale["min_ct"])


def normalise_dose(ds: np.ndarray, scale: Mapping) -> np.ndarray:
    return (ds - scale["min_ds"]) / (scale["max_ds"] - scale["min_ds"])


def normalise_energy(energy_mev: float, scale: Mapping) -> float:
    return (energy_mev - scale["min_energy"]) / (scale["max_energy"] - scale["min_energy"])


def preprocess_ct_dose(raw: RawRecord, scale: Mapping, method: str = "average") -> Tuple[np.ndarray, np.ndarray]:
    """Normalise then downsample ``raw.ct``/``raw.ds``, float32 out."""
    ct_pooled = downsample_grid(normalise_ct(raw.ct, scale), method)
    dose_pooled = downsample_grid(normalise_dose(raw.ds, scale), method)
    return ct_pooled, dose_pooled


def preprocess_flux(flux_f64: np.ndarray, method: str = "average") -> np.ndarray:
    return downsample_grid(flux_f64, method)


def finish_record(
    raw: RawRecord,
    flux_f64: np.ndarray,
    scale: Mapping,
    sigmas_xy: Tuple[float, float],
    method: str = "average",
) -> PreprocessedRecord:
    """Compose the full per-record recipe into a :class:`PreprocessedRecord`."""
    ct_pooled, dose_pooled = preprocess_ct_dose(raw, scale, method)
    flux_pooled = preprocess_flux(flux_f64, method)
    energy_mev = float(raw.metadata["simulation_log"]["energy"][0])
    initial_energy_norm = normalise_energy(energy_mev, scale)
    return PreprocessedRecord(
        sample_id=raw.sample_id,
        ct=ct_pooled,
        dose=dose_pooled,
        flux=flux_pooled,
        initial_energy_norm=initial_energy_norm,
        energy_mev=energy_mev,
        sigmas_xy=(float(sigmas_xy[0]), float(sigmas_xy[1])),
    )


def skip_reason(ct: np.ndarray, dose: np.ndarray, flux: Optional[np.ndarray] = None) -> Optional[str]:
    """``"zero_dose"`` | ``"bad_shape"`` | ``"short_depth"`` | ``None``, in that order."""
    if dose.sum() == 0:
        return "zero_dose"
    arrays = (ct, dose) if flux is None else (ct, dose, flux)
    for arr in arrays:
        if tuple(arr.shape[:2]) != EXPECTED_LATERAL:
            return "bad_shape"
    for arr in arrays:
        if arr.shape[2] < MIN_DEPTH:
            return "short_depth"
    return None


def spots_reason(metadata: dict) -> Optional[str]:
    """``"n_spots"`` when the record is not single-spot, else ``None``."""
    sim = metadata["simulation_log"]
    if sim["n_spots"] != 1 or len(sim["bixelgrid_shifts_xy"]) != 1:
        return "n_spots"
    return None
