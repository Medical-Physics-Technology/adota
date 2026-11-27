"""The commissioned pencil beam on beamlet-pool records, on the model grid.

A pool record keeps only its 1 mm crop (CT and Monte Carlo dose) and the metadata
of its simulation. The pencil beam needs the whole field: its ray trace runs on the
rotated patient CT. So each field (one patient, one gantry angle, about 30 beamlets)
is rebuilt exactly as the pool generator built it (:func:`field_geometry` on the CT
the generator loaded), every beamlet's crop window is recomputed with the dataset's
own :func:`extract_beamlet_roi`, and the rebuilt CT crop must equal the stored one
voxel for voxel before any dose is kept. The dose of each beamlet is then taken on
that window (1 mm, ``(z, y, x)``), averaged down to 2 mm the way the export averaged
the Monte Carlo dose, and permuted to the model layout ``(D, H, W)``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from src.beamlets.cropping import extract_beamlet_roi
from src.datasets.beamlet_record import downsample_grid
from src.mc_generation.geometry import reduce_vacuum_to_air, resample_to_isotropic
from src.mc_generation.robustness import field_geometry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PoolBeamlet:
    """One stored pool record, as the pencil beam needs it."""

    sample_id: str
    patient_id: str
    gantry_deg: float
    energy_mev: float
    spot: Tuple[float, float]
    roi_size: Tuple[int, int, int]
    shard: Path


def pool_fields(store_root: Path, split: str, sample_ids: Optional[Sequence[str]] = None
                ) -> Dict[Tuple[str, float], List[PoolBeamlet]]:
    """The stored beamlets of one split, grouped by field ``(patient_id, gantry_deg)``.

    ``sample_ids`` restricts the selection (order is not kept). The energy, spot and
    crop size come from each record's own ``sim_res``.
    """
    store_root = Path(store_root)
    index = pd.concat([pd.read_csv(p) for p in sorted((store_root / "index").glob("batch_*.csv"))],
                      ignore_index=True)
    index = index[(index["split"] == split) & (index["status"] == "stored")]
    index["candidate_id"] = index["candidate_id"].astype(str)
    if sample_ids is not None:
        index = index[index["candidate_id"].isin(set(map(str, sample_ids)))]
    fields: Dict[Tuple[str, float], List[PoolBeamlet]] = {}
    for shard, rows in index.groupby("shard", sort=True):
        with h5py.File(store_root / shard, "r") as handle:
            for r in rows.itertuples(index=False):
                sim = json.loads(handle[r.candidate_id].attrs["sim_res"])
                beamlet = PoolBeamlet(
                    sample_id=r.candidate_id, patient_id=str(r.patient_id), gantry_deg=float(r.gantry_deg),
                    energy_mev=float(sim["initial_energy"]), spot=tuple(float(v) for v in sim["spot_position"]),
                    roi_size=tuple(int(v) for v in sim["roi_size"]), shard=store_root / shard)
                fields.setdefault((beamlet.patient_id, beamlet.gantry_deg), []).append(beamlet)
    return fields


def stored_crop(beamlet: PoolBeamlet, name: str = "ct") -> np.ndarray:
    """The stored 1 mm ``(z, y, x)`` array of a record (``ct`` or ``dose``)."""
    with h5py.File(beamlet.shard, "r") as handle:
        return np.asarray(handle[beamlet.sample_id][name][...])


def beamlet_windows(geom, bdl, beamlets: Sequence[PoolBeamlet]) -> Dict[str, Tuple[slice, slice, slice]]:
    """The ``(z, y, x)`` window of every beamlet on the rebuilt field grid, each checked against
    the stored CT crop voxel for voxel.

    Raises:
        RuntimeError: naming the beamlets whose rebuilt crop differs or leaves the grid.
    """
    windows, bad = {}, []
    for b in beamlets:
        h, w, d = b.roi_size
        crop, _, crp, oob = extract_beamlet_roi(geom.ct, *bdl.distances, list(b.spot), geom.iso_ext, b.roi_size,
                                                ct_array=geom.ct_array)
        stored = stored_crop(b, "ct")
        if oob or not np.array_equal(np.asarray(crop).astype(stored.dtype), stored):
            bad.append(b.sample_id)
            continue
        iz, iy, _ = crp
        windows[b.sample_id] = (slice(iz - h // 2, iz - h // 2 + h), slice(iy - w // 2, iy - w // 2 + w),
                                slice(0, d))
    if bad:
        raise RuntimeError(f"{len(bad)}/{len(beamlets)} rebuilt CT crops differ from the stored ones "
                           f"(e.g. {bad[:3]}); check the pool config, the CT source and the code version")
    return windows


def to_model_grid(dose_zyx_1mm: np.ndarray, method: str = "average") -> np.ndarray:
    """A 1 mm ``(z, y, x)`` crop on the 2 mm model grid ``(D, H, W)``, the export's averaging."""
    return np.ascontiguousarray(downsample_grid(np.asarray(dose_zyx_1mm, dtype=np.float32), method)
                                .transpose(2, 0, 1))


def field_doses(engine, ct, gantry_deg: float, rob_cfg, bdl, beamlets: Sequence[PoolBeamlet],
                method: str = "average") -> Tuple[Dict[str, np.ndarray], dict]:
    """Pencil-beam dose of every beamlet of one field on the model grid, and timings.

    ``ct`` is the patient CT as the generator loaded it (isotropic, vacuum set to air).
    """
    t0 = perf_counter()
    geom = field_geometry(ct, float(gantry_deg), rob_cfg, bdl)
    windows = beamlet_windows(geom, bdl, beamlets)
    t_geom = perf_counter() - t0
    field = engine.prepare_field(geom.ct, geom.iso_mc, geom.mc_gantry)
    doses, spot_seconds = {}, 0.0
    for b in beamlets:
        result = engine.dose_in_window(field, b.energy_mev, [b.spot], windows[b.sample_id])
        doses[b.sample_id] = to_model_grid(result.dose, method)
        spot_seconds += result.seconds
    return doses, {"geometry_seconds": t_geom, "raytrace_seconds": field.seconds, "spot_seconds": spot_seconds,
                   "model_energy_example": result.model_energy}


def patient_cts(resolver, entries: Mapping[str, object], patient_ids: Sequence[str], iso_spacing_mm: float
                ) -> Iterator[Tuple[str, object]]:
    """``(patient_id, CT)`` as the pool generator loaded each: resampled, vacuum set to air."""
    for pid in patient_ids:
        rec = resolver.record(entries[pid])
        yield pid, reduce_vacuum_to_air(resample_to_isotropic(rec.load_image(), iso_spacing_mm))


def save_dose(out_dir: Path, sample_id: str, dose: np.ndarray) -> Path:
    """Write one model-grid dose, never replacing an existing file."""
    path = Path(out_dir) / f"{sample_id}.npy"
    with open(path, "xb") as handle:
        np.save(handle, dose.astype(np.float32))
    return path


__all__ = ["PoolBeamlet", "beamlet_windows", "field_doses", "patient_cts", "pool_fields", "save_dose",
           "stored_crop", "to_model_grid"]
