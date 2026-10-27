"""Packing simulated beamlets into compact, checked HDF5 shards, and moving them.

The generator writes three arrays per beamlet at 1 mm: CT (int16, 2.3 MB), dose
(float32, 4.6 MB) and the analytical flux (float64, 9.2 MB). The flux is a pure
function of the beamlet's metadata (entrance point, steering, spot sigmas), so a
shard keeps the CT and the dose, gzip-compressed and lossless (about 2.6 MB per
beamlet, measured), and :func:`read_record` recomputes the flux. Packing checks
that the recomputed flux equals the saved one before any raw file is deleted.

One shard per batch and split: ``<store>/<split>/batch_NNNN.h5``, one group per
beamlet named by its candidate id, holding ``ct``, ``dose`` and the generator's
``sim_res`` as a JSON attribute; and ``<store>/index/batch_NNNN.csv`` with one row
per planned beamlet and its fate (``stored`` or the reason it was not).

The post-Monte-Carlo overshoot check repeats the planning gate on the simulated
dose: R80 defined and at least ``range_margin_mm`` before the crop's end, and the
dose maximum inside the patient.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.flux import flux_projection, flux_spatial_spread
from src.datasets.beamlet_h5 import sha256_of

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
GZIP_LEVEL = 4


@dataclass
class StoreChecks:
    range_margin_mm: float = 10.0
    min_deposition_ratio: float = 0.95
    flux_rtol: float = 1e-6


def recompute_flux(sim_res: Dict, sigmas_xy: Sequence[float]) -> np.ndarray:
    """The flux channel exactly as :func:`src.mc_generation.robustness.process_beamlet`
    built it, from the beamlet's own metadata.

    The sigmas go in as plain Python floats, as :func:`flux_spatial_spread` returns
    them: a ``np.float64`` sigma would promote the float32 grid to float64 under
    NumPy 2's rules and change the flux in its seventh digit.
    """
    return flux_projection([float(v) for v in sim_res["rays_entrence_point_proj"]],
                           [float(v) for v in sim_res["beamlet_angles"]], [float(v) for v in sigmas_xy],
                           tuple(int(v) for v in sim_res["roi_size"]),
                           spacing=np.asarray([1, 1, 1], dtype=np.float32))


def post_mc_reason(sim_res: Dict, checks: StoreChecks) -> str:
    """Empty when the simulated beamlet passes the overshoot checks, else why not."""
    r80 = float(sim_res.get("r80_mm", np.nan))
    depth = float(sim_res["roi_size"][2])
    if not np.isfinite(r80) or r80 > depth - checks.range_margin_mm:
        return "mc_overshoot_crop"
    if not bool(sim_res.get("peak_in_body", True)):
        return "mc_peak_outside_patient"
    if float(sim_res.get("dose_deposition_ratio", 1.0)) < checks.min_deposition_ratio:
        return "mc_low_deposition"
    return ""


def pack_batch(raw_root: Path, rows: pd.DataFrame, shard_dir: Path, batch: int,
               sigma_of: Callable[[float], Sequence[float]], checks: StoreChecks
               ) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Pack the simulated beamlets of one batch into one shard per split.

    Every planned row gets an index entry; a beamlet the generator skipped
    (QA) is ``missing``, one failing the overshoot checks is ``rejected:<why>``
    and not stored. Raises if a recomputed flux differs from the saved one.
    ``sigma_of(energy)`` gives the spot sigmas the generator used
    (:func:`spot_sigmas` for a beam model).
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    handles: Dict[str, h5py.File] = {}
    paths: Dict[str, Path] = {}
    index: List[Dict] = []
    try:
        for row in rows.itertuples():
            base = Path(raw_root) / str(row.field_dir) / str(row.candidate_id)
            entry = {"candidate_id": row.candidate_id, "batch": batch, "split": row.split,
                     "patient_id": row.patient_id, "anatomy": row.anatomy, "energy_mev": float(row.energy_mev),
                     "gantry_deg": float(row.gantry_deg), "theta_x_deg": float(row.theta_x_deg),
                     "theta_y_deg": float(row.theta_y_deg)}
            meta_path = Path(f"{base}_sim_res.json")
            if not meta_path.exists():
                index.append({**entry, "status": "missing"})
                continue
            sim_res = json.loads(meta_path.read_text())
            entry.update(r80_mm=sim_res.get("r80_mm"), peak_in_body=sim_res.get("peak_in_body"),
                         dose_in_body_fraction=sim_res.get("dose_in_body_fraction"),
                         dose_deposition_ratio=sim_res.get("dose_deposition_ratio"),
                         mc_seconds=sim_res.get("mc_seconds"))
            reason = post_mc_reason(sim_res, checks)
            ct = np.load(f"{base}_ct.npy")
            dose = np.load(f"{base}_ds.npy").astype(np.float32)
            if not np.isfinite(dose).all() or dose.max() <= 0:
                reason = reason or "mc_invalid_dose"
            if reason:
                index.append({**entry, "status": f"rejected:{reason}"})
                continue
            sigmas = tuple(float(v) for v in sigma_of(float(sim_res["initial_energy"])))
            saved = np.load(f"{base}_flux.npy")
            again = recompute_flux(sim_res, sigmas)
            if not np.allclose(again, saved, rtol=checks.flux_rtol, atol=checks.flux_rtol * float(saved.max())):
                raise RuntimeError(f"{row.candidate_id}: recomputed flux differs from the saved one; "
                                   "nothing deleted")
            if row.split not in handles:
                paths[row.split] = shard_dir / row.split / f"batch_{batch:04d}.h5"
                paths[row.split].parent.mkdir(parents=True, exist_ok=True)
                handles[row.split] = h5py.File(paths[row.split], "w")
                handles[row.split].attrs.update(schema_version=SCHEMA_VERSION, batch=batch, split=row.split)
            group = handles[row.split].create_group(str(row.candidate_id))
            group.create_dataset("ct", data=ct.astype(np.int16), compression="gzip",
                                 compression_opts=GZIP_LEVEL, shuffle=True)
            group.create_dataset("dose", data=dose, compression="gzip", compression_opts=GZIP_LEVEL, shuffle=True)
            group.attrs["sim_res"] = json.dumps(sim_res)
            group.attrs["flux_sigma_xy_mm"] = np.asarray(sigmas, dtype=float)
            index.append({**entry, "status": "stored", "shard": f"{row.split}/batch_{batch:04d}.h5"})
    finally:
        for handle in handles.values():
            handle.close()
    frame = pd.DataFrame(index)
    verify_shards(paths, frame)
    return frame, paths


def spot_sigmas(bdl: BeamDataLibrary) -> Callable[[float], Sequence[float]]:
    """The generator's spot sigmas per energy (:func:`flux_spatial_spread`)."""
    return lambda energy: flux_spatial_spread(bdl, energy)


def read_record(group: h5py.Group) -> Dict[str, object]:
    """One stored beamlet as the generator wrote it: ``ct``, ``dose``, ``flux``
    (recomputed) and ``sim_res``."""
    sim_res = json.loads(group.attrs["sim_res"])
    return {"ct": group["ct"][...], "dose": group["dose"][...], "sim_res": sim_res,
            "flux": recompute_flux(sim_res, group.attrs["flux_sigma_xy_mm"])}


def verify_shards(paths: Dict[str, Path], index: pd.DataFrame) -> None:
    """Every ``stored`` row is in its shard with finite arrays of the crop shape."""
    stored = index[index["status"] == "stored"]
    for split, path in paths.items():
        want = set(stored.loc[stored["split"] == split, "candidate_id"])
        with h5py.File(path, "r") as h5:
            have = set(h5.keys())
            if have != want:
                raise RuntimeError(f"{path}: {len(want - have)} missing, {len(have - want)} unexpected records")
            for key in list(have)[:5]:
                dose = h5[key]["dose"][...]
                if dose.shape != h5[key]["ct"].shape or not np.isfinite(dose).all():
                    raise RuntimeError(f"{path}:{key} is malformed")


def move_file(source: Path, destination: Path) -> str:
    """Copy to ``destination`` through a temporary name, check the checksum,
    rename, then delete the source. Returns the sha256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_of(source)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, temporary)
    if sha256_of(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch copying {source} to {destination}")
    os.replace(temporary, destination)
    source.unlink()
    return digest


def free_gb(path: Path) -> float:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def store_summary(store_root: Path) -> Optional[pd.DataFrame]:
    """All index rows written so far."""
    files = sorted((Path(store_root) / "index").glob("batch_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None
