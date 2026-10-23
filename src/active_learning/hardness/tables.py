"""The two analysis tables: physics metrics of every beamlet joined with every
engine's errors.

- **Table A**, HPO validation records of the beamlet HDF5: the metrics come from
  :func:`src.acquisition.reference.record_features` in ``analytic`` mode (the
  input-only arm the deployed score was fitted on), the errors from the
  engines' per-record CSVs.
- **Table B**, the saved beamlets of the engine-comparison blocks: the metrics
  come from the saved CT and flux through
  :func:`src.acquisition.candidates.model_grid_inputs` and
  :func:`~src.acquisition.candidates.analytic_features`, the path a candidate
  takes before simulation; the errors from :func:`.engines.score_blocks`.

Error columns are named ``err_<metric>__<engine>`` with the metrics of
:data:`src.active_learning.ensemble.agreement.ERROR_METRICS`, oriented so that
larger is worse.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

from src.acquisition.candidates import analytic_features, model_grid_inputs
from src.acquisition.features import FEATURE_NAMES, FeatureConfig
from src.acquisition.reference import read_stored_record, record_features
from src.active_learning.ensemble.agreement import ERROR_METRICS
from src.active_learning.retrospective.sampling import energy_bin
from src.datasets.beamlet_h5 import sha256_of

logger = logging.getLogger(__name__)

SITE_OF_ANATOMY = {"thorax": "lung", "thoracic": "lung", "pelvic_abdominal": "colorectal",
                   "abdominal": "colorectal"}
"""Both tables' anatomy labels mapped onto one site name."""


def error_column(metric: str, engine: str) -> str:
    return f"err_{metric}__{engine}"


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def _h5_chunk(h5_path: str, ids: Sequence[str]) -> List[Dict]:
    rows = []
    with h5py.File(h5_path, "r") as h5:
        for sid in ids:
            out = record_features(read_stored_record(h5[sid], sid), "analytic", FeatureConfig(),
                                  max_energy_mev=None)
            rows.append(out if out is not None else {"sample_id": sid, "peak_inside_crop": False,
                                                     "feature_skip": "no_flux"})
    return rows


def _block_chunk(entries: Sequence[Sequence[str]]) -> List[Dict]:
    rows = []
    for sample_id, directory, stem in entries:
        base = Path(directory) / stem
        meta = json.loads(Path(f"{base}_sim_res.json").read_text())
        energy = float(meta["initial_energy"])
        features, inside = analytic_features(*model_grid_inputs(np.load(f"{base}_ct.npy"),
                                                                np.load(f"{base}_flux.npy"), energy), energy)
        rows.append({"sample_id": sample_id, **features, "peak_inside_crop": inside})
    return rows


def _run_pool(fn, chunks: Sequence, workers: int, what: str, *args) -> pd.DataFrame:
    rows: List[Dict] = []
    started = perf_counter()
    if workers <= 1:
        _worker_init()
        for i, chunk in enumerate(chunks, 1):
            rows += fn(*args, chunk)
            logger.info("%s features: chunk %d/%d, %.0f s", what, i, len(chunks), perf_counter() - started)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = [pool.submit(fn, *args, chunk) for chunk in chunks]
            for i, fut in enumerate(as_completed(futures), 1):
                rows += fut.result()
                if i % max(1, len(chunks) // 10) == 0 or i == len(chunks):
                    logger.info("%s features: chunk %d/%d, %.0f s", what, i, len(chunks), perf_counter() - started)
    return pd.DataFrame(rows)


def h5_features(h5_path: Path, ids: Sequence[str], workers: int = 1, chunk: int = 100) -> pd.DataFrame:
    """The thirty metrics of HDF5 records, analytic arm, every energy kept."""
    ids = list(ids)
    chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    frame = _run_pool(_h5_chunk, chunks, workers, "table A", str(h5_path))
    return frame.set_index("sample_id").loc[ids].reset_index()


def block_features(entries: pd.DataFrame, workers: int = 1, chunk: int = 100) -> pd.DataFrame:
    """The thirty metrics of saved block beamlets (``sample_id``, ``block_dir``, ``stem``)."""
    triples = entries[["sample_id", "block_dir", "stem"]].astype(str).to_numpy().tolist()
    chunks = [triples[i:i + chunk] for i in range(0, len(triples), chunk)]
    frame = _run_pool(_block_chunk, chunks, workers, "table B")
    return frame.set_index("sample_id").loc[entries["sample_id"]].reset_index()


def error_columns(errors: pd.DataFrame, engine: str, metrics: Sequence[str]) -> pd.DataFrame:
    """``err_<metric>__<engine>`` columns of one engine's per-record errors, by ``sample_id``."""
    frame = errors.set_index("sample_id")
    return pd.DataFrame({error_column(m, engine): ERROR_METRICS[m](frame).astype(float) for m in metrics})


# --------------------------------------------------------------------------- table A

def table_a_metadata(index_csv: Path, records_csv: Optional[Path], ids: Sequence[str]) -> pd.DataFrame:
    """Scan, energy, 20 MeV bin, cell and site of every HPO validation record."""
    index = pd.read_csv(index_csv, usecols=["sample_id", "patient_key", "energy_mev", "source_dataset"],
                        dtype={"sample_id": str, "patient_key": str})
    index = index.drop_duplicates("sample_id").set_index("sample_id").loc[list(ids)]
    out = pd.DataFrame({"sample_id": list(ids), "scan": index["patient_key"].to_numpy(),
                        "energy_mev": index["energy_mev"].to_numpy(dtype=float),
                        "source_dataset": index["source_dataset"].to_numpy()})
    if records_csv is not None and Path(records_csv).exists():
        anatomy = pd.read_csv(records_csv, usecols=["sample_id", "anatomy"], dtype=str) \
            .drop_duplicates("sample_id").set_index("sample_id")["anatomy"]
        out["anatomy"] = anatomy.reindex(out["sample_id"]).to_numpy()
    else:
        out["anatomy"] = out["source_dataset"].map({"trainset_pelvis": "pelvic_abdominal",
                                                    "initial_test_one_ct": "thorax"})
    out["site"] = out["anatomy"].map(SITE_OF_ANATOMY)
    out["energy_bin"] = energy_bin(out["energy_mev"]).to_numpy()
    out["cell"] = out["scan"] + "|" + out["energy_bin"]
    return out


# --------------------------------------------------------------------------- table B

def table_b_entries(all_beamlets_csv: Path) -> pd.DataFrame:
    """The comparison's beamlets with their scan, energy, cell and site; the
    sample id is ``<block>/<stem>``."""
    frame = pd.read_csv(all_beamlets_csv)
    out = pd.DataFrame({"sample_id": frame["block"].astype(str) + "/" + frame["stem"].astype(str),
                        "block": frame["block"].astype(str), "stem": frame["stem"].astype(str),
                        "block_dir": frame["block_dir"].astype(str),
                        "scan": frame["patient_id"].astype(str), "anatomy": frame["anatomy"].astype(str),
                        "energy_mev": frame["energy"].astype(float), "gantry_deg": frame["gantry"].astype(float),
                        "grid_ix": frame["grid_ix"], "grid_iy": frame["grid_iy"]})
    out["site"] = out["anatomy"].map(SITE_OF_ANATOMY)
    out["energy_bin"] = energy_bin(out["energy_mev"]).to_numpy()
    out["cell"] = out["scan"] + "|" + out["energy_bin"]
    if out["sample_id"].duplicated().any():
        raise ValueError("duplicate block/stem pairs in the comparison table")
    return out


def file_error_columns(all_beamlets_csv: Path, entries: pd.DataFrame, prefixes: Mapping[str, str]) -> pd.DataFrame:
    """The comparison file's own errors (1 mm grid, its gamma code), kept beside
    the recomputed ones as a cross-check: ``file_<engine>_<column>``."""
    frame = pd.read_csv(all_beamlets_csv)
    frame.index = frame["block"].astype(str) + "/" + frame["stem"].astype(str)
    out = {}
    for engine, prefix in prefixes.items():
        for column in [c for c in frame.columns if c.startswith(prefix + "_")]:
            out[f"file_{engine}_{column[len(prefix) + 1:]}"] = frame[column]
    return pd.DataFrame(out).reindex(entries["sample_id"]).reset_index(drop=True)


def geometry_leakage(entries: pd.DataFrame, index_csv: Path, tol_mm: float = 1.0) -> pd.DataFrame:
    """Per comparison CT, the training-set scans with the same grid along the
    axes the gantry rotation leaves alone (x size, z size, z origin, z spacing).

    The scan key of the beamlet HDF5 hashes the grid of the unrotated CT, which
    a rotated block no longer has, so this is the check available; an empty
    ``matches`` column means no training scan shares that geometry.
    """
    index = pd.read_csv(index_csv, usecols=["patient_key", "image_size_vox_0", "image_size_vox_2",
                                            "image_origin_mm_2", "image_spacing_mm_2"], dtype={"patient_key": str})
    scans = index.drop_duplicates("patient_key")
    rows = []
    for scan, part in entries.groupby("scan"):
        stem = part.iloc[0]
        meta = json.loads((Path(stem["block_dir"]) / f"{stem['stem']}_sim_res.json").read_text())
        size, origin, spacing = meta["image_size"], meta["image_origin"], meta["image_spacing"]
        hit = scans[(scans["image_size_vox_0"] == size[0]) & (scans["image_size_vox_2"] == size[2])
                    & ((scans["image_origin_mm_2"] - origin[2]).abs() <= tol_mm)
                    & ((scans["image_spacing_mm_2"] - spacing[2]).abs() <= 0.01)]
        rows.append({"scan": scan, "size_x": size[0], "size_z": size[2], "origin_z_mm": origin[2],
                     "spacing_z_mm": spacing[2], "n_matches": len(hit), "matches": " ".join(hit["patient_key"])})
    return pd.DataFrame(rows)


def input_provenance(path: Path) -> Dict[str, Any]:
    """What is known about one input: a hash for a file, and the git commit
    recorded by whichever run wrote it (``manifest.json`` or a block's
    ``*_provenance.json``)."""
    path = Path(path)
    out: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file() and path.stat().st_size < 2 * 1024 ** 3:
        out["sha256"] = sha256_of(path)
    for directory in (path if path.is_dir() else path.parent, *list(path.parents)[:3]):
        manifest = directory / "manifest.json"
        if manifest.exists():
            git = json.loads(manifest.read_text()).get("git") or {}
            out.update(commit=git.get("commit"), commit_dirty=git.get("dirty"), manifest=str(manifest))
            break
    else:
        for prov in sorted((path if path.is_dir() else path.parent).glob("*_provenance.json")):
            data = json.loads(prov.read_text())
            out.setdefault("provenance", {})[prov.name] = {k: data.get(k) for k in
                                                          ("adota_commit", "adota_uncommitted_changes",
                                                           "model_fingerprint", "model_dir") if k in data}
    return out


def feature_names() -> List[str]:
    return list(FEATURE_NAMES)
