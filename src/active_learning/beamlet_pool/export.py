"""Export a split of the beamlet pool to the v3 training HDF5 layout.

The pool stores each beamlet at 1 mm on its own (60, 60, 320) crop, with the
flux recomputed on read (:mod:`.store`). Every reader of the training data
(:class:`src.loaders.generator.H5PYGenerator`, the retrospective validation
code, the ensemble members) expects the v3 layout instead: normalised CT and
dose and the flux, average-pooled to 2 mm, one group per record with the v2 +
v3 attributes. This module writes exactly that, through the same recipe the v3
build uses (:func:`src.datasets.beamlet_record.finish_record`,
:func:`src.datasets.beamlet_h5.write_record`), so a pool record and a training
record of the same beamlet would be identical.

The pool crop is 60 mm wide and 320 mm deep, so at 2 mm it is (30, 30, 160),
the model input itself: the validation loader's crop around the peak is the
identity on it. The pool's own identity rides along as ``pool_*`` attributes
(candidate id, patient, anatomy, collection, split, shard), which the index CSV
picks up.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import h5py
import numpy as np

from src.active_learning.beamlet_pool.store import read_record
from src.datasets.beamlet_h5 import RecordProvenance, index_row, write_file_attrs, write_index_csv, write_record
from src.datasets.beamlet_record import PreprocessedRecord, RawRecord, finish_record

logger = logging.getLogger(__name__)

SOURCE_DATASET = "beamlet_pool"
FLUX_COMPUTE = "flux_projection/cpu (src.active_learning.beamlet_pool.store.recompute_flux)"
POOL_ATTRS = ("pool_candidate_id", "pool_patient_id", "pool_anatomy", "pool_dataset", "pool_split", "pool_shard")


def pool_metadata(sim_res: Mapping) -> dict:
    """The pool's ``sim_res`` in the shape the v3 writer reads.

    The pool keeps ``beamlet_angles`` and ``num_primaries`` at the top level; the
    v3 writer reads them from ``simulation_log`` (and the primaries from its
    ``sim_params``), as the training generator wrote them.
    """
    meta = copy.deepcopy(dict(sim_res))
    sim = meta["simulation_log"]
    sim.setdefault("beamlet_angles", list(meta["beamlet_angles"]))
    sim.setdefault("sim_params", {"Num_Primaries": float(meta["num_primaries"])})
    sim.setdefault("n_spots", 1)
    return meta


def preprocess_pool_record(candidate_id: str, record: Mapping, scale: Mapping,
                           method: str = "average") -> Tuple[PreprocessedRecord, dict, str]:
    """One stored pool beamlet through the v3 recipe: ``(record, metadata, metadata_text)``."""
    meta = pool_metadata(record["sim_res"])
    text = json.dumps(meta)
    raw = RawRecord(sample_id=str(candidate_id), ct=np.asarray(record["ct"]),
                    ds=np.asarray(record["dose"], dtype=np.float32), metadata=meta, metadata_text=text)
    sigmas = tuple(float(v) for v in record["sigmas_xy"])
    pre = finish_record(raw, np.asarray(record["flux"], dtype=np.float64), scale, sigmas, method)
    return pre, meta, text


def iter_split(store_root: Path, split: str, limit: Optional[int] = None) -> Iterator[Tuple[str, str, dict]]:
    """``(shard, candidate_id, record)`` over a split's shards, in shard then id order."""
    shards = sorted((Path(store_root) / split).glob("batch_*.h5"))
    if not shards:
        raise FileNotFoundError(f"no shards under {Path(store_root) / split}")
    n = 0
    for shard in shards:
        with h5py.File(shard, "r") as h5:
            for candidate_id in sorted(h5.keys()):
                if limit is not None and n >= limit:
                    return
                group = h5[candidate_id]
                record = read_record(group)
                record["sigmas_xy"] = np.asarray(group.attrs["flux_sigma_xy_mm"], dtype=float).tolist()
                yield f"{split}/{shard.name}", candidate_id, record
                n += 1


def export_split(store_root: Path, split: str, out_path: Path, *, scale: Mapping, bdl_file: str,
                 bdl_sha256: str, method: str = "average", limit: Optional[int] = None,
                 file_attrs: Optional[Mapping[str, object]] = None) -> List[dict]:
    """Write ``split`` of the pool at ``store_root`` to ``out_path`` in the v3
    layout, plus ``<stem>_index.csv`` beside it; returns the index rows."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(f"{out_path} exists")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prov = RecordProvenance(source_dataset=SOURCE_DATASET, flux_compute=FLUX_COMPUTE, bdl_file=bdl_file,
                            bdl_sha256=bdl_sha256, downsample_method=method)
    rows: List[dict] = []
    with h5py.File(out_path, "w") as h5:
        for shard, candidate_id, record in iter_split(store_root, split, limit):
            pre, meta, text = preprocess_pool_record(candidate_id, record, scale, method)
            write_record(h5, pre, meta, text, prov)
            group = h5[pre.sample_id]
            extra: Dict[str, str] = {
                "pool_candidate_id": str(candidate_id), "pool_patient_id": str(meta.get("patient_id", "")),
                "pool_anatomy": str(meta.get("anatomy", "")), "pool_dataset": str(meta.get("dataset_name", "")),
                "pool_split": split, "pool_shard": shard}
            for key in POOL_ATTRS:
                group.attrs[key] = extra[key]
            rows.append(index_row(pre.sample_id, dict(group.attrs)))
            if len(rows) % 500 == 0:
                logger.info("exported %d records", len(rows))
        write_file_attrs(h5, {"schema_version": 3, "source": SOURCE_DATASET, "store_root": str(store_root),
                              "split": split, "n_records": len(rows), "scale_json": dict(scale),
                              "downsample_method": method, "bdl_file": bdl_file, "bdl_sha256": bdl_sha256,
                              **dict(file_attrs or {})})
    write_index_csv(rows, out_path.with_name(out_path.stem + "_index.csv"))
    logger.info("%d records -> %s", len(rows), out_path)
    return rows
