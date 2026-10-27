"""The batch loop: simulate a batch on fast local storage, pack it, move it, clean up.

One batch at a time, per process:

1. Monte Carlo of the batch's beamlets through
   :func:`src.active_learning.oracle.label_batch` (beamlet mode, one MCsquare call
   per field with every beamlet at its own energy) into ``<work>/batch_NNNN/raw``;
2. :func:`.store.pack_batch`: overshoot checks, shards per split, flux check;
3. :func:`.store.move_file` of every shard and the index to ``<store>``, with
   checksums;
4. a ``<store>/state/batch_NNNN.json`` marker, then the raw files are deleted.

A batch with a marker is never redone; an interrupted batch resumes beamlet by
beamlet (the generator skips beamlets already on disk). Processes share the plan
and take batches ``b`` with ``b % n_shards == shard``, so two processes on one
machine, or the tasks of a SLURM array, never touch the same batch. Before each
batch the free space of both locations is checked against what the batch needs.
"""
from __future__ import annotations

import json
import logging
import shutil
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.active_learning.beamlet_pool.planning import PACKED_MB, RAW_MB
from src.active_learning.beamlet_pool.store import StoreChecks, free_gb, move_file, pack_batch, spot_sigmas
from src.active_learning.oracle import label_batch
from src.active_learning.pool import PoolEntry, RecordResolver
from src.beamlets.bdl import BeamDataLibrary
from src.mc_generation.mcsquare_runner import MCSquareRunner
from src.mc_generation.sweep import RobustnessConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchSettings:
    work_root: str                      # fast local storage for one raw batch (/scratch)
    store_root: str                     # where the shards live (/RadiotherapyData/adota_pool/<version>)
    keep_raw: bool = False
    transient_gb: float = 20.0          # MCsquare's dense dose grids of one beamlet block
    reserve_work_gb: float = 30.0       # never fill the work disk beyond this margin
    reserve_store_gb: float = 50.0


def marker(store_root: Path, batch: int) -> Path:
    return Path(store_root) / "state" / f"batch_{batch:04d}.json"


def my_batches(plan: pd.DataFrame, shard: int, n_shards: int) -> List[int]:
    return [int(b) for b in sorted(plan["batch"].unique()) if int(b) % n_shards == shard]


def space_check(settings: BatchSettings, n_beamlets: int) -> Optional[str]:
    """Why the next batch cannot start, or ``None``."""
    work_need = n_beamlets * RAW_MB / 1000.0 + settings.transient_gb + settings.reserve_work_gb
    store_need = n_beamlets * PACKED_MB / 1000.0 * 1.2 + settings.reserve_store_gb
    if free_gb(Path(settings.work_root)) < work_need:
        return f"work disk: {free_gb(Path(settings.work_root)):.0f} GB free, {work_need:.0f} GB needed"
    if free_gb(Path(settings.store_root)) < store_need:
        return f"store: {free_gb(Path(settings.store_root)):.0f} GB free, {store_need:.0f} GB needed"
    return None


def run_batch(batch: int, rows: pd.DataFrame, entries: Sequence[PoolEntry], runner: MCSquareRunner,
              bdl: BeamDataLibrary, rob_cfg: RobustnessConfig, settings: BatchSettings, checks: StoreChecks,
              resolver: RecordResolver) -> Dict:
    """Simulate, pack, move and mark one batch; returns its marker content."""
    work = Path(settings.work_root) / f"batch_{batch:04d}"
    store = Path(settings.store_root)
    started = time.perf_counter()
    stats = label_batch(rows, entries, runner, bdl, rob_cfg, work / "raw", resolver=resolver,
                        per_beamlet_energy=True)
    mc_done = time.perf_counter()
    index, shards = pack_batch(work / "raw", rows, work / "shards", batch, spot_sigmas(bdl), checks)
    index_local = work / "shards" / f"batch_{batch:04d}.csv"
    index.to_csv(index_local, index=False)
    digests = {f"{split}/batch_{batch:04d}.h5": move_file(path, store / split / path.name)
               for split, path in shards.items()}
    digests[f"index/batch_{batch:04d}.csv"] = move_file(index_local, store / "index" / index_local.name)
    status = index["status"].value_counts().to_dict()
    content = {"batch": batch, "n_planned": int(len(rows)), "status": status,
               "stored_by_split": index[index["status"] == "stored"].groupby("split").size().to_dict(),
               "mc_seconds": stats.get("mc_seconds"), "qa_skipped": stats.get("qa_skipped"),
               "wall_seconds_mc": mc_done - started, "wall_seconds_total": time.perf_counter() - started,
               "files": digests, "host": socket.gethostname(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
               "num_primaries": rob_cfg.num_primaries, "num_threads": rob_cfg.num_threads,
               "checks": asdict(checks)}
    path = marker(store, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, default=str))
    if not settings.keep_raw:
        shutil.rmtree(work, ignore_errors=True)
    logger.info("batch %d: %s in %.2f h (MC %.2f h)", batch, status, content["wall_seconds_total"] / 3600.0,
                content["wall_seconds_mc"] / 3600.0)
    return content


def run_batches(plan: pd.DataFrame, entries: Sequence[PoolEntry], runner: MCSquareRunner, bdl: BeamDataLibrary,
                rob_cfg: RobustnessConfig, settings: BatchSettings, checks: StoreChecks, shard: int = 0,
                n_shards: int = 1, max_batches: Optional[int] = None) -> List[Dict]:
    """This process's batches, in order, skipping finished ones; stops at
    ``max_batches`` or when a location runs short of space."""
    resolver = RecordResolver(entries)
    done: List[Dict] = []
    todo = [b for b in my_batches(plan, shard, n_shards) if not marker(Path(settings.store_root), b).exists()]
    logger.info("shard %d/%d: %d batches to do", shard, n_shards, len(todo))
    for batch in todo:
        if max_batches is not None and len(done) >= max_batches:
            break
        rows = plan[plan["batch"] == batch]
        problem = space_check(settings, len(rows))
        if problem:
            logger.error("stopping before batch %d: %s", batch, problem)
            break
        done.append(run_batch(batch, rows, entries, runner, bdl, rob_cfg, settings, checks, resolver))
    return done


def progress(plan: pd.DataFrame, store_root: Path) -> Dict:
    """Finished batches, beamlets stored per split, and the rate so far."""
    markers = [json.loads(p.read_text()) for p in sorted((Path(store_root) / "state").glob("batch_*.json"))]
    stored: Dict[str, int] = {}
    for m in markers:
        for split, n in m.get("stored_by_split", {}).items():
            stored[split] = stored.get(split, 0) + int(n)
    hours = sum(m.get("wall_seconds_total", 0.0) for m in markers) / 3600.0
    planned = int(plan["batch"].nunique())
    return {"batches_done": len(markers), "batches_planned": planned, "stored_by_split": stored,
            "beamlets_stored": int(sum(stored.values())), "process_hours": hours,
            "hours_per_batch": hours / len(markers) if markers else None,
            "status": pd.Series([k for m in markers for k, v in m.get("status", {}).items() for _ in range(int(v))])
            .value_counts().to_dict() if markers else {}}
