"""Recompute the difficulty metrics of the reference beamlets from the inputs alone.

For every record of the reference HDF5 set this computes the thirty metrics of
the difficulty score twice: located by the Monte Carlo ground truth exactly as
the study did (``gt``), and located by the analytic dose surrogate that a
candidate beamlet would get before any simulation (``analytic``). The ``gt``
pass must reproduce the study's ``results.csv``; the ``analytic`` pass is the
feature set the score is refit on. Their record-by-record disagreement is the
diagnostic that says which metrics the surrogate moved.

Usage:
    uv run python scripts/analysis/acquisition_input_only_features.py --stride 23 --out /scratch/.../subset
    uv run python scripts/analysis/acquisition_input_only_features.py --workers 8 --out /scratch/.../full

Reads no model and touches no GPU; the cost is HDF5 I/O plus the metrics.
"""
from __future__ import annotations

import os

# One BLAS thread per process: the work is fanned out over a process pool below,
# and a multithreaded numpy in every worker oversubscribes the machine (load 45
# on 8 workers). Must precede the first numpy import, hence the E402 exemption
# for this file in pyproject.toml.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Sequence

import h5py
import pandas as pd
import typer

from src.acquisition.features import FeatureConfig
from src.acquisition.reference import peak_agreement, read_stored_record, record_features

REFERENCE_H5 = Path("/scratch/mstryja/DoTA_dataset_v2/"
                    "trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5")
REFERENCE_RESULTS = Path("/scratch/mstryja/adota_runs/20260707_124010/results.csv")
MODES = ("gt", "analytic")

logger = logging.getLogger("acquisition_input_only_features")


def process_chunk(h5_path: str, sample_ids: Sequence[str], modes: Sequence[str]) -> List[Dict]:
    """Features of ``sample_ids`` in every mode; one H5 handle per chunk."""
    rows: List[Dict] = []
    config = FeatureConfig()
    with h5py.File(h5_path, "r") as h5:
        for sid in sample_ids:
            record = read_stored_record(h5[sid], sid)
            depth_err_mm, lateral_err_vox = peak_agreement(record, config.resolution_mm[0])
            for mode in modes:
                t0 = time.perf_counter()
                out = record_features(record, mode, config)
                if out is None:
                    continue
                out.update({"peak_depth_err_mm": depth_err_mm, "peak_lateral_err_vox": lateral_err_vox,
                            "feature_time_s": time.perf_counter() - t0})
                rows.append(out)
    return rows


def select_ids(results_csv: Path, stride: int, limit: Optional[int]) -> List[str]:
    """The study's sample ids, in its order, thinned by ``stride`` and capped."""
    ids = pd.read_csv(results_csv, usecols=["sample_id"])["sample_id"].tolist()[::stride]
    return ids[:limit] if limit else ids


def main(
    out: Annotated[Path, typer.Option(help="Output directory (on /scratch).")],
    h5: Annotated[Path, typer.Option(help="Reference HDF5 set.")] = REFERENCE_H5,
    results: Annotated[Path, typer.Option(help="Study results.csv; defines the id set and order.")] = REFERENCE_RESULTS,
    stride: Annotated[int, typer.Option(min=1, help="Take every N-th record.")] = 1,
    limit: Annotated[Optional[int], typer.Option(help="Stop after this many records.")] = None,
    workers: Annotated[int, typer.Option(min=1, help="Process-pool size.")] = 1,
    chunk: Annotated[int, typer.Option(min=1, help="Records per worker task.")] = 200,
    mode: Annotated[List[str], typer.Option(help="gt and/or analytic.")] = list(MODES),
) -> None:
    """Write ``features_<mode>.csv`` per mode plus a ``manifest.json``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for m in mode:
        if m not in MODES:
            raise typer.BadParameter(f"mode must be one of {MODES}, got {m!r}")
    out.mkdir(parents=True, exist_ok=True)
    ids = select_ids(results, stride, limit)
    chunks = [ids[i:i + chunk] for i in range(0, len(ids), chunk)]
    logger.info("%d records in %d chunks, modes %s, %d workers", len(ids), len(chunks), mode, workers)

    started = time.perf_counter()
    rows: List[Dict] = []
    if workers == 1:
        for i, c in enumerate(chunks, 1):
            rows += process_chunk(str(h5), c, mode)
            logger.info("chunk %d/%d done, %d rows, %.0f s", i, len(chunks), len(rows), time.perf_counter() - started)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_chunk, str(h5), c, mode) for c in chunks]
            for i, fut in enumerate(as_completed(futures), 1):
                rows += fut.result()
                logger.info("chunk %d/%d done, %d rows, %.0f s", i, len(chunks), len(rows),
                            time.perf_counter() - started)

    frame = pd.DataFrame(rows)
    written = {}
    for m in mode:
        path = out / f"features_{m}.csv"
        frame[frame["mode"] == m].drop(columns=["mode"]).to_csv(path, index=False)
        written[m] = str(path)
        logger.info("wrote %s (%d rows)", path, int((frame["mode"] == m).sum()))
    (out / "manifest.json").write_text(json.dumps({
        "h5": str(h5), "results": str(results), "stride": stride, "limit": limit,
        "n_requested": len(ids), "n_rows": len(frame), "modes": list(mode),
        "elapsed_s": time.perf_counter() - started, "feature_config": FeatureConfig().__dict__,
        "outputs": written}, indent=2, default=str))


if __name__ == "__main__":
    typer.run(main)
