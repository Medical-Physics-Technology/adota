"""Build the v3 training HDF5 from raw Monte Carlo beamlet records.

    uv run python scripts/build_beamlet_h5.py \\
        --out /scratch/mstryja/DoTA_dataset_v3/trainset_v3.h5 \\
        --bdl /home/mstryja/tools/mcsquare/BDL/hptc_beam_model_rsnone.txt \\
        --exclusion-list /home/mstryja/projects/dota_pytorch/auxilary_files/IndexesExclude_....txt

Reads every raw ``<id>_ct.npy`` / ``<id>_ds.npy`` / ``<id>_metadata.json`` triple
under ``--raw-root/<source>`` for each ``--source``, applies the exclusion list and
the skip rules of ``docs/dev/h5_v3_spec.md`` section 3, and writes one HDF5 group
per surviving record with the v2 six attrs plus the v3 provenance attrs (section 4).

Process model: a ``multiprocessing`` "spawn" pool does the CPU-only load / screen /
pool step (:func:`process_candidate`); the main process batches the GPU flux
projection per raw shape, screens again with the flux present, and does every HDF5
write. ``--workers`` 0 or 1 skips the pool and runs in-process, for debugging.
Writes ``<out>.partial`` and renames it to ``--out`` on success; refuses if
``--out`` already exists. ``--resume`` reopens an existing ``.partial``.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
import torch
import typer
from tqdm import tqdm

from src.active_learning.retrospective.dataset import read_exclusion_list
from src.adota.config import DEFAULT_SCALE
from src.beamlets.bdl import BeamDataLibrary
from src.datasets.beamlet_h5 import (
    SCHEMA_VERSION,
    RecordProvenance,
    index_row,
    is_complete_group,
    plan_candidates,
    sha256_of,
    write_file_attrs,
    write_index_csv,
    write_record,
    write_skip_csv,
)
from src.datasets.beamlet_record import (
    FluxInputs,
    PreprocessedRecord,
    RawRecord,
    flux_batch,
    flux_inputs,
    list_record_ids,
    load_raw_record,
    normalise_energy,
    preprocess_ct_dose,
    preprocess_flux,
    skip_reason,
    spots_reason,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("build_beamlet_h5")
app = typer.Typer(help="Build the v3 beamlet HDF5 from raw Monte Carlo records.")

# ── Worker process (spawn pool): module-level so it pickles by reference ────

_WORKER_BDL: Optional[BeamDataLibrary] = None
_WORKER_RAW_ROOT: Optional[Path] = None
_WORKER_SCALE: Optional[Mapping] = None
_WORKER_METHOD: str = "average"


def init_worker(raw_root: str, bdl_path: str, scale: Mapping, method: str) -> None:
    """Pool initializer: pin torch threads and parse the BDL once per process."""
    global _WORKER_BDL, _WORKER_RAW_ROOT, _WORKER_SCALE, _WORKER_METHOD
    torch.set_num_threads(1)
    _WORKER_BDL = BeamDataLibrary.from_file(Path(bdl_path))
    _WORKER_RAW_ROOT = Path(raw_root)
    _WORKER_SCALE = scale
    _WORKER_METHOD = method


def process_candidate(item: Tuple[str, str]) -> dict:
    """Worker body: load, screen and pre-process one candidate. Never touches CUDA.

    Returns a skip row (``status="skip"``) or a payload (``status="ok"``) carrying
    the pooled ``ct``/``dose`` and the raw-shape :class:`FluxInputs` the main
    process needs for batched flux and final assembly.
    """
    source, sample_id = item
    record_dir = _WORKER_RAW_ROOT / source

    def _skip(reason: str, detail: str = "") -> dict:
        return {"status": "skip", "sample_id": sample_id, "source": source, "reason": reason, "detail": detail}

    try:
        raw: RawRecord = load_raw_record(record_dir, sample_id)
    except json.JSONDecodeError as exc:
        return _skip("json_error", str(exc))
    except KeyError as exc:
        return _skip("json_error", f"missing metadata key: {exc}")
    except Exception as exc:  # noqa: BLE001 - a record that cannot be read is skipped, as in v2
        return _skip("load_error", repr(exc))

    reason = spots_reason(raw.metadata)
    if reason is not None:
        return _skip(reason)

    try:
        ct, dose = preprocess_ct_dose(raw, _WORKER_SCALE, _WORKER_METHOD)
        reason = skip_reason(ct, dose)
        if reason is not None:
            return _skip(reason)
        fi = flux_inputs(raw, _WORKER_BDL)
    except Exception as exc:  # noqa: BLE001 - any preprocessing failure is a load_error (section 3)
        return _skip("load_error", repr(exc))

    return {"status": "ok", "source": source, "sample_id": sample_id, "metadata": raw.metadata,
            "metadata_text": raw.metadata_text, "ct": ct, "dose": dose, "flux_inputs": fi}


def _iter_results(candidates, raw_root, bdl, scale, downsample, workers):
    """Yield worker results in order; in-process (no pool) for ``workers`` in (0, 1)."""
    if workers in (0, 1):
        init_worker(str(raw_root), str(bdl), scale, downsample)
        for item in candidates:
            yield process_candidate(item)
        return
    ctx = multiprocessing.get_context("spawn")
    # Windowed imap: Pool.imap has no backpressure, so a writer slower than the pool
    # would buffer every pooled record (about 2.6 MB each) in the main process.
    window = 8 * workers
    with ctx.Pool(workers, initializer=init_worker,
                  initargs=(str(raw_root), str(bdl), scale, downsample)) as pool:
        for start in range(0, len(candidates), window):
            for result in pool.imap(process_candidate, candidates[start:start + window]):
                yield result


# ── Main-process orchestration ──────────────────────────────────────────────


def resolve_device(requested: str) -> str:
    """``requested`` unchanged, or ``"cpu"`` with a warning when CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested (%s) but unavailable; falling back to cpu", requested)
        return "cpu"
    return requested


def flux_compute_string(device: str) -> str:
    return f"flux_projection_gpu_batched/float64/{device.split(':', 1)[0]}"


def _flush_group(
    payloads: List[dict], device: str, downsample_method: str, scale: Mapping, prov_common: dict,
    h5: h5py.File, skip_rows: List[dict],
) -> int:
    """Flux-batch, pool, screen and write one same-raw-shape group. Returns n written."""
    if not payloads:
        return 0
    inputs: List[FluxInputs] = [p["flux_inputs"] for p in payloads]
    try:
        flux_arrays = flux_batch(inputs, device=device)
    except torch.cuda.OutOfMemoryError:
        if len(payloads) == 1:
            raise
        mid = len(payloads) // 2
        logger.warning("CUDA OOM at flux batch size %d; halving to %d/%d", len(payloads), mid,
                        len(payloads) - mid)
        return (_flush_group(payloads[:mid], device, downsample_method, scale, prov_common, h5, skip_rows)
                + _flush_group(payloads[mid:], device, downsample_method, scale, prov_common, h5, skip_rows))

    written = 0
    for payload, flux_f64 in zip(payloads, flux_arrays):
        flux_pooled = preprocess_flux(flux_f64, downsample_method)
        reason = skip_reason(payload["ct"], payload["dose"], flux_pooled)
        if reason is not None:
            skip_rows.append({"sample_id": payload["sample_id"], "source": payload["source"],
                               "reason": reason, "detail": ""})
            continue
        fi: FluxInputs = payload["flux_inputs"]
        pre = PreprocessedRecord(
            sample_id=payload["sample_id"], ct=payload["ct"], dose=payload["dose"], flux=flux_pooled,
            initial_energy_norm=normalise_energy(fi.energy_mev, scale), energy_mev=fi.energy_mev,
            sigmas_xy=fi.sigmas_xy,
        )
        prov = RecordProvenance(source_dataset=payload["source"], downsample_method=downsample_method,
                                 **prov_common)
        write_record(h5, pre, payload["metadata"], payload["metadata_text"], prov)
        written += 1
    return written


def _open_output(out: Path, resume: bool) -> Tuple[h5py.File, Set[str]]:
    """Open ``<out>.partial``; return the handle and the ids already complete in it."""
    partial = out.with_suffix(out.suffix + ".partial")
    if out.exists():
        raise FileExistsError(f"{out} already exists; refusing to overwrite a finished build")
    if resume and partial.exists():
        h5 = h5py.File(partial, "a")
        complete = {gid for gid in h5.keys() if is_complete_group(h5[gid])}
        for gid in list(h5.keys()):
            if gid not in complete:
                del h5[gid]
        return h5, complete
    return h5py.File(partial, "w"), set()


def build(
    raw_root: Path,
    sources: Sequence[str],
    out: Path,
    bdl: Path,
    exclusion_list: Path,
    scale: Optional[Mapping[str, float]] = None,
    device: str = "cuda",
    flux_batch_size: int = 32,
    downsample: str = "average",
    limit: Optional[int] = None,
    ids_file: Optional[Path] = None,
    resume: bool = False,
    workers: int = 8,
) -> dict:
    """Run the full build (spec section 6.2). Returns the file attrs dict written."""
    raw_root, out, bdl, exclusion_list = Path(raw_root), Path(out), Path(bdl), Path(exclusion_list)
    scale = dict(scale) if scale is not None else dict(DEFAULT_SCALE)

    exclusion_ids = set(read_exclusion_list(exclusion_list))  # raises if missing
    bdl_sha256 = sha256_of(bdl)  # raises if missing/unreadable
    device = resolve_device(device)

    log_path = out.parent / f"{out.stem}_build.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    try:
        return _build_inner(raw_root, sources, out, bdl, bdl_sha256, exclusion_ids, exclusion_list, scale,
                             device, flux_batch_size, downsample, limit, ids_file, resume, workers)
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def _build_inner(
    raw_root, sources, out, bdl, bdl_sha256, exclusion_ids, exclusion_list, scale, device, flux_batch_size,
    downsample, limit, ids_file, resume, workers,
) -> dict:
    retained, skip_rows = plan_candidates(raw_root, sources, exclusion_ids, ids_file, limit)
    all_raw_ids: Set[str] = set()
    for source in sources:
        all_raw_ids.update(list_record_ids(Path(raw_root) / source))
    n_excluded_not_found = len([i for i in exclusion_ids if i not in all_raw_ids])

    h5, already_complete = _open_output(out, resume)
    partial_path = Path(h5.filename)
    to_process = [c for c in retained if c[1] not in already_complete]
    prov_common = {"flux_compute": flux_compute_string(device), "bdl_file": bdl.name, "bdl_sha256": bdl_sha256,
                   "flux_model": "SingleGaussian"}

    n_processed, t0 = 0, time.time()
    pending: Dict[Tuple[int, ...], List[dict]] = {}
    try:
        progress = tqdm(_iter_results(to_process, raw_root, bdl, scale, downsample, workers),
                         total=len(to_process), disable=not sys.stdout.isatty(), desc="build_beamlet_h5")
        for result in progress:
            n_processed += 1
            if result["status"] == "skip":
                skip_rows.append({"sample_id": result["sample_id"], "source": result["source"],
                                   "reason": result["reason"], "detail": result.get("detail", "")})
            else:
                shape = tuple(result["flux_inputs"].shape)
                pending.setdefault(shape, []).append(result)
                if len(pending[shape]) >= flux_batch_size:
                    _flush_group(pending.pop(shape), device, downsample, scale, prov_common, h5, skip_rows)
            if n_processed % 1000 == 0:
                rate = n_processed / max(time.time() - t0, 1e-9)
                logger.info("processed %d/%d candidates (%.1f records/s)", n_processed, len(to_process), rate)
        for shape in list(pending):
            _flush_group(pending.pop(shape), device, downsample, scale, prov_common, h5, skip_rows)

        n_records = len(h5.keys())
        file_attrs = _file_attrs(out, bdl, bdl_sha256, exclusion_list, scale, device, flux_batch_size,
                                  downsample, n_excluded_not_found, sources, raw_root, retained, skip_rows,
                                  n_records)
        write_file_attrs(h5, file_attrs)
        index_rows = [index_row(gid, dict(h5[gid].attrs)) for gid in h5.keys()]
    finally:
        h5.close()

    write_index_csv(index_rows, out.parent / f"{out.stem}_index.csv")
    write_skip_csv(skip_rows, out.parent / f"{out.stem}_skipped.csv")
    partial_path.rename(out)
    logger.info("wrote %s: %d records, %d skipped", out, n_records, len(skip_rows))
    return file_attrs


def _generator_string() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                                     check=True).stdout.strip())
        return f"scripts/build_beamlet_h5.py@{commit}{'+dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001 - informational only
        return "scripts/build_beamlet_h5.py@unknown"


def _sources_json(sources: Sequence[str], raw_root: Path, retained: List[Tuple[str, str]],
                   skip_rows: List[dict]) -> dict:
    sources_json: dict = {"note": "retained = v2 count minus excluded"}
    for source in sources:
        n_raw = len(list_record_ids(Path(raw_root) / source))
        n_excl = sum(1 for r in skip_rows if r["source"] == source and r["reason"] == "excluded")
        n_other_skip = sum(1 for r in skip_rows if r["source"] == source and r["reason"] != "excluded")
        n_retained_source = sum(1 for s, _ in retained if s == source)
        sources_json[source] = {
            "root": str(Path(raw_root) / source), "n_raw_json": n_raw,
            "n_candidates": n_retained_source + n_excl, "n_excluded": n_excl,
            "n_written": n_retained_source - n_other_skip, "n_skipped": n_other_skip,
        }
    return sources_json


def _file_attrs(
    out: Path, bdl: Path, bdl_sha256: str, exclusion_list: Path, scale: Mapping, device: str,
    flux_batch_size: int, downsample: str, n_excluded_not_found: int, sources: Sequence[str], raw_root: Path,
    retained: List[Tuple[str, str]], skip_rows: List[dict], n_records: int,
) -> dict:
    n_excluded = sum(1 for r in skip_rows if r["reason"] == "excluded")
    gpu_name = torch.cuda.get_device_name(device) if device.startswith("cuda") and torch.cuda.is_available() else "none"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": _generator_string(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": str(torch.__version__),  # torch.__version__ is a str subclass; h5py needs a plain str
        "h5py": h5py.version.version,
        "cuda": torch.version.cuda or "none",
        "gpu_name": gpu_name,
        "scale_json": json.dumps(dict(scale)),
        "bdl_file": str(bdl.resolve()),
        "bdl_sha256": bdl_sha256,
        "flux_compute": flux_compute_string(device),
        "flux_batch": flux_batch_size,
        "downsample_method": downsample,
        "exclusion_list_path": str(exclusion_list.resolve()),
        "exclusion_list_sha256": sha256_of(exclusion_list),
        "n_excluded": n_excluded,
        "n_excluded_not_found": n_excluded_not_found,
        "sources_json": _sources_json(sources, raw_root, retained, skip_rows),
        "n_records": n_records,
        "n_skipped": len(skip_rows) - n_excluded,
        "skip_log_path": str((out.parent / f"{out.stem}_skipped.csv").resolve()),
        "index_path": str((out.parent / f"{out.stem}_index.csv").resolve()),
        "source_v2_path": "",
        "flux_equivalence": "unchecked",
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def main(
    raw_root: Annotated[Path, typer.Option(help="Root of the raw record tree.")] = Path(
        "/RadiotherapyData/dataset_v0"),
    source: Annotated[Optional[List[str]], typer.Option(
        help="Raw source subdirectory, repeatable; order matters for --limit. "
             "Defaults to trainset_pelvis, initial_test_one_ct.")] = None,
    out: Annotated[Path, typer.Option(help="Final .h5 path; writes <out>.partial first.")] = ...,
    bdl: Annotated[Path, typer.Option(help="MCsquare beam-data-library file.")] = ...,
    scale_json: Annotated[Optional[Path], typer.Option(
        help="JSON with the six scale keys; defaults to src.adota.config.DEFAULT_SCALE.")] = None,
    exclusion_list: Annotated[Path, typer.Option(
        help="Ids to drop, one per line. Required: no default; a missing file is a hard error.")] = ...,
    device: Annotated[str, typer.Option(
        help="cuda, cuda:N or cpu; falls back to cpu with a warning if CUDA is unavailable.")] = "cuda",
    flux_batch: Annotated[int, typer.Option(
        help="Flux projections per batched GPU call; halved on CUDA OOM down to 1.")] = 32,
    downsample: Annotated[str, typer.Option(help="average, linear or trilinear.")] = "average",
    limit: Annotated[Optional[int], typer.Option(help="Truncate the candidate list.")] = None,
    ids_file: Annotated[Optional[Path], typer.Option(
        help="One id per line; restricts the candidate set before exclusion.")] = None,
    resume: Annotated[bool, typer.Option(
        help="Reopen the existing .partial: skip complete groups, rebuild incomplete ones.")] = False,
    workers: Annotated[int, typer.Option(
        help="Process-pool size for the CPU load/screen step. 0 or 1 runs in-process (debugging).")] = 8,
) -> None:
    sources = source or ["trainset_pelvis", "initial_test_one_ct"]
    scale = json.loads(Path(scale_json).read_text()) if scale_json is not None else None

    try:
        attrs = build(
            raw_root=raw_root, sources=sources, out=out, bdl=bdl, exclusion_list=exclusion_list, scale=scale,
            device=device, flux_batch_size=flux_batch, downsample=downsample, limit=limit, ids_file=ids_file,
            resume=resume, workers=workers,
        )
    except (FileNotFoundError, FileExistsError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    logger.info("n_records=%s n_skipped=%s n_excluded=%s", attrs["n_records"], attrs["n_skipped"],
                attrs["n_excluded"])


if __name__ == "__main__":
    app()
