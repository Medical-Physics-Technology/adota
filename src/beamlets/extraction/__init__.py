"""Stage 1: per-spot beamlet extraction.

Orchestrates the full extraction for a plan: rotate the CT around the isocenter
once per field, then for every spot crop the BEV CT, build the flux projection,
and save the ADoTA inputs. Outputs land under ``<plan_dir>/adota_beamlets/``:

* ``{id}_ct.npy``       -- BEV CT crop, shape ``(60, 60, 320)`` (z, y, x),
* ``{id}_flux.npy``     -- proton-flux projection, same shape,
* ``{id}_sim_res.json`` -- per-spot metadata (energy, angles, entrance, crp, ...),
* ``manifest.json``     -- run summary,
* ``overlays/``         -- per-field visual sanity-check PNGs.

The geometry follows ``src/beamlets/__init__.py``: isocenter via
``TransformContinuousIndexToPhysicalPoint``, rotation around the isocenter
(SimpleITK), and the air-padded depth-from-entrance crop.
Split by role to stay inside the 500-line limit; the public names are
re-exported here, so ``from src.beamlets.extraction import run_extraction``
keeps working.

* this module -- config, per-field orchestration, run manifest;
* :mod:`~src.beamlets.extraction.spot` -- the per-spot crop + flux projection;
* :mod:`~src.beamlets.extraction.io` -- output tree and per-spot writes;
* :mod:`~src.beamlets.extraction.overlay` -- the per-field sanity-check PNG.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk

from src.beamlets import ROI_SIZE, roi_for_factor
from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.extraction.io import _prepare_output_dir, _save_spot
from src.beamlets.extraction.overlay import _save_field_overlay
from src.beamlets.extraction.spot import _build_sim_res, _process_spot
from src.beamlets.isocenter import isocenter_physical
from src.beamlets.plan_spots import expand_plan_to_spots, group_by_field
from src.beamlets.rotation import rotate_ct_around_isocenter
from src.loaders.plan_directory import PlanDirectory
from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)

# Re-exported so the package presents the same surface the single module did.
# The underscored names are internals, but tests and sibling code already import
# some of them from here, so removing them would be a silent breaking change.
__all__ = [
    "ExtractionConfig",
    "ROI_SIZE",
    "_FieldTiming",
    "_build_manifest",
    "_build_sim_res",
    "_extract_impl",
    "_prepare_output_dir",
    "_process_spot",
    "_save_field_overlay",
    "_save_spot",
    "_union_seconds",
    "run_extraction",
    "run_extraction_pooled",
]

@dataclass
class ExtractionConfig:
    """Configuration for :func:`run_extraction`.

    Attributes:
        roi_size: ``(H, W, D)`` ROI size.
        n_spots: If set, extract only the first ``n_spots`` (per the global spot
            order) -- a cheap subset for smoke runs.
        beams: If set, only extract these beam (field) indices.
        overwrite: Allow writing into a non-empty output directory.
        save_overlays: Save a per-field visual sanity-check PNG.
        bdl_path: Override the beam data library path (default: plan-local).
        flux_on_gpu: Compute the per-spot flux projection on ``flux_device`` via
            :func:`src.beamlets.flux.flux_projection_gpu` instead of NumPy. The
            result is numerically identical (float64, verified by tests); this is
            purely a speed option. Default ``False`` (the NumPy path, unchanged).
        flux_device: Torch device used when ``flux_on_gpu`` is set.
    """

    roi_size: tuple = ROI_SIZE
    n_spots: Optional[int] = None
    beams: Optional[List[int]] = None
    overwrite: bool = False
    save_overlays: bool = True
    bdl_path: Optional[Path] = None
    flux_on_gpu: bool = False
    flux_device: str = "cuda"
    grid_factor: int = 1
    """Field-level resampling factor (1 = current 1mm path, byte-identical; 2 =
    rotate/crop/flux on the 2mm grid so each crop is already the model grid). The
    per-spot ``roi_size``/geometry written to ``sim_res`` reflect the factor, so
    accumulation de-rotates back to the 1mm CT grid with no extra configuration."""


@dataclass
class _FieldTiming:
    """Per-field timing as absolute ``(start, end)`` intervals per step.

    Storing intervals (not durations) lets us report the **real wall-clock time**
    each step was active under the thread pool via :func:`_union_seconds` -- the
    union collapses overlapping concurrent intervals instead of summing them. In
    the serial path the intervals are disjoint, so the union equals the plain sum
    (the numbers are unchanged).
    """

    rotation_s: float = 0.0
    crop_iv: List[tuple] = dataclass_field(default_factory=list)
    flux_iv: List[tuple] = dataclass_field(default_factory=list)
    save_iv: List[tuple] = dataclass_field(default_factory=list)


def _union_seconds(intervals: List[tuple]) -> float:
    """Total wall-clock length covered by a set of ``(start, end)`` intervals.

    Overlapping intervals (concurrent threads) are merged, so the result is the
    real time during which the step was active -- never more than the elapsed
    wall time -- rather than the inflated sum of per-thread durations.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total


def run_extraction(
    plan_directory: PlanDirectory,
    output_dir: Path,
    config: Optional[ExtractionConfig] = None,
) -> dict:
    """Extract per-spot ADoTA inputs for a whole plan (sequential reference).

    This is the serial implementation kept as the trusted reference;
    :func:`run_extraction_pooled` is the thread-pooled twin producing
    **bit-identical** output. Both share the same orchestration and per-spot
    worker (:func:`_process_spot`), differing only in serial vs concurrent spots.

    Args:
        plan_directory: The loaded plan directory (CT, parsed plan, BDL path).
        output_dir: Where the per-spot files / manifest / overlays are written.
        config: Extraction options.

    Returns:
        The manifest dict (also written to ``output_dir/manifest.json``).

    Raises:
        FileExistsError: If ``output_dir`` is non-empty and ``overwrite`` is off.
    """
    return _extract_impl(plan_directory, output_dir, config, parallel=False, workers=0)


def run_extraction_pooled(
    plan_directory: PlanDirectory,
    output_dir: Path,
    config: Optional[ExtractionConfig] = None,
    workers: int = 0,
) -> dict:
    """Thread-pooled twin of :func:`run_extraction` (bit-identical output).

    Processes each field's spots concurrently with a thread pool. The per-spot
    work (SimpleITK crop, flux projection, ``np.save``) releases the GIL, so the
    threads overlap it while sharing the rotated CT array and the CUDA context
    zero-copy. Because both paths call the same :func:`_process_spot` and each spot
    writes only its own files, the output matches the serial path exactly.

    Args:
        plan_directory: The loaded plan directory.
        output_dir: Destination directory.
        config: Extraction options.
        workers: Thread count (``0`` = auto: ``min(32, os.cpu_count())``).

    Returns:
        The manifest dict.
    """
    return _extract_impl(
        plan_directory, output_dir, config, parallel=True, workers=workers
    )


def _extract_impl(
    plan_directory: PlanDirectory,
    output_dir: Path,
    config: Optional[ExtractionConfig],
    *,
    parallel: bool,
    workers: int,
) -> dict:
    """Shared extraction core; ``parallel`` selects serial vs thread-pool spots."""
    config = config or ExtractionConfig()
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, config.overwrite)

    bdl_path = config.bdl_path or plan_directory.bdl_path
    bdl = BeamDataLibrary.from_file(Path(bdl_path))
    d_nozzle, d_smx, d_smy = bdl.distances

    ct = plan_directory.ct
    spots = expand_plan_to_spots(plan_directory.plan)
    if config.beams is not None:
        spots = [s for s in spots if s["beam"] in set(config.beams)]
    if config.n_spots is not None:
        spots = spots[: config.n_spots]
    grouped = group_by_field(spots)

    logger.info(
        "Extracting %d spots across %d field(s) into %s",
        len(spots),
        len(grouped),
        output_dir,
    )

    overlays_dir = output_dir / "overlays"
    if config.save_overlays:
        overlays_dir.mkdir(exist_ok=True)

    effective_workers = (
        (workers or min(32, (os.cpu_count() or 4))) if parallel else 0
    )

    # Field-level resampling factor. gf=1 keeps every operation byte-identical;
    # gf=2 rotates/crops/fluxes on the 2mm grid. roi/flux-spacing are derived from
    # gf once (constant across fields); ``[1,1,1]`` (float32) is exactly the flux
    # default so gf=1 stays byte-identical.
    gf = config.grid_factor
    roi = config.roi_size if gf == 1 else roi_for_factor(gf)
    flux_spacing = np.asarray([gf, gf, gf], dtype=np.float32)

    started = perf_counter()
    timings: Dict[int, _FieldTiming] = {}
    oob_count = 0

    for beam, field_spots in grouped.items():
        timing = _FieldTiming()
        timings[beam] = timing

        iso_index = field_spots[0]["simulation_log"]["isocenter"]
        adjusted_angle = field_spots[0]["simulation_log"]["gantry_angle"]
        # The plan isocenter x is flipped relative to the CT (S3); this lands the
        # rotation pivot on the true target.
        iso_phys = isocenter_physical(iso_index, ct)

        logger.info(
            "Field beam=%d: %d spots, gantry(adj)=%.1f deg, iso_index=%s -> phys=%s",
            beam,
            len(field_spots),
            adjusted_angle,
            tuple(round(float(c), 2) for c in iso_index),
            tuple(round(float(c), 2) for c in iso_phys),
        )

        # Rotate into an EXPANDED grid so the off-isocenter rotation clips no
        # patient information (Phase 1/2). The crop and the stored geometry then
        # live in this expanded frame; accumulation de-rotates back to the
        # original CT grid.
        rot_t = perf_counter()
        rotated_ct = rotate_ct_around_isocenter(
            ct, adjusted_angle, iso_phys, expand=True, out_spacing_factor=gf
        )
        timing.rotation_s = perf_counter() - rot_t

        # Copy the rotated grid to numpy once per field (not once per spot): the
        # crops only slice into it, so this is the dominant-cost optimization.
        rotated_ct_array = sitk.GetArrayFromImage(rotated_ct)

        image_origin = rotated_ct.GetOrigin()
        image_spacing = rotated_ct.GetSpacing()
        image_size = rotated_ct.GetSize()

        # The per-spot work is independent (each writes its own files; the rotated
        # CT is read-only), so the serial and pooled paths are bit-identical.
        worker = partial(
            _process_spot,
            rotated_ct=rotated_ct,
            rotated_ct_array=rotated_ct_array,
            iso_phys=iso_phys,
            d_nozzle=d_nozzle,
            d_smx=d_smx,
            d_smy=d_smy,
            bdl=bdl,
            image_origin=image_origin,
            image_spacing=image_spacing,
            image_size=image_size,
            config=config,
            output_dir=output_dir,
            roi=roi,
            flux_spacing=flux_spacing,
        )
        if parallel:
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                results = list(pool.map(worker, field_spots))
        else:
            results = [worker(record) for record in field_spots]

        # Collect per-spot (start, end) intervals; unioned later for real wall time.
        for res in results:
            timing.crop_iv.append(res["crop"])
            timing.flux_iv.append(res["flux"])
            timing.save_iv.append(res["save"])
            oob_count += res["oob"]

        if config.save_overlays:
            _save_field_overlay(
                overlays_dir,
                beam,
                rotated_ct,
                field_spots,
                iso_phys,
                bdl,
                roi,
            )

    elapsed = perf_counter() - started
    manifest = _build_manifest(
        plan_directory,
        output_dir,
        bdl,
        spots,
        grouped,
        config,
        timings,
        oob_count,
        elapsed,
        parallel=parallel,
        workers=effective_workers,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, cls=NumpyEncoder)
    )
    logger.info(
        "Extraction complete: %d spots in %.1fs (%d out-of-bounds crops). Output: %s",
        len(spots),
        elapsed,
        oob_count,
        output_dir,
    )
    return manifest


def _build_manifest(
    plan_directory: PlanDirectory,
    output_dir: Path,
    bdl: BeamDataLibrary,
    spots: List[dict],
    grouped: Dict[int, List[dict]],
    config: ExtractionConfig,
    timings: Dict[int, _FieldTiming],
    oob_count: int,
    elapsed: float,
    parallel: bool = False,
    workers: int = 0,
) -> dict:
    """Assemble the run manifest."""
    return {
        "plan_dir": str(plan_directory.plan_dir),
        "output_dir": str(output_dir),
        "bdl_path": str(bdl.source_path),
        "bdl_distances": {
            "d_nozzle": bdl.d_nozzle,
            "d_smx": bdl.d_smx,
            "d_smy": bdl.d_smy,
        },
        "roi_size": list(config.roi_size),
        "n_spots": len(spots),
        "n_fields": len(grouped),
        "spots_per_field": {beam: len(s) for beam, s in grouped.items()},
        "oob_crops": oob_count,
        "elapsed_s": elapsed,
        "parallel": parallel,
        "workers": workers,
        # Real wall-clock time each step was active (union of concurrent
        # intervals); equals the plain sum in the serial path.
        "timing_per_field": {
            beam: {
                "rotation_s": t.rotation_s,
                "crop_s_total": _union_seconds(t.crop_iv),
                "flux_s_total": _union_seconds(t.flux_iv),
                "save_s_total": _union_seconds(t.save_iv),
            }
            for beam, t in timings.items()
        },
        "spot_ids": [s["id"] for s in spots],
    }
