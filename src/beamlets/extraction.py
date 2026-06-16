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
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dataclass_field
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk

from src.beamlets import ROI_SIZE
from src.beamlets.bdl import BeamDataLibrary, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import flux_projection, flux_projection_gpu, flux_spatial_spread
from src.beamlets.isocenter import isocenter_physical
from src.beamlets.plan_spots import expand_plan_to_spots, group_by_field
from src.beamlets.rotation import rotate_ct_around_isocenter
from src.loaders.plan_directory import PlanDirectory
from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)

__all__ = ["ExtractionConfig", "run_extraction", "run_extraction_pooled"]


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
            ct, adjusted_angle, iso_phys, expand=True
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
                config.roi_size,
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


def _process_spot(
    record: dict,
    *,
    rotated_ct: sitk.Image,
    rotated_ct_array: np.ndarray,
    iso_phys: tuple,
    d_nozzle: float,
    d_smx: float,
    d_smy: float,
    bdl: BeamDataLibrary,
    image_origin: tuple,
    image_spacing: tuple,
    image_size: tuple,
    config: ExtractionConfig,
    output_dir: Path,
) -> dict:
    """Crop + flux + save one spot; return its per-step timings and ``oob`` flag.

    This is the single per-spot implementation shared by the serial
    (:func:`run_extraction`) and pooled (:func:`run_extraction_pooled`) paths, so
    both produce byte-identical outputs. It reads the shared ``rotated_ct`` /
    ``rotated_ct_array`` (read-only) and writes only this spot's own files, so it
    is safe to run concurrently across spots.
    """
    sim_log = record["simulation_log"]
    spot_position = sim_log["bixelgrid_shifts_xy"][0]
    energy = sim_log["energy"][0]

    crop_t0 = perf_counter()
    cropped_ct, entrance, crp, oob = extract_beamlet_roi(
        rotated_ct,
        d_nozzle,
        d_smx,
        d_smy,
        spot_position,
        iso_phys,
        config.roi_size,
        ct_array=rotated_ct_array,
    )
    crop_t1 = perf_counter()

    flux_t0 = perf_counter()
    beamlet_angles = spot_position_to_angles(
        spot_position[0], spot_position[1], d_smx, d_smy
    )
    sigmas = flux_spatial_spread(bdl, energy)
    re_proj = [entrance[1], entrance[2], entrance[0]]
    if config.flux_on_gpu:
        flux = flux_projection_gpu(
            re_proj, beamlet_angles, sigmas, cropped_ct.shape, device=config.flux_device
        )
    else:
        flux = flux_projection(re_proj, beamlet_angles, sigmas, cropped_ct.shape)
    flux_t1 = perf_counter()

    sim_res = _build_sim_res(
        record,
        beamlet_angles,
        entrance,
        re_proj,
        crp,
        oob,
        image_origin,
        image_spacing,
        image_size,
        config.roi_size,
    )

    save_t0 = perf_counter()
    _save_spot(output_dir, record["id"], cropped_ct, flux, sim_res)
    save_t1 = perf_counter()

    # Absolute (start, end) timestamps so concurrent intervals can be unioned
    # into real wall-clock per-step times (see _union_seconds).
    return {
        "crop": (crop_t0, crop_t1),
        "flux": (flux_t0, flux_t1),
        "save": (save_t0, save_t1),
        "oob": int(oob),
    }


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create the output dir; on overwrite WIPE it first so no stale files survive.

    Accumulation reads every ``*_sim_res.json`` in the directory, so per-spot
    files left over from a previous run (e.g. a different grid / spot subset)
    must be removed -- not just written over -- or they would be mixed into the
    accumulated dose.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {output_dir} is not empty; pass overwrite=True "
                "to extract into it anyway."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _build_sim_res(
    record: dict,
    beamlet_angles: tuple,
    entrance: np.ndarray,
    re_proj: list,
    crp: tuple,
    oob: bool,
    image_origin: tuple,
    image_spacing: tuple,
    image_size: tuple,
    roi_size: tuple,
) -> dict:
    """Assemble the per-spot metadata dict (notebook schema + extras)."""
    sim_log = dict(record["simulation_log"])
    sim_log["beamlet_angles"] = list(beamlet_angles)
    return {
        "id": record["id"],
        "beam": record["beam"],
        "layer": record["layer"],
        "spot": record["spot"],
        "field_id": record["field_id"],
        "simulation_log": sim_log,
        "initial_energy": float(sim_log["energy"][0]),
        "gantry_angle": float(sim_log["gantry_angle"]),
        "relative_weight": float(sim_log["relative_weight"]),
        "roi_size": list(roi_size),
        "image_origin": list(image_origin),
        "image_spacing": list(image_spacing),
        "image_size": list(image_size),
        "rays_entrence_point": entrance.tolist(),
        "rays_entrence_point_proj": [float(v) for v in re_proj],
        "crp_numpy_ct": list(crp),
        "oob": bool(oob),
    }


def _save_spot(
    output_dir: Path,
    spot_id_str: str,
    cropped_ct: np.ndarray,
    flux: np.ndarray,
    sim_res: dict,
) -> None:
    """Write the CT crop, flux projection and metadata for one spot."""
    np.save(output_dir / f"{spot_id_str}_ct.npy", cropped_ct)
    np.save(output_dir / f"{spot_id_str}_flux.npy", flux)
    (output_dir / f"{spot_id_str}_sim_res.json").write_text(
        json.dumps(sim_res, indent=4, cls=NumpyEncoder)
    )


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


def _save_field_overlay(
    overlays_dir: Path,
    beam: int,
    rotated_ct: sitk.Image,
    field_spots: List[dict],
    isocenter_physical: tuple,
    bdl: BeamDataLibrary,
    roi_size: tuple,
) -> None:
    """Save a visual sanity-check PNG for a field's most-weighted spot.

    Left: the rotated CT axial slice through the isocenter with the isocenter and
    the +x beam axis marked. Right: that spot's crop (depth x vs lateral y) with
    the flux projection overlaid, to confirm the beam enters at x=0 and the flux
    traces the ray.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Representative spot = the most-weighted one in the field.
    record = max(field_spots, key=lambda r: r["simulation_log"]["relative_weight"])
    sim_log = record["simulation_log"]
    spot_position = sim_log["bixelgrid_shifts_xy"][0]
    energy = sim_log["energy"][0]
    d_nozzle, d_smx, d_smy = bdl.distances

    cropped_ct, entrance, crp, _oob = extract_beamlet_roi(
        rotated_ct, d_nozzle, d_smx, d_smy, spot_position, isocenter_physical, roi_size
    )
    beamlet_angles = spot_position_to_angles(
        spot_position[0], spot_position[1], d_smx, d_smy
    )
    sigmas = flux_spatial_spread(bdl, energy)
    re_proj = [entrance[1], entrance[2], entrance[0]]
    flux = flux_projection(re_proj, beamlet_angles, sigmas, cropped_ct.shape)

    iso_idx = rotated_ct.TransformPhysicalPointToIndex(
        [float(c) for c in isocenter_physical]
    )
    ct_arr = sitk.GetArrayFromImage(rotated_ct)  # (z, y, x)
    iz, iy, ix = iso_idx[2], iso_idx[1], iso_idx[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    # Left: rotated CT axial slice through the isocenter.
    axes[0].imshow(ct_arr[iz, :, :], cmap="gray", origin="lower")
    axes[0].scatter([ix], [iy], color="red", s=30, label="isocenter")
    axes[0].axhline(iy, color="cyan", lw=0.6, ls="--", label="beam axis (+x)")
    axes[0].set_title(
        f"beam {beam}: rotated CT axial @ iso z={iz}\n"
        f"gantry(adj)={sim_log['gantry_angle']:.0f} deg"
    )
    axes[0].set_xlabel("x (depth ->)")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=8)

    # Right: the spot's crop, depth (x) vs lateral (y) at mid-z, flux overlaid.
    mid_z = cropped_ct.shape[0] // 2
    axes[1].imshow(cropped_ct[mid_z, :, :], cmap="gray", origin="lower", aspect="auto")
    flux_slice = flux[mid_z, :, :]
    if float(flux_slice.max()) > 0:
        axes[1].contour(
            flux_slice,
            levels=[flux_slice.max() * lvl for lvl in (0.1, 0.5, 0.9)],
            colors="orange",
            linewidths=0.8,
        )
    axes[1].set_title(f"spot {record['id']}: CT crop + flux (mid-z)\nE={energy:.1f} MeV")
    axes[1].set_xlabel("x (depth, 0 = entrance)")
    axes[1].set_ylabel("y (lateral)")

    fig.tight_layout()
    fig.savefig(overlays_dir / f"beam{beam:02d}_{record['id']}.png", bbox_inches="tight")
    plt.close(fig)
