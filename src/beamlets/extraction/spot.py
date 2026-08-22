"""Per-spot beamlet extraction: BEV CT crop plus proton-flux projection.

Flow (:func:`_process_spot`, one spot):
1. Convert the spot's bixel-grid position into beamlet angles via the beam data
   library.
2. Crop the rotated CT to the beamlet ROI, air-padded from the entrance point.
3. Project the proton flux through that ROI (GPU when available).
4. Assemble the per-spot metadata record (:func:`_build_sim_res`).

Returns arrays and metadata; persisting them is
:mod:`~src.beamlets.extraction.io`'s job.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Optional

import numpy as np
import SimpleITK as sitk

from src.beamlets import ROI_SIZE
from src.beamlets.bdl import BeamDataLibrary, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.extraction.io import _save_spot
from src.beamlets.flux import flux_projection, flux_projection_gpu, flux_spatial_spread

if TYPE_CHECKING:  # annotation only; importing it eagerly would be circular
    from src.beamlets.extraction import ExtractionConfig

logger = logging.getLogger(__name__)

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
    roi: tuple = ROI_SIZE,
    flux_spacing: Optional[np.ndarray] = None,
) -> dict:
    """Crop + flux + save one spot; return its per-step timings and ``oob`` flag.

    This is the single per-spot implementation shared by the serial
    (:func:`run_extraction`) and pooled (:func:`run_extraction_pooled`) paths, so
    both produce byte-identical outputs. It reads the shared ``rotated_ct`` /
    ``rotated_ct_array`` (read-only) and writes only this spot's own files, so it
    is safe to run concurrently across spots. ``roi`` and ``flux_spacing`` come
    from the field-level resampling factor (``(60,60,320)`` + ``[1,1,1]`` at gf=1;
    ``(30,30,160)`` + ``[2,2,2]`` at gf=2) -- the crop and flux are then built on
    the same grid the model consumes.
    """
    if flux_spacing is None:
        flux_spacing = np.asarray([1, 1, 1], dtype=np.float32)
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
        roi,
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
            re_proj, beamlet_angles, sigmas, cropped_ct.shape,
            spacing=flux_spacing, device=config.flux_device,
        )
    else:
        flux = flux_projection(
            re_proj, beamlet_angles, sigmas, cropped_ct.shape, spacing=flux_spacing
        )
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
        roi,
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
