"""Fast streaming plan pipeline: fused extract -> infer -> accumulate, no disk.

A single-pass alternative to the staged ``extract,infer,accumulate`` stages that
never writes a beamlet to disk. Per field it rotates the CT around the isocenter,
then streams the field's spots in batches: crop the BEV CT, build the flux
projection (GPU), preprocess + down-sample (GPU), run a batched forward, up-sample
+ de-normalize (GPU), and deposit each prediction into the field's grid -- exactly
the same operations as the staged path (it reuses the shared
:func:`src.loaders.dir_based.prepare_input_from_arrays` /
:func:`~src.loaders.dir_based.postprocess_prediction` and
:func:`src.beamlets.accumulation.deposit_crop`), so the accumulated dose is
numerically identical to the staged pipeline. Only one batch is ever live, so peak
memory is bounded (~one batch + a few CT-sized grids) regardless of spot count --
which is what the per-beamlet disk round-trip was originally working around.

The staged pipeline (``extraction`` / ``inference`` / ``accumulation``) is left
completely untouched; this is a separate, additive path selected by the ``stream``
stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import numpy as np
import SimpleITK as sitk
import torch

from src.adota.config import DEFAULT_SCALE
from src.beamlets import ROI_SIZE
from src.beamlets.accumulation import deposit_crop
from src.beamlets.bdl import BeamDataLibrary, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import flux_projection, flux_projection_gpu, flux_spatial_spread
from src.beamlets.isocenter import isocenter_physical
from src.beamlets.plan_spots import expand_plan_to_spots, group_by_field
from src.beamlets.rotation import rotate_ct_around_isocenter
from src.loaders.dir_based import postprocess_prediction, prepare_input_from_arrays
from src.loaders.plan_directory import PlanDirectory

logger = logging.getLogger(__name__)

__all__ = ["StreamingConfig", "run_streaming_pipeline"]


@dataclass
class StreamingConfig:
    """Configuration for :func:`run_streaming_pipeline`.

    Mirrors the relevant fields of the staged ``ExtractionConfig`` /
    ``InferenceConfig`` / ``AccumulationConfig`` so the streaming run matches them.
    """

    roi_size: tuple = ROI_SIZE
    n_spots: Optional[int] = None
    beams: Optional[List[int]] = None
    bdl_path: Optional[Path] = None
    batch_size: int = 56
    flux_on_gpu: bool = True
    flux_device: str = "cuda"
    normalize_flux: bool = True
    downsampling_method: str = "interpolation"
    scale: dict = field(default_factory=lambda: dict(DEFAULT_SCALE))
    calibration_factor: float = 1.0
    clip_negative: bool = True


def run_streaming_pipeline(
    plan_directory: PlanDirectory,
    model: torch.nn.Module,
    device: torch.device,
    output_path: Path,
    config: Optional[StreamingConfig] = None,
) -> dict:
    """Fused, disk-free plan dose computation; writes ``output_path`` and returns a summary.

    Args:
        plan_directory: The loaded plan directory (CT, parsed plan, BDL path).
        model: The loaded ADoTA model (on ``device``, eval mode).
        device: Target device for inference / GPU flux / resize.
        output_path: Destination ``.mhd`` (e.g. ``<plan_dir>/Dose_ADoTA.mhd``).
        config: Streaming options.

    Returns:
        A summary dict (timing per step, spot/field counts, output path).

    Raises:
        ValueError: If ``output_path`` would overwrite the MC reference dose.
    """
    config = config or StreamingConfig()
    output_path = Path(output_path)
    if output_path.name.lower() in {"dose.mhd", "dose.raw"}:
        raise ValueError(
            f"Refusing to write to {output_path.name}: the MC reference dose is "
            "read-only. Choose a different output name."
        )

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
        "Streaming %d spots across %d field(s) (batch=%d, flux_on_gpu=%s) on %s",
        len(spots),
        len(grouped),
        config.batch_size,
        config.flux_on_gpu,
        device,
    )

    model.eval()
    total = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.float32)  # (z, y, x)
    timing = {k: 0.0 for k in ("rotation", "crop", "flux", "prep", "forward", "post", "deposit", "derotate")}
    n_spots = 0
    started = perf_counter()

    for beam, field_spots in grouped.items():
        iso_index = field_spots[0]["simulation_log"]["isocenter"]
        angle = field_spots[0]["simulation_log"]["gantry_angle"]
        iso_phys = isocenter_physical(iso_index, ct)

        rot_t = perf_counter()
        rotated_ct = rotate_ct_around_isocenter(ct, angle, iso_phys, expand=True)
        rotated_ct_array = sitk.GetArrayFromImage(rotated_ct)
        timing["rotation"] += perf_counter() - rot_t

        ex_nx, ex_ny, ex_nz = rotated_ct.GetSize()
        deposit_grid = np.zeros((ex_nz, ex_ny, ex_nx), dtype=np.float32)

        batches = [
            field_spots[i : i + config.batch_size]
            for i in range(0, len(field_spots), config.batch_size)
        ]
        for batch in batches:
            inputs, energies, deposits = [], [], []  # deposits: (crp, weight)
            for record in batch:
                sim_log = record["simulation_log"]
                spot_position = sim_log["bixelgrid_shifts_xy"][0]
                energy = sim_log["energy"][0]

                crop_t = perf_counter()
                cropped_ct, entrance, crp, _oob = extract_beamlet_roi(
                    rotated_ct, d_nozzle, d_smx, d_smy, spot_position, iso_phys,
                    config.roi_size, ct_array=rotated_ct_array,
                )
                timing["crop"] += perf_counter() - crop_t

                flux_t = perf_counter()
                beamlet_angles = spot_position_to_angles(
                    spot_position[0], spot_position[1], d_smx, d_smy
                )
                sigmas = flux_spatial_spread(bdl, energy)
                re_proj = [entrance[1], entrance[2], entrance[0]]
                if config.flux_on_gpu:
                    flux = flux_projection_gpu(
                        re_proj, beamlet_angles, sigmas, cropped_ct.shape,
                        device=config.flux_device,
                    )
                else:
                    flux = flux_projection(re_proj, beamlet_angles, sigmas, cropped_ct.shape)
                timing["flux"] += perf_counter() - flux_t

                prep_t = perf_counter()
                x, e = prepare_input_from_arrays(
                    cropped_ct, flux, energy, scale=config.scale,
                    normalize_flux=config.normalize_flux,
                    downsampling_method=config.downsampling_method, device=device,
                )
                timing["prep"] += perf_counter() - prep_t

                inputs.append(x)
                energies.append(e)
                deposits.append((crp, float(sim_log["relative_weight"])))

            x_batch = torch.stack(inputs).to(device)
            e_batch = torch.stack(energies).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fwd_t = perf_counter()
            with torch.no_grad():
                pred = model(x_batch, e_batch)[0]  # (B, 1, 160, 30, 30)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timing["forward"] += perf_counter() - fwd_t

            for i, (crp, weight) in enumerate(deposits):
                post_t = perf_counter()
                dose_pred = postprocess_prediction(pred[i : i + 1], config.scale)
                # (1,1,320,60,60) -> (60,60,320) = (z,y,x) crop, as accumulation does.
                dose_crop = np.moveaxis(np.squeeze(dose_pred), 0, -1)
                timing["post"] += perf_counter() - post_t

                dep_t = perf_counter()
                deposit_crop(deposit_grid, dose_crop, crp, weight, config.roi_size)
                timing["deposit"] += perf_counter() - dep_t
                n_spots += 1

        # De-rotate the field's deposited grid back to the original CT grid and add.
        der_t = perf_counter()
        rotated_image = sitk.GetImageFromArray(deposit_grid)
        rotated_image.SetOrigin(rotated_ct.GetOrigin())
        rotated_image.SetSpacing(rotated_ct.GetSpacing())
        rotated_image.SetDirection(ct.GetDirection())
        derotated = rotate_ct_around_isocenter(
            rotated_image, -angle, iso_phys, reference=ct, default_value=0.0
        )
        total += sitk.GetArrayFromImage(derotated)
        timing["derotate"] += perf_counter() - der_t
        logger.info("Field beam=%d: streamed %d spots", beam, len(field_spots))

    if config.clip_negative:
        total = np.clip(total, 0.0, None)
    if config.calibration_factor != 1.0:
        total *= np.float32(config.calibration_factor)

    dose_image = sitk.GetImageFromArray(total)
    dose_image.CopyInformation(ct)
    write_t = perf_counter()
    sitk.WriteImage(dose_image, str(output_path))
    timing["write"] = perf_counter() - write_t

    elapsed = perf_counter() - started
    summary = {
        "n_spots": n_spots,
        "n_fields": len(grouped),
        "elapsed_s": elapsed,
        "calibration_factor": float(config.calibration_factor),
        "dose_max": float(total.max()),
        "dose_sum": float(total.sum()),
        "grid_size": list(ct.GetSize()),
        "output_path": str(output_path),
        "timing": timing,
    }
    logger.info(
        "Streaming complete: %d spots across %d field(s) -> %s (max=%.4g) in %.1fs",
        n_spots,
        len(grouped),
        output_path,
        summary["dose_max"],
        elapsed,
    )
    return summary
