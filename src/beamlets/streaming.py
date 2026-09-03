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
numerically identical to the staged pipeline. On the ``grid_factor != 1`` path the
post-processing runs through the batched
:func:`~src.loaders.dir_based.postprocess_predictions_batched` instead, which is
bit-identical to the per-spot loop but de-normalizes and reorders on the device and
returns the batch in one pinned copy. Only one batch is ever live, so peak
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
from src.beamlets import ROI_SIZE, roi_for_factor
from src.beamlets.accumulation import deposit_crop
from src.beamlets.bdl import BeamDataLibrary, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import (
    flux_projection,
    flux_projection_gpu,
    flux_projection_gpu_batched,
    flux_spatial_spread,
)
from src.beamlets.isocenter import isocenter_physical
from src.beamlets.plan_spots import expand_plan_to_spots, group_by_field
from src.beamlets.rotation import derotation_subgrid, rotate_ct_around_isocenter
from src.image_processing.rotation import rotate_beamlet_crops_batched
from src.loaders.dir_based import (
    postprocess_prediction,
    postprocess_predictions_batched,
    prepare_input_from_arrays,
    prepare_inputs_from_arrays_batched,
)
from src.loaders.plan_directory import PlanDirectory

logger = logging.getLogger(__name__)

__all__ = ["StreamingConfig", "run_streaming_pipeline"]


def _nonzero_bounds(grid: np.ndarray):
    """Inclusive ``((z,y,x) lo, (z,y,x) hi)`` bounds of ``grid``'s non-zero voxels.

    ``None`` when the grid is entirely zero. Tests ``!= 0`` rather than ``> 0``
    because the model may predict negative dose (clipping happens once, at the end).
    """
    nonzero = grid != 0
    if not nonzero.any():
        return None
    axes = ((1, 2), (0, 2), (0, 1))
    found = [np.where(nonzero.any(axis=ax))[0] for ax in axes]
    return (
        tuple(int(f[0]) for f in found),
        tuple(int(f[-1]) for f in found),
    )


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
    flux_batched: bool = False
    """Build the whole batch's flux in one :func:`flux_projection_gpu_batched`
    call instead of per-spot (GPU only). ``False`` (default) keeps the per-spot
    path byte-identical to the staged pipeline; ``True`` is the optimized ADoTA
    reinterpretation used for the fair DoTA-vs-ADoTA plan timing (the counterpart
    to the batched GPU BEV rotation)."""
    flux_batched_dtype: str = "float64"
    """Compute dtype of the batched flux (``flux_batched`` only). ``"float64"``
    (default) reproduces the per-spot :func:`flux_projection_gpu` exactly -- it is
    the same float64 math, only evaluated for the whole batch at once -- so the
    batched path stays numerically equivalent to the production one at a few
    tenths of a second per plan. ``"float32"`` is ~2x faster on the flux alone and
    agrees to ~5e-7 relative; use it only where that has been validated."""
    batched_prep: bool = False
    """Build the whole batch's model input in one
    :func:`~src.loaders.dir_based.prepare_inputs_from_arrays_batched` call instead
    of per-spot. The ``B`` CT crops are staged into one contiguous pinned block and
    copied host-to-device **once**, and when ``flux_batched`` is on the flux tensor
    is consumed straight off the device, so the flux never makes a device -> host
    -> device round trip. ``False`` (default) keeps the per-spot loop."""
    flux_device: str = "cuda"
    normalize_flux: bool = True
    downsampling_method: str = "interpolation"
    scale: dict = field(default_factory=lambda: dict(DEFAULT_SCALE))
    calibration_factor: float = 1.0
    clip_negative: bool = True
    grid_factor: int = 1
    """Field-level resampling factor (1 = current 1mm per-beamlet path, byte-
    identical; 2 = rotate/crop/flux/deposit on the 2mm grid -- the resize is done
    once per field by the rotate/de-rotate instead of per beamlet)."""
    precision: str = "fp32"
    """Forward-pass precision. ``"fp32"`` (default) runs the model in full
    precision, unchanged. ``"fp16"`` wraps **only the model forward** in
    ``torch.autocast`` (CUDA half precision) for a large speed-up; the prediction
    is cast back to fp32 before deposit so accumulation is unchanged. fp16 is a
    no-op on CPU. Validate the dose (gamma vs MC) before adopting."""
    reinterpretation_mode: str = "adota_flux"
    """Which per-beamlet direction handling to run (for the DoTA-vs-ADoTA timing
    study). ``"adota_flux"`` (default) is the real ADoTA pipeline, byte-identical
    to before: axis-aligned crop + analytical flux channel. ``"dota_rotation"``
    is the DoTA-like path, actually executed for a *measured* plan number: each
    batch of CT crops is rotated into the BEV (batched GPU ``grid_sample``,
    charged to ``rotate_to_bev``), the shared model runs on the rotated CT with a
    zero second channel (no flux is built or charged), and the predicted dose is
    rotated back to the field frame (``rotate_to_field``) before deposit. The
    dose it deposits is NOT validated (the shared 2-channel model is fed a zero
    flux); this mode exists only to time the reinterpretation on real plan data."""


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
        "Streaming %d spots across %d field(s) (batch=%d, flux_on_gpu=%s, "
        "flux_batched=%s, batched_prep=%s) on %s",
        len(spots),
        len(grouped),
        config.batch_size,
        config.flux_on_gpu,
        config.flux_batched,
        config.batched_prep,
        device,
    )

    # Field-level resampling factor. gf=1 keeps every operation byte-identical to
    # the original 1mm path; gf=2 rotates/crops/fluxes/deposits on the 2mm grid and
    # the per-field rotate/de-rotate carries the single resize (no per-beamlet
    # down/up-sample). roi/flux-spacing are derived from gf; ``[1,1,1]`` (float32)
    # is exactly the flux default so gf=1 stays byte-identical.
    gf = config.grid_factor
    roi = config.roi_size if gf == 1 else roi_for_factor(gf)
    flux_spacing = np.asarray([gf, gf, gf], dtype=np.float32)
    # fp16 wraps only the GPU forward in autocast; the prediction is cast back to
    # fp32 for deposit so accumulation is unchanged. No-op on CPU (autocast fp16
    # is CUDA-only), so the staged-equivalence tests stay byte-identical.
    use_fp16 = config.precision == "fp16" and device.type == "cuda"
    if config.precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be 'fp32' or 'fp16', got {config.precision!r}")

    # Batched host<->device staging. The flux only stays resident when it was
    # actually built on the device in one call; otherwise the batched prep still
    # helps (one pinned CT copy instead of B pageable ones) but takes host arrays.
    if config.flux_batched_dtype not in ("float32", "float64"):
        raise ValueError(
            "flux_batched_dtype must be 'float32' or 'float64', got "
            f"{config.flux_batched_dtype!r}"
        )
    flux_dtype = getattr(torch, config.flux_batched_dtype)
    batched_flux = bool(config.flux_on_gpu and config.flux_batched)
    keep_flux_on_device = bool(batched_flux and config.batched_prep)

    model.eval()
    dota_mode = config.reinterpretation_mode == "dota_rotation"
    if config.reinterpretation_mode not in ("adota_flux", "dota_rotation"):
        raise ValueError(
            "reinterpretation_mode must be 'adota_flux' or 'dota_rotation', "
            f"got {config.reinterpretation_mode!r}"
        )
    if dota_mode:
        logger.warning(
            "reinterpretation_mode=dota_rotation: timing the DoTA BEV rotations on "
            "real plan data; the deposited dose is NOT validated (zero flux channel)."
        )

    total = np.zeros(sitk.GetArrayFromImage(ct).shape, dtype=np.float32)  # (z, y, x)
    timing = {
        k: 0.0
        for k in (
            "rotation", "crop", "flux", "rotate_to_bev", "prep", "forward",
            "post", "rotate_to_field", "deposit", "derotate",
        )
    }
    n_spots = 0
    # Reused pinned host buffer for the batched post-processing copy (gf != 1 on
    # CUDA); allocated on the first batch, once the prediction's shape is known.
    post_buffer: Optional[torch.Tensor] = None
    # Reused pinned host staging buffer for the batched CT host->device copy
    # (``batched_prep`` on CUDA); allocated on the first batch, once the crop
    # shape is known.
    prep_buffer: Optional[torch.Tensor] = None
    started = perf_counter()

    for beam, field_spots in grouped.items():
        iso_index = field_spots[0]["simulation_log"]["isocenter"]
        angle = field_spots[0]["simulation_log"]["gantry_angle"]
        iso_phys = isocenter_physical(iso_index, ct)

        rot_t = perf_counter()
        rotated_ct = rotate_ct_around_isocenter(
            ct, angle, iso_phys, expand=True, out_spacing_factor=gf
        )
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
            # Crop every record first; collect what each reinterpretation needs.
            crops, angles_list, energy_list, flux_list = [], [], [], []
            flux_params = []  # (re_proj, angles, sigmas) for the ADoTA flux
            for record in batch:
                sim_log = record["simulation_log"]
                spot_position = sim_log["bixelgrid_shifts_xy"][0]
                energy = sim_log["energy"][0]

                crop_t = perf_counter()
                cropped_ct, entrance, crp, _oob = extract_beamlet_roi(
                    rotated_ct, d_nozzle, d_smx, d_smy, spot_position, iso_phys,
                    roi, ct_array=rotated_ct_array,
                )
                timing["crop"] += perf_counter() - crop_t

                beamlet_angles = spot_position_to_angles(
                    spot_position[0], spot_position[1], d_smx, d_smy
                )
                crops.append(cropped_ct)
                angles_list.append(beamlet_angles)
                energy_list.append(energy)
                deposits.append((crp, float(sim_log["relative_weight"])))

                if not dota_mode:
                    # Charged to "flux": resolving the spot sigmas is part of
                    # building the flux channel, and leaving it untimed hid ~18%
                    # of the stream stage from the timing table.
                    sig_t = perf_counter()
                    sigmas = flux_spatial_spread(bdl, energy)
                    re_proj = [entrance[1], entrance[2], entrance[0]]
                    flux_params.append((re_proj, beamlet_angles, sigmas))
                    timing["flux"] += perf_counter() - sig_t

            # ADoTA reinterpretation: analytical flux (the charged cost). Batched
            # (one GPU call) when flux_batched, else per-spot; both feed the model
            # the axis-aligned crop + flux channel.
            if not dota_mode:
                shape = crops[0].shape
                flux_t = perf_counter()
                if batched_flux:
                    flux_batch = flux_projection_gpu_batched(
                        [p[0] for p in flux_params], [p[1] for p in flux_params],
                        [p[2] for p in flux_params], shape, spacing=flux_spacing,
                        device=config.flux_device, dtype=flux_dtype,
                        return_numpy=not keep_flux_on_device,
                    )
                    # Handed to the batched prep as a device tensor (no host round
                    # trip); only materialised on the host for the per-spot path.
                    flux_list = flux_batch if keep_flux_on_device else list(flux_batch)
                elif config.flux_on_gpu:
                    flux_list = [
                        flux_projection_gpu(p[0], p[1], p[2], shape,
                                            spacing=flux_spacing, device=config.flux_device)
                        for p in flux_params
                    ]
                else:
                    flux_list = [
                        flux_projection(p[0], p[1], p[2], shape, spacing=flux_spacing)
                        for p in flux_params
                    ]
                timing["flux"] += perf_counter() - flux_t
                model_cts = crops
                normalize_flux = config.normalize_flux

            # DoTA reinterpretation (1/2): batched CT-patch -> BEV rotation (GPU).
            if dota_mode:
                r2b_t = perf_counter()
                rot = rotate_beamlet_crops_batched(
                    crops, angles_list, inverse=False, device=str(device),
                    dtype=torch.float32, repeats=1, return_numpy=True,
                )
                model_cts = list(rot.rotated_numpy)
                timing["rotate_to_bev"] += perf_counter() - r2b_t
                # Zero second channel: a real DoTA model is single-channel, so no
                # flux is built or charged; the shared model still needs 2 inputs.
                flux_list = [np.zeros_like(c) for c in model_cts]
                normalize_flux = False

            if config.batched_prep:
                # One contiguous pinned host->device copy for the CT channel and
                # (with flux_batched) zero copies for the flux, instead of 2*B
                # per-spot transfers. Numerically the same as the loop below.
                prep_t = perf_counter()
                if prep_buffer is None and device.type == "cuda":
                    prep_buffer = torch.empty(
                        (config.batch_size, *model_cts[0].shape),
                        dtype=torch.float32, pin_memory=True,
                    )
                x_batch, e_batch = prepare_inputs_from_arrays_batched(
                    model_cts, flux_list, energy_list, scale=config.scale,
                    normalize_flux=normalize_flux,
                    downsampling_method=config.downsampling_method, device=device,
                    resize=(gf == 1), ct_buffer=prep_buffer,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timing["prep"] += perf_counter() - prep_t
            else:
                for cropped_ct, flux, energy in zip(model_cts, flux_list, energy_list):
                    prep_t = perf_counter()
                    x, e = prepare_input_from_arrays(
                        cropped_ct, flux, energy, scale=config.scale,
                        normalize_flux=normalize_flux,
                        downsampling_method=config.downsampling_method, device=device,
                        resize=(gf == 1),
                    )
                    timing["prep"] += perf_counter() - prep_t
                    inputs.append(x)
                    energies.append(e)

                x_batch = torch.stack(inputs).to(device)
                e_batch = torch.stack(energies).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fwd_t = perf_counter()
            with torch.no_grad():
                if use_fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred = model(x_batch, e_batch)[0]
                    pred = pred.float()  # back to fp32 for an unchanged deposit
                else:
                    pred = model(x_batch, e_batch)[0]  # (B, 1, 160, 30, 30)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timing["forward"] += perf_counter() - fwd_t

            dose_crops = []
            if gf == 1:
                # 1mm path: the per-beamlet trilinear up-sample to (320,60,60) is
                # per-spot and its batch would not fit a pinned buffer, so this
                # stays the original loop (and stays byte-identical).
                for i in range(len(deposits)):
                    post_t = perf_counter()
                    dose_pred = postprocess_prediction(
                        pred[i : i + 1], config.scale, upsample=True
                    )
                    # (1,1,D,H,W) -> (H,W,D) = (z,y,x) crop, as accumulation does.
                    dose_crop = np.moveaxis(np.squeeze(dose_pred), 0, -1)
                    timing["post"] += perf_counter() - post_t
                    dose_crops.append(dose_crop)
            else:
                # Field-grid path: de-normalize and reorder to (z,y,x) on the
                # device, then bring the whole batch back in one copy into a
                # reused pinned buffer. Bit-identical to the loop above, but it
                # replaces B pageable copies with one pinned copy and hands the
                # deposit a contiguous crop instead of a strided moveaxis view.
                post_t = perf_counter()
                if post_buffer is None and device.type == "cuda":
                    post_buffer = torch.empty(
                        (config.batch_size, pred.shape[3], pred.shape[4], pred.shape[2]),
                        dtype=torch.float32,
                        pin_memory=True,
                    )
                dose_batch = postprocess_predictions_batched(
                    pred, config.scale, out=post_buffer
                )
                # Views into ``post_buffer``; consumed by the deposit below, before
                # the next batch overwrites them.
                dose_crops = list(dose_batch)
                timing["post"] += perf_counter() - post_t

            # DoTA reinterpretation (2/2): batched BEV dose -> field-frame (GPU).
            if dota_mode:
                d2f_t = perf_counter()
                rot_back = rotate_beamlet_crops_batched(
                    dose_crops, angles_list, inverse=True, device=str(device),
                    dtype=torch.float32, repeats=1, return_numpy=True,
                )
                dose_crops = list(rot_back.rotated_numpy)
                timing["rotate_to_field"] += perf_counter() - d2f_t

            for (crp, weight), dose_crop in zip(deposits, dose_crops):
                dep_t = perf_counter()
                deposit_crop(deposit_grid, dose_crop, crp, weight, roi)
                timing["deposit"] += perf_counter() - dep_t
                n_spots += 1

        # De-rotate the field's deposited grid back to the original CT grid and add.
        der_t = perf_counter()
        rotated_image = sitk.GetImageFromArray(deposit_grid)
        rotated_image.SetOrigin(rotated_ct.GetOrigin())
        rotated_image.SetSpacing(rotated_ct.GetSpacing())
        rotated_image.SetDirection(ct.GetDirection())
        # Only the part of the CT grid that can sample a non-zero deposit voxel is
        # de-rotated; everywhere else all eight trilinear neighbours are zero, so
        # the result is exactly zero and adding it is a no-op. A plan's dose
        # typically occupies about a tenth of the CT, so both the resample and the
        # accumulation shrink with it. Nothing is dropped; the values inside the
        # sub-grid can differ from a full-grid resample by an ulp or two from the
        # resampler's coordinate arithmetic (see :func:`derotation_subgrid`).
        bounds = _nonzero_bounds(deposit_grid)
        if bounds is not None:
            sub_reference, sub_slices = derotation_subgrid(
                rotated_image, bounds, -angle, iso_phys, ct
            )
            if sub_reference is not None:
                derotated = rotate_ct_around_isocenter(
                    rotated_image, -angle, iso_phys,
                    reference=sub_reference, default_value=0.0,
                )
                total[sub_slices] += sitk.GetArrayFromImage(derotated)
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
        "grid_factor": gf,
        "grid_mode": "1mm" if gf == 1 else f"{gf}mm_field",
        "reinterpretation_mode": config.reinterpretation_mode,
        "batch_size": int(config.batch_size),
        "flux_batched": bool(config.flux_batched),
        "batched_prep": bool(config.batched_prep),
        "precision": "fp16" if use_fp16 else "fp32",
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
