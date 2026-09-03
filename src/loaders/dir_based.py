"""Load a single beamlet record from a directory of numpy arrays.

The directory layout is the one the paper's test sets ship in: one ``.npy`` per
volume (CT, flux, dose) per sample id.

Flow:
1. Read the CT / flux / dose arrays for one sample id.
2. Optionally downsample to the (160, 30, 30) model grid (interpolation or
   average pooling).
3. Min-max normalise with the training ``scale`` dict and stack into the
   two-channel model input.
4. ``postprocess_prediction`` / ``save_prediction`` invert that for output.
"""

import json
import logging
import os
from time import perf_counter
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Default scale corresponds to the low-range model, trained on the lung dataset.
DEFAULT_SCALE = {
    "min_ds": 0.0,
    "max_ds": 24732944.0,
    "min_ct": -1024,
    "max_ct": 3063,
    "min_energy": 70.00221819271046,
    "max_energy": 179.99924071411004,
}


def get_single_record(
    id: str,
    storage_path: str,
    scale: dict = None,
    normalize_flux: bool = True,
    downsampling_method: str = "interpolation",
    beamlet_angle: float = False,
) -> Tuple[torch.Tensor]:
    scale = DEFAULT_SCALE if scale is None else scale
    print("Using scale: ", scale)

    x = np.load(os.path.join(storage_path, f"{id}_ct.npy"))
    flux = np.load(os.path.join(storage_path, f"{id}_flux.npy"))
    y = np.load(os.path.join(storage_path, f"{id}_ds.npy"))
    with open(os.path.join(storage_path, f"{id}_sim_res.json"), "r") as f:
        meta = json.load(f)

    energy = meta["simulation_log"]["energy"][0]
    beamlet_angle_ = meta["simulation_log"].get("beamlet_angles", None)
    # Convert numpy arrays to PyTorch tensors
    ct_grid = torch.tensor(x, dtype=torch.float32)
    dose_grid = torch.tensor(y, dtype=torch.float32)
    flux_grid = torch.tensor(flux, dtype=torch.float32)
    e = energy

    ct_grid = (ct_grid - scale["min_ct"]) / (scale["max_ct"] - scale["min_ct"])
    dose_grid = (dose_grid - scale["min_ds"]) / (scale["max_ds"] - scale["min_ds"])
    e = (meta["initial_energy"] - scale["min_energy"]) / (
        scale["max_energy"] - scale["min_energy"]
    )

    # Permute dimensions to (D, H, W)
    ct_grid = ct_grid.permute(2, 0, 1)
    dose_grid = dose_grid.permute(2, 0, 1)
    flux_grid = flux_grid.permute(2, 0, 1)
    # Apply channel dimension
    ct_grid = ct_grid.unsqueeze(0)
    dose_grid = dose_grid.unsqueeze(0)
    flux_grid = flux_grid.unsqueeze(0)
    if normalize_flux:
        flux_grid = (flux_grid - flux_grid.min()) / (flux_grid.max() - flux_grid.min())

    logger.info(
        f"Loaded data for ID: {id}. Shapes - CT: {ct_grid.shape}, Dose: {dose_grid.shape}, Flux: {flux_grid.shape}"
    )
    # Perform Avarage Pooling on dimension physical dimensions
    if downsampling_method == "avg_pooling":
        ct_grid = F.avg_pool3d(ct_grid, kernel_size=2, stride=2)
        dose_grid = F.avg_pool3d(dose_grid, kernel_size=2, stride=2)
        flux_grid = F.avg_pool3d(flux_grid, kernel_size=2, stride=2)
        logger.info(
            f"Downsampled using Average Pooling. New shape: CT: {ct_grid.shape}, "
            f"Dose: {dose_grid.shape}, Flux: {flux_grid.shape}"
        )

    # Perform Interpolation using F.interpolate to resize to (160, 30, 30)
    if downsampling_method == "interpolation":
        ct_grid = F.interpolate(
            ct_grid.unsqueeze(0),
            size=(160, 30, 30),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        dose_grid = F.interpolate(
            dose_grid.unsqueeze(0),
            size=(160, 30, 30),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        flux_grid = F.interpolate(
            flux_grid.unsqueeze(0),
            size=(160, 30, 30),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    # Concatente flux and ct grid
    x = torch.cat((ct_grid, flux_grid), dim=0)
    initial_energy = torch.tensor(e, dtype=torch.float32)
    initial_energy = initial_energy.unsqueeze(0)
    if beamlet_angle and beamlet_angle_ is not None:
        return x, initial_energy, dose_grid, beamlet_angle_

    return x, initial_energy, dose_grid


def prepare_input_from_arrays(
    ct_crop: np.ndarray,
    flux_crop: np.ndarray,
    initial_energy_mev: float,
    scale: dict = None,
    normalize_flux: bool = True,
    downsampling_method: str = "interpolation",
    device: Optional["torch.device"] = None,
    timing: Optional[dict] = None,
    resize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the 2-channel ADoTA model input from in-memory CT/flux crops.

    This is the array-level core shared by the disk loader
    (:func:`get_single_record_no_gt`) and the streaming pipeline
    (:mod:`src.beamlets.streaming`), so both produce numerically identical inputs.
    It converts the CT/flux crops to tensors (on ``device`` when given, so the
    normalization + trilinear down-sample run there), normalizes, and concatenates
    them with the normalized energy. The model always consumes the ``(160, 30, 30)``
    grid; what differs between the two extraction paths is only whether a resize is
    needed to reach it.

    Args:
        ct_crop: BEV CT crop ``(z, y, x)`` in HU -- ``(60, 60, 320)`` on the 1mm
            grid (``grid_factor=1``) or ``(30, 30, 160)`` on the 2mm grid
            (``grid_factor=2``).
        flux_crop: Flux projection, same shape.
        initial_energy_mev: Beam energy in MeV (normalized with the energy scale).
        scale: Min-max scaling dict (defaults to :data:`DEFAULT_SCALE`).
        normalize_flux: Min-max normalize the flux channel per crop.
        downsampling_method: ``"interpolation"`` (trilinear) or ``"avg_pooling"``.
        device: Torch device for the tensors / resize (``None`` = CPU).
        timing: Optional measurement hook; accumulates the resize seconds under
            ``"downsample"``.
        resize: When ``True`` (default, ``grid_factor=1``) the permuted 1mm crop
            ``(320, 60, 60)`` is trilinearly resized to the ``(160, 30, 30)`` model
            grid exactly as before. When ``False`` (``grid_factor=2``) the crop was
            already cropped on the 2mm grid and permutes straight to ``(160, 30, 30)``
            so the resize -- which would be a no-op resize-to-self -- is skipped.

    Returns:
        ``(x, energy)`` where ``x`` is ``(2, 160, 30, 30)`` and ``energy`` is the
        normalized scalar energy ``(1,)``, both on ``device``.
    """
    scale = DEFAULT_SCALE if scale is None else scale

    # Convert numpy arrays to PyTorch tensors (on ``device`` when given, so the
    # normalization + trilinear resize below run there).
    ct_grid = torch.tensor(ct_crop, dtype=torch.float32, device=device)
    flux_grid = torch.tensor(flux_crop, dtype=torch.float32, device=device)

    ct_grid = (ct_grid - scale["min_ct"]) / (scale["max_ct"] - scale["min_ct"])
    e = (initial_energy_mev - scale["min_energy"]) / (
        scale["max_energy"] - scale["min_energy"]
    )

    # Permute dimensions to (D, H, W)
    ct_grid = ct_grid.permute(2, 0, 1)
    flux_grid = flux_grid.permute(2, 0, 1)
    # Apply channel dimension
    ct_grid = ct_grid.unsqueeze(0)
    flux_grid = flux_grid.unsqueeze(0)

    if normalize_flux:
        flux_grid = (flux_grid - flux_grid.min()) / (flux_grid.max() - flux_grid.min())

    interp_t = perf_counter()
    # Perform Avarage Pooling on dimension physical dimensions
    if downsampling_method == "avg_pooling":
        ct_grid = F.avg_pool3d(ct_grid, kernel_size=2, stride=2)
        flux_grid = F.avg_pool3d(flux_grid, kernel_size=2, stride=2)

    if downsampling_method == "interpolation" and resize:
        ct_grid = F.interpolate(
            ct_grid.unsqueeze(0),
            size=(160, 30, 30),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        flux_grid = F.interpolate(
            flux_grid.unsqueeze(0),
            size=(160, 30, 30),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    if timing is not None:
        # Sync so the async GPU resize is attributed here, not to a later op
        # (measurement-only; the normal path is untouched).
        if ct_grid.is_cuda:
            torch.cuda.synchronize(ct_grid.device)
        timing["downsample"] = timing.get("downsample", 0.0) + (perf_counter() - interp_t)

    # Concatente flux and ct grid
    x = torch.cat((ct_grid, flux_grid), dim=0)
    initial_energy = torch.tensor(e, dtype=torch.float32)
    initial_energy = initial_energy.unsqueeze(0)

    return x, initial_energy


def _stage_batch(
    crops,
    device: Optional["torch.device"] = None,
    buffer: Optional["torch.Tensor"] = None,
) -> torch.Tensor:
    """Stack per-spot crops into one contiguous ``(B, z, y, x)`` device tensor.

    ``crops`` may already be a device tensor (the batched GPU flux hands one over
    directly, so no host round-trip happens at all), a stacked NumPy array, or a
    list of per-spot NumPy crops. In the list case the crops are stacked into a
    single contiguous block and moved to ``device`` in **one** copy instead of
    ``B`` separate ones; passing a pinned ``buffer`` (shape ``(>= B, z, y, x)``,
    ``pin_memory=True``) keeps that copy off the pageable-allocation path.

    Args:
        crops: Device tensor, ``(B, z, y, x)`` array, or a list of ``(z, y, x)``
            crops (all the same shape).
        device: Destination torch device (``None`` = CPU).
        buffer: Optional pinned host staging tensor reused across batches. It is
            only a transfer staging area -- the returned tensor is the device
            copy, so the buffer may be overwritten by the next batch.

    Returns:
        A ``(B, z, y, x)`` float32 tensor on ``device``.
    """
    if isinstance(crops, torch.Tensor):
        return crops.to(device=device, dtype=torch.float32)
    if isinstance(crops, np.ndarray) and crops.ndim == 4:
        return torch.as_tensor(np.ascontiguousarray(crops, dtype=np.float32)).to(
            device=device, non_blocking=True
        )

    n_batch = len(crops)
    if buffer is not None and buffer.shape[0] >= n_batch:
        staged = buffer[:n_batch]
        np.stack([np.asarray(c, dtype=np.float32) for c in crops], axis=0,
                 out=staged.numpy())
    else:
        staged = torch.from_numpy(
            np.stack([np.asarray(c, dtype=np.float32) for c in crops], axis=0)
        )
    return staged.to(device=device, non_blocking=True)


def prepare_inputs_from_arrays_batched(
    ct_crops,
    flux_crops,
    initial_energies_mev: "Sequence[float]",
    scale: dict = None,
    normalize_flux: bool = True,
    downsampling_method: str = "interpolation",
    device: Optional["torch.device"] = None,
    resize: bool = True,
    ct_buffer: Optional["torch.Tensor"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched twin of :func:`prepare_input_from_arrays` for a whole batch at once.

    Numerically equivalent to calling :func:`prepare_input_from_arrays` per spot
    and ``torch.stack``-ing the results -- every step here is the same elementwise
    op or the same per-sample reduction, just applied over the batch axis -- but it
    replaces the per-spot work with three batched steps:

    1. the ``B`` CT crops are staged into one contiguous (optionally pinned) block
       and moved to ``device`` in a **single** host-to-device copy;
    2. ``flux_crops`` may be handed in as a tensor **already on the device** (what
       :func:`~src.beamlets.flux.flux_projection_gpu_batched` returns), so the
       flux channel never makes a device -> host -> device round trip;
    3. normalization, the permute to ``(D, H, W)`` and the trilinear resize run
       once over the batch instead of ``B`` times.

    Args:
        ct_crops: ``B`` BEV CT crops ``(z, y, x)`` in HU -- a list, a stacked
            ``(B, z, y, x)`` array, or a device tensor.
        flux_crops: The matching flux projections, same accepted forms. A device
            tensor is used in place (no host copy).
        initial_energies_mev: ``(B,)`` beam energies in MeV.
        scale: Min-max scaling dict (defaults to :data:`DEFAULT_SCALE`).
        normalize_flux: Min-max normalize the flux channel **per crop** (the
            per-sample reduction matches the per-spot path exactly).
        downsampling_method: ``"interpolation"`` (trilinear) or ``"avg_pooling"``.
        device: Torch device for the batch.
        resize: As in :func:`prepare_input_from_arrays` -- ``True`` on the 1mm
            (``grid_factor=1``) path resizes to ``(160, 30, 30)``; ``False`` on the
            2mm path skips the resize-to-self.
        ct_buffer: Optional pinned host staging tensor ``(>= B, z, y, x)`` reused
            across batches for the CT copy.

    Returns:
        ``(x, energies)`` where ``x`` is ``(B, 2, 160, 30, 30)`` and ``energies``
        is ``(B, 1)``, both on ``device``.
    """
    scale = DEFAULT_SCALE if scale is None else scale

    ct_grid = _stage_batch(ct_crops, device=device, buffer=ct_buffer)
    flux_grid = _stage_batch(flux_crops, device=device)

    ct_grid = (ct_grid - scale["min_ct"]) / (scale["max_ct"] - scale["min_ct"])

    # (B, z, y, x) -> (B, 1, D, H, W); the per-spot path's permute(2, 0, 1) plus
    # the channel axis, done over the batch.
    ct_grid = ct_grid.permute(0, 3, 1, 2).unsqueeze(1)
    flux_grid = flux_grid.permute(0, 3, 1, 2).unsqueeze(1)

    if normalize_flux:
        # Per-sample min/max: min and max are exact reductions, so reducing over
        # the batch axis in one call matches the per-crop ``.min()`` / ``.max()``.
        dims = (1, 2, 3, 4)
        lo = torch.amin(flux_grid, dim=dims, keepdim=True)
        hi = torch.amax(flux_grid, dim=dims, keepdim=True)
        flux_grid = (flux_grid - lo) / (hi - lo)

    if downsampling_method == "avg_pooling":
        ct_grid = F.avg_pool3d(ct_grid, kernel_size=2, stride=2)
        flux_grid = F.avg_pool3d(flux_grid, kernel_size=2, stride=2)

    if downsampling_method == "interpolation" and resize:
        ct_grid = F.interpolate(
            ct_grid, size=(160, 30, 30), mode="trilinear", align_corners=False
        )
        flux_grid = F.interpolate(
            flux_grid, size=(160, 30, 30), mode="trilinear", align_corners=False
        )

    x = torch.cat((ct_grid, flux_grid), dim=1)
    e = (
        np.asarray(initial_energies_mev, dtype=np.float64) - scale["min_energy"]
    ) / (scale["max_energy"] - scale["min_energy"])
    energies = torch.as_tensor(e, dtype=torch.float32).unsqueeze(1).to(device)
    return x, energies


def get_single_record_no_gt(
    id: str,
    storage_path: str,
    scale: dict = None,
    normalize_flux: bool = True,
    downsampling_method: str = "interpolation",
    timing: Optional[dict] = None,
    device: Optional["torch.device"] = None,
    resize: bool = True,
) -> Tuple[torch.Tensor]:
    """Load and prepare a single inference record (reads files, then delegates).

    ``timing`` is an optional measurement-only hook: when a dict is passed, the
    seconds spent reading the per-spot files are accumulated under ``"read"`` and
    the seconds spent down-sampling to the ADoTA grid under ``"downsample"``. It
    does not affect the returned tensors (the training/eval path passes no dict).

    ``device`` (default ``None`` = CPU, unchanged) moves the CT/flux tensors onto
    the given Torch device, so the normalization + trilinear down-sampling run
    there. The file read + preprocessing is identical to before; the preprocessing
    now lives in :func:`prepare_input_from_arrays` (shared with the streaming path).
    """
    scale = DEFAULT_SCALE if scale is None else scale

    read_t = perf_counter()
    x = np.load(os.path.join(storage_path, f"{id}_ct.npy"))
    flux = np.load(os.path.join(storage_path, f"{id}_flux.npy"))
    with open(os.path.join(storage_path, f"{id}_sim_res.json"), "r") as f:
        meta = json.load(f)
    if timing is not None:
        timing["read"] = timing.get("read", 0.0) + (perf_counter() - read_t)

    return prepare_input_from_arrays(
        x,
        flux,
        meta["initial_energy"],
        scale=scale,
        normalize_flux=normalize_flux,
        downsampling_method=downsampling_method,
        device=device,
        timing=timing,
        resize=resize,
    )


def postprocess_prediction(
    pred: torch.Tensor,
    scale: dict = None,
    timing: Optional[dict] = None,
    upsample: bool = True,
) -> np.ndarray:
    """Up-sample a model prediction to the ROI grid and de-normalize to dose.

    The array-level core shared by the disk saver (:func:`save_prediction`) and the
    streaming pipeline (:mod:`src.beamlets.streaming`). The trilinear up-sampling
    goes back to the **beamlet ROI grid** ``(320, 60, 60)`` (the extraction
    resolution), not the plan grid -- the resampling onto the plan grid happens
    later in accumulation/de-rotation. When ``timing`` is passed, the up-sample
    seconds are accumulated under ``"upsample"``.

    Args:
        pred: Model output ``(1, 1, 160, 30, 30)`` on any device.
        scale: Min-max scaling dict (defaults to :data:`DEFAULT_SCALE`).
        timing: Optional measurement hook.
        upsample: When ``True`` (default, ``grid_factor=1``) the prediction is
            trilinearly up-sampled to the 1mm beamlet ROI grid ``(320, 60, 60)`` as
            before. When ``False`` (``grid_factor=2``) the prediction is kept at
            ``(160, 30, 30)`` -- it will be deposited directly on the 2mm field grid
            and the de-rotation back to the 1mm CT grid does the up-sampling once per
            field -- so the per-beamlet up-sample is skipped.

    Returns:
        The de-normalized dose as a NumPy array: ``(1, 1, 320, 60, 60)`` when
        ``upsample`` (1mm ROI), else ``(1, 1, 160, 30, 30)`` (the 2mm field grid).
    """
    scale = DEFAULT_SCALE if scale is None else scale
    interp_t = perf_counter()
    if upsample:
        pred_upsampled = F.interpolate(
            pred, size=(320, 60, 60), mode="trilinear", align_corners=False
        )
    else:
        pred_upsampled = pred
    if timing is not None:
        # Sync so the async GPU interpolate is fully attributed here, not to the
        # following .cpu() copy (only when measuring; the normal path is untouched).
        if pred_upsampled.is_cuda:
            torch.cuda.synchronize(pred_upsampled.device)
        timing["upsample"] = timing.get("upsample", 0.0) + (perf_counter() - interp_t)
    pred_upsampled_np = pred_upsampled.detach().cpu().numpy()
    return inverse_minmax(pred_upsampled_np, scale["min_ds"], scale["max_ds"])


def postprocess_predictions_batched(
    pred: torch.Tensor,
    scale: dict = None,
    out: Optional["torch.Tensor"] = None,
) -> np.ndarray:
    """De-normalize a whole batch on-device and bring it back in one copy.

    The batched twin of :func:`postprocess_prediction` for the ``grid_factor != 1``
    path (no up-sample), used by :mod:`src.beamlets.streaming`. It replaces the
    per-spot loop of "device-to-host copy, de-normalize on the host, reorder the
    axes with a non-contiguous ``moveaxis`` view" with three batched steps:

    1. permute ``(B, 1, D, H, W)`` to ``(B, H, W, D) = (B, z, y, x)`` and make it
       contiguous **on the device**, which is the layout
       :func:`src.beamlets.accumulation.deposit_crop` reads;
    2. de-normalize on the device (one fused multiply-add over the batch);
    3. one device-to-host copy for the whole batch.

    The result is bit-identical to calling :func:`postprocess_prediction` with
    ``upsample=False`` per sample and applying ``np.moveaxis(np.squeeze(...), 0,
    -1)``: the permute is a pure reindex and the de-normalization is the same
    float32 multiply-add (numpy keeps float32 under NEP 50, so both sides round
    identically). It is faster for two reasons -- one copy instead of ``B``, and a
    contiguous crop for the deposit instead of a strided view.

    Args:
        pred: Model output ``(B, 1, 160, 30, 30)`` on any device.
        scale: Min-max scaling dict (defaults to :data:`DEFAULT_SCALE`).
        out: Optional pre-allocated host tensor of shape ``(>= B, H, W, D)`` to
            receive the batch. Pin it (``pin_memory=True``) to keep the copy off
            the pageable-allocation path, which is what makes the large-batch cost
            scale linearly. **The returned array is then a view into ``out`` and is
            overwritten by the next call**, so consume it before the next batch.

    Returns:
        ``(B, z, y, x)`` contiguous de-normalized dose as a NumPy array.
    """
    scale = DEFAULT_SCALE if scale is None else scale
    # (B, 1, D, H, W) -> (B, H, W, D) = (B, z, y, x), the deposit's layout.
    dose = pred.detach().squeeze(1).permute(0, 2, 3, 1).contiguous()
    span = np.float32(scale["max_ds"] - scale["min_ds"])
    dose = dose * span + np.float32(scale["min_ds"])
    if out is None:
        return dose.cpu().numpy()
    destination = out[: dose.shape[0]]
    destination.copy_(dose, non_blocking=True)
    if dose.is_cuda:
        torch.cuda.synchronize(dose.device)
    return destination.numpy()


def save_prediction(
    pred: torch.Tensor,
    id: str,
    path: str,
    scale: dict = None,
    logging: bool = False,
    timing: Optional[dict] = None,
    upsample: bool = True,
) -> None:
    """Up-sample, de-normalize (via :func:`postprocess_prediction`) and write.

    Writes ``{id}_ds_pred.npy``. When ``timing`` is passed, the disk-write seconds
    are accumulated under ``"write"`` (the up-sample is timed under ``"upsample"``
    inside :func:`postprocess_prediction`). ``upsample=False`` (the ``grid_factor=2``
    path) keeps the prediction on the ``(160, 30, 30)`` model grid -- accumulation
    deposits it on the 2mm field grid and the per-field de-rotation up-samples once.
    """
    scale = DEFAULT_SCALE if scale is None else scale
    pred_upsampled_np = postprocess_prediction(pred, scale, timing, upsample=upsample)
    write_t = perf_counter()
    save_path = os.path.join(path, f"{id}_ds_pred.npy")
    np.save(save_path, pred_upsampled_np)
    if timing is not None:
        timing["write"] = timing.get("write", 0.0) + (perf_counter() - write_t)
