import torch
import numpy as np

from time import perf_counter
from typing import Optional, Tuple
import os
import json
import torch.nn.functional as F

from src.utils.scallers import inverse_minmax

import logging

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
            f"Downsampled using Average Pooling. New shape: CT: {ct_grid.shape}, Dose: {dose_grid.shape}, Flux: {flux_grid.shape}"
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


def get_single_record_no_gt(
    id: str,
    storage_path: str,
    scale: dict = None,
    normalize_flux: bool = True,
    downsampling_method: str = "interpolation",
    timing: Optional[dict] = None,
    device: Optional["torch.device"] = None,
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


def save_prediction(
    pred: torch.Tensor,
    id: str,
    path: str,
    scale: dict = None,
    logging: bool = False,
    timing: Optional[dict] = None,
) -> None:
    """Up-sample, de-normalize (via :func:`postprocess_prediction`) and write.

    Writes ``{id}_ds_pred.npy``. When ``timing`` is passed, the disk-write seconds
    are accumulated under ``"write"`` (the up-sample is timed under ``"upsample"``
    inside :func:`postprocess_prediction`).
    """
    scale = DEFAULT_SCALE if scale is None else scale
    pred_upsampled_np = postprocess_prediction(pred, scale, timing)
    write_t = perf_counter()
    save_path = os.path.join(path, f"{id}_ds_pred.npy")
    np.save(save_path, pred_upsampled_np)
    if timing is not None:
        timing["write"] = timing.get("write", 0.0) + (perf_counter() - write_t)
