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


def get_single_record_no_gt(
    id: str,
    storage_path: str,
    scale: dict = None,
    normalize_flux: bool = True,
    downsampling_method: str = "interpolation",
    timing: Optional[dict] = None,
    device: Optional["torch.device"] = None,
) -> Tuple[torch.Tensor]:
    """Load and prepare a single inference record.

    ``timing`` is an optional measurement-only hook: when a dict is passed, the
    seconds spent reading the per-spot files are accumulated under ``"read"`` and
    the seconds spent down-sampling to the ADoTA grid under ``"downsample"``. It
    does not affect the returned tensors (the training/eval path passes no dict).

    ``device`` (default ``None`` = CPU, unchanged) moves the CT/flux tensors onto
    the given Torch device right after loading, so the normalization and the
    trilinear down-sampling run there (e.g. on the GPU). The returned tensors live
    on that device.
    """
    scale = DEFAULT_SCALE if scale is None else scale

    read_t = perf_counter()
    x = np.load(os.path.join(storage_path, f"{id}_ct.npy"))
    flux = np.load(os.path.join(storage_path, f"{id}_flux.npy"))
    # y = np.load(os.path.join(storage_path, f"{id}_ds.npy"))
    with open(os.path.join(storage_path, f"{id}_sim_res.json"), "r") as f:
        meta = json.load(f)
    if timing is not None:
        timing["read"] = timing.get("read", 0.0) + (perf_counter() - read_t)

    energy = meta["simulation_log"]["energy"][0]
    # Convert numpy arrays to PyTorch tensors (on ``device`` when given, so the
    # normalization + trilinear resize below run there).
    ct_grid = torch.tensor(x, dtype=torch.float32, device=device)
    # dose_grid = torch.tensor(y, dtype=torch.float32)
    flux_grid = torch.tensor(flux, dtype=torch.float32, device=device)
    e = energy

    ct_grid = (ct_grid - scale["min_ct"]) / (scale["max_ct"] - scale["min_ct"])
    # dose_grid = (dose_grid - scale["min_ds"]) / (
    #     scale["max_ds"] - scale["min_ds"]
    # )
    e = (meta["initial_energy"] - scale["min_energy"]) / (
        scale["max_energy"] - scale["min_energy"]
    )

    # Permute dimensions to (D, H, W)
    ct_grid = ct_grid.permute(2, 0, 1)
    # dose_grid = dose_grid.permute(2, 0, 1)
    flux_grid = flux_grid.permute(2, 0, 1)
    # Apply channel dimension
    ct_grid = ct_grid.unsqueeze(0)
    # dose_grid = dose_grid.unsqueeze(0)
    flux_grid = flux_grid.unsqueeze(0)

    if normalize_flux:
        flux_grid = (flux_grid - flux_grid.min()) / (flux_grid.max() - flux_grid.min())

    interp_t = perf_counter()
    # Perform Avarage Pooling on dimension physical dimensions
    if downsampling_method == "avg_pooling":
        ct_grid = F.avg_pool3d(ct_grid, kernel_size=2, stride=2)
        # dose_grid = F.avg_pool3d(dose_grid, kernel_size=2, stride=2)
        flux_grid = F.avg_pool3d(flux_grid, kernel_size=2, stride=2)

    if downsampling_method == "interpolation":
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

    return x, initial_energy  # , # dose_grid


def save_prediction(
    pred: torch.Tensor,
    id: str,
    path: str,
    scale: dict = None,
    logging: bool = False,
    timing: Optional[dict] = None,
) -> None:
    """Upsample, de-normalize and write a single prediction.

    The trilinear upsampling here goes back to the **beamlet ROI grid**
    ``(320, 60, 60)`` (the extraction resolution), not the plan grid -- the
    resampling onto the plan grid happens later in the accumulation stage. When a
    ``timing`` dict is passed (measurement-only), the seconds spent on that
    up-sampling are accumulated under ``"upsample"`` and the seconds spent
    de-normalizing and writing the ``.npy`` under ``"write"``.
    """
    scale = DEFAULT_SCALE if scale is None else scale
    # print(pred.min(), pred.max())
    interp_t = perf_counter()
    pred_upsampled = F.interpolate(
        pred, size=(320, 60, 60), mode="trilinear", align_corners=False
    )
    if timing is not None:
        # Sync so the async GPU interpolate is fully attributed here, not to the
        # following .cpu() copy (only when measuring; the normal path is untouched).
        if pred_upsampled.is_cuda:
            torch.cuda.synchronize(pred_upsampled.device)
        timing["upsample"] = timing.get("upsample", 0.0) + (perf_counter() - interp_t)
    # print(pred_upsampled.min(), pred_upsampled.max())
    write_t = perf_counter()
    pred_upsampled_np = pred_upsampled.detach().cpu().numpy()
    pred_upsampled_np = inverse_minmax(
        pred_upsampled_np, scale["min_ds"], scale["max_ds"]
    )
    # print(pred_upsampled_np.min(), pred_upsampled_np.max())
    save_path = os.path.join(path, f"{id}_ds_pred.npy")
    np.save(save_path, pred_upsampled_np)
    if timing is not None:
        timing["write"] = timing.get("write", 0.0) + (perf_counter() - write_t)
    # print(f"Prediction saved to: {save_path}")
