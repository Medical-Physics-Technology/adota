"""3-D rotation utilities for arrays stored as (D, H, W)."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


@dataclass
class VolumeRotation:
    rotated: Optional[np.ndarray]
    pure_seconds_all: list[float] = field(default_factory=list)
    practical_seconds_all: list[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def pure_median(self) -> float:
        return float(np.median(self.pure_seconds_all)) if self.pure_seconds_all else float("nan")

    @property
    def practical_median(self) -> float:
        return float(np.median(self.practical_seconds_all)) if self.practical_seconds_all else float("nan")


def center_pivot_dhw(shape_dhw: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return the voxel-space center pivot as (d, h, w)."""
    depth, height, width = shape_dhw
    return ((depth - 1) / 2.0, (height - 1) / 2.0, (width - 1) / 2.0)


def build_lateral_axis_rotation_matrix_3d(
    axis: str,
    angle_deg: float,
    pivot_dhw: tuple[float, float, float],
) -> np.ndarray:
    """Build a 4x4 output-to-input affine matrix for a lateral-axis rotation.

    Arrays are shaped (D, H, W). Supported axes are:
    - "h"/"y": rotate around the H axis, mixing D and W coordinates.
    - "w"/"x": rotate around the W axis, mixing D and H coordinates.

    The returned matrix maps output voxel coordinates to input coordinates and
    is suitable for scipy.ndimage.affine_transform and torch grid_sample.
    """
    axis_normalized = axis.lower()
    pivot = np.asarray([*pivot_dhw, 1.0], dtype=np.float64)
    theta = np.deg2rad(float(angle_deg))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))

    matrix = np.eye(4, dtype=np.float64)
    if axis_normalized in {"h", "y"}:
        matrix[:3, :3] = np.array(
            [
                [cos_theta, 0.0, -sin_theta],
                [0.0, 1.0, 0.0],
                [sin_theta, 0.0, cos_theta],
            ],
            dtype=np.float64,
        )
    elif axis_normalized in {"w", "x"}:
        matrix[:3, :3] = np.array(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported rotation axis {axis!r}; expected 'h'/'y' or 'w'/'x'.")

    matrix[:3, 3] = pivot[:3] - matrix[:3, :3] @ pivot[:3]
    return matrix


def compose_affine_matrices(*matrices: np.ndarray) -> np.ndarray:
    """Compose output-to-input affine matrices in application order."""
    composed = np.eye(4, dtype=np.float64)
    for matrix in matrices:
        composed = np.asarray(matrix, dtype=np.float64) @ composed
    return composed


def rotate_volume_scipy(
    volume_dhw: np.ndarray,
    matrix4: np.ndarray,
    repeats: int = 1,
) -> VolumeRotation:
    """Rotate one (D, H, W) volume with SciPy and return timing details."""
    matrix3 = matrix4[:3, :3]
    offset = matrix4[:3, 3]
    volume_f = volume_dhw.astype(np.float32, copy=False)
    cval = float(volume_f.min())

    _ = ndimage.affine_transform(
        volume_f, matrix=matrix3, offset=offset, order=1, mode="constant", cval=cval
    )

    pure_times: list[float] = []
    rotated: Optional[np.ndarray] = None
    for _ in range(repeats):
        start = perf_counter()
        rotated = ndimage.affine_transform(
            volume_f,
            matrix=matrix3,
            offset=offset,
            order=1,
            mode="constant",
            cval=cval,
        )
        pure_times.append(perf_counter() - start)

    return VolumeRotation(
        rotated=np.asarray(rotated),
        pure_seconds_all=pure_times,
        practical_seconds_all=list(pure_times),
    )


def _torch_norm_theta(matrix4_dhw: np.ndarray, shape_dhw: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = shape_dhw
    reorder_dhw_to_xyz = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    matrix_xyz = reorder_dhw_to_xyz @ matrix4_dhw @ reorder_dhw_to_xyz
    norm_to_pix = np.array(
        [
            [(width - 1) / 2.0, 0.0, 0.0, (width - 1) / 2.0],
            [0.0, (height - 1) / 2.0, 0.0, (height - 1) / 2.0],
            [0.0, 0.0, (depth - 1) / 2.0, (depth - 1) / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pix_to_norm = np.array(
        [
            [2.0 / (width - 1), 0.0, 0.0, -1.0],
            [0.0, 2.0 / (height - 1), 0.0, -1.0],
            [0.0, 0.0, 2.0 / (depth - 1), -1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix_norm = pix_to_norm @ matrix_xyz @ norm_to_pix
    return torch.from_numpy(matrix_norm[:3, :]).float()


def _make_torch_grid(theta: torch.Tensor, size: torch.Size) -> torch.Tensor:
    n_batches, _, depth, height, width = size
    z = torch.linspace(-1.0, 1.0, depth, device=theta.device, dtype=theta.dtype)
    y = torch.linspace(-1.0, 1.0, height, device=theta.device, dtype=theta.dtype)
    x = torch.linspace(-1.0, 1.0, width, device=theta.device, dtype=theta.dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    xx = xx.expand(n_batches, -1, -1, -1)
    yy = yy.expand(n_batches, -1, -1, -1)
    zz = zz.expand(n_batches, -1, -1, -1)
    coeff = theta.view(n_batches, 3, 4, 1, 1, 1)
    grid_x = coeff[:, 0, 0] * xx + coeff[:, 0, 1] * yy + coeff[:, 0, 2] * zz + coeff[:, 0, 3]
    grid_y = coeff[:, 1, 0] * xx + coeff[:, 1, 1] * yy + coeff[:, 1, 2] * zz + coeff[:, 1, 3]
    grid_z = coeff[:, 2, 0] * xx + coeff[:, 2, 1] * yy + coeff[:, 2, 2] * zz + coeff[:, 2, 3]
    return torch.stack((grid_x, grid_y, grid_z), dim=-1)


def rotate_volume_torch(
    volume_dhw: np.ndarray,
    matrix4: np.ndarray,
    repeats: int = 1,
    device: str | torch.device = "cpu",
) -> VolumeRotation:
    """Rotate one (D, H, W) volume with torch grid_sample and return timings."""
    torch_device = torch.device(device)
    volume_f = volume_dhw.astype(np.float32, copy=False)
    cval = float(volume_f.min())
    volume_t = torch.from_numpy(volume_f).to(torch_device).unsqueeze(0).unsqueeze(0)
    theta = _torch_norm_theta(matrix4, volume_f.shape).to(torch_device).unsqueeze(0)

    def rotate_tensor(input_volume: torch.Tensor, input_theta: torch.Tensor) -> torch.Tensor:
        grid = _make_torch_grid(input_theta, input_volume.shape)
        shifted = input_volume - cval
        rotated = F.grid_sample(
            shifted, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        outside = torch.any((grid < -1.0) | (grid > 1.0), dim=-1, keepdim=True)
        return rotated.masked_fill(outside.permute(0, 4, 1, 2, 3), 0.0) + cval

    _ = rotate_tensor(volume_t, theta)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)

    pure_times: list[float] = []
    practical_times: list[float] = []
    rotated_np: Optional[np.ndarray] = None
    for _ in range(repeats):
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        start = perf_counter()
        rotated_t = rotate_tensor(volume_t, theta)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        pure_times.append(perf_counter() - start)

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        start = perf_counter()
        volume_t_p = torch.from_numpy(volume_f).to(torch_device).unsqueeze(0).unsqueeze(0)
        theta_p = _torch_norm_theta(matrix4, volume_f.shape).to(torch_device).unsqueeze(0)
        rotated_p = rotate_tensor(volume_t_p, theta_p)
        rotated_np = rotated_p.squeeze(0).squeeze(0).detach().cpu().numpy().copy()
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        practical_times.append(perf_counter() - start)

    return VolumeRotation(
        rotated=rotated_np,
        pure_seconds_all=pure_times,
        practical_seconds_all=practical_times,
    )


def rotate_lateral_axes_sequential(
    volume_dhw: np.ndarray,
    angle_y_deg: float,
    angle_x_deg: float,
    repeats: int = 1,
    backend: str = "scipy",
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, VolumeRotation, VolumeRotation]:
    """Rotate around Y/H first, then X/W, returning final volume and timings."""
    pivot = center_pivot_dhw(volume_dhw.shape)
    matrix_y = build_lateral_axis_rotation_matrix_3d("y", angle_y_deg, pivot)
    matrix_x = build_lateral_axis_rotation_matrix_3d("x", angle_x_deg, pivot)

    if backend == "scipy":
        first = rotate_volume_scipy(volume_dhw, matrix_y, repeats=repeats)
        if first.rotated is None:
            raise RuntimeError("SciPy Y-axis rotation did not produce an output volume.")
        second = rotate_volume_scipy(first.rotated, matrix_x, repeats=repeats)
    elif backend == "torch":
        first = rotate_volume_torch(volume_dhw, matrix_y, repeats=repeats, device=device)
        if first.rotated is None:
            raise RuntimeError("Torch Y-axis rotation did not produce an output volume.")
        second = rotate_volume_torch(first.rotated, matrix_x, repeats=repeats, device=device)
    else:
        raise ValueError(f"Unsupported rotation backend {backend!r}; expected 'scipy' or 'torch'.")

    if second.rotated is None:
        raise RuntimeError(f"{backend} X-axis rotation did not produce an output volume.")
    return second.rotated, first, second
