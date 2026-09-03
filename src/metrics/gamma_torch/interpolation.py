# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Uniform-grid description and the fused interpolate-and-reduce step.

This is where the GPU win comes from. The CPU implementation materialises a
``(shell_points, ref_points, ndim)`` float64 coordinate array, interpolates it,
and only then reduces -- an intermediate that reaches ~48 GB at r = 3 mm for a
clinical 3D case, which is why that code slices it into hundreds of RAM chunks.
Here the coordinates are never written out: each ``(shell_tile, ref_tile)`` block
is interpolated and folded straight into a running ``torch.minimum``.

Because both grids are uniform, a voxel lookup is ``floor((p - x0) / dx)`` rather
than a binary search. Uniformity is therefore verified on entry rather than
assumed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "DEFAULT_TILE_ELEMENTS",
    "UniformGrid",
    "as_tensor",
    "min_dose_difference",
    "normalise_axes",
    "to_numpy",
]

# Target element count of the (shell_points x ref_points) intermediate. The
# working set is roughly 110 bytes per element in float64 -- three fractional
# weights, six flat corner offsets, and the interpolation temporaries -- so the
# default asks for about 1 GB of device memory. Lower it on a small GPU; the
# only cost is more kernel launches.
DEFAULT_TILE_ELEMENTS = 1 << 23


def to_numpy(values):
    """Detach a torch tensor to numpy; pass anything else through."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return values


def normalise_axes(axes, label: str) -> Tuple[np.ndarray, ...]:
    """Coerce the axes argument to a tuple of 1-D float64 arrays.

    ``pymedphys.gamma`` accepts a bare 1-D array for the one-dimensional case;
    the same shorthand is accepted here.

    Args:
        axes: A sequence of 1-D coordinate arrays, or one such array in 1D.
        label: Argument name, used in the error message.

    Returns:
        One contiguous 1-D float64 array per axis.

    Raises:
        ValueError: If ``axes`` is a bare scalar.
    """
    values = to_numpy(axes)
    if isinstance(values, np.ndarray) and values.ndim == 1:
        values = (values,)
    if isinstance(values, (int, float)):
        raise ValueError(f"{label} must be a sequence of 1-D coordinate arrays.")
    return tuple(
        np.asarray(to_numpy(axis), dtype=np.float64).ravel() for axis in values
    )


def as_tensor(values, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Move an array or tensor onto ``device`` with ``dtype``.

    Widening happens on the device rather than the host, so a float32 dose does
    not cross the bus at float64 width. For a 100 M-voxel plan that halves the
    upload.
    """
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.from_numpy(np.ascontiguousarray(values)).to(device).to(dtype)


class UniformGrid:
    """A uniformly spaced axis set, described by start / step / size per axis.

    Attributes:
        starts: First coordinate of each axis.
        steps: Spacing of each axis.
        sizes: Number of points along each axis.
    """

    def __init__(self, axes: Sequence[np.ndarray], label: str) -> None:
        """Describe and validate the axes.

        Args:
            axes: One 1-D float64 coordinate array per dimension.
            label: Argument name, used in the error messages.

        Raises:
            ValueError: If an axis has fewer than two points, does not increase
                monotonically, or is not uniformly spaced.
        """
        self.starts: List[float] = []
        self.steps: List[float] = []
        self.sizes: List[int] = []
        for index, values in enumerate(axes):
            if values.size < 2:
                raise ValueError(
                    f"{label}[{index}] needs at least two points; got {values.size}."
                )
            diffs = np.diff(values)
            step = float(diffs[0])
            if step <= 0:
                raise ValueError(
                    f"{label}[{index}] must increase monotonically; its first step "
                    f"is {step}."
                )
            if not np.allclose(diffs, step, rtol=1e-6, atol=1e-6 * abs(step)):
                raise ValueError(
                    f"{label}[{index}] must be uniformly spaced for the torch gamma "
                    f"backend; its steps span [{diffs.min()}, {diffs.max()}]."
                )
            self.starts.append(float(values[0]))
            self.steps.append(step)
            self.sizes.append(int(values.size))

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.sizes)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Grid shape implied by the axes."""
        return tuple(self.sizes)

    @property
    def strides(self) -> List[int]:
        """Row-major strides, so a corner's flat index is a sum of per-axis terms."""
        strides = [1] * len(self.sizes)
        for axis in range(len(self.sizes) - 2, -1, -1):
            strides[axis] = strides[axis + 1] * self.sizes[axis + 1]
        return strides


def _multilinear(dose_flat, lower, upper, weights) -> torch.Tensor:
    """Nested lerp over the ``2**ndim`` corners, innermost axis first.

    Written as ``low * (1 - w) + high * w`` and nested in the CPU
    implementation's order, rather than as a flat sum over the corners, so the
    two agree to within rounding of the same expression instead of differing by
    a reassociation.
    """

    def corner(axis: int, offset: torch.Tensor) -> torch.Tensor:
        if axis == len(lower) - 1:
            low = dose_flat[offset + lower[axis]]
            high = dose_flat[offset + upper[axis]]
        else:
            low = corner(axis + 1, offset + lower[axis])
            high = corner(axis + 1, offset + upper[axis])
        return low.mul_(1.0 - weights[axis]).add_(high.mul_(weights[axis]))

    return corner(0, torch.zeros((), dtype=torch.int64, device=dose_flat.device))


def _interpolate_shell_tile(
    dose_flat: torch.Tensor,
    grid: UniformGrid,
    strides: Sequence[int],
    ref_index_coords: Sequence[torch.Tensor],
    shell_index_offsets: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Multilinear interpolation of the evaluation dose at ref + shell offsets.

    Coordinates are carried in *index space* (units of voxels) rather than in mm:
    the reference coordinate is converted once per tile in float64, so a float32
    working dtype rounds a number of order the grid size rather than one of order
    the patient coordinate.

    Args:
        dose_flat: The evaluation dose, flattened, on the device.
        grid: The evaluation grid description.
        strides: Row-major strides of the evaluation grid.
        ref_index_coords: Per-axis index-space coordinates of this tile's
            reference points, each ``(m,)``.
        shell_index_offsets: Per-axis index-space shell offsets, each ``(s,)``.

    Returns:
        An ``(s, m)`` tensor of interpolated doses, ``+inf`` outside the grid.
    """
    lower: List[torch.Tensor] = []
    upper: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    out_of_bounds: Optional[torch.Tensor] = None

    for axis in range(grid.ndim):
        size = grid.sizes[axis]
        position = ref_index_coords[axis].unsqueeze(0) + shell_index_offsets[
            axis
        ].unsqueeze(1)
        outside = (position < 0) | (position > size - 1)
        out_of_bounds = outside if out_of_bounds is None else out_of_bounds | outside
        del outside

        # floor() alone would index one plane past the end at p == size - 1,
        # where np.searchsorted lands on (size - 2, weight 1); clamping both
        # corner indices reproduces that value.
        index_low = position.floor().clamp_(0, size - 1)
        weights.append(position.sub_(index_low))
        index_low = index_low.to(torch.int64)
        lower.append(index_low * strides[axis])
        upper.append(torch.clamp(index_low + 1, max=size - 1) * strides[axis])
        del index_low

    value = _multilinear(dose_flat, lower, upper, weights)
    return value.masked_fill_(out_of_bounds, float("inf"))


def _unravel_to_index_space(
    flat_indices: torch.Tensor,
    shape: Tuple[int, ...],
    index_axes: Sequence[torch.Tensor],
) -> List[torch.Tensor]:
    """Flat reference indices to per-axis index-space coordinates on the eval grid.

    Decomposing the flat index on the fly avoids materialising the
    ``(ndim, n_voxels)`` meshgrid the CPU path builds -- 2.4 GB for a 100 M-voxel
    plan, before any interpolation happens.
    """
    coords: List[torch.Tensor] = []
    remainder = flat_indices
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    for axis in range(len(shape)):
        block = strides[axis]
        if block == 1:
            axis_index = remainder
        else:
            axis_index = torch.div(remainder, block, rounding_mode="floor")
            remainder = remainder - axis_index * block
        coords.append(index_axes[axis][axis_index])
    return coords


def min_dose_difference(
    dose_eval_flat: torch.Tensor,
    eval_grid: UniformGrid,
    ref_index_axes: Sequence[torch.Tensor],
    ref_shape: Tuple[int, ...],
    flat_dose_reference: torch.Tensor,
    active: torch.Tensor,
    shell: Sequence[np.ndarray],
    local_gamma: bool,
    global_normalisation: float,
    tile_elements: int = DEFAULT_TILE_ELEMENTS,
) -> Tuple[torch.Tensor, int]:
    """Minimum absolute relative dose difference over one shell.

    Args:
        dose_eval_flat: Flattened evaluation dose on the device.
        eval_grid: The evaluation grid description.
        ref_index_axes: Per-axis lookup from a reference index to its index-space
            coordinate on the evaluation grid.
        ref_shape: Shape of the reference dose grid.
        flat_dose_reference: Flattened reference dose on the device.
        active: Flat indices of the reference voxels still searching, ``(M,)``.
        shell: Per-axis shell offsets, in the coordinates' units.
        local_gamma: Divide by the local reference dose rather than by the
            normalisation. A zero reference voxel then yields ``inf`` or ``nan``,
            matching the CPU path rather than guarding it away.
        global_normalisation: The global normalisation dose.
        tile_elements: Target size of the ``(s, m)`` intermediate.

    Returns:
        ``(min_relative_dose_difference, interpolated_samples)``; the first a
        ``(M,)`` tensor aligned with ``active``.
    """
    device, dtype = dose_eval_flat.device, dose_eval_flat.dtype
    n_active = int(active.numel())
    n_shell = int(shell[0].size)
    strides = eval_grid.strides

    shell_offsets = [
        torch.as_tensor(offsets / eval_grid.steps[axis], device=device, dtype=dtype)
        for axis, offsets in enumerate(shell)
    ]

    result = torch.empty(n_active, device=device, dtype=dtype)
    ref_tile = max(1, min(n_active, tile_elements))
    shell_tile = max(1, tile_elements // ref_tile)
    samples = 0

    for start in range(0, n_active, ref_tile):
        indices = active[start : start + ref_tile]
        ref_coords = _unravel_to_index_space(indices, ref_shape, ref_index_axes)
        ref_dose = flat_dose_reference[indices]
        denominator = ref_dose if local_gamma else global_normalisation

        accumulator = torch.full(
            (indices.numel(),), float("inf"), device=device, dtype=dtype
        )
        for shell_start in range(0, n_shell, shell_tile):
            tile = [
                offsets[shell_start : shell_start + shell_tile]
                for offsets in shell_offsets
            ]
            value = _interpolate_shell_tile(
                dose_eval_flat, eval_grid, strides, ref_coords, tile
            )
            samples += value.numel()
            # An out-of-bounds +inf survives as an infinite difference and loses
            # every minimum, which is what the CPU path's extrap_fill_value does.
            value = value.sub_(ref_dose).div_(denominator).abs_()
            torch.minimum(accumulator, value.amin(dim=0), out=accumulator)
            del value
        result[start : start + ref_tile] = accumulator
        del ref_coords, ref_dose, accumulator

    return result, samples
