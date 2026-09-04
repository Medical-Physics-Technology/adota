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

"""The gamma convergence loop and the public entry points.

The loop stays on the host: it runs 12-21 iterations for a clinical case, each
one launching the device work in :mod:`~src.metrics.gamma_torch.interpolation`,
so porting it to the device would buy nothing. What it must get exactly right is
the radius schedule -- snapped to ``distance_mm_threshold`` and grown by
``max(r / interp_fraction / max_gamma, delta)`` -- because a different schedule
samples different points and makes parity with the CPU path meaningless.
"""

from __future__ import annotations

import numpy as np
import torch

from .interpolation import (
    DEFAULT_TILE_ELEMENTS,
    UniformGrid,
    as_tensor,
    min_dose_difference,
    normalise_axes,
    to_numpy,
)
from .shells import calculate_coordinates_shell

__all__ = ["gamma_index_torch_core", "gamma_torch"]


def _scalar_threshold(value, name: str) -> float:
    """Unwrap a scalar threshold, rejecting the sequence form pymedphys allows."""
    array = np.asarray(to_numpy(value), dtype=np.float64)
    if array.ndim != 0 and array.size != 1:
        raise NotImplementedError(
            f"{name} must be a scalar for the torch gamma backend. Sequences of "
            "thresholds -- which pymedphys.gamma answers with a dict of gamma "
            "arrays -- are a known gap; call the backend once per threshold."
        )
    return float(array.reshape(()))


def _random_subset_mask(points_to_calc: torch.Tensor, random_subset: int):
    """Keep ``random_subset`` randomly chosen points of the cutoff mask.

    Uses numpy's global RNG and ``np.random.shuffle`` exactly as pymedphys does,
    so seeding numpy reproduces the same subset in both implementations.
    """
    to_calc_index = np.where(points_to_calc.cpu().numpy())[0]
    np.random.shuffle(to_calc_index)
    mask = torch.zeros_like(points_to_calc)
    keep = torch.as_tensor(
        np.sort(to_calc_index[0:random_subset]).astype(np.int64),
        device=points_to_calc.device,
    )
    mask[keep] = True
    return mask


def _reference_index_axes(reference_axes, eval_grid, device, dtype):
    """Per-axis lookup from a reference index to its index-space eval coordinate.

    Built in float64 and cast once, so a float32 working dtype never has to
    represent a patient coordinate -- only a voxel index.
    """
    return [
        (
            (
                torch.as_tensor(axis, device=device, dtype=torch.float64)
                - eval_grid.starts[index]
            )
            / eval_grid.steps[index]
        ).to(dtype)
        for index, axis in enumerate(reference_axes)
    ]


def gamma_index_torch_core(
    axes_reference,
    dose_reference,
    axes_evaluation,
    dose_evaluation,
    dose_percent_threshold,
    distance_mm_threshold,
    *,
    lower_percent_dose_cutoff=20,
    interp_fraction=10,
    max_gamma=None,
    local_gamma=False,
    global_normalisation=None,
    skip_once_passed=False,
    random_subset=None,
    device=None,
    dtype=None,
    tile_elements=DEFAULT_TILE_ELEMENTS,
    stats=None,
) -> torch.Tensor:
    """Compute the gamma index on a torch device.

    The positional arguments and every keyword up to ``random_subset`` mirror
    :func:`pymedphys.gamma`; the rest are torch-specific extras.

    Args:
        axes_reference: Per-axis 1-D uniform coordinate arrays for the reference
            grid (numpy or torch).
        dose_reference: The reference dose. Each point becomes the centre of a
            gamma ellipsoid.
        axes_evaluation: Per-axis coordinates of the evaluation grid.
        dose_evaluation: The evaluation dose -- the grid that is interpolated and
            searched at increasing distances.
        dose_percent_threshold: Percent dose threshold. Scalar only.
        distance_mm_threshold: Distance threshold, in the coordinates' units.
            Scalar only.
        lower_percent_dose_cutoff: Percent of the normalisation below which gamma
            is not calculated. Applied to the reference grid only.
        interp_fraction: The distance threshold is divided into this many steps
            for the search.
        max_gamma: Largest gamma searched for; the search stops once no larger
            radius could improve on it. Defaults to infinity.
        local_gamma: Normalise the dose difference by the local reference dose
            rather than by the global normalisation.
        global_normalisation: Dose the percent inputs are taken from. Defaults to
            ``max(dose_reference)``.
        skip_once_passed: Stop searching a voxel as soon as its gamma is below 1.
        random_subset: Evaluate only this many randomly chosen reference points.
        device: Torch device. Defaults to CUDA when available, else CPU.
        dtype: ``torch.float32`` (default) or ``torch.float64``.
        tile_elements: Target element count of the ``(shell, reference)``
            intermediate. Trades device memory for kernel-launch overhead.
        stats: Optional dict, filled in with ``iterations``, ``shell_points`` and
            ``interp_samples`` for benchmarking.

    Returns:
        A tensor of gamma values on ``device``, shaped like ``dose_reference``.
        NaN marks voxels that were not evaluated -- below the cutoff, outside the
        random subset, or with no in-bounds evaluation point.

    Raises:
        NotImplementedError: If either threshold is a sequence.
        ValueError: If an axis is not uniform, or the shapes disagree.
    """
    dose_percent_threshold = _scalar_threshold(
        dose_percent_threshold, "dose_percent_threshold"
    )
    distance_mm_threshold = _scalar_threshold(
        distance_mm_threshold, "distance_mm_threshold"
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    dtype = dtype or torch.float32

    reference_axes = normalise_axes(axes_reference, "axes_reference")
    evaluation_axes = normalise_axes(axes_evaluation, "axes_evaluation")
    ref_grid = UniformGrid(reference_axes, "axes_reference")
    eval_grid = UniformGrid(evaluation_axes, "axes_evaluation")

    dose_ref = as_tensor(dose_reference, device, dtype)
    dose_eval = as_tensor(dose_evaluation, device, dtype)
    if tuple(dose_ref.shape) != ref_grid.shape:
        raise ValueError(
            f"Length of items in axes_reference ({ref_grid.shape}) does not match "
            f"the shape of dose_reference ({tuple(dose_ref.shape)})"
        )
    if tuple(dose_eval.shape) != eval_grid.shape:
        raise ValueError(
            "Length of items in axes_evaluation does not match the shape of "
            "dose_evaluation"
        )
    if ref_grid.ndim != eval_grid.ndim:
        raise ValueError("The dimensions of the input data do not match")

    max_gamma = float("inf") if max_gamma is None else float(max_gamma)
    if global_normalisation is None:
        global_normalisation = float(dose_ref.max())
    global_normalisation = float(global_normalisation)
    lower_dose_cutoff = lower_percent_dose_cutoff / 100 * global_normalisation
    maximum_test_distance = distance_mm_threshold * max_gamma

    flat_dose_reference = dose_ref.reshape(-1)
    dose_eval_flat = dose_eval.reshape(-1)
    points_to_calc = flat_dose_reference >= lower_dose_cutoff
    if random_subset is not None:
        points_to_calc = _random_subset_mask(points_to_calc, int(random_subset))
    ref_index_axes = _reference_index_axes(reference_axes, eval_grid, device, dtype)

    # The loop's bookkeeping runs over the candidate voxels only, not the whole
    # volume. The CPU implementation masks a full-length array every iteration;
    # on a 100 M-voxel plan where a few million voxels clear the cutoff, that
    # scan costs more than the interpolation it is scheduling. Compacting once
    # up front is arithmetically identical -- voxels outside `points_to_calc`
    # are never checked either way.
    candidate_indices = torch.nonzero(points_to_calc, as_tuple=False).reshape(-1)
    candidate_gamma = torch.full(
        (candidate_indices.numel(),), float("inf"), device=device, dtype=dtype
    )
    still_searching = torch.ones_like(candidate_indices, dtype=torch.bool)

    distance = 0.0
    distance_step_size = distance_mm_threshold / interp_fraction
    force_search_distances = [distance_mm_threshold]
    counters = {"iterations": 0, "shell_points": 0, "interp_samples": 0}

    while distance <= maximum_test_distance:
        active = candidate_indices[still_searching]
        shell = calculate_coordinates_shell(
            distance, ref_grid.ndim, distance_step_size
        )
        min_relative_dose_difference, samples = min_dose_difference(
            dose_eval_flat,
            eval_grid,
            ref_index_axes,
            ref_grid.shape,
            flat_dose_reference,
            active,
            shell,
            local_gamma,
            global_normalisation,
            tile_elements,
        )
        counters["iterations"] += 1
        counters["shell_points"] += int(shell[0].size)
        counters["interp_samples"] += samples

        gamma_at_distance = torch.sqrt(
            (min_relative_dose_difference / (dose_percent_threshold / 100)) ** 2
            + (distance / distance_mm_threshold) ** 2
        )
        candidate_gamma[still_searching] = torch.minimum(
            gamma_at_distance, candidate_gamma[still_searching]
        )
        del gamma_at_distance, min_relative_dose_difference, active

        # No larger radius can improve a voxel whose gamma is already at or below
        # r / distance_mm_threshold, so it drops out of the search here.
        still_searching = candidate_gamma > (distance / distance_mm_threshold)
        if skip_once_passed:
            still_searching &= candidate_gamma >= 1
        if not bool(still_searching.any()):
            break

        distance_step_size = max(
            distance / interp_fraction / max_gamma,
            distance_mm_threshold / interp_fraction,
        )
        distance += distance_step_size
        if force_search_distances and distance >= force_search_distances[0]:
            distance = force_search_distances.pop(0)

    if stats is not None:
        stats.update(counters)

    # Scatter the candidates back into the full grid. Everything else stays
    # +inf, which becomes the NaN that marks "not evaluated". Done in place:
    # the `where` form would allocate four more full-volume copies, 2 GB on a
    # 100 M-voxel plan, for no benefit.
    del dose_eval, dose_eval_flat
    gamma = torch.full_like(flat_dose_reference, float("inf"))
    gamma[candidate_indices] = candidate_gamma
    gamma = gamma.reshape(ref_grid.shape)
    gamma.masked_fill_(torch.isinf(gamma), float("nan"))
    if max_gamma != float("inf"):
        # clamp_ leaves NaN alone, exactly as `gamma[gamma > max_gamma] = ...` does.
        gamma.clamp_(max=max_gamma)
    return gamma


def gamma_torch(*args, **kwargs) -> np.ndarray:
    """Numpy in, numpy out convenience wrapper over :func:`gamma_index_torch_core`.

    Takes and returns the same things ``pymedphys.gamma`` does. Prefer the core
    when the doses are already device-resident: this wrapper adds a round trip
    that exists only to hand back an :class:`numpy.ndarray`.
    """
    return gamma_index_torch_core(*args, **kwargs).detach().cpu().numpy()
