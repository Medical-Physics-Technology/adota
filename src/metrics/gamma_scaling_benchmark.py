# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Runtime against problem size, with everything else held fixed.

The plan-scale and beamlet-scale benchmarks differ in more than grid size:
different dose content, voxel spacing, search resolution and timing protocol.
This module isolates size. It cuts a series of nested sub-volumes out of one
plan dose pair, all centred on the same point in the high-dose region, and
times the same backends on each with an identical gamma configuration.

Two settings are pinned explicitly rather than derived from the crop, because
otherwise they would drift with it: the global normalisation dose is the full
plan's reference maximum, so the dose tolerance and the cutoff dose are the
same at every size, and the timing protocol is the beamlet protocol (one
warm-up, then repeated calls, every repetition kept).

Crop content still affects convergence: a crop around the high-dose region has
a larger fraction of voxels above the cutoff than the whole plan does. The
nested design reduces, but does not remove, that dependence, which is why the
evaluated reference-point count is recorded and reported beside the voxel
count.
"""

from __future__ import annotations

import hashlib
import logging
import resource
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.metrics.gamma_beamlet_specs import GammaCase, RungSpec
from src.metrics.gamma_pass_rate import gamma_index

logger = logging.getLogger(__name__)

__all__ = ["Crop", "nested_crops", "time_crop"]


@dataclass
class Crop:
    """One sub-volume of a plan dose pair.

    Attributes:
        target_voxels: The size that was asked for.
        bounds_zyx: ``((z0, z1), (y0, y1), (x0, x1))`` half-open index bounds.
        reference: Reference dose crop, in the plan's dose units.
        evaluation: Evaluation dose crop.
    """

    target_voxels: int
    bounds_zyx: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
    reference: np.ndarray
    evaluation: np.ndarray

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.reference.shape)

    @property
    def voxels(self) -> int:
        return int(np.prod(self.shape))

    def describe(self, cutoff_dose: float) -> Dict[str, Any]:
        """Everything about the crop the record needs, hashes included."""
        return {
            "target_voxels": self.target_voxels,
            "bounds_zyx": [list(b) for b in self.bounds_zyx],
            "shape_zyx": list(self.shape),
            "voxels": self.voxels,
            "reference_max": float(self.reference.max()),
            "n_above_cutoff": int(np.count_nonzero(self.reference >= cutoff_dose)),
            "reference_sha256": hashlib.sha256(np.ascontiguousarray(self.reference).tobytes()).hexdigest(),
            "evaluation_sha256": hashlib.sha256(np.ascontiguousarray(self.evaluation).tobytes()).hexdigest(),
        }


def high_dose_centre(reference: np.ndarray, fraction: float = 0.5) -> Tuple[int, ...]:
    """Centroid of the voxels at or above ``fraction`` of the maximum, as indices.

    The argmax alone can sit on an isolated hot voxel at the grid edge, which
    would centre every crop on empty space; the centroid of the high-dose
    region is where the clinically relevant dose is.
    """
    mask = reference >= fraction * reference.max()
    coordinates = np.argwhere(mask)
    return tuple(int(round(v)) for v in coordinates.mean(axis=0))


def nested_crops(
    reference: np.ndarray,
    evaluation: np.ndarray,
    targets: Sequence[int],
    centre: Optional[Tuple[int, ...]] = None,
) -> List[Crop]:
    """Nested sub-volumes of increasing size around one centre.

    Each crop's edge lengths are the full grid's scaled by the cube root of the
    size ratio, so the aspect ratio is preserved, then clamped to the grid.
    Nesting is enforced: a larger crop always contains the smaller ones, so
    the series differs only by what is added at the outside.

    Args:
        reference: Full reference dose ``(z, y, x)``.
        evaluation: Full evaluation dose, same shape.
        targets: Voxel counts to aim for, ascending; the full grid size is
            included automatically as the last crop.
        centre: Index centre; defaults to :func:`high_dose_centre`.

    Returns:
        The crops, smallest first.
    """
    full_shape = np.array(reference.shape, dtype=int)
    total = int(np.prod(full_shape))
    if centre is None:
        centre = high_dose_centre(reference)
    centre_array = np.array(centre, dtype=int)

    crops: List[Crop] = []
    previous: Optional[np.ndarray] = None
    for target in sorted(set(int(t) for t in targets) | {total}):
        factor = min(1.0, (target / total) ** (1.0 / 3.0))
        shape = np.maximum(2, np.round(full_shape * factor).astype(int))
        shape = np.minimum(shape, full_shape)
        start = np.clip(centre_array - shape // 2, 0, full_shape - shape)
        stop = start + shape
        if previous is not None:
            start = np.minimum(start, previous[0])
            stop = np.maximum(stop, previous[1])
        previous = (start, stop)
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, stop))
        crops.append(
            Crop(
                target_voxels=target,
                bounds_zyx=tuple((int(a), int(b)) for a, b in zip(start, stop)),
                reference=np.ascontiguousarray(reference[slices]),
                evaluation=np.ascontiguousarray(evaluation[slices]),
            )
        )
    return crops


def time_crop(
    crop: Crop,
    spacing_zyx: Sequence[float],
    case: GammaCase,
    rung: RungSpec,
    global_normalisation: float,
    *,
    device: Optional[str],
    repeats: int,
) -> Dict[str, Any]:
    """Warm up, then time ``repeats`` calls of the array entry point on a crop.

    The timed region is the public ``gamma_index`` call: the backend's gamma
    evaluation including host-device transfers, and the pass-rate reduction.
    The input arrays are not copied per call; with a zero pre-cutoff the entry
    point's in-place masking is a no-op on non-negative dose, which is asserted.
    """
    if not (crop.reference >= 0).all() or not (crop.evaluation >= 0).all():
        raise ValueError("crop doses must be non-negative for the copy-free timed region")
    params = {**case.as_params(), "global_normalisation": float(global_normalisation)}
    resolution = tuple(float(s) for s in spacing_zyx)
    scale = {"y_min": 0.0, "y_max": float(global_normalisation)}
    target = rung.resolve_device(device)
    stats: Dict[str, Any] = {}
    options = rung.backend_options(device)
    if options is not None:
        options = {**options, "stats": stats}
    on_cuda = bool(target and str(target).startswith("cuda") and rung.backend == "torch")

    def call():
        return gamma_index(
            ground_truth=crop.reference,
            prediction=crop.evaluation,
            scale=scale,
            gamma_params=params,
            resolution=resolution,
            cutoff=0,
            backend=rung.backend,
            backend_options=options,
        )

    def sync():
        if on_cuda:
            torch.cuda.synchronize(torch.device(target))

    gamma_map, pass_rate = call()
    sync()
    n_evaluated = int(np.count_nonzero(gamma_map > 0))
    del gamma_map
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(torch.device(target))

    times: List[float] = []
    for _ in range(max(1, repeats)):
        sync()
        start = perf_counter()
        _, pass_rate = call()
        sync()
        times.append(perf_counter() - start)
    ordered = sorted(times)
    quartiles = np.percentile(ordered, [25, 50, 75]) if len(ordered) > 1 else [ordered[0]] * 3
    samples = stats.get("interp_samples")
    return {
        "rung": rung.rung,
        "label": rung.label,
        "device": target if rung.backend == "torch" else "cpu",
        "dtype": rung.dtype,
        "voxels": crop.voxels,
        "shape_zyx": list(crop.shape),
        "n_evaluated": n_evaluated,
        "pass_rate_pct": float(100.0 * pass_rate[0]),
        "seconds_all": [float(t) for t in times],
        "seconds_median": float(quartiles[1]),
        "seconds_q1": float(quartiles[0]),
        "seconds_q3": float(quartiles[2]),
        "seconds_best": float(ordered[0]),
        "repeats": len(times),
        "iterations": stats.get("iterations"),
        "shell_points": stats.get("shell_points"),
        "interp_samples": samples,
        "samples_per_s": float(samples / quartiles[1]) if samples else None,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(torch.device(target))) if on_cuda else None,
        "host_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "config": {
            **params,
            "spacing_mm": list(resolution),
            "device": target if rung.backend == "torch" else "cpu",
            "dtype": rung.dtype if rung.backend == "torch" else "float64",
        },
    }
