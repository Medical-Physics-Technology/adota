# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Beamlet-scale counterpart to :mod:`src.metrics.gamma_benchmark`.

The plan-level ladder in :mod:`src.metrics.gamma_benchmark` measures the gamma
backends on 70-100 million-voxel patient grids. A beamlet is three orders of
magnitude smaller -- 160 x 30 x 30 = 144,000 voxels at 2 mm -- and it is the size
that matters during training, where the gamma pass rate is wanted every few
epochs over a pool of validation records rather than once per plan. Nothing
about the plan-scale result carries over: at 144 k voxels the fixed costs
(kernel launches, host-device transfers, the Python-side convergence loop) are a
large fraction of the total, so the speed-ups have to be measured, not scaled.

The same four rungs are used, so the two studies read on one axis:

===== ================================================== ==================
Rung   What it is                                         Isolates
===== ================================================== ==================
1      ``pymedphys.gamma`` on the host                     the baseline
2      ``gamma_torch`` on torch-CPU, float64               the implementation
3      ``gamma_torch`` on one GPU, float64                 the device
4      ``gamma_torch`` on one GPU, float32                 precision
===== ================================================== ==================

Rung 0 (a historical recording) has no beamlet-scale analogue, so it is absent.

Two timings are reported per case, because they answer different questions:

``array``
    numpy in, numpy out, through :func:`src.metrics.gamma_pass_rate.gamma_index`.
    This is the offline analysis path, and the one comparable with the
    plan-level table.
``tensor``
    torch in, through :func:`src.metrics.gamma_pass_rate.gamma_index_torch` with
    both volumes already resident on the GPU. This is the training path: under
    the pymedphys backend it pays two full host transfers that the torch backend
    does not, and that difference is part of what is being measured.

The dose pairs come from :mod:`src.metrics.gamma_beamlet_pairs` and the rung
and criterion definitions from :mod:`src.metrics.gamma_beamlet_specs`, both
split off by role: one builds the data, one names what is measured, this one
does the measuring. Their public names are re-exported here so a caller has a
single import site.

Like :mod:`src.metrics.gamma_benchmark` this module takes no YAML config. Its
only inputs are a cached pair file, a device and the criteria.
"""

from __future__ import annotations

import logging
import platform
import resource
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.metrics.gamma_beamlet_pairs import (
    BeamletPair,
    beamlet_resolution_mm,
    build_pairs,
    load_pairs,
    save_pairs,
)
from src.metrics.gamma_beamlet_specs import (
    RUNGS,
    GammaCase,
    RungSpec,
    criterion_label,
    default_cases,
)
from src.metrics.gamma_pass_rate import gamma_index, gamma_index_torch

logger = logging.getLogger(__name__)

__all__ = [
    "BeamletPair",
    "RungSpec",
    "RUNGS",
    "GammaCase",
    "beamlet_resolution_mm",
    "build_pairs",
    "save_pairs",
    "load_pairs",
    "criterion_label",
    "default_cases",
    "environment_stamp",
    "PreparedCall",
    "build_gamma_call",
    "time_case",
    "sweep",
]


def environment_stamp() -> Dict[str, Any]:
    """Machine, library versions and GPU name, recorded beside every result."""
    import pymedphys

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pymedphys": pymedphys.__version__,
        "numpy": np.__version__,
        "gpu": gpu,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ── Timing ──────────────────────────────────────────────────────────────────


def _synchronise(rung: RungSpec, device: Optional[str]) -> None:
    """Wait for the device queue, so a timing is not just a launch time."""
    target = device or rung.device
    if rung.backend == "torch" and target and target.startswith("cuda"):
        torch.cuda.synchronize(torch.device(target))


def _denormalised(pair: BeamletPair, scale: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Physical-unit copies of the pair, fresh each call.

    ``gamma_index`` zeroes its inputs below the cutoff in place, so it must never
    be handed the cached arrays.
    """
    from src.utils.scallers import inverse_minmax

    return (
        inverse_minmax(pair.reference.astype(np.float64), scale["min_ds"], scale["max_ds"]),
        inverse_minmax(pair.evaluation.astype(np.float64), scale["min_ds"], scale["max_ds"]),
    )


def _gamma_scale(scale: Dict[str, float]) -> Dict[str, float]:
    """The ``y_min`` / ``y_max`` view of the scale dict the gamma helpers expect."""
    merged = dict(scale)
    merged.setdefault("y_min", scale["min_ds"])
    merged.setdefault("y_max", scale["max_ds"])
    return merged


def _resolved_config(
    pair: BeamletPair, case: GammaCase, rung: RungSpec, target: Optional[str], scale: Dict[str, float]
) -> Dict[str, Any]:
    """Every setting that decides the result, spelled out rather than implied.

    The normalisation dose is the physical maximum of the reference volume,
    which is what both backends use under global normalisation.
    """
    from src.metrics.gamma_torch import DEFAULT_TILE_ELEMENTS

    span = scale["max_ds"] - scale["min_ds"]
    return {
        "dose_percent_threshold": case.dose_percent_threshold,
        "distance_mm_threshold": case.distance_mm_threshold,
        "lower_percent_dose_cutoff": case.lower_percent_dose_cutoff,
        "local_gamma": False,
        "global_normalisation_dose": float(pair.reference.max()) * span + scale["min_ds"],
        "interp_fraction": case.interp_fraction,
        "max_gamma": case.max_gamma,
        "skip_once_passed": False,
        "random_subset": None,
        "spacing_mm": list(beamlet_resolution_mm()),
        "grid_shape": list(pair.shape),
        "dtype": rung.dtype if rung.backend == "torch" else "float64",
        "device": target if rung.backend == "torch" else "cpu",
        "tile_elements": DEFAULT_TILE_ELEMENTS if rung.backend == "torch" else None,
    }


@dataclass
class PreparedCall:
    """A zero-argument gamma evaluation plus what is needed to time it.

    Attributes:
        call: Runs one evaluation and returns ``(gamma_values, pass_rate)``.
        target: The device the torch backend runs on, or ``None`` for pymedphys.
        on_cuda: Whether ``target`` is a CUDA device, so the caller synchronises.
        stats: Filled by the torch backend with its search counters per call.
    """

    call: Any
    target: Optional[str]
    on_cuda: bool
    stats: Dict[str, Any]


def build_gamma_call(
    pair: BeamletPair,
    case: GammaCase,
    rung: RungSpec,
    scale: Dict[str, float],
    *,
    device: Optional[str] = None,
    path: str = "array",
) -> PreparedCall:
    """Prepare one evaluation of ``pair`` through the public entry point.

    The array path de-normalises the cached pair inside the call, as the
    analysis scripts do; the tensor path uploads the normalised volumes once,
    outside the call, and clones them per call so a mutating backend cannot
    contaminate the next repetition.

    Raises:
        ValueError: If ``path`` is neither ``"array"`` nor ``"tensor"``.
    """
    if path not in ("array", "tensor"):
        raise ValueError(f"path must be 'array' or 'tensor'; got {path!r}")
    params = case.as_params()
    resolution = beamlet_resolution_mm()
    gamma_scale = _gamma_scale(scale)
    target = rung.resolve_device(device)
    stats: Dict[str, Any] = {}
    options = rung.backend_options(device)
    if options is not None:
        options = {**options, "stats": stats}
    on_cuda = bool(target and str(target).startswith("cuda") and rung.backend == "torch")

    if path == "tensor":
        # The pymedphys rung has no device of its own, but the point of the
        # tensor path is that the volumes start on the GPU; it is the transfer
        # back to the host that the comparison is about.
        tensor_device = torch.device(target if rung.backend == "torch" else (device or "cuda:0"))
        reference = torch.as_tensor(pair.reference).reshape(1, 1, *pair.shape).to(tensor_device)
        evaluation = torch.as_tensor(pair.evaluation).reshape(1, 1, *pair.shape).to(tensor_device)

        def call():
            return gamma_index_torch(
                ground_truth=reference.clone(),
                prediction=evaluation.clone(),
                scale=gamma_scale,
                gamma_params=params,
                resolution=resolution,
                cutoff=0,
                backend=rung.backend,
                backend_options=options,
            )

    else:

        def call():
            reference, evaluation = _denormalised(pair, scale)
            return gamma_index(
                ground_truth=reference,
                prediction=evaluation,
                scale=gamma_scale,
                gamma_params=params,
                resolution=resolution,
                cutoff=0,
                backend=rung.backend,
                backend_options=options,
            )

    return PreparedCall(call=call, target=target, on_cuda=on_cuda, stats=stats)


def time_case(
    pair: BeamletPair,
    case: GammaCase,
    rung: RungSpec,
    scale: Dict[str, float],
    *,
    device: Optional[str] = None,
    repeats: int = 3,
    path: str = "array",
) -> Dict[str, Any]:
    """Time one (pair, criterion, rung) case and return its pass rate.

    The timed region is the public entry point end to end: for the array path
    that is de-normalisation of the cached pair, the backend's gamma
    evaluation including any host-device transfers, and the pass-rate
    reduction; for the tensor path it starts from volumes already resident on
    the device. Every repetition is retained.

    Args:
        pair: The beamlet dose pair.
        case: The gamma recipe.
        rung: Backend, device and precision.
        scale: Training scale dict.
        device: Overrides ``rung.device`` (the CLI's ``--device``).
        repeats: Timed repetitions after one untimed warm-up.
        path: ``"array"`` for the numpy entry point, ``"tensor"`` for the
            device-resident one.

    Returns:
        A row dict with the pass rate, every repetition's time and the summary
        statistics over them, the resolved gamma configuration, the torch
        backend's search counters, and peak memory where measurable.

    Raises:
        ValueError: If ``path`` is neither ``"array"`` nor ``"tensor"``.
    """
    if path not in ("array", "tensor"):
        raise ValueError(f"path must be 'array' or 'tensor'; got {path!r}")

    prepared = build_gamma_call(pair, case, rung, scale, device=device, path=path)
    call, target, on_cuda, stats = prepared.call, prepared.target, prepared.on_cuda, prepared.stats

    # One untimed call absorbs the CUDA context creation and the first kernel
    # compilation, neither of which recurs during a sweep.
    _, pass_rate = call()
    _synchronise(rung, target)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(torch.device(target))
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    times: List[float] = []
    for _ in range(max(1, repeats)):
        _synchronise(rung, target)
        start = perf_counter()
        _, pass_rate = call()
        _synchronise(rung, target)
        times.append(perf_counter() - start)

    ordered = sorted(times)
    quartiles = np.percentile(ordered, [25, 50, 75]) if len(ordered) > 1 else [ordered[0]] * 3
    return {
        "sample_id": pair.sample_id,
        "energy_mev": pair.energy_mev,
        "voxels": pair.voxels,
        "criterion": criterion_label(case),
        "interp_fraction": case.interp_fraction,
        "max_gamma": case.max_gamma,
        "rung": rung.rung,
        "backend": rung.backend,
        "device": target if rung.backend == "torch" else "cpu",
        "dtype": rung.dtype,
        "path": path,
        "pass_rate_pct": float(100.0 * pass_rate[0]),
        "seconds_all": [float(t) for t in times],
        "seconds_best": float(ordered[0]),
        "seconds_median": float(quartiles[1]),
        "seconds_q1": float(quartiles[0]),
        "seconds_q3": float(quartiles[2]),
        "seconds_mean": float(sum(times) / len(times)),
        "repeats": len(times),
        "iterations": stats.get("iterations"),
        "shell_points": stats.get("shell_points"),
        "interp_samples": stats.get("interp_samples"),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(torch.device(target))) if on_cuda else None,
        # ru_maxrss is a process-lifetime high-water mark in KiB, so a delta of
        # zero means the case did not raise the process peak, not that it used
        # no memory. It is recorded because nothing better is portable.
        "host_maxrss_kib_after": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "host_maxrss_kib_delta": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before),
        "config": _resolved_config(pair, case, rung, target, scale),
    }


def sweep(
    pairs: Sequence[BeamletPair],
    cases: Sequence[GammaCase],
    rungs: Sequence[RungSpec],
    scale: Dict[str, float],
    *,
    device: Optional[str] = None,
    repeats: int = 3,
    paths: Sequence[str] = ("array",),
    rotate: bool = True,
) -> List[Dict[str, Any]]:
    """Time every (pair, case, rung, path) combination.

    The outer loop is over pairs, and within a pair the rungs are executed in
    an order rotated by the pair's index, so that no backend systematically
    runs first (cold) or last (after the machine has warmed) across the draw.
    The order actually used is recorded in every row.

    Args:
        pairs: Beamlet dose pairs.
        cases: Gamma recipes.
        rungs: Backends to measure.
        scale: Training scale dict.
        device: Overrides each rung's device.
        repeats: Timed repetitions per combination.
        paths: Which entry points to measure; see the module docstring.
        rotate: Rotate the rung order per pair; ``False`` keeps the given order.

    Returns:
        One row per combination, in execution order.
    """
    rows: List[Dict[str, Any]] = []
    total = len(pairs) * len(cases) * len(rungs) * len(paths)
    for pair_index, pair in enumerate(pairs):
        shift = pair_index % len(rungs) if rotate and rungs else 0
        ordered_rungs = list(rungs[shift:]) + list(rungs[:shift])
        order_label = [rung.label for rung in ordered_rungs]
        for case in cases:
            for position, rung in enumerate(ordered_rungs):
                for entry in paths:
                    row = time_case(
                        pair, case, rung, scale, device=device, repeats=repeats, path=entry
                    )
                    row["pair_index"] = pair_index
                    row["rung_order"] = order_label
                    row["order_position"] = position
                    row["sequence"] = len(rows)
                    rows.append(row)
        logger.info(
            "pair %d/%d (%s): %d/%d rows done, rung order %s",
            pair_index + 1,
            len(pairs),
            pair.sample_id[:8],
            len(rows),
            total,
            " > ".join(order_label),
        )
    return rows
