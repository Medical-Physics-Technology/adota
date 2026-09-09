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

The dose pairs themselves come from :mod:`src.metrics.gamma_beamlet_pairs`,
which is split off by role: that module builds and caches the data, this one
times the backends against it. Its public names are re-exported here so a caller
has a single import site.

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
    "time_case",
    "sweep",
]


@dataclass(frozen=True)
class RungSpec:
    """One backend / device / precision combination under test.

    Attributes:
        rung: Ladder position, matching :mod:`src.metrics.gamma_benchmark`.
        backend: ``"pymedphys"`` or ``"torch"``.
        device: Device string for the torch backend; ``None`` for pymedphys.
        dtype: ``"float32"`` or ``"float64"`` for the torch backend.
    """

    rung: int
    backend: str
    device: Optional[str] = None
    dtype: Optional[str] = None

    @property
    def label(self) -> str:
        """Short human-readable name, used as a table column."""
        if self.backend == "pymedphys":
            return "pymedphys cpu"
        return f"torch {self.device} {self.dtype}"

    def resolve_device(self, device: Optional[str]) -> Optional[str]:
        """The device this rung runs on, given the CLI's ``--device``.

        The override applies only to the GPU rungs. Rung 2 is the torch-CPU
        rung by definition, so ``--device cuda:0`` must not silently move it
        onto the GPU and make rungs 2 and 3 the same measurement.
        """
        if self.device is None:
            return None
        if self.device.startswith("cuda") and device:
            return device
        return self.device

    def backend_options(self, device: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Options dict for :func:`src.metrics.gamma_pass_rate.gamma_index`."""
        if self.backend == "pymedphys":
            return None
        return {"device": self.resolve_device(device), "dtype": self.dtype}


# The rungs, keyed by the name the CLI accepts. ``cuda`` is a placeholder that
# the CLI's --device argument replaces, so the same table can be produced on any
# GPU index without editing the specs.
RUNGS: Dict[str, RungSpec] = {
    "rung1": RungSpec(1, "pymedphys"),
    "rung2": RungSpec(2, "torch", "cpu", "float64"),
    "rung3": RungSpec(3, "torch", "cuda", "float64"),
    "rung4": RungSpec(4, "torch", "cuda", "float32"),
}


@dataclass(frozen=True)
class GammaCase:
    """One gamma recipe: a criterion plus the search parameters.

    Attributes:
        dose_percent_threshold: Dose-difference criterion, in percent.
        distance_mm_threshold: Distance-to-agreement criterion, in millimetres.
        lower_percent_dose_cutoff: Percent of the normalisation below which
            gamma is not evaluated.
        interp_fraction: Steps the distance threshold is divided into.
        max_gamma: Largest gamma searched for.
    """

    dose_percent_threshold: float
    distance_mm_threshold: float
    lower_percent_dose_cutoff: float = 10.0
    interp_fraction: int = 10
    max_gamma: float = 2.0

    def as_params(self) -> Dict[str, Any]:
        """The dict both backends take, in ``DEFAULT_GAMMA_PARAMS`` form."""
        return {
            "dose_percent_threshold": self.dose_percent_threshold,
            "distance_mm_threshold": self.distance_mm_threshold,
            "interp_fraction": self.interp_fraction,
            "max_gamma": self.max_gamma,
            "lower_percent_dose_cutoff": self.lower_percent_dose_cutoff,
            "random_subset": None,
            "local_gamma": False,
            "quiet": True,
        }


def criterion_label(case: GammaCase) -> str:
    """``"3%/3mm/10%"``-style label, matching the plan-level tables."""
    return (
        f"{case.dose_percent_threshold:g}%/{case.distance_mm_threshold:g}mm/"
        f"{case.lower_percent_dose_cutoff:g}%"
    )


def default_cases() -> Tuple[GammaCase, ...]:
    """The three headline criteria, at the repository's default search settings.

    3%/3mm is the reported headline; 2%/2mm is what the training configs use;
    1%/1mm is the tightest criterion, and the most expensive, because a tighter
    distance threshold makes the search radius grow in smaller steps.
    """
    return (
        GammaCase(1.0, 1.0),
        GammaCase(2.0, 2.0),
        GammaCase(3.0, 3.0),
    )


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
