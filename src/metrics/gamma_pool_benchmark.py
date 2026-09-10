# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Directly measured gamma cost over validation pools of beamlets.

EXP-0007 projected the cost of scoring 200 and 2000 beamlets by multiplying a
per-beamlet median. This module measures it: whole passes over fixed pools,
every beamlet timed individually and every pass retained, so the total, the
tail and the pass-to-pass variation are all observed rather than inferred.

Two kinds of pass are measured and kept apart:

``cached``
    The pool's dose pairs are already in memory; only gamma is timed. This is
    the cost of the metric itself.
``integrated``
    Each beamlet is read from the dataset, run through the model, and scored,
    as a training-time evaluation pass would do. Inference and gamma are timed
    separately within it, and the sum is what a pass actually costs.

Pools are defined by explicit record ids, saved with the results, so the same
pool can be rebuilt and re-timed.
"""

from __future__ import annotations

import logging
import resource
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.metrics.gamma_beamlet_benchmark import build_gamma_call
from src.metrics.gamma_beamlet_pairs import BeamletPair, beamlet_resolution_mm
from src.metrics.gamma_beamlet_specs import GammaCase, RungSpec
from src.metrics.gamma_pass_rate import gamma_index, gamma_index_torch

logger = logging.getLogger(__name__)

__all__ = [
    "test_pool_ids",
    "validation_split_pool_ids",
    "cached_passes",
    "integrated_passes",
]


def test_pool_ids(h5_path: Path, size: Optional[int], seed: int) -> List[str]:
    """Record ids of a held-out pool: the whole file, or a seeded draw of ``size``."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        all_ids = sorted(handle.keys())
    if size is None or size >= len(all_ids):
        return all_ids
    rng = np.random.RandomState(seed)
    return sorted(all_ids[i] for i in rng.choice(len(all_ids), size=size, replace=False))


def validation_split_pool_ids(
    train_h5: Path, excluded_path: Optional[Path], test_size: float, split_seed: int, size: Optional[int], seed: int
) -> Dict[str, Any]:
    """A seeded draw from the training run's own validation split.

    Reproduces the split exactly as ``scripts/train_adota.py`` makes it, so the
    pool is a subset of the records the training loop would monitor.
    """
    from src.training.data import load_record_ids, train_val_split

    usable = load_record_ids(train_h5, excluded_path)
    _, val_ids = train_val_split(usable, test_size=test_size, random_state=split_seed)
    if size is None or size >= len(val_ids):
        picked = list(val_ids)
    else:
        rng = np.random.RandomState(seed)
        picked = sorted(val_ids[i] for i in rng.choice(len(val_ids), size=size, replace=False))
    return {"ids": picked, "n_usable": len(usable), "n_validation_split": len(val_ids)}


def _criterion_label(case: GammaCase) -> str:
    return f"{case.dose_percent_threshold:g}%/{case.distance_mm_threshold:g}mm/{case.lower_percent_dose_cutoff:g}%"


def _sync(on_cuda: bool, target: Optional[str]) -> None:
    if on_cuda:
        torch.cuda.synchronize(torch.device(target))


def cached_passes(
    pairs: Sequence[BeamletPair],
    case: GammaCase,
    rung: RungSpec,
    scale: Dict[str, float],
    *,
    device: Optional[str],
    path: str,
    passes: int,
) -> Dict[str, Any]:
    """One untimed warm-up, then ``passes`` complete timed passes over the pool.

    Every beamlet is timed individually inside each pass; the pass wall time is
    measured separately around the whole loop, so the Python overhead between
    beamlets is visible as the difference.
    """
    prepared = [build_gamma_call(pair, case, rung, scale, device=device, path=path) for pair in pairs]
    first = prepared[0]
    first.call()
    _sync(first.on_cuda, first.target)

    results: List[Dict[str, Any]] = []
    pass_rates: List[float] = []
    for pass_index in range(passes):
        if first.on_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(first.target))
        per_beamlet: List[float] = []
        _sync(first.on_cuda, first.target)
        wall_start = perf_counter()
        for index, item in enumerate(prepared):
            start = perf_counter()
            _, pass_rate = item.call()
            _sync(item.on_cuda, item.target)
            per_beamlet.append(perf_counter() - start)
            if pass_index == 0:
                pass_rates.append(float(100.0 * pass_rate[0]))
        wall = perf_counter() - wall_start
        results.append(
            {
                "pass_index": pass_index,
                "wall_s": wall,
                "sum_beamlet_s": float(sum(per_beamlet)),
                "per_beamlet_s": per_beamlet,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(torch.device(first.target)))
                if first.on_cuda
                else None,
                "host_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
        )
        logger.info("%s | %s pass %d: %.1f s wall", rung.label, path, pass_index, wall)
    return {
        "rung": rung.rung,
        "label": rung.label,
        "path": path,
        "criterion": _criterion_label(case),
        "n_beamlets": len(pairs),
        "sample_ids": [pair.sample_id for pair in pairs],
        "pass_rates_pct": pass_rates,
        "passes": results,
    }


def integrated_passes(
    dataset,
    model,
    case: GammaCase,
    rung: RungSpec,
    scale: Dict[str, float],
    *,
    device: str,
    path: str,
    passes: int,
) -> Dict[str, Any]:
    """Read, infer and score every record of ``dataset``, ``passes`` times.

    ``path`` selects how the gamma step receives its inputs: ``"tensor"`` hands
    the model output and the reference to the torch backend on the device,
    ``"array"`` copies both to the host first, as the pymedphys path must.
    The inference time includes the dataset read and the transfer to the
    device; the gamma time is the rest.
    """
    params = case.as_params()
    resolution = beamlet_resolution_mm()
    gamma_scale = {**scale, "y_min": scale["min_ds"], "y_max": scale["max_ds"]}
    target = rung.resolve_device(device)
    options = rung.backend_options(device)
    infer_device = torch.device(device)
    on_cuda = infer_device.type == "cuda"
    span = scale["max_ds"] - scale["min_ds"]

    def score(y: torch.Tensor, prediction: torch.Tensor):
        if path == "tensor":
            return gamma_index_torch(
                ground_truth=y.reshape(1, 1, *y.shape[-3:]),
                prediction=prediction.reshape(1, 1, *prediction.shape[-3:]),
                scale=gamma_scale,
                gamma_params=params,
                resolution=resolution,
                cutoff=0,
                backend=rung.backend,
                backend_options=options,
            )
        reference = y.squeeze().cpu().numpy().astype(np.float64) * span + scale["min_ds"]
        evaluation = prediction.squeeze().detach().cpu().numpy().astype(np.float64) * span + scale["min_ds"]
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

    def one(index: int):
        start = perf_counter()
        x, energy, y = dataset[index]
        with torch.no_grad():
            prediction = model(x.unsqueeze(0).to(infer_device), energy.unsqueeze(0).to(infer_device))[0]
        y = y.to(infer_device)
        _sync(on_cuda, str(infer_device))
        infer_s = perf_counter() - start
        gamma_start = perf_counter()
        _, pass_rate = score(y, prediction)
        _sync(on_cuda, str(infer_device))
        return infer_s, perf_counter() - gamma_start, float(100.0 * pass_rate[0])

    one(0)  # warm-up
    results: List[Dict[str, Any]] = []
    pass_rates: List[float] = []
    for pass_index in range(passes):
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(infer_device)
        infer_all: List[float] = []
        gamma_all: List[float] = []
        wall_start = perf_counter()
        for index in range(len(dataset)):
            infer_s, gamma_s, rate = one(index)
            infer_all.append(infer_s)
            gamma_all.append(gamma_s)
            if pass_index == 0:
                pass_rates.append(rate)
        wall = perf_counter() - wall_start
        results.append(
            {
                "pass_index": pass_index,
                "wall_s": wall,
                "sum_inference_s": float(sum(infer_all)),
                "sum_gamma_s": float(sum(gamma_all)),
                "per_beamlet_inference_s": infer_all,
                "per_beamlet_gamma_s": gamma_all,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(infer_device)) if on_cuda else None,
                "host_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
        )
        logger.info("integrated %s | %s pass %d: %.1f s wall", rung.label, path, pass_index, wall)
    return {
        "rung": rung.rung,
        "label": rung.label,
        "path": path,
        "criterion": _criterion_label(case),
        "n_beamlets": len(dataset),
        "sample_ids": list(dataset.record_ids),
        "pass_rates_pct": pass_rates,
        "target_device": target,
        "passes": results,
    }
