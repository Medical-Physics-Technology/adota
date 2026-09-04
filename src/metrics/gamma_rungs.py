"""Running one rung of the gamma deviation ladder over a plan.

Both runners go through :func:`src.metrics.plan_gamma.plan_gamma` with a single
criterion at a time, so the CPU and torch rungs differ in the gamma backend and
in nothing else -- same dose direction, same normalisation, same pass-rate
arithmetic. Splitting them out of :mod:`src.metrics.gamma_benchmark` keeps both
modules inside the repository's 500-line limit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.metrics.gamma_benchmark import PlanCase
from src.metrics.plan_gamma import criterion_label, plan_gamma

logger = logging.getLogger(__name__)

__all__ = ["run_cpu_case", "run_torch_case"]


def _map_path(maps_dir: Path, plan_name: str, label: str, tag: str) -> Path:
    """``<maps_dir>/<plan>/<tag>__<criterion>.npy`` with a filename-safe label."""
    safe = label.replace("%", "pct").replace("/", "_")
    return maps_dir / plan_name / f"{tag}__{safe}.npy"


def _store_map(
    maps_dir: Optional[Path],
    case: PlanCase,
    label: str,
    tag: str,
    gamma_map: np.ndarray,
) -> Optional[str]:
    """Persist a gamma map as a float32 ``.npy`` and return its path, else None."""
    if maps_dir is None:
        return None
    path = _map_path(Path(maps_dir), case.name, label, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, gamma_map.astype(np.float32))
    logger.info("  wrote gamma map %s (%.0f MB)", path, path.stat().st_size / 1e6)
    return str(path)


def run_cpu_case(
    case: PlanCase,
    criteria: Optional[Sequence[Tuple[float, float, float]]] = None,
    maps_dir: Optional[Path] = None,
    tag: str = "rung1",
) -> dict:
    """Rung 1: re-run the recorded CPU path, one criterion at a time.

    Goes through :func:`src.metrics.plan_gamma.plan_gamma` with the plan's own
    ``gamma_params_base``, so the computation is the pipeline's -- only the
    pymedphys version differs from the recording.

    Args:
        case: The loaded plan.
        criteria: Subset of the plan's criteria; defaults to all of them.
        maps_dir: When given, each gamma map is written there as a float32
            ``.npy`` for the voxel-level comparison. ~400 MB per criterion.
        tag: Filename prefix for the persisted maps.

    Returns:
        A per-plan result dict (see :func:`write_rung_json`).
    """
    return _run_case(
        case,
        criteria,
        maps_dir,
        tag,
        lambda criterion: _cpu_criterion(case, criterion),
    )


def _cpu_criterion(case: PlanCase, criterion) -> Tuple[np.ndarray, float, dict]:
    """One pymedphys criterion; returns ``(gamma_map, pass_rate_pct, extra)``."""
    started = perf_counter()
    results = plan_gamma(
        case.dose_eval,
        case.dose_ref,
        case.spacing_zyx,
        [criterion],
        case.gamma_params_base,
    )
    elapsed = perf_counter() - started
    result = results[0]
    return result["gamma_map"], result["pass_rate_pct"], {"elapsed_s": elapsed}


def run_torch_case(
    case: PlanCase,
    device: str,
    dtype: str = "float64",
    criteria: Optional[Sequence[Tuple[float, float, float]]] = None,
    maps_dir: Optional[Path] = None,
    tag: Optional[str] = None,
) -> dict:
    """Rungs 2-4: the same criteria through :mod:`src.metrics.gamma_torch`.

    Goes through the same :func:`src.metrics.plan_gamma.plan_gamma` entry point
    as rung 1, with only the backend switched, so the two differ in the gamma
    kernel and nothing else.

    Args:
        case: The loaded plan.
        device: ``"cpu"`` or e.g. ``"cuda:0"``.
        dtype: ``"float32"`` or ``"float64"``.
        criteria: Subset of the plan's criteria; defaults to all of them.
        maps_dir: Optional directory for float32 ``.npy`` gamma maps.
        tag: Filename prefix; defaults to ``"<device>_<dtype>"``.

    Returns:
        A per-plan result dict, with ``peak_gpu_mem_bytes`` and
        ``interp_samples`` filled in per criterion.
    """
    resolved_tag = tag or f"{device.replace(':', '')}_{dtype}"
    warm_up_s = _warm_up_device(device, dtype)
    result = _run_case(
        case,
        criteria,
        maps_dir,
        resolved_tag,
        lambda criterion: _torch_criterion(case, criterion, device, dtype),
    )
    result["warm_up_s"] = warm_up_s
    return result


def _warm_up_device(device: str, dtype: str) -> float:
    """Pay the CUDA context and kernel-autotune cost once, outside the timings.

    Calls the kernel directly on a tiny synthetic grid rather than going through
    ``plan_gamma``: a small corner of a real plan can hold no dose above the
    cutoff at all, and the pass-rate denominator is then zero.

    Args:
        device: ``"cpu"`` or ``"cuda:<index>"``.
        dtype: ``"float32"`` or ``"float64"``.

    Returns:
        Seconds spent, reportable as a one-off cost. Zero on CPU, where there is
        no context to create.
    """
    import torch

    from src.metrics.gamma_torch import gamma_index_torch_core

    if not device.startswith("cuda"):
        return 0.0
    started = perf_counter()
    axes = tuple(np.arange(6, dtype=np.float64) for _ in range(3))
    dose = np.linspace(0.0, 1.0, 6**3, dtype=np.float32).reshape(6, 6, 6)
    gamma_index_torch_core(
        axes, dose, axes, dose * 1.01, 2.0, 2.0,
        lower_percent_dose_cutoff=10, interp_fraction=5, max_gamma=2,
        device=device, dtype=getattr(torch, dtype),
    )
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    logger.info("CUDA warm-up on %s (%s): %.1fs", device, dtype, elapsed)
    return elapsed


def _torch_criterion(
    case: PlanCase, criterion, device: str, dtype: str
) -> Tuple[np.ndarray, float, dict]:
    """One torch criterion; returns ``(gamma_map, pass_rate_pct, extra)``."""
    import torch

    stats: dict = {}
    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started = perf_counter()
    results = plan_gamma(
        case.dose_eval,
        case.dose_ref,
        case.spacing_zyx,
        [criterion],
        case.gamma_params_base,
        backend="torch",
        backend_options={"device": device, "dtype": dtype, "stats": stats},
    )
    if on_cuda:
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started

    extra = {"elapsed_s": elapsed, **stats}
    if on_cuda:
        extra["peak_gpu_mem_bytes"] = int(torch.cuda.max_memory_allocated(device))
    result = results[0]
    return result["gamma_map"], result["pass_rate_pct"], extra


def _run_case(case: PlanCase, criteria, maps_dir, tag, run_one) -> dict:
    """Shared per-plan loop: run each criterion, collect timings and maps."""
    selected = list(criteria) if criteria is not None else list(case.criteria)
    entries: List[dict] = []
    total = 0.0
    for criterion in selected:
        label = criterion_label(tuple(criterion))
        logger.info("[%s] %s %s ...", case.name, tag, label)
        gamma_map, pass_rate_pct, extra = run_one(criterion)
        elapsed = float(extra.get("elapsed_s", 0.0))
        total += elapsed
        entry = {
            "label": label,
            "criterion": [float(v) for v in criterion],
            "pass_rate_pct": float(pass_rate_pct),
            "elapsed_s": elapsed,
            "n_evaluated": int(
                np.count_nonzero(np.nan_to_num(gamma_map, nan=0.0) > 0)
            ),
        }
        entry.update({k: v for k, v in extra.items() if k != "elapsed_s"})
        entry["gamma_map_path"] = _store_map(maps_dir, case, label, tag, gamma_map)
        entries.append(entry)
        logger.info(
            "[%s] %s %s -> GPR %.4f%% in %.1fs",
            case.name,
            tag,
            label,
            pass_rate_pct,
            elapsed,
        )
        del gamma_map
    return {
        "grid_zyx": [int(v) for v in case.dose_ref.shape],
        "spacing_zyx": [float(v) for v in case.spacing_zyx],
        "n_voxels": case.n_voxels,
        "gamma_params_base": case.gamma_params_base,
        "recorded_elapsed_s": case.recorded_elapsed_s,
        "recorded_pass_rate_pct": case.recorded,
        "criteria": entries,
        "total_elapsed_s": total,
    }
