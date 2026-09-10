# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Agreement between gamma maps compared in their native precision.

The plan-level parity check of CHG-0005 compared gamma maps that had been saved
as float32, so it could not certify the float64 outputs it was meant to
certify. This module compares maps in memory, before any cast or
serialisation, and records enough about the comparison that the claim can be
re-checked without the maps: dtypes, shapes, a digest of each map's raw bytes,
the evaluated-mask disagreements, the distribution of absolute differences, the
number of points that cross the pass threshold, and the pass-rate difference.

The gates are predeclared verification thresholds, not tuning knobs: a
comparison either clears them or it does not, and the measured values are
reported either way.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.adota.config import DEFAULT_GAMMA_PARAMS
from src.metrics.gamma_beamlet_benchmark import GammaCase, RungSpec
from src.metrics.gamma_beamlet_pairs import BeamletPair, beamlet_resolution_mm
from src.metrics.gamma_pass_rate import _gamma_pass_rate, _gamma_values

__all__ = [
    "GATES",
    "PERCENTILES",
    "beamlet_gamma_map",
    "plan_gamma_map",
    "pass_rate_pct",
    "map_digest",
    "compare_maps",
    "check_gates",
]

# Predeclared verification gates, keyed by the working dtype of the map under
# test. A float64 map must reproduce the reference at the level of the pass
# rate, the evaluated mask and the pass threshold; float32 is held to the pass
# rate only.
GATES: Dict[str, Dict[str, float]] = {
    "float64": {"max_pass_rate_delta_pp": 0.01, "max_mask_disagreements": 0, "max_boundary_crossings": 0},
    "float32": {"max_pass_rate_delta_pp": 0.1},
}

PERCENTILES: Tuple[float, ...] = (50.0, 95.0, 99.0, 99.9, 99.99)


def _torch_options(rung: RungSpec, device: Optional[str]) -> Optional[Dict[str, Any]]:
    return rung.backend_options(device)


def beamlet_gamma_map(
    pair: BeamletPair,
    case: GammaCase,
    rung: RungSpec,
    scale: Dict[str, float],
    device: Optional[str] = None,
) -> np.ndarray:
    """The raw gamma map of one beamlet pair, NaN where not evaluated.

    Goes through the same de-normalisation and the same backend dispatch as the
    timed entry point, but stops before ``np.nan_to_num`` so the evaluated mask
    survives, and returns the array in the dtype the backend produced.
    """
    from src.utils.scallers import inverse_minmax

    reference = inverse_minmax(pair.reference.astype(np.float64), scale["min_ds"], scale["max_ds"])
    evaluation = inverse_minmax(pair.evaluation.astype(np.float64), scale["min_ds"], scale["max_ds"])
    axes = tuple(np.arange(size) * step for size, step in zip(pair.shape, beamlet_resolution_mm()))
    return np.asarray(
        _gamma_values(axes, reference, evaluation, case.as_params(), rung.backend, _torch_options(rung, device))
    )


def plan_gamma_map(
    dose_ref: np.ndarray,
    dose_eval: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    criterion: Tuple[float, float, float],
    gamma_params_base: Dict[str, Any],
    rung: RungSpec,
    device: Optional[str] = None,
) -> np.ndarray:
    """The raw gamma map of a plan-scale pair under the plan's recorded recipe.

    Mirrors :func:`src.metrics.plan_gamma.plan_gamma` parameter for parameter,
    but returns the map before the pass-rate reduction zeroes its NaNs.
    """
    dose_pct, dist_mm, cutoff_pct = criterion
    params = {
        **DEFAULT_GAMMA_PARAMS,
        **gamma_params_base,
        "dose_percent_threshold": dose_pct,
        "distance_mm_threshold": dist_mm,
        "lower_percent_dose_cutoff": cutoff_pct,
    }
    axes = tuple(np.arange(size) * step for size, step in zip(dose_ref.shape, spacing_zyx))
    return np.asarray(
        _gamma_values(axes, dose_ref.copy(), dose_eval.copy(), params, rung.backend, _torch_options(rung, device))
    )


def pass_rate_pct(gamma_map: np.ndarray) -> float:
    """The project's pass rate of a raw map, in percent, without mutating it."""
    _, rates = _gamma_pass_rate(np.array(gamma_map, copy=True))
    return float(100.0 * rates[0])


def map_digest(gamma_map: np.ndarray) -> str:
    """SHA-256 of the map's raw bytes in C order, so a saved map can be verified."""
    return hashlib.sha256(np.ascontiguousarray(gamma_map).tobytes()).hexdigest()


def compare_maps(reference: np.ndarray, other: np.ndarray) -> Dict[str, Any]:
    """Every agreement statistic of the map under test against the reference.

    Args:
        reference: The baseline raw map (NaN where not evaluated).
        other: The map under test, same shape, any float dtype.

    Returns:
        A dict of the statistics named in the module docstring. Differences are
        computed in float64 over the points both maps evaluated.

    Raises:
        ValueError: If the shapes differ.
    """
    if reference.shape != other.shape:
        raise ValueError(f"gamma maps must share shape, got {reference.shape} vs {other.shape}")
    flat_a = reference.ravel()
    flat_b = other.ravel()
    mask_a = ~np.isnan(flat_a)
    mask_b = ~np.isnan(flat_b)
    both = mask_a & mask_b
    n_both = int(np.count_nonzero(both))

    result: Dict[str, Any] = {
        "reference_dtype": str(reference.dtype),
        "other_dtype": str(other.dtype),
        "shape": list(reference.shape),
        "n_evaluated_reference": int(np.count_nonzero(mask_a)),
        "n_evaluated_other": int(np.count_nonzero(mask_b)),
        "n_evaluated_both": n_both,
        "mask_disagreements": int(np.count_nonzero(mask_a ^ mask_b)),
        "reference_pass_rate_pct": pass_rate_pct(reference),
        "other_pass_rate_pct": pass_rate_pct(other),
        "reference_digest": map_digest(reference),
        "other_digest": map_digest(other),
    }
    result["pass_rate_delta_pp"] = result["other_pass_rate_pct"] - result["reference_pass_rate_pct"]

    if n_both == 0:
        result.update(
            {"max_abs_delta": 0.0, "mean_abs_delta": 0.0, "boundary_crossings": 0, "boundary_crossing_fraction": 0.0}
        )
        result.update({f"p{p:g}_abs_delta": 0.0 for p in PERCENTILES})
        return result

    a = flat_a[both].astype(np.float64)
    b = flat_b[both].astype(np.float64)
    delta = np.abs(a - b)
    crossings = int(np.count_nonzero((a > 1.0) != (b > 1.0)))
    result.update(
        {
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "boundary_crossings": crossings,
            "boundary_crossing_fraction": crossings / n_both,
            "bitwise_identical": bool(
                reference.dtype == other.dtype and np.array_equal(flat_a, flat_b, equal_nan=True)
            ),
        }
    )
    for percentile, value in zip(PERCENTILES, np.percentile(delta, PERCENTILES)):
        result[f"p{percentile:g}_abs_delta"] = float(value)
    return result


def check_gates(comparison: Dict[str, Any], dtype: str) -> Dict[str, Any]:
    """Evaluate the predeclared gates for a comparison; report, never raise."""
    gates = GATES.get(dtype, GATES["float32"])
    checks = {"max_pass_rate_delta_pp": abs(comparison["pass_rate_delta_pp"]) <= gates["max_pass_rate_delta_pp"]}
    if "max_mask_disagreements" in gates:
        checks["max_mask_disagreements"] = comparison["mask_disagreements"] <= gates["max_mask_disagreements"]
    if "max_boundary_crossings" in gates:
        checks["max_boundary_crossings"] = comparison["boundary_crossings"] <= gates["max_boundary_crossings"]
    return {"gates": gates, "checks": checks, "passed": all(checks.values())}
