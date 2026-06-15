"""Plan-level gamma pass rate over several criteria.

Thin orchestration on top of :func:`src.metrics.gamma_pass_rate.gamma_index` (the
same function the per-beamlet evaluation uses), run once per ``(dose%, distance_mm,
dose_cutoff%)`` criterion on the full plan grid. Each criterion yields a gamma map
(same shape as the dose) and the pass rate among evaluated voxels.

``gamma_index`` mutates its inputs (it zeroes sub-cutoff voxels in place), so every
call here is given **copies** of the dose arrays.
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import List, Sequence, Tuple

import numpy as np

from src.adota.config import DEFAULT_GAMMA_PARAMS
from src.metrics.gamma_pass_rate import gamma_index

logger = logging.getLogger(__name__)

__all__ = ["parse_criteria", "criterion_label", "plan_gamma"]


def parse_criteria(raw: Sequence) -> List[Tuple[float, float, float]]:
    """Normalize the YAML criteria list into ``(dose%, dist_mm, cutoff%)`` tuples.

    Args:
        raw: Iterable of 3-element items ``[dose_percent, distance_mm, cutoff_percent]``.

    Returns:
        List of float tuples.

    Raises:
        ValueError: If any item does not have exactly three numbers.
    """
    criteria: List[Tuple[float, float, float]] = []
    for item in raw:
        values = list(item)
        if len(values) != 3:
            raise ValueError(
                f"Each gamma criterion must be [dose%, distance_mm, cutoff%]; got {item!r}"
            )
        criteria.append((float(values[0]), float(values[1]), float(values[2])))
    return criteria


def criterion_label(criterion: Tuple[float, float, float]) -> str:
    """Human-readable ``"1%/1mm/10%"`` label, trimming trailing ``.0``."""

    def fmt(value: float) -> str:
        return f"{value:g}"

    dose, dist, cutoff = criterion
    return f"{fmt(dose)}%/{fmt(dist)}mm/{fmt(cutoff)}%"


def plan_gamma(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    spacing_zyx: Tuple[float, float, float],
    criteria: Sequence[Tuple[float, float, float]],
    extra_params: dict | None = None,
) -> List[dict]:
    """Compute the gamma map + pass rate for each criterion on the plan grid.

    Args:
        dose_eval: Dose under test (e.g. ADoTA), ``(z, y, x)`` in Gy.
        dose_ref: Reference dose (e.g. MCsquare), same shape, in Gy.
        spacing_zyx: Voxel spacing ``(z, y, x)`` in mm (the gamma resolution).
        criteria: Iterable of ``(dose%, distance_mm, cutoff%)`` tuples.
        extra_params: Extra pymedphys gamma params merged over
            :data:`src.adota.config.DEFAULT_GAMMA_PARAMS` (e.g. ``interp_fraction``,
            ``max_gamma``, ``local_gamma``, ``random_subset``).

    Returns:
        One dict per criterion: ``{"criterion", "label", "pass_rate_pct",
        "gamma_map", "gamma_params"}``. ``gamma_map`` has the dose shape; 0 marks
        not-evaluated (sub-cutoff) voxels, ``<=1`` passes, ``>1`` fails.

    Raises:
        ValueError: If the shapes differ.
    """
    if dose_eval.shape != dose_ref.shape:
        raise ValueError(
            f"dose_eval and dose_ref must share shape, got {dose_eval.shape} "
            f"vs {dose_ref.shape}."
        )

    extra_params = extra_params or {}
    ref_max = max(float(dose_ref.max()), 1e-30)
    scale = {"y_max": ref_max, "y_min": 0.0}
    resolution = tuple(float(s) for s in spacing_zyx)

    results: List[dict] = []
    for criterion in criteria:
        dose_pct, dist_mm, cutoff_pct = criterion
        gamma_params = {
            **DEFAULT_GAMMA_PARAMS,
            **extra_params,
            "dose_percent_threshold": dose_pct,
            "distance_mm_threshold": dist_mm,
            "lower_percent_dose_cutoff": cutoff_pct,
        }
        label = criterion_label(criterion)
        logger.info(
            "Gamma %s (interp_fraction=%s, max_gamma=%s) ...",
            label,
            gamma_params.get("interp_fraction"),
            gamma_params.get("max_gamma"),
        )
        started = perf_counter()
        # gamma_index mutates its inputs -> pass copies. Reference is the ground
        # truth (first arg), prediction the eval dose (per-beamlet convention).
        gamma_map, pass_rate = gamma_index(
            ground_truth=dose_ref.copy(),
            prediction=dose_eval.copy(),
            scale=scale,
            gamma_params=gamma_params,
            resolution=resolution,
            cutoff=0,
        )
        pass_rate_pct = float(pass_rate[0]) * 100.0
        logger.info(
            "  %s -> GPR %.2f%% in %.1fs", label, pass_rate_pct, perf_counter() - started
        )
        results.append(
            {
                "criterion": criterion,
                "label": label,
                "pass_rate_pct": pass_rate_pct,
                "gamma_map": gamma_map,
                "gamma_params": gamma_params,
            }
        )
    return results
