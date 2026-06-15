"""Plan-level dose error metrics (ADoTA vs MCsquare reference).

These are the full-grid counterparts to the per-beamlet metrics: MAPE and RMSE are
computed over a **high-dose mask** (voxels above a fraction of the reference dose's
99th percentile, so out-of-field air does not deflate the numbers), while the
relative dose error mirrors the single-beamlet definition exactly
(:func:`src.metrics.classic.calculate_relative_dose_error`, whole grid, normalized
by the reference peak).
"""

from __future__ import annotations

import numpy as np

from src.metrics.classic import calculate_relative_dose_error, calculate_rmse

__all__ = ["high_dose_mask", "plan_dose_metrics"]


def high_dose_mask(
    dose_ref: np.ndarray, frac: float = 0.1, percentile: float = 99.0
) -> np.ndarray:
    """Boolean mask of voxels above ``frac`` of the reference's percentile dose.

    The threshold is ``frac * percentile(dose_ref, percentile)``. The 99th
    percentile (rather than the raw max) is used as a robust "plan dose" so a
    single hot voxel cannot move the threshold.

    Args:
        dose_ref: Reference dose volume (e.g. MCsquare), any shape.
        frac: Fraction of the percentile dose defining the threshold (0.1 = 10%).
        percentile: Percentile of the reference dose used as the dose level.

    Returns:
        Boolean array of ``dose_ref`` shape, ``True`` inside the high-dose region.
    """
    threshold = frac * float(np.percentile(dose_ref, percentile))
    return dose_ref > threshold


def plan_dose_metrics(
    dose_eval: np.ndarray,
    dose_ref: np.ndarray,
    frac: float = 0.1,
    percentile: float = 99.0,
) -> dict:
    """Plan-level error metrics comparing ``dose_eval`` to reference ``dose_ref``.

    Args:
        dose_eval: Dose under test (e.g. ADoTA), same shape as ``dose_ref``.
        dose_ref: Reference dose (e.g. MCsquare).
        frac: High-dose mask fraction (of the reference percentile dose).
        percentile: Percentile of the reference dose for the mask threshold.

    Returns:
        Dict with ``mape_pct`` and ``rmse_gy`` over the high-dose mask,
        ``relative_dose_error_pct`` over the whole grid, plus the mask metadata
        (``threshold_gy``, ``n_mask_voxels``, ``frac``, ``percentile``).

    Raises:
        ValueError: If the shapes differ.
    """
    if dose_eval.shape != dose_ref.shape:
        raise ValueError(
            f"dose_eval and dose_ref must share shape, got {dose_eval.shape} "
            f"vs {dose_ref.shape}."
        )

    threshold = frac * float(np.percentile(dose_ref, percentile))
    mask = dose_ref > threshold
    n_mask = int(mask.sum())

    if n_mask > 0:
        diff = np.abs(dose_eval[mask] - dose_ref[mask])
        mape = float(np.mean(diff / np.abs(dose_ref[mask])) * 100.0)
        rmse = float(calculate_rmse(dose_eval[mask], dose_ref[mask]))
    else:
        mape = float("nan")
        rmse = float("nan")

    rde = float(calculate_relative_dose_error(dose_eval, dose_ref))

    return {
        "mape_pct": mape,
        "rmse_gy": rmse,
        "relative_dose_error_pct": rde,
        "threshold_gy": threshold,
        "n_mask_voxels": n_mask,
        "frac": frac,
        "percentile": percentile,
    }
