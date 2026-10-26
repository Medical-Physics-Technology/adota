"""Dose-agreement metrics shared by the water validation and the engine comparison.

Gamma goes through :func:`src.metrics.gamma_pass_rate.gamma_index` (pymedphys or
the torch backend) with the criteria type of the angle-robustness analysis
(:class:`src.mc_generation.angle_robustness_analysis.GammaCriterion`); range
metrics through :mod:`src.metrics.range_metrics`. Everything here takes
``(z, y, x)`` volumes on one grid with the beam along x.
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.mc_generation.angle_robustness_analysis import GammaCriterion
from src.metrics.gamma_pass_rate import gamma_index
from src.metrics.range_metrics import compute_range_metrics, range_metric_deltas

__all__ = ["GammaCriterion", "criteria_from_config", "dose_agreement", "dose_bbox", "gamma_pass_rates"]

# 3%/3mm, 10% cutoff is the headline (CLAUDE.md); 2%/2mm matches training/ablation.
DEFAULT_CRITERIA = ((3.0, 3.0, 10.0), (2.0, 2.0, 10.0), (1.0, 1.0, 10.0))


def criteria_from_config(items: Optional[Sequence]) -> Tuple[GammaCriterion, ...]:
    """``[{dose, dist, cutoff}, ...]`` (or ``[d, d, c]`` triples) -> criteria."""
    items = DEFAULT_CRITERIA if not items else items
    out = []
    for it in items:
        d, r, c = (it["dose"], it["dist"], it["cutoff"]) if isinstance(it, Mapping) else it
        out.append(GammaCriterion(float(d), float(r), float(c)))
    return tuple(out)


def dose_bbox(volumes: Sequence[np.ndarray], low_frac: float = 0.001, margin: int = 5) -> Tuple[slice, ...]:
    """Bounding box (+margin) of the voxels above ``low_frac`` of each volume's max, unioned."""
    shape = volumes[0].shape
    lo = np.array(shape)
    hi = np.zeros(len(shape), dtype=int)
    for v in volumes:
        peak = float(np.max(v))
        idx = np.argwhere(v > low_frac * peak) if peak > 0 else np.empty((0, len(shape)))
        if idx.size:
            lo = np.minimum(lo, idx.min(0))
            hi = np.maximum(hi, idx.max(0) + 1)
    if np.any(hi <= lo):
        return tuple(slice(0, n) for n in shape)
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def gamma_pass_rates(
    reference: np.ndarray,
    test: np.ndarray,
    criteria: Sequence[GammaCriterion],
    spacing_mm: float = 1.0,
    local: bool = False,
    interp_fraction: int = 5,
    backend: str = "pymedphys",
    backend_options: Optional[dict] = None,
) -> Dict[str, float]:
    """Gamma pass rate (%) of ``test`` against ``reference`` for every criterion.

    Global gamma is normalised to the reference maximum; voxels below the
    criterion's cutoff (% of that maximum) in the reference are excluded.
    """
    logging.getLogger("pymedphys").setLevel(logging.WARNING)
    peak = float(np.max(reference))
    out = {}
    for c in criteria:
        if peak <= 0:
            out[c.key] = float("nan")
            continue
        params = {"dose_percent_threshold": c.dose, "distance_mm_threshold": c.dist,
                  "interp_fraction": interp_fraction, "max_gamma": 2,
                  "lower_percent_dose_cutoff": c.cutoff, "random_subset": None,
                  "local_gamma": local, "quiet": True}
        _, gpr = gamma_index(reference.astype(np.float64).copy(), test.astype(np.float64).copy(),
                             {"y_max": peak, "y_min": 0.0}, params, (spacing_mm,) * 3,
                             backend=backend, backend_options=dict(backend_options or {}))
        out[c.key] = float(gpr[0]) * 100.0
    return out


def dose_agreement(reference: np.ndarray, test: np.ndarray, spacing_mm: float = 1.0,
                   high_dose_frac: float = 0.1) -> Dict[str, float]:
    """Range, scale and voxel-difference agreement of ``test`` with ``reference``.

    Range metrics use the laterally integrated depth dose along x. Voxel
    differences are reported in % of the reference maximum over the voxels where
    either dose exceeds ``high_dose_frac`` of that maximum.
    """
    peak = float(np.max(reference))
    idd_ref = reference.sum(axis=(0, 1)) * spacing_mm ** 2
    idd_test = test.sum(axis=(0, 1)) * spacing_mm ** 2
    rm_ref = compute_range_metrics(idd_ref, spacing_mm)
    rm_test = compute_range_metrics(idd_test, spacing_mm)
    mask = (reference > high_dose_frac * peak) | (test > high_dose_frac * peak)
    diff = (test - reference)[mask] / peak * 100.0 if peak > 0 else np.array([np.nan])
    out = {
        "peak_dose_ratio": float(np.max(test)) / peak if peak > 0 else float("nan"),
        "integral_dose_ratio": float(test.sum()) / float(reference.sum()) if reference.sum() > 0 else float("nan"),
        "idd_peak_ratio": rm_test.peak_dose / rm_ref.peak_dose if rm_ref.peak_dose > 0 else float("nan"),
        "mean_abs_diff_pct": float(np.mean(np.abs(diff))),
        "max_abs_diff_pct": float(np.max(np.abs(diff))),
        "mean_diff_pct": float(np.mean(diff)),
        "r80_ref_mm": rm_ref.r80_mm,
    }
    out.update(range_metric_deltas(rm_test, rm_ref))
    return out
