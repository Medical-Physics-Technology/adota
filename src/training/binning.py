"""Per-sample validation records and how they are aggregated.

Flow:
1. :class:`_SampleMetrics` is one validation sample's id, energy and metrics.
2. :func:`_bin_by_fixed_edges` groups those records into the fixed energy bands
   the paper reports; :func:`_bin_by_quantile` groups them into equal-count
   bands, which stays informative when the energy distribution is skewed.
3. :func:`_worst_k_records` returns the K worst samples by MAPE for the
   per-epoch failure log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ── Per-sample record ───────────────────────────────────────────────────────


@dataclass
class _SampleMetrics:
    energy_mev: float
    loss_mse: float
    loss_ps: float
    rmse_gy: float
    mape_pct: float
    rde_pct: float
    gpr: Optional[float] = None  # populated only for the GPR subset


# ── Energy binning ──────────────────────────────────────────────────────────


def _bin_by_fixed_edges(
    energies_mev: np.ndarray,
    values: np.ndarray,
    edges: Sequence[float],
) -> Dict[str, float]:
    """Mean of ``values`` per fixed-edge energy bin."""
    edges_arr = np.asarray(edges, dtype=float)
    indices = np.digitize(energies_mev, edges_arr) - 1
    out: Dict[str, float] = {}
    for k in range(len(edges_arr) - 1):
        mask = indices == k
        if mask.any():
            out[f"{edges_arr[k]:.0f}-{edges_arr[k + 1]:.0f}"] = float(values[mask].mean())
    return out


def _bin_by_quantile(
    energies_mev: np.ndarray,
    values: np.ndarray,
    n_bins: int,
) -> Dict[str, Any]:
    """Mean of ``values`` per quantile energy bin (returns edges + means)."""
    if n_bins < 1 or energies_mev.size < n_bins:
        return {"edges": [], "means": {}}
    edges = np.quantile(energies_mev, np.linspace(0.0, 1.0, n_bins + 1))
    indices = np.clip(np.digitize(energies_mev, edges) - 1, 0, n_bins - 1)
    means: Dict[str, float] = {}
    for k in range(n_bins):
        mask = indices == k
        if mask.any():
            means[f"q{k}"] = float(values[mask].mean())
    return {"edges": edges.tolist(), "means": means}


# ── Worst-K samples ─────────────────────────────────────────────────────────


def _worst_k_records(
    sample_ids: Sequence[str],
    samples: Sequence[_SampleMetrics],
    k: int,
) -> List[Dict[str, Any]]:
    indexed = sorted(
        enumerate(samples), key=lambda e: e[1].mape_pct, reverse=True
    )[:k]
    return [
        {
            "sample_id": sample_ids[i],
            "energy_mev": s.energy_mev,
            "rmse_gy": s.rmse_gy,
            "mape_pct": s.mape_pct,
            "rde_pct": s.rde_pct,
        }
        for i, s in indexed
    ]
