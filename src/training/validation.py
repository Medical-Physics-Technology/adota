"""Validation-time analytics for ADoTA training.

A single call to :func:`evaluate_validation` runs one full pass over the
validation loader and returns a structured dictionary containing:

- Mean training-loss components (``LMSE``, ``LPS``, combined).
- Per-sample physical metrics (RMSE [Gy], MAPE [%], RDE [%]) computed
  on the de-normalized dose.
- Per-energy breakdowns of those metrics using both fixed bin edges and
  quantile bins.
- The top-K worst samples by MAPE (id, energy, RMSE, MAPE, RDE).
- Optionally, gamma pass rate on a random subset of the validation set.

A separate :func:`save_attention_snapshot` helper persists attention maps
from a fixed canary sample for transformer interpretability.

All artifacts (per-sample CSV, worst-K JSON, attention maps) are written
under ``run_dir/validation/epoch_NNNN/`` and ``run_dir/attention/``
respectively.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.adota.config import denormalize_energy
from src.metrics.classic import (
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from src.training.losses import LMSE, LPS
from src.utils.scallers import inverse_minmax
from src.utils.serialization import NumpyEncoder
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)


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


# ── GPR helper ──────────────────────────────────────────────────────────────


def _gamma_pass_rate(
    y_pred_gy: np.ndarray,
    y_gt_gy: np.ndarray,
    resolution_mm: Tuple[float, float, float],
    gamma_params: Dict[str, Any],
) -> Optional[float]:
    """Compute the global gamma pass rate (fraction of voxels with γ ≤ 1).

    Returns ``None`` if pymedphys is unavailable or the computation fails;
    the caller logs and continues so a transient GPR failure can't kill
    the training run.
    """
    try:
        from pymedphys import gamma  # local import keeps the dep optional
    except ImportError:  # pragma: no cover
        logger.warning("pymedphys not installed; skipping GPR.")
        return None

    axes = tuple(
        np.arange(y_gt_gy.shape[d]) * resolution_mm[d] for d in range(3)
    )
    try:
        gamma_values = gamma(axes, y_gt_gy, axes, y_pred_gy, **gamma_params)
    except Exception as exc:  # pragma: no cover - pymedphys edge cases
        logger.warning("GPR computation failed: %s", exc)
        return None

    gamma_values = np.nan_to_num(gamma_values, nan=0.0)
    valid = gamma_values > 0
    if not valid.any():
        return None
    return float(np.sum(gamma_values[valid] <= 1.0) / np.count_nonzero(valid))


# ── Main entry point ────────────────────────────────────────────────────────


def evaluate_validation(
    *,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    val_sample_ids: Sequence[str],
    device: torch.device,
    config: Any,  # TrainingConfig (avoid hard import to break circularity)
    weight_mse: torch.Tensor,
    weight_ps: torch.Tensor,
    compute_gpr: bool,
    gpr_subset_indices: Sequence[int],
    run_dir: Path,
    epoch: int,
) -> Dict[str, Any]:
    """Run validation and produce structured metrics + side artifacts.

    Args:
        model: Already moved to ``device`` and in any mode (``.eval()``
            is called internally).
        val_dataloader: Yields ``(x, energy, y)`` tuples; ``shuffle=False``
            assumed so ``val_sample_ids`` aligns with the iteration order.
        val_sample_ids: Sample IDs in the same order as ``val_dataloader``.
        device: Compute device.
        config: ``TrainingConfig`` instance (we read ``scale``,
            ``energy_bins_fixed``, ``energy_bins_quantile_n``,
            ``worst_k_samples``, ``gpr_resolution_mm``, ``gamma_params``,
            ``lps_dx_mm``, ``lps_dy_mm`` from it).
        weight_mse, weight_ps: Loss balancing weights used for the
            combined validation loss.
        compute_gpr: Whether to compute GPR this epoch.
        gpr_subset_indices: Indices (into the iteration order) of the
            samples to run GPR on. Ignored if ``compute_gpr`` is False.
        run_dir: Root run directory; per-epoch artifacts go under
            ``run_dir/validation/epoch_NNNN/``.
        epoch: Current epoch (used for filenames).

    Returns:
        Dictionary with the aggregated metrics, ready to feed into
        :class:`~src.training.run.MetricsLog`.
    """
    model.eval()
    scale = config.scale

    loss_mse_fn = LMSE()
    loss_ps_fn = LPS(dx=config.lps_dx_mm, dy=config.lps_dy_mm)

    samples: List[_SampleMetrics] = []
    sample_ids_seen: List[str] = []
    gpr_subset_set = set(int(i) for i in gpr_subset_indices) if compute_gpr else set()

    # GPR progress heartbeat: ~10 lines total across the whole subset.
    from src.training.run import log_phase

    gpr_total = len(gpr_subset_set)
    gpr_heartbeat_every = max(1, gpr_total // 10) if gpr_total else 1
    gpr_done = 0
    gpr_running_sum = 0.0
    gpr_running_n = 0
    gpr_start_time = perf_counter()

    flat_idx = 0
    with torch.no_grad():
        for batch in val_dataloader:
            x, energy, y = batch
            x = x.to(device, non_blocking=True)
            energy = energy.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x, energy)
            y_pred = outputs[0] if isinstance(outputs, tuple) else outputs

            # Per-sample loop so we can index sample IDs and compute
            # GPR on a subset without rebuilding the batch boundaries.
            batch_size = y_pred.shape[0]
            for b in range(batch_size):
                y_pred_b = y_pred[b : b + 1]
                y_b = y[b : b + 1]

                l_mse = float(loss_mse_fn(y_pred_b, y_b).item())
                l_ps = float(loss_ps_fn(y_pred_b, y_b).item())

                y_pred_np = inverse_minmax(
                    y_pred_b.detach().cpu().numpy(),
                    scale["min_ds"],
                    scale["max_ds"],
                )
                y_np = inverse_minmax(
                    y_b.detach().cpu().numpy(), scale["min_ds"], scale["max_ds"]
                )
                y_pred_gy = to_gy(y_pred_np)
                y_gt_gy = to_gy(y_np)

                rmse = float(calculate_rmse(y_pred_gy, y_gt_gy))
                mask = y_pred_np > 0.1 * float(np.max(y_pred_np))
                mape = (
                    float(calculate_pure_mape(y_np[mask], y_pred_np[mask]))
                    if mask.any()
                    else 0.0
                )
                rde = float(calculate_relative_dose_error(y_pred_gy, y_gt_gy))
                e_mev = denormalize_energy(float(energy[b].item()), scale)

                gpr_value: Optional[float] = None
                if flat_idx in gpr_subset_set:
                    gpr_value = _gamma_pass_rate(
                        y_pred_gy.squeeze(),
                        y_gt_gy.squeeze(),
                        config.gpr_resolution_mm,
                        config.gamma_params,
                    )
                    gpr_done += 1
                    if gpr_value is not None:
                        gpr_running_sum += gpr_value
                        gpr_running_n += 1

                    if gpr_done % gpr_heartbeat_every == 0 or gpr_done == gpr_total:
                        elapsed = perf_counter() - gpr_start_time
                        per_sample = elapsed / max(1, gpr_done)
                        remaining = per_sample * (gpr_total - gpr_done)
                        running_mean = (
                            gpr_running_sum / gpr_running_n
                            if gpr_running_n > 0
                            else float("nan")
                        )
                        pct = 100.0 * gpr_done / gpr_total
                        log_phase(
                            "GPR",
                            f"{gpr_done}/{gpr_total} ({pct:.0f}%)  "
                            f"running mean={running_mean:.4f}  "
                            f"ETA {int(remaining // 60):02d}:"
                            f"{int(remaining % 60):02d}",
                        )

                samples.append(
                    _SampleMetrics(
                        energy_mev=e_mev,
                        loss_mse=l_mse,
                        loss_ps=l_ps,
                        rmse_gy=rmse,
                        mape_pct=mape,
                        rde_pct=rde,
                        gpr=gpr_value,
                    )
                )
                if flat_idx < len(val_sample_ids):
                    sample_ids_seen.append(val_sample_ids[flat_idx])
                else:
                    sample_ids_seen.append(f"idx_{flat_idx}")
                flat_idx += 1

    if not samples:
        return {"n_samples": 0}

    # ── Aggregates ───────────────────────────────────────────────────
    energies = np.array([s.energy_mev for s in samples])
    rmse_arr = np.array([s.rmse_gy for s in samples])
    mape_arr = np.array([s.mape_pct for s in samples])
    rde_arr = np.array([s.rde_pct for s in samples])
    l_mse_arr = np.array([s.loss_mse for s in samples])
    l_ps_arr = np.array([s.loss_ps for s in samples])
    gpr_values = [s.gpr for s in samples if s.gpr is not None]

    w_mse = float(weight_mse.item() if torch.is_tensor(weight_mse) else weight_mse)
    w_ps = float(weight_ps.item() if torch.is_tensor(weight_ps) else weight_ps)
    combined_loss = w_mse * l_mse_arr + w_ps * l_ps_arr

    metrics: Dict[str, Any] = {
        "n_samples": len(samples),
        "loss_combined_mean": float(combined_loss.mean()),
        "loss_mse_mean": float(l_mse_arr.mean()),
        "loss_ps_mean": float(l_ps_arr.mean()),
        "rmse_gy_mean": float(rmse_arr.mean()),
        "rmse_gy_std": float(rmse_arr.std()),
        "mape_pct_mean": float(mape_arr.mean()),
        "mape_pct_std": float(mape_arr.std()),
        "rde_pct_mean": float(rde_arr.mean()),
        "rde_pct_std": float(rde_arr.std()),
    }

    if gpr_values:
        gpr_arr = np.array(gpr_values)
        metrics.update(
            {
                "gpr_mean": float(gpr_arr.mean()),
                "gpr_std": float(gpr_arr.std()),
                "gpr_n_samples": len(gpr_values),
            }
        )

    # ── Per-energy breakdowns ────────────────────────────────────────
    fixed_edges = config.energy_bins_fixed
    metrics["per_energy_fixed"] = {
        "rmse_gy": _bin_by_fixed_edges(energies, rmse_arr, fixed_edges),
        "mape_pct": _bin_by_fixed_edges(energies, mape_arr, fixed_edges),
        "rde_pct": _bin_by_fixed_edges(energies, rde_arr, fixed_edges),
    }
    metrics["per_energy_quantile"] = {
        "rmse_gy": _bin_by_quantile(energies, rmse_arr, config.energy_bins_quantile_n),
        "mape_pct": _bin_by_quantile(energies, mape_arr, config.energy_bins_quantile_n),
        "rde_pct": _bin_by_quantile(energies, rde_arr, config.energy_bins_quantile_n),
    }

    # ── Worst-K ─────────────────────────────────────────────────────
    metrics["worst_k"] = _worst_k_records(
        sample_ids_seen, samples, config.worst_k_samples
    )

    # ── Side artifacts ──────────────────────────────────────────────
    val_dir = run_dir / "validation" / f"epoch_{epoch:04d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample CSV (one row per validation sample)
    csv_path = val_dir / "per_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "energy_mev",
                "loss_mse",
                "loss_ps",
                "rmse_gy",
                "mape_pct",
                "rde_pct",
                "gpr",
            ]
        )
        for sid, s in zip(sample_ids_seen, samples):
            writer.writerow(
                [
                    sid,
                    f"{s.energy_mev:.4f}",
                    f"{s.loss_mse:.6e}",
                    f"{s.loss_ps:.6e}",
                    f"{s.rmse_gy:.6e}",
                    f"{s.mape_pct:.6f}",
                    f"{s.rde_pct:.6f}",
                    f"{s.gpr:.6f}" if s.gpr is not None else "",
                ]
            )

    with open(val_dir / "summary.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    return metrics


# ── Attention canary snapshot ───────────────────────────────────────────────


def save_attention_snapshot(
    *,
    model: torch.nn.Module,
    canary_x: torch.Tensor,
    canary_energy: torch.Tensor,
    run_dir: Path,
    epoch: int,
) -> Optional[Path]:
    """Save attention maps for a fixed validation sample.

    Skipped silently when the model has no transformer blocks (the
    forward pass returns a zero placeholder in that case).

    Args:
        model: Model in eval mode (we don't toggle modes here).
        canary_x: Single-sample input ``(1, C, D, H, W)`` already on
            the right device.
        canary_energy: Single-sample energy ``(1, 1)`` already on the
            right device.
        run_dir: Run directory; the file is written to
            ``run_dir/attention/epoch_NNNN.npy``.
        epoch: Current epoch.

    Returns:
        Path to the saved file, or ``None`` if attention is not
        meaningful for this model.
    """
    num_transformers = getattr(model, "num_transformers", 0)
    if num_transformers == 0:
        return None

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(canary_x, canary_energy)
    finally:
        if was_training:
            model.train()

    if not isinstance(outputs, tuple) or len(outputs) < 2:
        return None
    attn = outputs[1]
    out_path = run_dir / "attention" / f"epoch_{epoch:04d}.npy"
    np.save(out_path, attn.detach().cpu().numpy())
    return out_path
