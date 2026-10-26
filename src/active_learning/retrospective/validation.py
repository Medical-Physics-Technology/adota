"""The frozen validation set as it is measured: the fixed evaluation subsample,
the per-epoch loss, and the cycle metric set with dR80.

Cadence, so epochs-to-quality can be read off the log afterwards:

- the loss on the full validation set every epoch;
- the full metric set every ``eval_every_n_epochs`` on a fixed subsample of the
  validation set, drawn once with a fixed seed and identical across strategies;
- the full metric set on the whole validation set at every cycle boundary.

Metrics per sample: gamma pass rate on the torch backend, MAPE, relative dose
error, and **dR80** from :mod:`src.metrics.range_metrics` with its plateau
guard (undefined when the integrated depth dose never falls back below the
level). The tails are summarised by :func:`src.active_learning.validation.summarise`:
the fraction of records under 95 percent pass rate, the 5th percentile pass rate,
the 95th percentile of the absolute range error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.active_learning.validation import MODEL_DZ_MM, summarise
from src.adota.config import denormalize_energy
from src.metrics.classic import calculate_pure_mape, calculate_relative_dose_error
from src.metrics.gamma_pass_rate import gamma_index
from src.metrics.range_metrics import compute_range_metrics
from src.training.losses import LMSE, LPS
from src.training.validation import pick_gpr_subset
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)


def draw_eval_subsample(validation_ids: Sequence[str], size: int, seed: int) -> List[str]:
    """The fixed evaluation subsample, by the mechanism the training gamma pool
    uses; identical across strategies because the ids and the seed are."""
    positions = pick_gpr_subset(len(validation_ids), size, np.random.RandomState(seed))
    return [validation_ids[i] for i in positions]


@dataclass
class MetricSettings:
    """Everything the per-sample metrics need besides the tensors."""

    scale: Dict[str, float]
    gamma_params: Dict
    resolution_mm: Sequence[float] = (2.0, 2.0, 2.0)
    gamma_cutoff_percent: float = 10.0
    gamma_backend: str = "torch"
    lps_dx_mm: float = 2.0
    lps_dy_mm: float = 2.0

    @property
    def gamma_label(self) -> str:
        p = self.gamma_params
        return (f"{p['dose_percent_threshold']:g}pct_{p['distance_mm_threshold']:g}mm"
                f"_cutoff{self.gamma_cutoff_percent:g}pct")


def _idd(dose: np.ndarray) -> np.ndarray:
    return np.squeeze(np.asarray(dose)).sum(axis=(1, 2))


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, *, device: torch.device,
                  weight_mse: float, weight_ps: float, settings: MetricSettings) -> Dict:
    """The mean validation loss and its two components over ``loader``."""
    model.eval()
    loss_mse_fn, loss_ps_fn = LMSE(), LPS(dx=settings.lps_dx_mm, dy=settings.lps_dy_mm)
    sum_mse = sum_ps = 0.0
    n = 0
    started = perf_counter()
    for x, energy, y in loader:
        x, energy, y = (t.to(device, non_blocking=True) for t in (x, energy, y))
        y_pred = model(x, energy)[0]
        batch = int(y.shape[0])
        sum_mse += float(loss_mse_fn(y_pred, y).item()) * batch
        sum_ps += float(loss_ps_fn(y_pred, y).item()) * batch
        n += batch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mse, ps = sum_mse / max(n, 1), sum_ps / max(n, 1)
    return {"n": n, "loss_mse_mean": mse, "loss_ps_mean": ps,
            "loss_combined_mean": weight_mse * mse + weight_ps * ps,
            "seconds": perf_counter() - started}


@torch.no_grad()
def evaluate_metrics(model: torch.nn.Module, loader: DataLoader, sample_ids: Sequence[str], *,
                     device: torch.device, settings: MetricSettings,
                     desc: str = "metrics") -> pd.DataFrame:
    """Per-sample gamma pass rate, MAPE, RDE and dR80 over ``loader``.

    ``loader`` must iterate in the order of ``sample_ids`` (``shuffle=False``).
    Gamma runs on the tensors' own device through the torch backend, so the
    whole validation set is affordable at a cycle boundary.
    """
    model.eval()
    scale = settings.scale
    backend_options = {"device": device} if settings.gamma_backend == "torch" else None
    loss_mse_fn, loss_ps_fn = LMSE(), LPS(dx=settings.lps_dx_mm, dy=settings.lps_dy_mm)
    rows: List[Dict] = []
    flat = 0
    started = perf_counter()
    heartbeat = max(1, len(sample_ids) // 10)
    for x, energy, y in loader:
        x, energy, y = (t.to(device, non_blocking=True) for t in (x, energy, y))
        y_pred = model(x, energy)[0]
        for b in range(int(y.shape[0])):
            sample_id = sample_ids[flat] if flat < len(sample_ids) else f"idx_{flat}"
            flat += 1
            yb, pb = y[b:b + 1], y_pred[b:b + 1]
            y_np = inverse_minmax(yb.detach().cpu().numpy(), scale["min_ds"], scale["max_ds"])
            p_np = inverse_minmax(pb.detach().cpu().numpy(), scale["min_ds"], scale["max_ds"])
            y_np, p_np = np.squeeze(y_np), np.squeeze(p_np)
            mask = p_np > 0.1 * float(np.max(p_np))
            mape = float(calculate_pure_mape(y_np[mask], p_np[mask])) if mask.any() else np.nan
            rde = float(calculate_relative_dose_error(p_np, y_np))
            peak = float(np.max(y_np))
            try:
                # As the prospective loop does it: the cutoff is a share of this
                # record's own peak dose, and the torch backend runs on ``device``.
                _, rates = gamma_index(
                    y_np.copy(), p_np.copy(), {"y_max": peak, "y_min": 0.0},
                    dict(settings.gamma_params), tuple(settings.resolution_mm),
                    cutoff=settings.gamma_cutoff_percent, backend=settings.gamma_backend,
                    backend_options=backend_options)
                gpr = float(rates[0])
            except Exception as exc:  # one bad beamlet must not end the evaluation
                logger.warning("gamma failed for %s: %s", sample_id, exc)
                gpr = np.nan
            gt_range = compute_range_metrics(_idd(y_np), MODEL_DZ_MM)
            pred_range = compute_range_metrics(_idd(p_np), MODEL_DZ_MM)
            rows.append({
                "sample_id": sample_id,
                "energy_mev": float(denormalize_energy(float(energy[b].item()), scale)),
                "loss_mse": float(loss_mse_fn(pb, yb).item()),
                "loss_ps": float(loss_ps_fn(pb, yb).item()),
                "gpr": gpr, "mape_pct": mape, "rde_pct": rde,
                "r80_gt_mm": float(gt_range.r80_mm), "r80_pred_mm": float(pred_range.r80_mm),
                "dr80_mm": float(pred_range.r80_mm - gt_range.r80_mm),
                "peak_dose_gt": peak,
            })
            if flat % heartbeat == 0 or flat == len(sample_ids):
                finite = [r["gpr"] for r in rows if np.isfinite(r["gpr"])]
                logger.info("[%s] %d/%d  running GPR %.4f  %.0fs", desc, flat, len(sample_ids),
                            float(np.mean(finite)) if finite else float("nan"),
                            perf_counter() - started)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return pd.DataFrame(rows)


def summarise_metrics(frame: pd.DataFrame, subset_ids: Optional[Sequence[str]] = None) -> Dict:
    """The headline and tail numbers of :func:`summarise`, plus how many range
    errors were defined, optionally restricted to ``subset_ids``."""
    if subset_ids is not None:
        frame = frame[frame["sample_id"].isin(set(subset_ids))]
    out = summarise(frame)
    dr80 = frame["dr80_mm"].to_numpy(dtype=float)
    out["dr80_defined_fraction"] = float(np.isfinite(dr80).mean()) if dr80.size else float("nan")
    out["loss_mse_mean"] = float(frame["loss_mse"].mean()) if "loss_mse" in frame else float("nan")
    return out
