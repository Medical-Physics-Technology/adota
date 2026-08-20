"""Analysis for the beamlet-angle robustness grids (R3).

For each generated experiment dir: run ADoTA inference on every beamlet, compute
the gamma pass rate vs the MC dose, and lay the per-angle GPR into an 18x18 grid
indexed by ``grid_index``. Supports per-patient grids and per-site aggregation
(per-cell mean across a site's patients), and computes the shared color scale
(vmin/vmax) per gamma criterion across all comparable panels.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.beamlets.inference import InferenceConfig, discover_spot_ids, run_inference
from src.metrics.gamma_pass_rate import gamma_index

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GammaCriterion:
    dose: float          # dose-difference threshold [%]
    dist: float          # distance-to-agreement [mm]
    cutoff: float        # low-dose cutoff [%]

    @property
    def key(self) -> str:
        # no '.' -> it would be mistaken for a file suffix (0.1 -> "0p1")
        return f"g{self.dose:g}_{self.dist:g}_{self.cutoff:g}".replace(".", "p")

    @property
    def cbar_label(self) -> str:
        return f"Γ({self.dose:g}%, {self.dist:g}mm, {self.cutoff:g}%) [%]"


@dataclass
class Panel:
    anatomy: str
    energy: float
    patient: Optional[str]                # None for aggregated panels
    n_patients: int
    grids: Dict[str, np.ndarray]          # criterion.key -> (n, n) GPR grid


def _load_pred_as_mc_layout(pred_path: Path, mc_shape) -> np.ndarray:
    """Load ``{stem}_ds_pred.npy`` (1,1,D,H,W) and orient to the MC crop (H,W,D)."""
    ad = np.squeeze(np.load(pred_path)).astype(np.float64)  # (D, H, W)
    ad = np.moveaxis(ad, 0, -1)                             # (H, W, D)
    if ad.shape != tuple(mc_shape):
        raise ValueError(f"prediction shape {ad.shape} != MC {tuple(mc_shape)}")
    return ad


def _bbox_crop(mc: np.ndarray, ad: np.ndarray, margin: int, low_frac: float = 0.01):
    """Crop both volumes to the high-dose bounding box (+margin) to cut gamma cost.

    Local gamma only searches within the distance threshold, so trimming the
    all-zero region outside the beam (dilated by margin >= max distance) leaves
    the pass rate unchanged while removing most of the (60,60,320) voxels.
    """
    peak = float(mc.max())
    mask = mc > low_frac * peak
    if not mask.any():
        return mc, ad
    lo = np.maximum(np.argwhere(mask).min(0) - margin, 0)
    hi = np.minimum(np.argwhere(mask).max(0) + margin + 1, mc.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return mc[sl], ad[sl]


# Interpolation fraction for pymedphys gamma (matches the per-layer GPR figures).
_INTERP_FRACTION = 5
_PEAK_PCTL = 99.5


def _score_stem(task: tuple) -> tuple:
    """Worker: score one beamlet on all criteria (runs in a subprocess)."""
    exp_dir, stem, criteria, grid_n = task
    exp_dir = Path(exp_dir)
    logging.getLogger("pymedphys").setLevel(logging.WARNING)
    sr = json.loads((exp_dir / f"{stem}_sim_res.json").read_text())
    ix, iy = sr["grid_index"]
    if not (0 <= ix < grid_n and 0 <= iy < grid_n):
        return None
    pred_path = exp_dir / f"{stem}_ds_pred.npy"
    if not pred_path.exists():
        return None
    mc = np.load(exp_dir / f"{stem}_ds.npy").astype(np.float64)
    ad = _load_pred_as_mc_layout(pred_path, mc.shape)
    pos = mc > 0
    peak = float(np.percentile(mc[pos], _PEAK_PCTL)) if pos.any() else float(mc.max())
    margin = int(np.ceil(max(c.dist for c in criteria))) + 2
    mc, ad = _bbox_crop(mc, ad, margin)
    scale = {"y_max": peak, "y_min": 0.0}
    result = {}
    for c in criteria:
        if peak <= 0:
            result[c.key] = float("nan"); continue
        gp = {"dose_percent_threshold": c.dose, "distance_mm_threshold": c.dist,
              "interp_fraction": _INTERP_FRACTION, "max_gamma": 2,
              "lower_percent_dose_cutoff": c.cutoff, "random_subset": None,
              "local_gamma": True, "quiet": True}
        try:
            _, gpr = gamma_index(mc.copy(), ad.copy(), scale, gp, (1.0, 1.0, 1.0))
            result[c.key] = float(gpr[0]) * 100.0
        except Exception:
            result[c.key] = float("nan")
    return ix, iy, result, (sr["anatomy"], float(sr["initial_energy"]), sr["patient_id"])


def infer_dir(exp_dir: Path, model, device, batch_size: int = 56) -> None:
    """Run ADoTA inference over a dir (GPU), writing ``{stem}_ds_pred.npy``."""
    run_inference(Path(exp_dir), model, device,
                  InferenceConfig(batch_size=batch_size, grid_factor=1))


def score_dir_grids(
    exp_dir: Path, criteria: Sequence[GammaCriterion], grid_n: int,
    n_workers: Optional[int] = None,
) -> Panel:
    """Parallel per-beamlet gamma for one dir (CPU pool, no model); return its Panel.

    Requires ``{stem}_ds_pred.npy`` to already exist (see :func:`infer_dir`).
    """
    exp_dir = Path(exp_dir)
    stems = discover_spot_ids(exp_dir)
    grids = {c.key: np.full((grid_n, grid_n), np.nan) for c in criteria}
    crit = tuple(criteria)
    tasks = [(str(exp_dir), s, crit, grid_n) for s in stems]
    workers = n_workers or max(1, min(os.cpu_count() - 2, 46))
    meta = None
    n_used = 0
    # BLAS threads are pinned to 1 (env, set by the driver) so the process pool scales.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_score_stem, tasks, chunksize=1):
            if res is None:
                continue
            ix, iy, result, m = res
            if meta is None:
                meta = m
            for c in criteria:
                grids[c.key][ix, iy] = result[c.key]
            n_used += 1
    if meta is None:
        raise FileNotFoundError(f"no usable beamlets in {exp_dir}")
    logger.info("  %s: %d beamlets scored (%d workers)", exp_dir.name, n_used, workers)
    return Panel(anatomy=meta[0], energy=meta[1], patient=meta[2], n_patients=1, grids=grids)


def aggregate_panels(panels: List[Panel]) -> Panel:
    """Per-cell mean GPR across panels (same anatomy+energy); ignores NaNs."""
    assert panels, "no panels to aggregate"
    keys = panels[0].grids.keys()
    grids = {}
    for k in keys:
        stack = np.stack([p.grids[k] for p in panels], axis=0)
        with np.errstate(invalid="ignore"):
            grids[k] = np.nanmean(stack, axis=0)
    return Panel(anatomy=panels[0].anatomy, energy=panels[0].energy, patient=None,
                 n_patients=len(panels), grids=grids)


def extreme_indices(grid: np.ndarray, n: int = 3):
    """Return the ``n`` lowest- and ``n`` highest-GPR cells as ``(ix, iy, gpr)``."""
    finite = [(ix, iy, float(grid[ix, iy]))
              for ix in range(grid.shape[0]) for iy in range(grid.shape[1])
              if np.isfinite(grid[ix, iy])]
    finite.sort(key=lambda t: t[2])
    worst = finite[:n]
    best = list(reversed(finite[-n:]))
    return worst, best


def save_panel_grids(panel: "Panel", out_dir: Path) -> Path:
    """Persist a panel's per-cell GPR grids so examples can be re-rendered cheaply."""
    who = panel.patient if panel.patient is not None else f"aggregate{panel.n_patients}"
    path = Path(out_dir) / f"{panel.anatomy}_{who}_e{int(round(panel.energy))}_grids.npz"
    np.savez(path, energy=panel.energy, anatomy=panel.anatomy,
             patient=str(panel.patient), **panel.grids)
    return path


def render_beamlet_examples(
    exp_dir: Path, grid: np.ndarray, crit: GammaCriterion, model, device, scale: dict,
    out_dir: Path, n: int = 3, beamlet_shape: bool = True,
) -> List[Path]:
    """Render publication_figure for the ``n`` worst and ``n`` best beamlets (by GPR)."""
    import torch

    from src.figures.single_beam import publication_figure
    from src.loaders.dir_based import get_single_record
    from src.metrics.classic import calculate_pure_mape, calculate_rmse
    from src.utils.scallers import inverse_minmax
    from src.utils.unit_conversions import to_gy

    exp_dir = Path(exp_dir)
    gamma_params = {"dose_percent_threshold": crit.dose,
                    "distance_mm_threshold": crit.dist,
                    "lower_percent_dose_cutoff": crit.cutoff}
    worst, best = extreme_indices(grid, n)
    written: List[Path] = []
    for rank_label, group in (("worst", worst), ("best", best)):
        for r, (ix, iy, gpr) in enumerate(group, start=1):
            stem = f"a{ix:02d}_{iy:02d}"
            x, e_norm, y = get_single_record(stem, str(exp_dir), scale=scale,
                                             normalize_flux=True,
                                             downsampling_method="interpolation")
            x = x.to(device)
            with torch.no_grad():
                y_pred = model(x.unsqueeze(0), e_norm.to(device).unsqueeze(0))[0]
            gt = inverse_minmax(y.cpu().numpy(), scale["min_ds"], scale["max_ds"]).squeeze()
            pred = inverse_minmax(y_pred.cpu().numpy(), scale["min_ds"], scale["max_ds"]).squeeze()
            rmse = float(calculate_rmse(to_gy(pred), to_gy(gt)))
            mask = gt > 0.1 * gt.max()
            mape = float(calculate_pure_mape(gt[mask], pred[mask])) if mask.any() else float("nan")
            sr = json.loads((exp_dir / f"{stem}_sim_res.json").read_text())
            e_mev = float(sr["initial_energy"])
            fname = (f"{sr['anatomy']}_{sr['patient_id']}_e{int(round(e_mev))}_{crit.key}"
                     f"_{rank_label}{r}_{stem}_gpr{gpr:.1f}")
            paths = publication_figure(
                x.cpu().numpy(), e_mev, gt, pred, str(Path(out_dir) / fname),
                rmse, mape, float(gpr), gamma_params=gamma_params, beamlet_shape=beamlet_shape)
            written.extend(paths or [Path(out_dir) / f"{fname}.png"])
            logger.info("  example %s%d %s: GPR=%.1f MAPE=%.2f", rank_label, r, stem, gpr, mape)
    return written


def shared_scale_per_criterion(
    panels: Sequence[Panel], criteria: Sequence[GammaCriterion],
    low_percentile: float = 0.0, vmax_cap: float = 100.0,
) -> Dict[str, Tuple[float, float]]:
    """vmin/vmax per criterion across ALL panels (comparable panels share a scale).

    ``low_percentile`` sets vmin to that percentile of the pooled GPR values
    rather than the raw minimum, so a few low-outlier beamlets do not compress
    the colour range and wash out the (high) bulk of the grid. vmax is capped at
    ``vmax_cap`` (100%). This keeps the scale stable and presents the pass rates
    fairly -- the high values stay clearly high.
    """
    out = {}
    for c in criteria:
        vals = np.concatenate([p.grids[c.key][np.isfinite(p.grids[c.key])].ravel()
                               for p in panels if c.key in p.grids])
        if vals.size == 0:
            out[c.key] = (0.0, vmax_cap)
            continue
        vmin = float(np.percentile(vals, low_percentile)) if low_percentile > 0 else float(vals.min())
        vmax = min(float(vmax_cap), float(vals.max()))
        out[c.key] = (vmin, vmax)
    return out
