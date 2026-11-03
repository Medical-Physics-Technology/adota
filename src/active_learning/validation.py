"""The frozen yardstick: how it is drawn, and what is measured on it.

The validation set is generated once, before cycle 1, and never changes. It is drawn
by :func:`select_stratified`: equal counts per anatomy and energy layer, then per
patient and per field inside those, among the valid candidates, **without the
difficulty score**. The score is under revision (a model-independent version is
planned), and a yardstick drawn by one score would change with it; per-decile curves
are still available from the ``score_decile`` column computed after the draw.
:func:`select_balanced`, the earlier recipe with equal counts per score decile, is
kept for the ``stratified_score`` strategy and for comparison. The pre-existing
held-out sets stay as an independent second reading.

What is measured on it: gamma pass rate (mean, and the tail that actually matters -
the fraction below 95 percent and the 5th percentile), MAPE, relative dose error, and
**dR80**, the distal range error. dR80 is not computed anywhere in training validation
today; it comes from :mod:`src.metrics.range_metrics`, which returns NaN rather than a
number when the depth-dose curve never falls back below the level, so a beamlet whose
dose leaves the crop is excluded from the range statistics instead of poisoning them.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.evaluation.engine import evaluate
from src.evaluation.sources import MultiDirSource
from src.metrics.classic import calculate_pure_mape, calculate_relative_dose_error
from src.metrics.gamma_pass_rate import gamma_index
from src.metrics.range_metrics import compute_range_metrics

logger = logging.getLogger(__name__)

MODEL_DZ_MM = 2.0
"""Depth spacing of the model grid: the 320 mm crop resampled to 160 voxels."""


def _water_fill(sizes: Sequence[int], n: int) -> List[int]:
    """Allocate ``n`` over groups as evenly as their sizes allow: every group gets
    the same count, except those too small, which give all they have; the rest is
    spread over the others. Remainders go to the groups with the most spare."""
    sizes = [int(s) for s in sizes]
    n = min(int(n), sum(sizes))
    take = [0] * len(sizes)
    open_ = [i for i, s in enumerate(sizes) if s > 0]
    left = n
    while left > 0 and open_:
        share = left // len(open_)
        if share == 0:
            for i in sorted(open_, key=lambda i: sizes[i] - take[i], reverse=True)[:left]:
                take[i] += 1
            break
        for i in list(open_):
            add = min(share, sizes[i] - take[i])
            take[i] += add
            left -= add
            if take[i] == sizes[i]:
                open_.remove(i)
    return take


def select_stratified(table: pd.DataFrame, n: int, strata: Sequence[str], *,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` valid candidates with equal counts per level of each stratum, in
    order: ``n`` is split evenly over the first column's values, each share evenly
    over the second column's values inside it, and so on; candidates are drawn at
    random inside the last level.

    A level too small for its share gives everything it has and its shortfall goes
    to its siblings, so the anatomy x energy balance is kept before the per-patient
    one. No score is used.
    """
    valid = table[table["valid"].astype(bool)]
    if len(valid) < n:
        raise ValueError(f"only {len(valid)} valid candidates for a validation set of {n}")

    def draw(part: pd.DataFrame, want: int, depth: int) -> List[pd.DataFrame]:
        if want <= 0:
            return []
        if depth == len(strata):
            return [part.sample(want, random_state=int(rng.integers(2 ** 31)))]
        groups = list(part.groupby(strata[depth], sort=True))
        shares = _water_fill([len(g) for _, g in groups], want)
        out: List[pd.DataFrame] = []
        for (_, group), share in zip(groups, shares):
            out += draw(group, share, depth + 1)
        return out

    chosen = pd.concat(draw(valid, int(n), 0), ignore_index=True)
    logger.info("validation selection: %d beamlets, %s", len(chosen),
                chosen.groupby(list(strata[:2])).size().to_dict())
    return chosen


def select_balanced(
    table: pd.DataFrame,
    n: int,
    *,
    rng: np.random.Generator,
    score_column: str = "score_full",
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Draw a difficulty-balanced set: equal per score decile, per anatomy, per energy.

    Deciles are computed over the whole valid population, so a cell that holds only
    easy beamlets contributes to the easy deciles and nothing else; the per-cell
    budget is then spread over the deciles that cell actually has, and whatever a
    thin cell cannot supply is redistributed rather than silently lost.
    """
    valid = table[table["valid"].astype(bool)].reset_index(drop=True)
    if len(valid) < n:
        raise ValueError(f"only {len(valid)} valid candidates for a validation set of {n}")
    valid = valid.assign(
        _decile=pd.qcut(valid[score_column], n_deciles, labels=False, duplicates="drop"))
    cells = list(valid.groupby(["anatomy", "energy_mev"], sort=True))
    picked: List[pd.DataFrame] = []
    shortfall = 0
    per_cell = n // max(len(cells), 1)
    for i, (_, cell) in enumerate(cells):
        want = per_cell + (1 if i < n - per_cell * len(cells) else 0) + shortfall
        deciles = list(cell.groupby("_decile", sort=True))
        got: List[pd.DataFrame] = []
        per_decile = want // max(len(deciles), 1)
        remainder = want - per_decile * len(deciles)
        for j, (_, group) in enumerate(deciles):
            take = min(per_decile + (1 if j < remainder else 0), len(group))
            if take:
                got.append(group.sample(take, random_state=int(rng.integers(2 ** 31))))
        chosen_cell = pd.concat(got) if got else cell.iloc[:0]
        shortfall = want - len(chosen_cell)
        picked.append(chosen_cell)
    out = pd.concat(picked, ignore_index=True)
    if shortfall > 0:
        rest = valid[~valid["candidate_id"].isin(set(out["candidate_id"]))]
        out = pd.concat([out, rest.sample(min(shortfall, len(rest)),
                                          random_state=int(rng.integers(2 ** 31)))],
                        ignore_index=True)
    logger.info("validation selection: %d beamlets over %d anatomy-energy cells, "
                "decile counts %s", len(out), len(cells),
                out["_decile"].value_counts().sort_index().to_dict())
    return out.rename(columns={"_decile": "score_decile"})


def _idd(dose: np.ndarray) -> np.ndarray:
    """Integrated depth dose of a model-frame ``(1, D, H, W)`` or ``(D, H, W)`` array."""
    arr = np.squeeze(np.asarray(dose))
    return arr.sum(axis=(1, 2))


def evaluate_frozen_set(
    model: torch.nn.Module,
    entries: Sequence[Tuple[str, str]],
    *,
    device: torch.device,
    scale: dict,
    gamma_params: dict,
    resolution_mm: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    gamma_cutoff_percent: float = 10.0,
    gamma_backend: str = "torch",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Per-sample GPR, MAPE, RDE and dR80 over a frozen beamlet set.

    Args:
        model: The checkpoint under test, already on ``device`` and in eval mode.
        entries: ``(directory, stem)`` pairs naming the frozen beamlets.
        device: Where inference and, with the torch backend, gamma run.
        scale: The run's MinMax scaling, for de-normalising both doses.
        gamma_params: Criteria dict; the headline is 3%/3mm with a 10% cutoff.
        resolution_mm: Model-grid spacing, ``(2, 2, 2)`` by construction.
        gamma_cutoff_percent: Low-dose cutoff applied before the gamma call.
        gamma_backend: ``"torch"`` (GPU, the practical choice at this many samples)
            or ``"pymedphys"``.

    Returns:
        One row per sample: ids, and the four metric families.
    """
    source = MultiDirSource(entries, scale=scale)
    backend_options = {"device": device} if gamma_backend == "torch" else None

    def per_sample(ctx):
        y_np, y_pred_np = ctx.denorm(scale)
        y_np = np.squeeze(y_np)
        y_pred_np = np.squeeze(y_pred_np)
        mask = y_pred_np > 0.1 * float(np.max(y_pred_np))
        mape = float(calculate_pure_mape(y_np[mask], y_pred_np[mask])) if mask.any() else np.nan
        rde = float(calculate_relative_dose_error(y_pred_np, y_np))
        peak = float(np.max(y_np))
        try:
            _, rates = gamma_index(
                y_np.copy(), y_pred_np.copy(), {"y_max": peak, "y_min": 0.0},
                dict(gamma_params), resolution_mm, cutoff=gamma_cutoff_percent,
                backend=gamma_backend, backend_options=backend_options)
            gpr = float(rates[0])
        except Exception as exc:  # a single bad beamlet must not end the evaluation
            logger.warning("gamma failed for %s: %s", ctx.sample_id, exc)
            gpr = np.nan
        gt_range = compute_range_metrics(_idd(y_np), MODEL_DZ_MM)
        pred_range = compute_range_metrics(_idd(y_pred_np), MODEL_DZ_MM)
        return {
            "sample_id": ctx.sample_id,
            "directory": ctx.extra.get("directory", ""),
            "gpr": gpr, "mape_pct": mape, "rde_pct": rde,
            "r80_gt_mm": float(gt_range.r80_mm), "r80_pred_mm": float(pred_range.r80_mm),
            "dr80_mm": float(pred_range.r80_mm - gt_range.r80_mm),
            "dfw_gt_mm": float(gt_range.dfw_mm),
            "ddfw_mm": float(pred_range.dfw_mm - gt_range.dfw_mm),
            "peak_dose_gt": peak,
        }

    rows = evaluate(model, source, device=device, per_sample_fn=per_sample,
                    show_progress=show_progress, desc="AL validation")
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, gpr_target: float = 0.95) -> Dict[str, float]:
    """The headline numbers, tails included.

    The mean gamma pass rate is the number that gets quoted; the fraction of beamlets
    below the target and the 5th percentile are the numbers that move when a model
    stops failing on hard geometry, which is what the loop is for.
    """
    gpr = frame["gpr"].to_numpy(dtype=float)
    gpr = gpr[np.isfinite(gpr)]
    dr80 = frame["dr80_mm"].to_numpy(dtype=float)
    dr80 = dr80[np.isfinite(dr80)]
    out: Dict[str, float] = {
        "n": float(len(frame)),
        "n_gpr": float(len(gpr)),
        "gpr_mean": float(np.mean(gpr)) if len(gpr) else float("nan"),
        "gpr_p05": float(np.percentile(gpr, 5)) if len(gpr) else float("nan"),
        f"gpr_frac_below_{int(gpr_target * 100)}": (
            float(np.mean(gpr < gpr_target)) if len(gpr) else float("nan")),
        "mape_pct_mean": float(frame["mape_pct"].mean()),
        "mape_pct_p95": float(frame["mape_pct"].quantile(0.95)),
        "rde_pct_mean": float(frame["rde_pct"].mean()),
        "n_dr80": float(len(dr80)),
        "dr80_median_mm": float(np.median(dr80)) if len(dr80) else float("nan"),
        "abs_dr80_median_mm": float(np.median(np.abs(dr80))) if len(dr80) else float("nan"),
        "abs_dr80_p95_mm": (float(np.percentile(np.abs(dr80), 95))
                            if len(dr80) else float("nan")),
    }
    return out


def summarise_by(frame: pd.DataFrame, table: Optional[pd.DataFrame] = None,
                 column: str = "score_decile") -> pd.DataFrame:
    """Per-group headline numbers, for the per-decile (or per-anatomy) curves.

    ``table`` is the validation manifest; it carries the grouping columns, which the
    metric frame does not, and is joined on the sample id.
    """
    merged = frame
    if table is not None:
        merged = frame.merge(table[["candidate_id", column]].rename(
            columns={"candidate_id": "sample_id"}), on="sample_id", how="left")
    if column not in merged.columns:
        raise KeyError(f"{column!r} is in neither the metrics frame nor the manifest")
    rows = []
    for key, group in merged.groupby(column, sort=True):
        rows.append({column: key, **summarise(group)})
    return pd.DataFrame(rows)
