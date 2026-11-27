"""Is a voxel's dose error about its own material or about the beamlet it sits in?

Input: the per-beamlet sums of :mod:`src.engine_drivers.voxel_hu_error` (one cell per
beamlet, engine, dose band and 50 HU bin, holding the voxel count and the summed error).
Per engine and dose band, the voxel-weighted cell means ``y`` are explained by

* **beamlet** alone (one value per beamlet),
* **HU bin** alone (one value per bin), and
* both, additively (weighted backfitting of the two fixed effects).

``hu_increment = R2(both) - R2(beamlet)`` is how much the voxel's HU adds once the beamlet is
known; ``beamlet_increment = R2(both) - R2(HU)`` the reverse. The additive model's HU effects,
averaged over the bins of a tissue class, give the within-beamlet contrast of bone and of
lung/air against soft tissue, in % of the beamlet's maximum dose. Intervals come from resampling
patients. R2 here is of the cell means, voxel-weighted: the scatter of single voxels inside a
cell is not part of it, so both shares are upper bounds of what each factor explains per voxel.

Second part: the beamlet effects (what the beamlet contributes in a band) against the input-only
beamlet features, by Spearman and by an out-of-fold R2 of the analyse stage's gradient-boosted
model with folds grouped by patient.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from src.engine_drivers.importance import _model

logger = logging.getLogger(__name__)

__all__ = ["additive_fit", "beamlet_effects_vs_features", "decompose", "decomposition_summary", "decomposition_table",
           "run_voxel_decomposition", "weighted_r2"]

TISSUE_BINS = {"lung_air": (-np.inf, -500.0), "soft": (-500.0, 300.0), "bone": (300.0, np.inf)}
COMPLEXITY = ("total_hu_change", "pflugfelder_hi", "isi_sum", "sobel_th_mean", "lowdensity_path_mm",
              "air_gap_mm", "max_grad_depth_mm", "energy_mev")


def _group_mean(values: np.ndarray, w: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    num = np.bincount(codes, weights=w * values, minlength=n)
    den = np.bincount(codes, weights=w, minlength=n)
    return np.divide(num, den, out=np.zeros(n), where=den > 0)


def weighted_r2(y: np.ndarray, w: np.ndarray, fit: np.ndarray) -> float:
    mean = np.average(y, weights=w)
    total = float(np.sum(w * (y - mean) ** 2))
    return 1.0 - float(np.sum(w * (y - fit) ** 2)) / total if total > 0 else np.nan


def additive_fit(y: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, n_a: int, n_b: int,
                 max_iter: int = 200, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least squares of ``y ~ alpha[a] + beta[b]`` by backfitting.

    Returns ``(fit, alpha, beta)`` with ``beta`` centred to voxel-weighted mean zero (the grand
    level sits in ``alpha``).
    """
    beta = np.zeros(n_b)
    alpha = _group_mean(y, w, a, n_a)
    for _ in range(max_iter):
        new_beta = _group_mean(y - alpha[a], w, b, n_b)
        new_alpha = _group_mean(y - new_beta[b], w, a, n_a)
        change = max(np.max(np.abs(new_beta - beta)), np.max(np.abs(new_alpha - alpha)))
        alpha, beta = new_alpha, new_beta
        if change < tol:
            break
    shift = np.average(beta[b], weights=w)
    alpha, beta = alpha + shift, beta - shift
    return alpha[a] + beta[b], alpha, beta


def decompose(y: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray, bin_low: np.ndarray) -> Dict[str, float]:
    """R2 of beamlet, HU and both, the two increments, and the within-beamlet tissue contrasts."""
    n_a, n_b = int(a.max()) + 1, int(b.max()) + 1
    r_a = weighted_r2(y, w, _group_mean(y, w, a, n_a)[a])
    r_b = weighted_r2(y, w, _group_mean(y, w, b, n_b)[b])
    fit, _, beta = additive_fit(y, w, a, b, n_a, n_b)
    r_ab = weighted_r2(y, w, fit)
    bin_w = np.bincount(b, weights=w, minlength=n_b)
    effect = {}
    for name, (lo, hi) in TISSUE_BINS.items():
        sel = (bin_low >= lo) & (bin_low < hi) & (bin_w > 0)
        effect[name] = float(np.average(beta[sel], weights=bin_w[sel])) if sel.any() else np.nan
    return {"r2_beamlet": r_a, "r2_hu": r_b, "r2_both": r_ab, "hu_increment": r_ab - r_a,
            "beamlet_increment": r_ab - r_b, "bone_minus_soft": effect["bone"] - effect["soft"],
            "lung_air_minus_soft": effect["lung_air"] - effect["soft"]}


def _arrays(part: pd.DataFrame, value: str) -> Dict[str, np.ndarray]:
    y = (part["sum_abs"] if value == "abs" else part["sum_signed"]).to_numpy(float) / part["n"].to_numpy(float)
    bins = np.sort(part["bin_low_hu"].unique())
    return {"y": y, "w": part["n"].to_numpy(float), "beamlet": pd.factorize(part["sample_id"])[0],
            "b": np.searchsorted(bins, part["bin_low_hu"].to_numpy()), "bins": bins,
            "patient": pd.factorize(part["patient_id"])[0]}


def _boot(arr: Mapping[str, np.ndarray], n_boot: int, seed: int) -> List[Dict[str, float]]:
    """Patient-resampled decompositions; a patient drawn twice contributes its beamlets twice as
    distinct beamlets."""
    rng = np.random.default_rng(seed)
    n_pat = int(arr["patient"].max()) + 1
    rows_of = [np.flatnonzero(arr["patient"] == p) for p in range(n_pat)]
    n_beam = int(arr["beamlet"].max()) + 1
    out = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_pat, n_pat)
        idx = np.concatenate([rows_of[p] for p in draw])
        copy = np.repeat(np.arange(n_pat), [rows_of[p].size for p in draw])
        beam = pd.factorize(arr["beamlet"][idx] + n_beam * copy)[0]
        b = arr["b"][idx]
        used = np.unique(b)
        out.append(decompose(arr["y"][idx], arr["w"][idx], beam, np.searchsorted(used, b), arr["bins"][used]))
    return out


def decomposition_table(cells: pd.DataFrame, meta: pd.DataFrame, bands: Sequence[str],
                        values: Sequence[str] = ("abs", "signed"), n_boot: int = 200, seed: int = 0,
                        scopes: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per engine, band, error value and scope (``all`` and each site): the decomposition
    with patient-resampling 95% intervals."""
    frame = cells.merge(meta[["sample_id", "patient_id", "site"]], on="sample_id", how="left", validate="m:1")
    scopes = list(scopes) if scopes else ["all", *sorted(frame["site"].unique())]
    rows = []
    for (engine, band), part in frame.groupby(["engine", "band"], observed=True):
        if band not in bands:
            continue
        for scope in scopes:
            sub = part if scope == "all" else part[part["site"] == scope]
            for value in values:
                arr = _arrays(sub, value)
                point = decompose(arr["y"], arr["w"], arr["beamlet"], arr["b"], arr["bins"])
                boot = pd.DataFrame(_boot(arr, n_boot, seed))
                row = {"engine": engine, "band": band, "value": value, "scope": scope,
                       "cells": len(sub), "voxels": int(arr["w"].sum()), "beamlets": int(arr["beamlet"].max()) + 1}
                for k, v in point.items():
                    lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
                    row.update({k: v, f"{k}_ci_low": lo, f"{k}_ci_high": hi})
                rows.append(row)
                logger.info("decomposition %s %s %s %s: hu_increment %.4f, beamlet_increment %.4f",
                            engine, band, value, scope, point["hu_increment"], point["beamlet_increment"])
    return pd.DataFrame(rows)


def beamlet_effects_vs_features(cells: pd.DataFrame, features: pd.DataFrame, band: str, value: str = "abs",
                                n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Per engine: the beamlet effect in ``band`` (from the additive model) against the input-only
    features: Spearman per complexity feature and the out-of-fold R2 of all features together."""
    rows = []
    feat = features.set_index("sample_id")
    columns = [c for c in feat.columns if c not in ("patient_id", "site", "peak_inside_crop", "analytic_bp_slice",
                                                     "gantry_deg") and pd.api.types.is_numeric_dtype(feat[c])]
    for engine, part in cells[cells["band"] == band].groupby("engine", observed=True):
        arr = _arrays(part.merge(feat[["patient_id"]], left_on="sample_id", right_index=True), value)
        n_a, n_b = int(arr["beamlet"].max()) + 1, int(arr["b"].max()) + 1
        _, alpha, _ = additive_fit(arr["y"], arr["w"], arr["beamlet"], arr["b"], n_a, n_b)
        ids = pd.factorize(part["sample_id"])[1]
        effect = pd.Series(alpha, index=ids, name="effect")
        data = feat.loc[effect.index].assign(effect=effect)
        for f in COMPLEXITY:
            rho = spearmanr(data[f], data["effect"], nan_policy="omit")[0]
            rows.append({"engine": engine, "band": band, "value": value, "statistic": "spearman", "feature": f,
                         "estimate": float(rho), "beamlets": len(data)})
        X, y = data[columns].to_numpy(float), data["effect"].to_numpy(float)
        groups = pd.factorize(data["patient_id"])[0]
        pred = np.full(y.size, np.nan)
        for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
            pred[te] = _model(seed).fit(X[tr], y[tr]).predict(X[te])
        r2 = 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append({"engine": engine, "band": band, "value": value, "statistic": "oof_r2", "feature": "all inputs",
                     "estimate": float(r2), "beamlets": len(data)})
    return pd.DataFrame(rows)


def _ci(row: pd.Series, key: str, digits: int = 3) -> str:
    return f"{row[key]:.{digits}f} [{row[key + '_ci_low']:.{digits}f}, {row[key + '_ci_high']:.{digits}f}]"


def decomposition_summary(table: pd.DataFrame, effects: pd.DataFrame, n_boot: int) -> List[str]:
    """``summary.md`` lines: definitions, the decomposition tables, the beamlet effects against features."""
    lines = ["# Beamlet or material? Decomposition of the per-voxel dose error", "",
             "Cells: one per beamlet and 50 HU bin within a dose band; value = voxel-weighted mean error of the "
             "cell (abs = |e|, signed = e, both in % of the beamlet's Monte Carlo maximum; see "
             "`../summary.md`). R2 is voxel-weighted and of the cell means (the scatter of voxels inside a cell is "
             "not part of it).", "",
             "- **R2 beamlet**: one value per beamlet. **R2 HU**: one value per HU bin. **R2 both**: additive.",
             "- **HU added after beamlet** = R2 both - R2 beamlet: what the voxel's own HU explains once the "
             "beamlet is known. **beamlet added after HU** = R2 both - R2 HU.",
             "- **bone - soft** and **lung/air - soft**: the additive model's HU effects averaged over the bins of "
             "each class (above 300 HU, below -500 HU, -500 to 300 HU): how much larger the error is in that "
             "tissue than in soft tissue of the same beamlet, in percentage points of the maximum dose.",
             f"- [ ] = 95% intervals from resampling patients ({n_boot} resamples).", ""]
    header = ("| engine | band | R2 beamlet | R2 HU | HU added after beamlet | beamlet added after HU | "
              "bone - soft | lung/air - soft |")
    for value, scope_title in (("abs", "absolute error |e|"), ("signed", "signed error e")):
        part = table[(table["value"] == value) & (table["scope"] == "all")]
        lines += [f"## All sites, {scope_title}", "", header, "|---|---|---|---|---|---|---|---|"]
        for _, r in part.iterrows():
            lines.append(f"| {r.engine} | {r.band} | {r.r2_beamlet:.3f} | {r.r2_hu:.3f} | {_ci(r, 'hu_increment')} | "
                         f"{_ci(r, 'beamlet_increment')} | {_ci(r, 'bone_minus_soft', 2)} | "
                         f"{_ci(r, 'lung_air_minus_soft', 2)} |")
        lines.append("")
    part = table[(table["value"] == "abs") & (table["scope"] != "all")]
    lines += ["## Per site, absolute error |e|", "", "| engine | band | site | R2 beamlet | HU added after beamlet | "
              "bone - soft | lung/air - soft |", "|---|---|---|---|---|---|---|"]
    for _, r in part.iterrows():
        lines.append(f"| {r.engine} | {r.band} | {r.scope} | {r.r2_beamlet:.3f} | {_ci(r, 'hu_increment')} | "
                     f"{_ci(r, 'bone_minus_soft', 2)} | {_ci(r, 'lung_air_minus_soft', 2)} |")
    lines += ["", "## What the beamlet effect follows", "",
              "Beamlet effect = the beamlet's level in the additive model (absolute error, per band): Spearman with "
              "input-only beamlet features, and the out-of-fold R2 of the analyse stage's gradient-boosted model on "
              "all input-only features (5 folds grouped by patient).", "",
              "| engine | band | feature | estimate |", "|---|---|---|---|"]
    for _, r in effects.iterrows():
        name = "out-of-fold R2, all inputs" if r.statistic == "oof_r2" else f"Spearman, {r.feature}"
        lines.append(f"| {r.engine} | {r.band} | {name} | {r.estimate:+.3f} |")
    lines += ["", "## Caveats", "",
              "- Within a beamlet, HU and position (entrance, peak, fall-off, penumbra) are tied: a material that "
              "sits mostly in one position takes that position's error. The dose bands limit this but do not "
              "remove it.",
              "- The 0.1-10% band is dominated by Monte Carlo noise at 1e6 primaries (no noise floor measured).",
              "- The pencil beam's bone excess is expected from dose to water (pencil beam) against dose to medium "
              "(Monte Carlo); it is the positive control that the decomposition can see a material effect.", ""]
    return lines


def run_voxel_decomposition(cfg: Mapping, run) -> List[str]:
    """The stage: reads ``analysis/voxel_hu_error/voxel_sums_per_beamlet.csv.gz`` and writes
    ``analysis/voxel_hu_error/decomposition/``."""
    from src.engine_drivers.voxel_hu_error import BANDS

    dcfg = cfg.get("voxel_decomposition", {})
    n_boot, seed = int(dcfg.get("n_boot", 200)), int(dcfg.get("seed", 20261126))
    base = run / "analysis" / "voxel_hu_error"
    out = base / "decomposition"
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(base / "voxel_sums_per_beamlet.csv.gz", dtype={"sample_id": str})
    meta = pd.read_csv(run / "analysis" / "dataset_overview" / "beamlet_metadata.csv", dtype={"sample_id": str})
    features = pd.read_csv(run / "analysis" / "beamlet_features.csv", dtype={"sample_id": str})
    bands = [b[0] for b in BANDS]
    table = decomposition_table(cells, meta, bands, n_boot=n_boot, seed=seed)
    table.to_csv(out / "decomposition.csv", index=False)
    effects = pd.concat([beamlet_effects_vs_features(cells, features, band, seed=seed) for band in bands[1:]],
                        ignore_index=True)
    effects.to_csv(out / "beamlet_effects_vs_features.csv", index=False)
    return decomposition_summary(pd.read_csv(out / "decomposition.csv"),
                                 pd.read_csv(out / "beamlet_effects_vs_features.csv"), n_boot)
