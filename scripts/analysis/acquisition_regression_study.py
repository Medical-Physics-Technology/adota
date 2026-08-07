"""Step 0/2: regression study for a combined difficulty score with high error-correlation.

For each continuous error target we fit, under patient-grouped 5-fold CV:
  * an interpretable linear model (Ridge on the de-correlated parsimonious basis),
  * a linear model on all metrics, and
  * a flexible ceiling (HistGradientBoostingRegressor on all metrics),
and report held-out Pearson + Spearman (mean +/- s.d.). A "no energy / BP-depth"
feature variant isolates heterogeneity-driven signal from a trivial depth/energy
confound. For the chosen target the interpretable coefficients are printed as an
explicit formula.

Input-only: every feature is an input-derived metric; the error targets are used
only offline, as the regression target and for validation.

Run: uv run --with scikit-learn python scripts/analysis/acquisition_regression_study.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# de-correlated parsimonious basis (one representative per cluster) + energy
BASIS = [
    "wepl_std", "pflugfelder_hi", "total_hu_change", "max_hu_jump",
    "sum_sobel_bp", "sigma_hu_bp", "isi_mean", "hetero_fraction",
    "sobel_th_beam_angle", "sobel_dw_anisotropy", "interface_bp_distance",
    "wepl_mean",
]
ALL_FEATS = [
    "energy_mev", "ct_max_hu", "bp_range_min_mm", "bp_range_max_mm", "max_grad_depth_mm",
    "n_density_regions", "total_hu_change", "max_hu_jump", "sigma_hu_bp", "max_hu_gradient",
    "lateral_hu_var_bp", "hetero_fraction", "interface_bp_distance", "mean_sobel_axial",
    "p95_sobel_bp", "sum_sobel_bp", "sobel_dw_mean", "sobel_dw_anisotropy", "sobel_dw_edge_energy",
    "sobel_th_mean", "sobel_th_anisotropy", "sobel_th_edge_energy", "pflugfelder_hi", "wepl_mean",
    "wepl_std", "isi_sum", "isi_max", "isi_mean", "isi_axial_sum", "lateral_edge_energy",
]
# "no depth" set drops energy and the energy-derived BP-range features
NO_DEPTH = [f for f in ALL_FEATS if f not in ("energy_mev", "bp_range_min_mm", "bp_range_max_mm")]
RANGE_VALID_MAX_MM = 20.0


def fit_percentiles(train_vals):
    srt = np.sort(train_vals); n = len(srt)
    return lambda v: np.searchsorted(srt, v, side="right") / n


def percentile_matrix(df, cols, tr, te):
    Ptr = np.empty((len(tr), len(cols))); Pte = np.empty((len(te), len(cols)))
    for j, c in enumerate(cols):
        f = fit_percentiles(df[c].values[tr])
        Ptr[:, j] = f(df[c].values[tr]); Pte[:, j] = f(df[c].values[te])
    return Ptr, Pte


def cv_eval(df, cols, y, mask, groups, model_fn):
    idx = np.where(mask)[0]
    pe, sp = [], []
    gkf = GroupKFold(n_splits=5)
    for tr_, te_ in gkf.split(idx, groups=groups[idx]):
        tr, te = idx[tr_], idx[te_]
        Xtr, Xte = percentile_matrix(df, cols, tr, te)
        m = model_fn(); m.fit(Xtr, y[tr]); pr = m.predict(Xte)
        pe.append(pearsonr(pr, y[te])[0]); sp.append(spearmanr(pr, y[te]).correlation)
    return np.mean(pe), np.std(pe), np.mean(sp), np.std(sp)


MODELS = {
    "ridge_basis": (BASIS, lambda: Ridge(alpha=1.0)),
    "ridge_all": (ALL_FEATS, lambda: Ridge(alpha=1.0)),
    "gbm_all": (ALL_FEATS, lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=0)),
    "gbm_nodepth": (NO_DEPTH, lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/mstryja/adota_runs/20260707_124010/results.csv")
    ap.add_argument("--prov", default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/uuid_provenance_map.csv")
    ap.add_argument("--outdir", default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition")
    ap.add_argument("--formula-target", default="log1p_rde",
                    help="target to print the interpretable Ridge(basis) formula for")
    args = ap.parse_args()

    df = pd.read_csv(args.results).merge(pd.read_csv(args.prov), on="sample_id", how="left")
    groups = df["patient_key"].values
    rr = df["r100_delta_mm"].abs().values
    valid = rr < RANGE_VALID_MAX_MM
    ones = np.ones(len(df), bool)
    targets = {
        "e_gpr": (100.0 - df["gpr"].values, ones),
        "rde": (df["rde"].values, ones),
        "log1p_rde": (np.log1p(df["rde"].values), ones),
        "range_valid": (rr, valid),
    }

    rows = []
    for tname, (y, mask) in targets.items():
        for mname, (cols, fn) in MODELS.items():
            pm, ps, sm, ss = cv_eval(df, cols, y, mask, groups, fn)
            rows.append(dict(target=tname, model=mname, pearson=pm, pearson_sd=ps,
                             spearman=sm, spearman_sd=ss))
    res = pd.DataFrame(rows)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    res.to_csv(Path(args.outdir) / "regression_ceiling.csv", index=False)

    print(f"{'target':12} {'model':12} {'Pearson':>13} {'Spearman':>13}")
    print("-" * 52)
    for _, r in res.iterrows():
        print(f"{r.target:12} {r.model:12} {r.pearson:5.2f}+/-{r.pearson_sd:.2f}   "
              f"{r.spearman:5.2f}+/-{r.spearman_sd:.2f}")

    # interpretable formula for the chosen target: Ridge on the basis, percentile features
    y, mask = targets[args.formula_target]
    idx = np.where(mask)[0]
    P = np.empty((len(idx), len(BASIS)))
    for j, c in enumerate(BASIS):
        f = fit_percentiles(df[c].values[idx]); P[:, j] = f(df[c].values[idx])
    ridge = Ridge(alpha=1.0).fit(P, y[idx])
    order = np.argsort(-np.abs(ridge.coef_))
    print(f"\nInterpretable Ridge(basis) coefficients for target='{args.formula_target}' "
          f"(percentile features in [0,1]):")
    print(f"  intercept = {ridge.intercept_:.4f}")
    for j in order:
        print(f"  {BASIS[j]:22} {ridge.coef_[j]:+.4f}")
    print(f"\nwrote {Path(args.outdir)/'regression_ceiling.csv'}")


if __name__ == "__main__":
    main()
