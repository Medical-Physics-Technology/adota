"""Step 2-4: finalize the combined RDE difficulty score and evaluate it.

Produces, for target log1p(RDE):
  * out-of-fold (patient-grouped) predictions -> honest held-out Pearson/Spearman
    for an interpretable linear model (all metrics), a compact linear model
    (top-k metrics), and a GBM ceiling; plus a predicted-vs-true scatter;
  * a cross-anatomy robustness check (train one anatomy, predict the other);
  * a tail-lift readout (does the continuous RDE-difficulty score also enrich the
    gamma-failure tail?);
  * a persisted, deployable interpretable scorer (feature list, per-feature
    percentile grid, linear coefficients) as JSON.

Input-only features; RDE used only as the offline regression target / validation.
Run: uv run --with scikit-learn python scripts/analysis/acquisition_rde_finalize.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from scripts.analysis.acquisition_regression_study import ALL_FEATS

COMPACT_K = 8


def percentile_grid(vals, n=101):
    """101-point quantile grid so the percentile transform is reproducible/deployable."""
    qs = np.linspace(0, 1, n)
    return np.quantile(vals, qs)


def apply_grid(vals, grid):
    n = len(grid)
    return np.clip(np.searchsorted(grid, vals, side="right") / n, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/mstryja/adota_runs/20260707_124010/results.csv")
    ap.add_argument(
        "--prov",
        default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/uuid_provenance_map.csv",
    )
    ap.add_argument("--outdir", default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results).merge(pd.read_csv(args.prov), on="sample_id", how="left")
    groups = df["patient_key"].values
    y = np.log1p(df["rde"].values)                      # regression target
    feats = ALL_FEATS

    # ---- out-of-fold predictions (patient-grouped) ----
    n = len(df)
    oof = {"linear_all": np.full(n, np.nan), "gbm": np.full(n, np.nan)}
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(np.arange(n), groups=groups):
        Ptr = np.column_stack([apply_grid(df[c].values[tr], percentile_grid(df[c].values[tr])) for c in feats])
        Pte = np.column_stack([apply_grid(df[c].values[te], percentile_grid(df[c].values[tr])) for c in feats])
        oof["linear_all"][te] = Ridge(alpha=1.0).fit(Ptr, y[tr]).predict(Pte)
        oof["gbm"][te] = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, random_state=0).fit(Ptr, y[tr]).predict(Pte)

    def corr(pred):
        return pearsonr(pred, y)[0], spearmanr(pred, y).correlation

    print("=== held-out (patient-grouped OOF) correlation vs log1p(RDE) ===")
    for k, p in oof.items():
        pe, sp = corr(p)
        print(f"  {k:12} Pearson {pe:.3f}   Spearman {sp:.3f}")
    # correlation against RAW rde too (interpretability)
    pe_raw = pearsonr(oof["linear_all"], df["rde"].values)[0]
    sp_raw = spearmanr(oof["linear_all"], df["rde"].values).correlation
    print(f"  linear_all vs RAW rde: Pearson {pe_raw:.3f}  Spearman {sp_raw:.3f}")

    # ---- compact interpretable model: top-K features by |coef| of full linear fit ----
    Pfull = np.column_stack([apply_grid(df[c].values, percentile_grid(df[c].values)) for c in feats])
    full = Ridge(alpha=1.0).fit(Pfull, y)
    topk = [feats[i] for i in np.argsort(-np.abs(full.coef_))[:COMPACT_K]]
    oof_c = np.full(n, np.nan)
    for tr, te in gkf.split(np.arange(n), groups=groups):
        Ptr = np.column_stack([apply_grid(df[c].values[tr], percentile_grid(df[c].values[tr])) for c in topk])
        Pte = np.column_stack([apply_grid(df[c].values[te], percentile_grid(df[c].values[tr])) for c in topk])
        oof_c[te] = Ridge(alpha=1.0).fit(Ptr, y[tr]).predict(Pte)
    pe_c, sp_c = corr(oof_c)
    print(f"  linear_compact (top-{COMPACT_K}: {topk}): Pearson {pe_c:.3f}  Spearman {sp_c:.3f}")

    # ---- cross-anatomy robustness (train one anatomy, predict the other) ----
    print("\n=== cross-anatomy (linear_all) ===")
    anat = df["anatomy"].values
    for held in sorted(pd.unique(anat)):
        te = np.where(anat == held)[0]
        tr = np.where(anat != held)[0]
        Ptr = np.column_stack([apply_grid(df[c].values[tr], percentile_grid(df[c].values[tr])) for c in feats])
        Pte = np.column_stack([apply_grid(df[c].values[te], percentile_grid(df[c].values[tr])) for c in feats])
        pr = Ridge(alpha=1.0).fit(Ptr, y[tr]).predict(Pte)
        print(f"  test={held:18} Pearson {pearsonr(pr,y[te])[0]:.3f}  Spearman {spearmanr(pr,y[te]).correlation:.3f}")

    # ---- tail-lift: does the RDE-difficulty score also enrich the gamma tail? ----
    s = oof["linear_all"]
    tau_gpr = (df["gpr"].values < 95).astype(int)
    for frac in (0.10, 0.05):
        topn = np.argsort(-s)[:int(frac * n)]
        lift = tau_gpr[topn].mean() / max(tau_gpr.mean(), 1e-9)
        print(f"  gamma<95 lift @top{int(frac*100)}% of RDE-score: {lift:.2f}")

    # ---- scatter ----
    fig, ax = plt.subplots(figsize=(5, 5))
    idx = np.random.default_rng(0).choice(n, min(6000, n), replace=False)
    ax.scatter(oof["linear_all"][idx], y[idx], s=3, alpha=0.15)
    lo, hi = y.min(), y.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    pe, sp = corr(oof["linear_all"])
    ax.set_xlabel("predicted difficulty (linear, held-out)")
    ax.set_ylabel("log1p(RDE)")
    ax.set_title(f"RDE combined score: Pearson {pe:.2f}, Spearman {sp:.2f}")
    fig.tight_layout()
    fig.savefig(out / "rde_combined_scatter.png", dpi=140)
    plt.close(fig)

    # ---- persist deployable interpretable scorer ----
    scorer = {
        "target": "log1p(rde)", "model": "ridge_alpha1_on_percentile_features",
        "features": feats, "intercept": float(full.intercept_),
        "coefficients": {f: float(c) for f, c in zip(feats, full.coef_)},
        "percentile_grids": {f: percentile_grid(df[f].values).tolist() for f in feats},
        "compact_features": topk,
        "heldout": {"linear_all_pearson": float(pe), "linear_all_spearman": float(sp),
                    "compact_pearson": float(pe_c), "compact_spearman": float(sp_c)},
    }
    (out / "rde_combined_scorer.json").write_text(json.dumps(scorer, indent=2))
    print(f"\nwrote {out/'rde_combined_scorer.json'} and {out/'rde_combined_scatter.png'}")


if __name__ == "__main__":
    main()
