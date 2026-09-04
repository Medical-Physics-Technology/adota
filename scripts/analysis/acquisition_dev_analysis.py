"""Development-pool-only achievability study and target comparison.

Re-runs the two exploratory analyses of the report (Figure 12 achievability and
Table 4 target/transform comparison) on the *development pool only*, i.e. after
removing the seven frozen-test patients, so that every number in the report is
computed without ever touching the frozen test set. The frozen-test patients are
read from frozen_test_ids.csv, so the exclusion is identical to the one used by
acquisition_frozen_test.py (seed 42, whole patients, anatomy-stratified).

Outputs (into the run's figures/acquisition/):
  * fig1_achievability.png  - regenerated with dev-pool numbers,
  * dev_achievability.csv   - RDE/gamma/range linear+GBM Pearson (dev pool),
  * dev_target_comparison.csv - RDE/MAPE raw+log linear+GBM Pearson/Spearman.

Run: uv run --with scikit-learn python scripts/analysis/acquisition_dev_analysis.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from scripts.analysis.acquisition_regression_study import (
    ALL_FEATS,
    NO_DEPTH,
    RANGE_VALID_MAX_MM,
    percentile_matrix,
)

INK, MUTED, BASE, SURF, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#fcfcfb", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": MUTED, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False})

def ridge():
    return Ridge(alpha=1.0)


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


def cv(df, cols, y, mask, groups, model_fn):
    """Held-out Pearson and Spearman under 5-fold patient-grouped CV on `mask` rows."""
    idx = np.where(mask)[0]
    pe, sp = [], []
    for tr_, te_ in GroupKFold(5).split(idx, groups=groups[idx]):
        tr, te = idx[tr_], idx[te_]
        Xtr, Xte = percentile_matrix(df, cols, tr, te)
        m = model_fn().fit(Xtr, y[tr])
        pr = m.predict(Xte)
        pe.append(pearsonr(pr, y[te])[0])
        sp.append(spearmanr(pr, y[te]).correlation)
    return float(np.mean(pe)), float(np.mean(sp))


def main():
    ap = argparse.ArgumentParser()
    base = "/scratch/mstryja/adota_runs/20260707_124010"
    ap.add_argument("--results", default=f"{base}/results.csv")
    ap.add_argument("--prov", default=f"{base}/figures/acquisition/uuid_provenance_map.csv")
    ap.add_argument("--mape", default=f"{base}/figures/acquisition/mape_metric.csv")
    ap.add_argument("--test-ids", default=f"{base}/figures/acquisition/frozen_test_ids.csv")
    ap.add_argument("--outdir", default=f"{base}/figures/acquisition")
    ap.add_argument(
        "--report-fig",
        default="/home/mstryja/projects/adota/research/figures/acquisition/fig1_achievability.png",
    )
    args = ap.parse_args()
    out = Path(args.outdir)

    df = (pd.read_csv(args.results)
          .merge(pd.read_csv(args.prov), on="sample_id", how="left")
          .merge(pd.read_csv(args.mape), on="sample_id", how="left")
          .reset_index(drop=True))
    test_ids = set(pd.read_csv(args.test_ids)["sample_id"])
    dev = ~df["sample_id"].isin(test_ids)
    df = df[dev].reset_index(drop=True)
    groups = df["patient_key"].values
    print(f"dev pool: {len(df)} beamlets, {len(pd.unique(groups))} patients "
          f"(excluded {len(test_ids)} frozen-test patients)")

    # ---- achievability (Figure 12): RDE / gamma / range, linear + GBM Pearson ----
    ones = np.ones(len(df), bool)
    rr = df["r100_delta_mm"].abs().values
    range_valid = rr < RANGE_VALID_MAX_MM
    ach_targets = [
        ("Relative dose error", np.log1p(df["rde"].values), ones),
        ("Gamma error", 100.0 - df["gpr"].values, ones),
        ("Range error", rr, range_valid),
    ]
    ach = []
    print(f"\n{'achievability target':22} {'linear P':>9} {'GBM P':>7}")
    for name, y, mask in ach_targets:
        lp, _ = cv(df, ALL_FEATS, y, mask, groups, ridge)
        gp, _ = cv(df, ALL_FEATS, y, mask, groups, gbm)
        ach.append(dict(target=name, linear_pearson=lp, gbm_pearson=gp))
        print(f"{name:22} {lp:9.3f} {gp:7.3f}")
    # no-depth RDE ceiling (heterogeneity, not depth)
    nd_p, _ = cv(df, NO_DEPTH, np.log1p(df["rde"].values), ones, groups, gbm)
    print(f"{'RDE (no depth) GBM':22} {'':9} {nd_p:7.3f}")
    pd.DataFrame(ach).assign(rde_nodepth_gbm_pearson=nd_p).to_csv(out / "dev_achievability.csv", index=False)

    # ---- regenerate Figure 12 ----
    labels = [a["target"] for a in ach]
    lin = [a["linear_pearson"] for a in ach]
    gb = [a["gbm_pearson"] for a in ach]
    x = np.arange(len(labels))
    w = 0.36
    fig, axf = plt.subplots(figsize=(8.2, 4.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)
    b1 = axf.bar(x - w/2, lin, w, color=BLUE, label="interpretable linear model")
    b2 = axf.bar(x + w/2, gb, w, color=ORANGE, label="non-linear reference (gradient boosting)")
    axf.axhline(0.80, ls="--", lw=1.4, color=INK)
    axf.text(len(labels) - 0.5, 0.815, "target 0.80", ha="right", fontsize=9.5, color=INK)
    for bars in (b1, b2):
        for r in bars:
            axf.text(r.get_x() + r.get_width()/2, r.get_height() + 0.012,
                     f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    axf.set_xticks(x)
    axf.set_xticklabels(labels, fontsize=11)
    axf.set_ylabel("Pearson correlation (held-out)")
    axf.set_ylim(0, 1.0)
    axf.set_yticks(np.arange(0, 1.01, 0.2))
    axf.grid(axis="y", color=GRID, lw=0.7)
    axf.set_axisbelow(True)
    axf.legend(loc="upper right", frameon=False, fontsize=9.5)
    axf.set_title("Achievable correlation between an input-only score and each error measure",
                  fontsize=12, fontweight="bold")
    for p in (out / "fig1_achievability.png", Path(args.report_fig)):
        fig.savefig(p, dpi=150)
        print(f"wrote {p}")
    plt.close(fig)

    # ---- target comparison (Table 4): RDE / MAPE, raw + log, linear + GBM P/S ----
    tc_targets = []
    for col in ["rde", "mape_5pct", "mape_10pct"]:
        v = df[col].values.astype(float)
        m = np.isfinite(v)
        tc_targets.append((f"{col} raw", v, m))
        tc_targets.append((f"{col} log1p", np.log1p(v), m))
    rows = []
    print(f"\n{'target':16} {'lin P':>7} {'lin S':>7} {'GBM P':>7} {'GBM S':>7}")
    for name, y, m in tc_targets:
        lp, ls = cv(df, ALL_FEATS, y, m, groups, ridge)
        gp, gs = cv(df, ALL_FEATS, y, m, groups, gbm)
        rows.append(dict(target=name, linear_pearson=lp, linear_spearman=ls,
                         gbm_pearson=gp, gbm_spearman=gs))
        print(f"{name:16} {lp:7.3f} {ls:7.3f} {gp:7.3f} {gs:7.3f}")
    pd.DataFrame(rows).to_csv(out / "dev_target_comparison.csv", index=False)

    # RDE-MAPE Spearman on the dev pool
    for c in ["mape_5pct", "mape_10pct"]:
        m = np.isfinite(df[c].values)
        s = spearmanr(df["rde"].values[m], df[c].values[m]).correlation
        print(f"corr(RDE, {c}) Spearman [dev] = {s:.3f}")
    print(f"\nwrote {out/'dev_achievability.csv'}, {out/'dev_target_comparison.csv'}")


if __name__ == "__main__":
    main()
