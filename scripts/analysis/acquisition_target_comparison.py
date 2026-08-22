"""Correlation ceiling per error target: RDE and MAPE, raw and log(1+.).

Joins the MAPE columns (from extract_mape.py) onto results.csv and reports, for
each target, the held-out Pearson/Spearman of an interpretable linear model and
the flexible GBM ceiling (patient-grouped 5-fold CV). Answers "how strongly can
an input-only score correlate with RDE vs MAPE, raw vs log-transformed".

Run: uv run --with scikit-learn python scripts/analysis/acquisition_target_comparison.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from scripts.analysis.acquisition_regression_study import ALL_FEATS, percentile_matrix


def cv(df, groups, cols, y, mask, model_fn):
    idx = np.where(mask)[0]
    pe, sp = [], []
    for tr, te in GroupKFold(5).split(idx, groups=groups[idx]):
        a, b = idx[tr], idx[te]
        Ptr, Pte = percentile_matrix(df, cols, a, b)
        m = model_fn().fit(Ptr, y[a])
        pr = m.predict(Pte)
        pe.append(pearsonr(pr, y[b])[0])
        sp.append(spearmanr(pr, y[b]).correlation)
    return np.mean(pe), np.mean(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/mstryja/adota_runs/20260707_124010/results.csv")
    ap.add_argument(
        "--prov",
        default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/uuid_provenance_map.csv",
    )
    ap.add_argument("--mape", default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/mape_metric.csv")
    ap.add_argument(
        "--out",
        default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/target_comparison.csv",
    )
    args = ap.parse_args()

    df = (pd.read_csv(args.results)
          .merge(pd.read_csv(args.prov), on="sample_id", how="left")
          .merge(pd.read_csv(args.mape), on="sample_id", how="left"))
    groups = df["patient_key"].values

    def tgt(col):
        v = df[col].values.astype(float)
        return v, np.isfinite(v)

    targets = []
    for col in ["rde", "mape_5pct", "mape_10pct"]:
        v, mask = tgt(col)
        targets.append((f"{col} (raw)", v, mask))
        targets.append((f"{col} log(1+.)", np.log1p(v), mask))

    def ridge():
        return Ridge(alpha=1.0)

    def gbm():
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)

    rows = []
    print(f"{'target':22} {'linear P':>9} {'linear S':>9} {'GBM P':>7} {'GBM S':>7}")
    print("-" * 60)
    for name, y, mask in targets:
        lp, ls = cv(df, groups, ALL_FEATS, y, mask, ridge)
        gp, gs = cv(df, groups, ALL_FEATS, y, mask, gbm)
        rows.append(dict(target=name, linear_pearson=lp, linear_spearman=ls,
                         gbm_pearson=gp, gbm_spearman=gs))
        print(f"{name:22} {lp:9.3f} {ls:9.3f} {gp:7.3f} {gs:7.3f}")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    # cross-check: how correlated are RDE and MAPE themselves
    for c in ["mape_5pct", "mape_10pct"]:
        m = np.isfinite(df[c].values)
        print(f"corr(RDE, {c}) Spearman = {spearmanr(df['rde'].values[m], df[c].values[m]).correlation:.3f}")


if __name__ == "__main__":
    main()
