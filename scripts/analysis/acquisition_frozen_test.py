"""Frozen-test verification of the difficulty score.

Protocol:
  1. Freeze a test set of whole patients (about 10 percent of beamlets), selected
     independently within each anatomy so that both anatomies are represented.
     These patients are never used for fitting or model selection, so the test
     certifies generalization to unseen patients.
  2. On the remaining patients (the development pool) run 5-fold patient-grouped
     cross-validation (whole patients held out) as the development estimate, and
     fit one final model on all of the development pool.
  3. Apply the frozen final weights to the frozen test set and report the
     correlation.

The percentile transform (quantile grids) is fitted on the training rows only, so
no test information enters the fit at any stage. The target is log(1 + relative
dose error). Outputs: the frozen test sample ids, the final linear weights, and a
summary table of development-CV and test correlations for three score variants.

Run: uv run --with scikit-learn python scripts/analysis/acquisition_frozen_test.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold

from scripts.analysis.acquisition_regression_study import ALL_FEATS

SEED = 42
TEST_FRAC = 0.10
SPARSE_ALPHA = 0.0008


def grids(df, rows):
    return {c: np.quantile(df[c].values[rows], np.linspace(0, 1, 101)) for c in ALL_FEATS}


def transform(df, rows, G):
    return np.column_stack(
        [np.clip(np.searchsorted(G[c], df[c].values[rows], side="right") / 101, 0, 1)
         for c in ALL_FEATS]
    )


def make_model(name):
    if name == "sparse (Lasso)":
        return Lasso(alpha=SPARSE_ALPHA, max_iter=5000)
    if name == "full (ridge)":
        return Ridge(alpha=1.0)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


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

    df = (pd.read_csv(args.results)
          .merge(pd.read_csv(args.prov), on="sample_id", how="left")
          .reset_index(drop=True))
    y = np.log1p(df["rde"].values)
    groups = df["patient_key"].values
    anatomy = df["anatomy"].values
    rng = np.random.default_rng(SEED)

    # ---- frozen ~10% test: whole patients held out, stratified by anatomy ----
    test_patients = []
    for a in sorted(pd.unique(anatomy)):
        pats = list(pd.unique(groups[anatomy == a]))
        rng.shuffle(pats)
        target = TEST_FRAC * int((anatomy == a).sum())
        cum = 0
        for p in pats:
            test_patients.append(p)
            cum += int((groups == p).sum())
            if cum >= target:
                break
    test_mask = np.isin(groups, test_patients)
    dev = np.where(~test_mask)[0]
    test = np.where(test_mask)[0]
    df.loc[test, ["sample_id", "anatomy"]].assign(split="test").to_csv(out / "frozen_test_ids.csv", index=False)
    by_anat = {a: sum(1 for p in test_patients if anatomy[groups == p][0] == a)
               for a in sorted(pd.unique(anatomy))}
    print(f"total patients={len(pd.unique(groups))}   held-out TEST patients={len(test_patients)}  {by_anat}")
    print(f"dev beamlets={len(dev)}  test beamlets={len(test)} ({100*len(test)/len(df):.1f}%)  "
          f"dev patients={len(pd.unique(groups[dev]))}")

    # ---- development 5-fold patient-grouped CV (whole patients held out) ----
    def cv_dev(name):
        gd = groups[dev]
        pe, sp = [], []
        for tr, te in GroupKFold(5).split(dev, groups=gd):
            rtr, rte = dev[tr], dev[te]
            G = grids(df, rtr)
            m = make_model(name).fit(transform(df, rtr, G), y[rtr])
            pr = m.predict(transform(df, rte, G))
            pe.append(pearsonr(pr, y[rte])[0])
            sp.append(spearmanr(pr, y[rte]).correlation)
        return float(np.mean(pe)), float(np.mean(sp))

    # ---- final fit on all dev, frozen evaluation on test ----
    G_dev = grids(df, dev)
    Xdev, Xtest = transform(df, dev, G_dev), transform(df, test, G_dev)
    rows = []
    weights_out = {}
    for name in ["sparse (Lasso)", "full (ridge)", "gbm (non-linear)"]:
        cvp, cvs = cv_dev(name)
        m = make_model(name).fit(Xdev, y[dev])
        pr = m.predict(Xtest)
        tp, ts = pearsonr(pr, y[test])[0], spearmanr(pr, y[test]).correlation
        nz = (int(np.sum(np.abs(m.coef_) > 1e-6)) if hasattr(m, "coef_") else len(ALL_FEATS))
        rows.append(dict(variant=name, n_terms=nz, dev_cv_pearson=cvp, dev_cv_spearman=cvs,
                         test_pearson=float(tp), test_spearman=float(ts)))
        if hasattr(m, "coef_"):
            weights_out[name] = {"intercept": float(m.intercept_),
                                 "coefficients": {f: float(c) for f, c in zip(ALL_FEATS, m.coef_) if abs(c) > 1e-6}}
        print(f"  {name:20} terms={nz:2}  dev-CV P/S {cvp:.3f}/{cvs:.3f}   TEST P/S {tp:.3f}/{ts:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(out / "frozen_test_results.csv", index=False)
    json.dump({"seed": SEED, "test_frac": TEST_FRAC, "n_dev": int(len(dev)), "n_test": int(len(test)),
               "percentile_grids_dev": {c: G_dev[c].tolist() for c in ALL_FEATS},
               "weights": weights_out},
              open(out / "frozen_final_scorer.json", "w"), indent=2)
    print(f"\nwrote {out/'frozen_test_results.csv'}, {out/'frozen_test_ids.csv'}, {out/'frozen_final_scorer.json'}")


if __name__ == "__main__":
    main()
