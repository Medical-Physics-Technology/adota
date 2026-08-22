"""Wave 1: retrospective acquisition ranking benchmark (physics-only, input-derived).

Question answered
-----------------
On the labelled reference run, how well do *input-only* acquisition functions
rank beamlets by *true* model difficulty (GPR failure / range failure), and
which candidate does it best? No Monte Carlo and no model retraining: the model
error labels already exist in ``results.csv`` and are used strictly offline, to
calibrate the supervised candidates and to validate all candidates. The scored
features are input-derived heterogeneity/physics metrics only.

Split
-----
The reference run is a single patient CT (`..._one_ct_...`), so patient-grouped
and leave-one-anatomy-out splits are not available here. We instead use
GroupKFold over the **gantry angle**: whole beam directions are held out, so the
test beamlets traverse geometry not seen when calibrating the acquisition
function. This is the leakage-aware generalisation test available on this
dataset; cross-patient transfer is a separate, later study.

Caveats (documented, not blockers for Wave 1)
---------------------------------------------
* This ``results.csv`` predates the MCsquare WEPL fix; the correction shifts the
  WEPL metrics by ~4% on pelvis and does not change their ranking value.
* ``*_bp`` metrics locate the Bragg-peak slice from the GT dose. Their fully
  input-only form recomputes that slice from beam energy (``range_energy``);
  this mild dependence is a refinement for a later re-extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# ── input-only physics feature set: parsimonious-basis representatives ───────
# (derived cluster reps substituted by an available cluster-mate)
BASIS_FEATURES = [
    "wepl_std", "pflugfelder_hi", "total_hu_change", "max_hu_jump",
    "sum_sobel_bp", "sigma_hu_bp", "isi_mean", "hetero_fraction",
    "sobel_th_beam_angle", "sobel_dw_anisotropy", "interface_bp_distance",
    "wepl_mean",
]
# energy is an input-only beam parameter (no dose dependence); add for the
# linear/learned candidates.
LEARN_FEATURES = BASIS_FEATURES + ["energy_mev"]
EDGE_REP, WEPL_REP = "sum_sobel_bp", "wepl_std"

# offline range-error validity mask: |dR100| above this are failed fits
RANGE_VALID_MAX_MM = 20.0
GPR_FAIL = 95.0
RANGE_FAIL_MM = 3.0
TOP_FRACS = (0.10, 0.05)


def train_percentiles(train_vals: np.ndarray):
    """Return an empirical-CDF transform fitted on train values (leakage-free)."""
    srt = np.sort(train_vals)
    n = len(srt)
    return lambda v: np.searchsorted(srt, v, side="right") / n


def percentile_matrix(df, cols, tr, te):
    """Fit per-column percentile transform on train rows, apply to train+test."""
    Ptr = np.empty((len(tr), len(cols)))
    Pte = np.empty((len(te), len(cols)))
    for j, c in enumerate(cols):
        f = train_percentiles(df[c].values[tr])
        Ptr[:, j] = f(df[c].values[tr])
        Pte[:, j] = f(df[c].values[te])
    return Ptr, Pte


def lift(score, tau, frac):
    n = max(int(frac * len(score)), 1)
    top = np.argsort(-score)[:n]
    base = tau.mean()
    return tau[top].mean() / base if base > 0 else np.nan


def evaluate(score, tau):
    out = {"auroc": roc_auc_score(tau, score) if tau.any() and not tau.all() else np.nan}
    for f in TOP_FRACS:
        out[f"lift{int(f*100)}"] = lift(score, tau, f)
    return out


def build_candidates(df, cols_learn, tr, te, tau_gpr, tau_rng, e_gpr, e_rng):
    """Return {name: test-fold score array} for all Wave-1 candidates."""
    Ptr_b, Pte_b = percentile_matrix(df, BASIS_FEATURES, tr, te)      # basis only
    Ptr_l, Pte_l = percentile_matrix(df, cols_learn, tr, te)         # + energy
    idx = {c: i for i, c in enumerate(BASIS_FEATURES)}
    scores = {}

    # ── label-free candidates (no performance metric used at any stage) ──
    scores["single_edge (sum_sobel_bp)"] = Pte_b[:, idx[EDGE_REP]]
    scores["single_wepl (wepl_std)"] = Pte_b[:, idx[WEPL_REP]]
    scores["physics_prior_equal"] = Pte_b.mean(axis=1)
    edge, wepl = Pte_b[:, idx[EDGE_REP]], Pte_b[:, idx[WEPL_REP]]
    scores["noisy_OR (edge,wepl)"] = 1.0 - (1.0 - edge) * (1.0 - wepl)

    # ── supervised: linear, correlation-weighted (weights fit on TRAIN) ──
    def corr_w(target):
        w = np.array([abs(spearmanr(Ptr_b[:, j], target[tr]).correlation)
                      for j in range(Ptr_b.shape[1])])
        w = np.nan_to_num(w)
        return w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    scores["linear_gpr_weighted (A)"] = Pte_b @ corr_w(e_gpr)
    scores["linear_range_weighted (B)"] = Pte_b @ corr_w(e_rng)

    # ── supervised: learned heads (fit on TRAIN tail labels) ──
    def logit(tau):
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(Ptr_l, tau[tr])
        return m.predict_proba(Pte_l)[:, 1], m.predict_proba(Ptr_l)[:, 1]
    s_gpr_te, s_gpr_tr = logit(tau_gpr)
    s_rng_te, s_rng_tr = logit(tau_rng)
    scores["logistic_gpr (D)"] = s_gpr_te
    scores["logistic_range"] = s_rng_te
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         class_weight="balanced", random_state=0)
    gbm.fit(Ptr_l, tau_gpr[tr])
    scores["gbm_gpr"] = gbm.predict_proba(Pte_l)[:, 1]

    # ── dual-target: rank-max of the two learned heads ──
    def rank01(a):
        r = np.empty(len(a))
        r[np.argsort(a)] = np.arange(len(a))
        return r / max(len(a) - 1, 1)
    scores["dual_rankmax (gpr,range)"] = np.maximum(rank01(s_gpr_te), rank01(s_rng_te))
    return scores


def make_folds(df, split, n_folds):
    """Yield (fold_label, train_idx, test_idx) for the requested split mode."""
    if split == "gantry":
        groups = df["gantry_angle_deg"].round().astype(int).values
        gkf = GroupKFold(n_splits=n_folds)
        for i, (tr, te) in enumerate(gkf.split(df, groups=groups)):
            yield f"fold{i}", tr, te
    elif split == "anatomy":
        # leave-one-anatomy-out: train on one anatomy, test on the other
        anat = df["anatomy"].values
        for held in sorted(pd.unique(anat)):
            te = np.where(anat == held)[0]
            tr = np.where(anat != held)[0]
            yield f"test={held}", tr, te
    elif split == "patient":
        groups = df["patient_key"].values
        gkf = GroupKFold(n_splits=min(n_folds, len(pd.unique(groups))))
        for i, (tr, te) in enumerate(gkf.split(df, groups=groups)):
            yield f"fold{i}", tr, te
    else:
        raise ValueError(f"unknown split {split!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/mstryja/adota_runs/20260707_124010/results.csv")
    ap.add_argument("--outdir", default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition")
    ap.add_argument("--split", choices=["gantry", "anatomy", "patient"], default="anatomy")
    ap.add_argument("--anatomy-map", default=None,
                    help="CSV with sample_id,anatomy[,patient_key] provenance")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.results).reset_index(drop=True)
    if args.anatomy_map:
        prov = pd.read_csv(args.anatomy_map)
        df = df.merge(prov, on="sample_id", how="left")
        miss = df["anatomy"].isna().sum() if "anatomy" in df else len(df)
        if miss:
            raise SystemExit(f"{miss} beamlets have no provenance; check --anatomy-map")

    # offline labels (used only to calibrate supervised candidates + validate)
    e_gpr = (100.0 - df["gpr"].values)
    rr = df["r100_delta_mm"].abs().values
    valid = rr < RANGE_VALID_MAX_MM
    e_rng = np.where(valid, rr, np.nan)
    tau_gpr = (df["gpr"].values < GPR_FAIL).astype(int)
    tau_rng = ((rr > RANGE_FAIL_MM) & valid).astype(int)

    per_fold = {}          # name -> list of {metric: value}
    fold_labels = []       # per-fold identity (for anatomy transfer readout)
    for label, tr, te in make_folds(df, args.split, args.folds):
        fold_labels.append(label)
        scores = build_candidates(df, LEARN_FEATURES, tr, te,
                                  tau_gpr, tau_rng, e_gpr, e_rng)
        v = valid[te]
        for name, s in scores.items():
            rec = per_fold.setdefault(name, [])
            g = evaluate(s, tau_gpr[te])
            r = evaluate(s[v], tau_rng[te][v])
            rec.append({"fold": label,
                        **{f"gpr_{k}": val for k, val in g.items()},
                        **{f"rng_{k}": val for k, val in r.items()}})

    base_gpr = tau_gpr.mean() * 100
    base_rng = tau_rng[valid].mean() * 100
    rows = []
    for name, recs in per_fold.items():
        d = pd.DataFrame(recs).drop(columns=["fold"])
        row = {"candidate": name}
        for col in d.columns:
            row[col] = d[col].mean()
            row[col + "_sd"] = d[col].std()
        rows.append(row)
    res = pd.DataFrame(rows).sort_values("gpr_lift10", ascending=False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = args.split
    res.to_csv(outdir / f"ranking_wave1_{suffix}.csv", index=False)
    # per-fold detail (matters for anatomy transfer, where folds are asymmetric)
    detail = pd.concat([pd.DataFrame(r).assign(candidate=n) for n, r in per_fold.items()])
    detail.to_csv(outdir / f"ranking_wave1_{suffix}_perfold.csv", index=False)

    def fmt(name, key):
        r = res.loc[res.candidate == name].iloc[0]
        return f"{r[key]:.2f}±{r[key+'_sd']:.2f}"
    print(f"Split: {args.split}   folds: {fold_labels}")
    print(f"Base rates (pooled): GPR<{GPR_FAIL} = {base_gpr:.2f}%   "
          f"|dR100|>{RANGE_FAIL_MM}mm = {base_rng:.2f}% (valid)")
    print(f"{df.shape[0]} beamlets\n")
    hdr = f"{'candidate':30} {'GPR AUROC':>11} {'GPR lift10':>11} {'GPR lift5':>10} " \
          f"{'RNG AUROC':>11} {'RNG lift10':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name in res.candidate:
        print(f"{name:30} {fmt(name,'gpr_auroc'):>11} {fmt(name,'gpr_lift10'):>11} "
              f"{fmt(name,'gpr_lift5'):>10} {fmt(name,'rng_auroc'):>11} "
              f"{fmt(name,'rng_lift10'):>11}")
    if args.split == "anatomy":
        print("\nPer-fold (transfer direction) GPR lift10:")
        for name in res.candidate:
            d = pd.DataFrame(per_fold[name])
            cells = "  ".join(f"{r['fold']}: {r['gpr_lift10']:.2f}" for _, r in d.iterrows())
            print(f"  {name:28} {cells}")
    print(f"\nwrote {outdir/f'ranking_wave1_{suffix}.csv'}")


if __name__ == "__main__":
    main()
