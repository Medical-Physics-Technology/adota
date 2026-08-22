"""Single-metric correlation study: motivation for fitting a combined score.

For each error target (relative dose error, mean absolute percentage error, and
gamma pass-rate error) this computes the Spearman rank correlation of every input
metric on its own, ranks the metrics, and draws one figure per target showing the
best single metrics against two reference lines: the correlation reached by the
fitted linear combination and by the non-linear reference. The story is that no
single metric is sufficient, which is why the metrics are combined by a fit.

All correlations are computed on the development pool (frozen-test patients
removed), consistent with the rest of the report.

Run: uv run --with scikit-learn python scripts/analysis/acquisition_single_metric_corr.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from scripts.analysis.acquisition_dev_analysis import cv, gbm, ridge
from scripts.analysis.acquisition_regression_study import ALL_FEATS

INK, MUTED, GRID, BASE, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, RED, GREEN = "#2a78d6", "#eb6834", "#e34948", "#008300"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False})

BASE_RUN = "/scratch/mstryja/adota_runs/20260707_124010"
ACQ = f"{BASE_RUN}/figures/acquisition"
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
TOPN = 12


def main():
    df = (pd.read_csv(f"{BASE_RUN}/results.csv")
          .merge(pd.read_csv(f"{ACQ}/uuid_provenance_map.csv"), on="sample_id", how="left")
          .merge(pd.read_csv(f"{ACQ}/mape_metric.csv"), on="sample_id", how="left")
          .reset_index(drop=True))
    test_ids = set(pd.read_csv(f"{ACQ}/frozen_test_ids.csv")["sample_id"])
    df = df[~df["sample_id"].isin(test_ids)].reset_index(drop=True)
    groups = df["patient_key"].values
    print(f"dev pool: {len(df)} beamlets, {len(pd.unique(groups))} patients")

    targets = [
        (
            "rde",
            "relative dose error (RDE)",
            df["rde"].values.astype(float),
            "single_corr_rde.png",
        ),
        (
            "mape_5pct",
            "mean absolute percentage error (MAPE, 5% mask)",
            df["mape_5pct"].values.astype(float),
            "single_corr_mape.png",
        ),
        (
            "egpr",
            "gamma pass-rate error (100 - GPR)",
            100.0 - df["gpr"].values.astype(float),
            "single_corr_gamma.png",
        ),
    ]

    summary = []
    for key, label, y, fname in targets:
        m = np.isfinite(y)
        # single-metric Spearman (magnitude-ranked, sign kept)
        rows = []
        for f in ALL_FEATS:
            v = df[f].values.astype(float)
            ok = m & np.isfinite(v)
            s = spearmanr(v[ok], y[ok]).correlation
            rows.append((f, s))
        srt = sorted(rows, key=lambda r: -abs(r[1]))
        best_f, best_s = srt[0]
        # multivariate fit references (dev CV Spearman)
        _, lin_s = cv(df, ALL_FEATS, y, m, groups, ridge)
        _, gbm_s = cv(df, ALL_FEATS, y, m, groups, gbm)
        summary.append(dict(target=key, best_metric=best_f, best_single_spearman=best_s,
                            linear_spearman=lin_s, gbm_spearman=gbm_s))
        print(f"\n=== {label} ===")
        print(f"  best single metric: {best_f}  Spearman={best_s:.3f}")
        print(f"  fitted linear (all metrics)  Spearman={lin_s:.3f}")
        print(f"  non-linear reference (GBM)   Spearman={gbm_s:.3f}")
        for f, s in srt[:TOPN]:
            print(f"    {f:22} {s:+.3f}")

        # ---- figure ----
        top = srt[:TOPN][::-1]  # smallest at bottom for barh
        names = [f for f, _ in top]
        vals = [s for _, s in top]
        colors = [BLUE if v >= 0 else ORANGE for v in vals]
        fig, ax = plt.subplots(figsize=(9.6, 6.2))
        fig.subplots_adjust(left=0.30, right=0.965, top=0.86, bottom=0.12)
        yy = np.arange(len(names))
        ax.barh(yy, [abs(v) for v in vals], color=colors, height=0.66, zorder=3)
        for i, v in enumerate(vals):
            ax.text(abs(v) + 0.008, i, f"{v:+.2f}", va="center", ha="left", fontsize=9.5, color=INK)
        ax.set_yticks(yy)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("|Spearman rank correlation| with the target")
        xmax = max(abs(gbm_s), max(abs(v) for v in vals)) + 0.12
        ax.set_xlim(0, xmax)
        # reference lines: fitted linear and non-linear (identified by a legend, no clipping)
        ax.axvline(abs(lin_s), color=RED, ls="--", lw=1.8, zorder=4)
        ax.axvline(abs(gbm_s), color=GREEN, ls=":", lw=1.8, zorder=4)
        handles = [Line2D([0], [0], color=RED, ls="--", lw=1.8,
                          label=f"fitted linear score = {lin_s:.2f}"),
                   Line2D([0], [0], color=GREEN, ls=":", lw=1.8,
                          label=f"non-linear reference = {gbm_s:.2f}")]
        ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9.5, borderpad=0.7)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(f"Single metrics vs {label}\nbest single metric reaches "
                     f"{abs(best_s):.2f}; the fitted combination reaches {abs(lin_s):.2f}",
                     fontsize=12.5, fontweight="bold")
        fig.savefig(OUT / fname, dpi=150)
        plt.close(fig)
        print(f"  wrote {OUT/fname}")

    pd.DataFrame(summary).to_csv(f"{ACQ}/single_metric_corr_summary.csv", index=False)
    print(f"\nwrote {ACQ}/single_metric_corr_summary.csv")


if __name__ == "__main__":
    main()
