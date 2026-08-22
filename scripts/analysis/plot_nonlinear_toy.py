"""Toy illustration for Methods: why the non-linear reference can exceed the linear score.

An idealized two-feature regression in which the target depends only on the
*interaction* of the two features (a checkerboard), with no main effect of either
feature alone. A linear model, being a sum of per-feature effects, cannot
represent it and reaches held-out Pearson near zero; the gradient-boosted tree
ensemble splits on both features and recovers it. This is the mechanism behind
the gap between the fitted linear score and the non-linear reference in Section 3.

Run: uv run --with scikit-learn python scripts/analysis/plot_nonlinear_toy.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

INK, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False})
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
rng = np.random.default_rng(0)

# two features on [0,1]; target is a pure interaction (no main effect of either)
n = 6000
U = rng.uniform(0, 1, (n, 2))
u, v = U[:, 0], U[:, 1]
y = 4.0 * (u - 0.5) * (v - 0.5) + rng.normal(0, 0.15, n)
tr = rng.random(n) < 0.7
te = ~tr

lin = Ridge(alpha=1e-3).fit(U[tr], y[tr])
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(U[tr], y[tr])
r_lin = pearsonr(lin.predict(U[te]), y[te])[0]
r_gbm = pearsonr(gbm.predict(U[te]), y[te])[0]
print(f"held-out Pearson: linear={r_lin:.3f}  gradient boosting={r_gbm:.3f}")

# evaluation grid for the surfaces
g = np.linspace(0, 1, 120)
GU, GV = np.meshgrid(g, g)
grid = np.column_stack([GU.ravel(), GV.ravel()])
true_s = 4.0 * (GU - 0.5) * (GV - 0.5)
lin_s = lin.predict(grid).reshape(GU.shape)
gbm_s = gbm.predict(grid).reshape(GU.shape)

vmax = 1.0
panels = [("true target y(u,v)\n(depends only on the interaction u·v)", true_s, None),
          (f"linear model  b + w₁u + w₂v\nheld-out Pearson r = {r_lin:.2f}", lin_s, r_lin),
          (f"gradient boosting (tree ensemble)\nheld-out Pearson r = {r_gbm:.2f}", gbm_s, r_gbm)]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
fig.subplots_adjust(left=0.06, right=0.9, top=0.80, bottom=0.13, wspace=0.28)
for ax, (title, S, _) in zip(axes, panels):
    im = ax.imshow(S, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("feature u  (e.g. depth percentile)")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
axes[0].set_ylabel("feature v  (e.g. heterogeneity percentile)")
cax = fig.add_axes([0.915, 0.13, 0.015, 0.67])
fig.colorbar(im, cax=cax, label="target / prediction")
fig.suptitle("Why a non-linear reference is needed: an interaction the linear score cannot represent",
             fontsize=13, fontweight="bold", x=0.48, y=0.965)
fig.savefig(OUT / "nonlinear_toy.png", dpi=150)
print("wrote nonlinear_toy.png")
