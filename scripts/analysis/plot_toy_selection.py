"""Toy illustrations for 4.3: (1) redundancy clustering, (2) Lasso sparsity path."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import lasso_path

INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YEL, MAG, GRN, VIO, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                               "#e87ba4", "#008300", "#4a3aa7", "#e34948")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False})
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
rng = np.random.default_rng(0)
n = 800

# ============ TOY 1: redundancy / clustering ============
L = rng.normal(size=n)          # latent "length"
Wd = rng.normal(size=n)         # latent "width"
feats = {
    "length_cm":    L + 0.10*rng.normal(size=n),
    "length_cm_2":  1.01*L + 0.10*rng.normal(size=n),
    "length_inch":  0.39*L + 0.05*rng.normal(size=n),
    "width_cm":     Wd + 0.10*rng.normal(size=n),
    "width_inch":   0.39*Wd + 0.05*rng.normal(size=n),
    "weight_kg":    rng.normal(size=n),
}
names = list(feats)
X = np.column_stack([feats[k] for k in names])
C = np.corrcoef(X.T)
D = 1 - np.abs(C)
Z = linkage(squareform(D, checks=False), method="average")

fig, (axm, axd) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1, 1.05]))
fig.subplots_adjust(left=0.16, right=0.98, top=0.82, bottom=0.28, wspace=0.55)
im = axm.imshow(np.abs(C), cmap="Reds", vmin=0, vmax=1)
axm.set_xticks(range(6))
axm.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
axm.set_yticks(range(6))
axm.set_yticklabels(names, fontsize=9)
for i in range(6):
    for j in range(6):
        axm.text(j, i, f"{abs(C[i,j]):.2f}", ha="center", va="center",
                 color="white" if abs(C[i,j]) > 0.5 else INK2, fontsize=8)
axm.set_title("1) correlation between toy metrics\n(dark red = they move together)", fontsize=11.5)
fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04, label="|correlation|")

dn = dendrogram(Z, labels=names, ax=axd, color_threshold=0.5,
                above_threshold_color=MUTED, leaf_rotation=45, leaf_font_size=9)
axd.axhline(0.5, ls="--", lw=1.4, color=RED)
axd.text(0.02, 0.53, "cut here", color=RED, fontsize=9, transform=axd.get_yaxis_transform())
axd.set_ylabel("distance  (1 − |correlation|)", fontsize=10)
axd.set_title("2) cluster them, then KEEP ONE PER GROUP\n"
              "→ 3 independent axes instead of 6", fontsize=11.5)
axd.spines["left"].set_visible(True)
fig.suptitle("Remove redundancy — toy example: 6 metrics, but only 3 really-different things "
             "(length, width, weight)", fontsize=12.5, fontweight="bold", y=0.965)
fig.savefig(OUT / "toy_redundancy.png", dpi=150)
plt.close(fig)
print("wrote toy_redundancy.png")

# ============ TOY 2: Lasso sparsity path ============
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
Z2 = np.column_stack([
    z1,                                   # informative (big weight)
    z2,                                   # informative (small weight)
    0.9*z1 + 0.15*rng.normal(size=n),     # redundant with z1
    rng.normal(size=n),                   # useless
    rng.normal(size=n),                   # useless
    rng.normal(size=n),                   # useless
])
labels = ["informative #1", "informative #2", "redundant w/ #1",
          "useless a", "useless b", "useless c"]
cols = [BLUE, ORANGE, AQUA, MUTED, MUTED, MUTED]
y = 3.0*z1 + 1.5*z2 + 0.5*rng.normal(size=n)
Z2 = (Z2 - Z2.mean(0)) / Z2.std(0)
alphas, coefs, _ = lasso_path(Z2, y, n_alphas=60)
xa = -np.log10(alphas)

fig, ax = plt.subplots(figsize=(8.6, 4.8))
fig.subplots_adjust(left=0.09, right=0.72, top=0.88, bottom=0.14)
for k in range(6):
    ax.plot(xa, coefs[k], color=cols[k], lw=2.4 if k < 3 else 1.6,
            ls="-" if k < 3 else ":", label=labels[k], zorder=5 if k < 3 else 3)
ax.axhline(0, color=BASE, lw=1)
ax.set_xlabel("← stronger penalty      (relax penalty →)")
ax.set_ylabel("metric weight in the score")
ax.set_title("Keep only what pays — toy Lasso path:\nuseless metrics are driven to exactly 0; "
             "of two redundant metrics, one is kept", fontsize=11.5)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9.5)
ax.grid(color=GRID, lw=0.7)
ax.text(1.15, 0.16, "useless & 1-of-2 redundant metrics  →  weight 0",
        fontsize=9.5, color=INK2, style="italic")
fig.savefig(OUT / "toy_sparsity.png", dpi=150)
plt.close(fig)
print("wrote toy_sparsity.png")
