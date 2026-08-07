"""Phantom illustration of the edge metrics: shapes/orientations -> anisotropy, orientation."""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter

INK, MUTED, GRID, BASE, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, RED = "#2a78d6", "#e34948"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 11, "font.family": "sans-serif"})
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
N = 48
zz, yy, xx = np.indices((N, N, N))  # axis 0 = z = BEAM direction
cz = cy = cx = N // 2


def st_metrics(vol):
    """Structure-tensor descriptors (matches src/metrics/sobel.py):
    anisotropy A=(l1-l3)/sum, orientation theta=arccos|v1.z_hat|, edge_energy=trace,
    lateral fraction = sin^2(theta) = share of edge energy across the beam."""
    g = np.stack([sobel(vol, 0), sobel(vol, 1), sobel(vol, 2)], axis=-1).reshape(-1, 3)
    J = g.T @ g
    w, v = np.linalg.eigh(J)
    lam = w[::-1]
    v1 = v[:, 2]
    A = float((lam[0] - lam[2]) / lam.sum())
    theta = float(np.degrees(np.arccos(min(1.0, abs(v1[0])))))
    lat_frac = float(np.sin(np.radians(theta)) ** 2)
    return A, theta, lat_frac


def phantom(kind):
    v = np.zeros((N, N, N), np.float32)
    r = 11
    if kind == "sphere":
        v[(zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2 <= r*r] = 1000
    elif kind == "slab_perp":      # thin in z -> interface crossed head-on
        v[cz-5:cz+5, :, :] = 1000
    elif kind == "wall_par":       # thin in x -> beam grazes a lateral interface
        v[:, :, cx-5:cx+5] = 1000
    elif kind == "cyl_par":        # cylinder axis along beam (z)
        v[(yy-cy)**2 + (xx-cx)**2 <= r*r] = 1000
    elif kind == "cuboid_45":      # cuboid rotated 45 deg in the z-x plane
        u = (zz - cz) + (xx - cx)
        w_ = (zz - cz) - (xx - cx)
        v[(np.abs(u) < 9) & (np.abs(w_) < 9) & (np.abs(yy - cy) < 11)] = 1000
    return gaussian_filter(v, 1.0)


CASES = [
    ("sphere",    "Sphere\n(isotropic)"),
    ("slab_perp", "Slab across beam\n(head-on interface)"),
    ("wall_par",  "Wall along beam\n(grazing interface)"),
    ("cyl_par",   "Cylinder along beam"),
    ("cuboid_45", "Cuboid 45°"),
]

fig, axes = plt.subplots(2, len(CASES), figsize=(15, 6.2))
fig.subplots_adjust(left=0.05, right=0.995, top=0.80, bottom=0.14, wspace=0.12, hspace=0.28)
for j, (kind, label) in enumerate(CASES):
    v = phantom(kind)
    A, theta, lat = st_metrics(v)
    gmag = np.sqrt(sobel(v, 0)**2 + sobel(v, 1)**2 + sobel(v, 2)**2)
    sl_v = v[:, cy, :].T        # (x, z): beam horizontal (z), lateral vertical (x)
    sl_g = gmag[:, cy, :].T
    ax0, ax1 = axes[0, j], axes[1, j]
    ax0.imshow(sl_v, cmap="gray", origin="lower", aspect="equal")
    ax0.set_title(label, fontsize=11.5, color=INK)
    ax1.imshow(sl_g, cmap="hot", origin="lower", aspect="equal")
    # beam arrow (left->right = depth)
    ax0.annotate("", xy=(N*0.92, N*0.06), xytext=(N*0.08, N*0.06),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax0.text(N*0.5, N*0.13, "beam", color=BLUE, fontsize=8.5, ha="center", va="bottom")
    for ax in (ax0, ax1):
        ax.set_xticks([]); ax.set_yticks([])
    # metrics under the sobel row (orientation is undefined when nearly isotropic)
    if A < 0.12:
        txt = f"anisotropy A = {A:.2f}  (low)\norientation θ = n/a (isotropic)\nno preferred edge direction"
        col = INK
    else:
        txt = (f"anisotropy A = {A:.2f}\norientation θ = {theta:.0f}°\n"
               f"lateral share sin²θ = {lat*100:.0f}%")
        col = RED if lat > 0.6 else (BLUE if lat < 0.3 else INK)
    ax1.set_xlabel(txt, fontsize=9.5, color=col, labelpad=6)
axes[0, 0].set_ylabel("phantom (HU)", fontsize=11)
axes[1, 0].set_ylabel("edge |∇HU|", fontsize=11)
fig.suptitle("Edge metrics on simple phantoms — anisotropy = how directional the edges are; "
             "orientation θ = edge direction vs the beam", fontsize=13.5, fontweight="bold", y=0.955)
fig.text(0.5, 0.885, "Blue label = edges ALONG the beam (head-on, low range mixing)   ·   "
         "Red label = edges ACROSS the beam (grazing → range mixing, high lateral_edge_energy)",
         ha="center", fontsize=10, color=MUTED)
fig.savefig(OUT / "edge_phantoms.png", dpi=150)
print("wrote edge_phantoms.png")
for kind, _ in CASES:
    A, theta, lat = st_metrics(phantom(kind))
    print(f"  {kind:12} A={A:.2f} theta={theta:5.1f} lat={lat*100:4.0f}%")
