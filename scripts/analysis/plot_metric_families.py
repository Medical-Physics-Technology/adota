"""Per-family 'input -> how it is computed' figures for the metrics in diff_score."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import sobel as ndsobel

from scripts.training_set_analysis_advanced_metrics import analyse_density_regions, compute_advanced_metrics
from src.figures.ct_visualizations import HU_LUT, segment_hu
from src.loaders.generator import H5PYGenerator
from src.processing.mcsquare_calibration import hu_to_rsp_mcsquare
from src.processing.pflugfelder_hi import compute_pflugfelder_hi, compute_wepl_map
from src.utils.dose_grid_utils import estimate_bp_range
from src.utils.scallers import inverse_minmax

UUID = "10432ff0-e33b-48f4-ba14-29e771718abb"
H5 = "/scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
S = dict(min_ct=-1024., max_ct=3071., min_ds=0., max_ds=25277028.)
RES = (2., 2., 2.)
DZ = 2.0
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
INK, MUTED, GRID, BASE, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, RED, ORANGE = "#2a78d6", "#e34948", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 10.5, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False})

ds = H5PYGenerator(file_path=H5, augmentation=False, cropp=True, normalize=False, normalize_flux_only=True)
i = list(ds.record_ids).index(UUID)
x, _e, y = ds[i]
ct = inverse_minmax(x[0].numpy(), S["min_ct"], S["max_ct"])
flux = x[1].numpy()
dose = inverse_minmax(np.squeeze(y.numpy()), S["min_ds"], S["max_ds"])
D, H, W = ct.shape
midy = H // 2
z_min, z_max = estimate_bp_range(ct, dose)
k0, k1 = int(np.ceil(z_min)), int(np.floor(z_max))
bp_idx = int(np.argmax(dose.sum(axis=(1, 2))))
nreg, total_hu_change, regions = analyse_density_regions(ct, flux, z_min, z_max)
adv = compute_advanced_metrics(ct, flux, dose, z_min, z_max, regions)

# full-depth flux-weighted mean-HU profile (for display); metrics use the BP zone
def profile(a, b):
    prof = np.full(b - a, np.nan)
    for j, k in enumerate(range(a, b)):
        fs = np.abs(flux[k])
        fm = fs.max()
        if fm < 1e-12:
            continue
        m = fs >= 0.10 * fm
        if m.sum():
            prof[j] = np.average(ct[k][m], weights=fs[m])
    return prof
Hk = profile(0, D)
depth_mm = np.arange(D) * DZ

def sag(vol): return vol[:, midy, :].T  # (lateral_x, depth)


# ============================ FAMILY A: HU-along-beam profile ============================
fig = plt.figure(figsize=(13, 8.2))
gs = GridSpec(2, 1, height_ratios=[0.6, 1.0], hspace=0.30, left=0.08, right=0.97, top=0.93, bottom=0.26)
axI = fig.add_subplot(gs[0])
axP = fig.add_subplot(gs[1], sharex=None)
# input: CT sagittal + flux overlay
axI.imshow(sag(ct), cmap="gray", origin="lower", aspect="auto",
           extent=[0, D * DZ, 0, W], vmin=-1000, vmax=800)
fl = sag(flux)
fl = np.ma.masked_less(fl, 0.10 * fl.max())
axI.imshow(fl, cmap="autumn", origin="lower", aspect="auto", extent=[0, D * DZ, 0, W], alpha=0.55)
axI.axvspan(k0 * DZ, k1 * DZ, color=BLUE, alpha=0.10)
axI.set_ylabel("lateral [vox]")
axI.set_title(
    "INPUT: CT (grayscale) + beam flux (orange) — sagittal.  The metrics read the density ALONG the beam.",
    fontsize=11.5, color=INK)
axI.tick_params(labelbottom=False)
# profile with tissue shading + annotations
for k in range(D - 1):
    c = int(segment_hu(np.array([Hk[k]]))[0]) if np.isfinite(Hk[k]) else 0
    axP.axvspan(k * DZ, (k + 1) * DZ, color=HU_LUT[c][3], alpha=0.5, lw=0)
axP.axvspan(k0 * DZ, k1 * DZ, color=BLUE, alpha=0.06)
axP.plot(depth_mm, Hk, color=INK, lw=2.0, zorder=5)
# sigma band
mu = np.nanmean(Hk[k0:k1])
sg = adv["sigma_hu_bp"]
axP.axhspan(mu - sg, mu + sg, xmin=(k0*DZ)/(D*DZ), xmax=(k1*DZ)/(D*DZ), color=ORANGE, alpha=0.12, zorder=1)
axP.text(k0*DZ + 4, mu + sg, f"±sigma_hu_bp ({sg:.0f})", color=ORANGE, fontsize=9, va="bottom", ha="left")
# biggest jump
diffs = [abs(regions[j]["mean_hu"] - regions[j-1]["mean_hu"]) for j in range(1, len(regions))]
jm = int(np.argmax(diffs)) + 1
xj = regions[jm]["start_slice"] * DZ
axP.annotate("", xy=(xj, regions[jm]["mean_hu"]), xytext=(xj, regions[jm-1]["mean_hu"]),
             arrowprops=dict(arrowstyle="<->", color=RED, lw=2.2))
axP.annotate(
    f"max_hu_jump\n= {adv['max_hu_jump']:.0f} HU",
    xy=(xj, np.mean([regions[jm]['mean_hu'], regions[jm-1]['mean_hu']])),
         xytext=(-8,0), textcoords='offset points', color=RED, fontsize=9.5, va='center', ha='right', fontweight='bold')
# interface -> BP distance
axP.axvline(bp_idx * DZ, color=BLUE, ls="--", lw=1.6)
axP.text(bp_idx*DZ - 4, np.nanmax(Hk), "Bragg peak", color=BLUE, fontsize=9, va="top", ha="right")
axP.set_xlabel("depth along beam [mm]")
axP.set_ylabel("H(k): flux-weighted mean HU")
axP.set_title("HOW IT IS COMPUTED: segment the profile into tissue regions; metrics summarize its jumps & spread",
              fontsize=11.5, color=INK)
box = (f"total_hu_change = {total_hu_change:.0f}  (Σ |region jumps|)      "
       f"max_hu_gradient = {adv['max_hu_gradient']:.0f}  (steepest slice step)      "
       f"hetero_fraction = {adv['hetero_fraction']:.2f}  (fraction off dominant tissue)\n"
       f"n_density_regions = {nreg}       "
       f"interface_bp_distance = {adv['interface_bp_distance']:.1f} slices")
axI.set_xlim(0, D*DZ + 10)
axP.set_xlim(0, D*DZ + 10)
# tissue legend and metric summary placed BELOW the x-axis, so neither overlaps the profile
seen = {}
[seen.setdefault(int(segment_hu(np.array([h]))[0]), True) for h in Hk if np.isfinite(h)]
handles = [plt.Rectangle((0, 0), 1, 1, color=HU_LUT[c][3]) for c in seen]
labels = [HU_LUT[c][0] for c in seen]
fig.legend(handles, labels, title="tissue", loc="lower center", bbox_to_anchor=(0.5, 0.10),
           ncol=len(labels), fontsize=9.5, title_fontsize=10, frameon=True,
           columnspacing=1.5, handlelength=1.3, borderpad=0.6)
fig.text(0.5, 0.02, box, ha="center", va="bottom", fontsize=9.5, color=INK,
         bbox=dict(boxstyle="round", fc="white", ec=BASE, alpha=0.95))
fig.savefig(OUT / "family_hu_profile.png", dpi=150)
plt.close(fig)
print("family_hu_profile.png")


# ============================ FAMILY B: WEPL ============================
rsp = hu_to_rsp_mcsquare(ct)
wepl = compute_wepl_map(ct, RES, bp_idx * DZ)
pf = compute_pflugfelder_hi(wepl, flux.sum(axis=0))
mask2d = flux.sum(axis=0) > 0.10 * flux.sum(axis=0).max()
fig = plt.figure(figsize=(13, 4.6))
gs = GridSpec(1, 4, width_ratios=[1, 1, 0.9, 0.7],
    wspace=0.42, left=0.05, right=0.97, top=0.82, bottom=0.16)
a0 = fig.add_subplot(gs[0])
im0 = a0.imshow(sag(ct), cmap="gray", origin="lower", aspect="auto", vmin=-1000, vmax=800)
a0.set_title("1) CT (HU)")
a0.set_xlabel("depth [vox]")
a0.set_ylabel("lateral")
a1 = fig.add_subplot(gs[1])
im1 = a1.imshow(sag(rsp), cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=1.8)
a1.set_title("2) RSP = ρ·S_mat/S_water")
a1.set_xlabel("depth [vox]")
fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)
a2 = fig.add_subplot(gs[2])
wp = np.where(mask2d, wepl, np.nan)
im2 = a2.imshow(wp, cmap="viridis", origin="lower", aspect="auto")
a2.set_title("3) WEPL map  W(i,j)")
a2.set_xlabel("lat x")
a2.set_ylabel("lat y")
fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04, label="mm")
a3 = fig.add_subplot(gs[3])
vals = wepl[mask2d]
a3.hist(vals, bins=25, color=BLUE, alpha=0.85, orientation="horizontal")
a3.axhline(pf["wepl_mean"], color=INK, lw=1.5)
a3.axhspan(pf["wepl_mean"]-pf["wepl_std"], pf["wepl_mean"]+pf["wepl_std"], color=ORANGE, alpha=0.25)
a3.set_title(f"WEPL spread\nstd={pf['wepl_std']:.1f} mm")
a3.set_xlabel("count")
fig.suptitle("WEPL family — accumulate relative stopping power along the beam; wepl_std = lateral spread of range "
             "('half bone / half air')", fontsize=12.5, fontweight="bold", y=0.97)
fig.savefig(OUT / "family_wepl.png", dpi=150)
plt.close(fig)
print("family_wepl.png")


# ============================ FAMILY C: edges (Sobel + structure tensor) ============================
gmag = np.sqrt(ndsobel(ct, 0)**2 + ndsobel(ct, 1)**2 + ndsobel(ct, 2)**2)
fig = plt.figure(figsize=(13, 4.6))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1],
    wspace=0.3, left=0.05, right=0.965, top=0.82, bottom=0.16)
b0 = fig.add_subplot(gs[0])
b0.imshow(sag(ct), cmap="gray", origin="lower", aspect="auto", vmin=-1000, vmax=800)
b0.set_title("1) CT (HU)")
b0.set_xlabel("depth [vox]")
b0.set_ylabel("lateral")
b1 = fig.add_subplot(gs[1])
im = b1.imshow(sag(gmag), cmap="hot", origin="lower", aspect="auto")
b1.set_title("2) |∇HU| (3-D Sobel)")
b1.set_xlabel("depth [vox]")
fig.colorbar(im, ax=b1, fraction=0.046, pad=0.04)
# along-beam vs across-beam gradient share
gz = np.abs(ndsobel(ct, 0))
glat = np.sqrt(ndsobel(ct, 1)**2 + ndsobel(ct, 2)**2)
b2 = fig.add_subplot(gs[2])
seg = slice(k0, k1 + 1)
m = (flux[seg].sum() > 0)
along = float(gz[seg].sum())
across = float(glat[seg].sum())
b2.bar(["across beam\n(lateral)", "along beam\n(axial)"], [across, along], color=[RED, BLUE], width=0.6)
b2.set_title("3) edge orientation\n(structure tensor → anisotropy, θ)")
b2.set_ylabel("Σ|∇HU| in BP zone")
b2.text(0, across, "lateral_edge_energy\n∝ tr(J)·sin²θ", ha="center", va="bottom", fontsize=8.5, color=RED)
fig.suptitle("Edge family — |∇HU| highlights the tissue boundaries the beam crosses",
             fontsize=13, fontweight="bold", y=0.995)
fig.savefig(OUT / "family_edges.png", dpi=150)
plt.close(fig)
print("family_edges.png")
print("done")
