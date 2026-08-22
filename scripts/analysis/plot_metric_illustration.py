"""Visual illustration of how the key metrics are computed on one beamlet."""
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import sobel as ndsobel

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scripts.training_set_analysis_advanced_metrics import analyse_density_regions, compute_advanced_metrics
from src.figures.ct_visualizations import HU_LUT, segment_hu
from src.loaders.generator import H5PYGenerator
from src.metrics.sobel import compute_sobel_metrics
from src.processing.pflugfelder_hi import compute_pflugfelder_hi, compute_wepl_map
from src.utils.dose_grid_utils import estimate_bp_range
from src.utils.scallers import inverse_minmax

UUID = "10432ff0-e33b-48f4-ba14-29e771718abb"
H5 = "/scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
S = dict(min_ct=-1024., max_ct=3071., min_ds=0., max_ds=25277028.)
RES = (2., 2., 2.)
DZ = RES[0]
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, RED = "#2a78d6", "#e34948"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False})

ds = H5PYGenerator(file_path=H5, augmentation=False, cropp=True, normalize=False, normalize_flux_only=True)
i = list(ds.record_ids).index(UUID)
x, _e, y = ds[i]
ct = inverse_minmax(x[0].numpy(), S["min_ct"], S["max_ct"])
flux = x[1].numpy()
dose = inverse_minmax(np.squeeze(y.numpy()), S["min_ds"], S["max_ds"])

z_min, z_max = estimate_bp_range(ct, dose)
k0, k1 = int(np.ceil(z_min)), int(np.floor(z_max))
bp_idx = int(np.argmax(dose.sum(axis=(1, 2))))

# flux-weighted mean-HU profile over BP zone
depths = np.arange(k0, k1 + 1) * DZ
mean_hu = np.zeros(k1 - k0 + 1)
for j, k in enumerate(range(k0, k1 + 1)):
    fs = np.abs(flux[k])
    fm = fs.max()
    if fm < 1e-12:
        mean_hu[j] = ct[k].mean()
        continue
    m = fs >= 0.10 * fm
    mean_hu[j] = np.average(ct[k][m], weights=fs[m]) if m.sum() else ct[k].mean()
cls = segment_hu(mean_hu)

nreg, total_hu_change, regions = analyse_density_regions(ct, flux, z_min, z_max)
adv = compute_advanced_metrics(ct, flux, dose, z_min, z_max, regions)
sob = compute_sobel_metrics(ct, flux, z_min, z_max)
wepl = compute_wepl_map(ct, RES, bp_idx * DZ)
pf = compute_pflugfelder_hi(wepl, flux.sum(axis=0))
print("computed:", dict(total_hu_change=round(total_hu_change,1), **{k: round(v,3) for k,v in adv.items()},
      sum_sobel_bp=round(sob["sum_sobel_bp"],0), wepl_std=round(pf["wepl_std"],2), wepl_mean=round(pf["wepl_mean"],1)))

# ---------- figure ----------
fig = plt.figure(figsize=(13, 8.5))
gs = GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.22,
              left=0.07, right=0.965, top=0.9, bottom=0.08)

# Panel A: HU-along-beam profile with tissue regions + annotations
axA = fig.add_subplot(gs[0, :])
for r in regions:                       # shade each tissue region
    a, b = r["start_slice"] * DZ, (r["end_slice"] + 1) * DZ
    axA.axvspan(a, b, color=HU_LUT[r["class_idx"]][3], alpha=0.75, lw=0)
axA.plot(depths, mean_hu, color=INK, lw=2.2, zorder=5)
axA.scatter(depths, mean_hu, s=10, color=INK, zorder=6)
# biggest jump between consecutive regions
if len(regions) >= 2:
    diffs = [abs(regions[j]["mean_hu"] - regions[j-1]["mean_hu"]) for j in range(1, len(regions))]
    jmax = int(np.argmax(diffs)) + 1
    xj = regions[jmax]["start_slice"] * DZ
    y0, y1 = regions[jmax-1]["mean_hu"], regions[jmax]["mean_hu"]
    axA.annotate("", xy=(xj, y1), xytext=(xj, y0),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=2.2))
    axA.text(xj + 3, (y0+y1)/2, f"max_hu_jump\n= {adv['max_hu_jump']:.0f} HU",
             color=RED, fontsize=9.5, va="center", fontweight="bold")
axA.axvline(bp_idx * DZ, color=BLUE, ls="--", lw=1.8, zorder=4)
axA.text(bp_idx * DZ, axA.get_ylim()[1], " Bragg peak", color=BLUE, fontsize=9.5, va="top", ha="left")
axA.set_xlabel("depth along beam  [mm]")
axA.set_ylabel("flux-weighted mean HU")
axA.set_title("HU-along-beam profile  H(k)  —  drives total_hu_change, max_hu_jump, sigma_hu_bp, hetero_fraction",
              fontsize=12.5, color=INK, pad=8)
axA.grid(axis="y", color=GRID, lw=0.7)
txt = (f"total_hu_change = {total_hu_change:.0f} HU  (Σ region jumps)\n"
       f"sigma_hu_bp = {adv['sigma_hu_bp']:.0f} HU  (spread of H)\n"
       f"hetero_fraction = {adv['hetero_fraction']:.2f}\n"
       f"n_density_regions = {nreg}")
axA.text(0.40, 0.70, txt, transform=axA.transAxes, fontsize=9.5, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec=BASE, alpha=0.9))
# tissue legend
seen = {}
for r in regions:
    seen[r["class_idx"]] = HU_LUT[r["class_idx"]][0]
handles = [plt.Rectangle((0,0),1,1, color=HU_LUT[c][3]) for c in seen]
axA.legend(handles, list(seen.values()), title="tissue", loc="upper left",
           fontsize=8.5, title_fontsize=9, frameon=True, ncol=len(seen))

# Panel B: WEPL lateral map
axB = fig.add_subplot(gs[1, 0])
mask2d = flux.sum(axis=0) > 0.10 * flux.sum(axis=0).max()
wepl_show = np.where(mask2d, wepl, np.nan)
im = axB.imshow(wepl_show, cmap="viridis", origin="lower", aspect="auto")
axB.set_title(f"WEPL map to Bragg peak  —  wepl_std = {pf['wepl_std']:.1f} mm\n"
              f"(lateral spread of water-equiv. path length; wepl_mean = {pf['wepl_mean']:.0f} mm)",
              fontsize=11, color=INK)
axB.set_xlabel("lateral x [voxel]")
axB.set_ylabel("lateral y [voxel]")
cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
cb.set_label("WEPL [mm]", fontsize=9)

# Panel C: Sobel gradient magnitude (sagittal slice through BP zone)
gmag = np.sqrt(ndsobel(ct,0)**2 + ndsobel(ct,1)**2 + ndsobel(ct,2)**2)
midy = ct.shape[1] // 2
axC = fig.add_subplot(gs[1, 1])
imc = axC.imshow(gmag[k0:k1+1, midy, :].T, cmap="hot", origin="lower", aspect="auto")
axC.set_title("Sobel edge magnitude |∇HU| near the Bragg peak\n"
              "(tissue edges the beam crosses — feeds sum_sobel_bp / lateral_edge_energy)", fontsize=11, color=INK)
axC.set_xlabel("depth along beam [slice]")
axC.set_ylabel("lateral x [voxel]")
cbc = fig.colorbar(imc, ax=axC, fraction=0.046, pad=0.04)
cbc.set_label("|∇HU|", fontsize=9)

fig.suptitle(f"How the metrics are computed — one heterogeneous beamlet ({UUID[:8]}, 174.6 MeV)",
             fontsize=14, fontweight="bold", color=INK, y=0.965)
fig.savefig(OUT / "fig4_metric_illustration.png", dpi=150)
print("wrote", OUT / "fig4_metric_illustration.png")
