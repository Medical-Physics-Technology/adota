"""Verify the analytic beam centerline against the flux ridge (raw record frame).

For a handful of records it (1) builds the centerline from JSON entrance +
steering angles via src.beamlets.centerline, (2) measures the per-slice deviation
from the flux-weighted ridge, and (3) renders CT | flux | centerline_soft |
centerline_binary with the analytic line overlaid, for a visual correctness check.

Run: uv run python scripts/projection_ablation/verify_centerline.py
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/mstryja/projects/adota")
from src.beamlets.centerline import beam_line_from_metadata, render_centerline

H5 = "/scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
JDIRS = ["/RadiotherapyData/dataset_v0/trainset_pelvis",
         "/RadiotherapyData/dataset_v0/initial_test_one_ct"]
OUT = Path("/home/mstryja/projects/adota/research/figures/projection_ablation")
OUT.mkdir(parents=True, exist_ok=True)
INK, SURF = "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.size": 10, "text.color": INK})


def find_json(uid):
    for d in JDIRS:
        p = os.path.join(d, uid + "_metadata.json")
        if os.path.exists(p):
            return p
    return None


def flux_ridge(flux):
    H, W, D = flux.shape
    ii = np.arange(H)[:, None]; jj = np.arange(W)[None, :]
    c0 = np.full(D, np.nan); c1 = np.full(D, np.nan)
    for d in range(D):
        s = flux[:, :, d]; tot = s.sum()
        if tot > 0:
            c0[d] = (s * ii).sum() / tot; c1[d] = (s * jj).sum() / tot
    return c0, c1


def main():
    # pick a spread of steering magnitudes
    with h5py.File(H5, "r") as f:
        keys = list(f.keys())
        cand = keys[:400]
        rows = []
        for k in cand:
            ba = np.array(f[k].attrs["beamlet_angles"], dtype=float)
            rows.append((k, float(np.hypot(*ba))))
        rows.sort(key=lambda r: r[1])
        picks = [rows[0][0], rows[len(rows) // 2][0], rows[-1][0],
                 rows[-2][0], rows[-8][0]]

        fig, axes = plt.subplots(len(picks), 4, figsize=(15, 3.0 * len(picks)))
        print(f"{'uuid[:8]':10} {'steer(θx,θy)':20} {'max|dev| axis0,axis1 [vox]':28} {'mean|dev|':10}")
        for r, k in enumerate(picks):
            g = f[k]
            ct = g["ct"][:]; flux = g["flux"][:]
            ba = np.array(g.attrs["beamlet_angles"], dtype=float)
            jp = find_json(k); meta = json.load(open(jp))
            roi = meta.get("roi_size")
            ds = (roi[0] / ct.shape[0]) if roi else 2.0
            line = beam_line_from_metadata(meta["rays_entrence_point"], ba, ds)

            # agreement vs flux ridge
            D = flux.shape[2]; d = np.arange(D)
            c0f, c1f = flux_ridge(flux)
            c0l, c1l = line.lateral_center(d)
            m = np.isfinite(c0f)
            dev0 = np.abs(c0l[m] - c0f[m]); dev1 = np.abs(c1l[m] - c1f[m])
            print(f"{k[:8]:10} {str(np.round(ba,2).tolist()):20} "
                  f"{f'{dev0.max():.3f}, {dev1.max():.3f}':28} "
                  f"{f'{dev0.mean():.3f}, {dev1.mean():.3f}'}")

            soft = render_centerline(line, ct.shape, mode="soft", sigma=1.5)
            binr = render_centerline(line, ct.shape, mode="binary")

            a1c = int(round(np.nanmean(c1l)))  # slice through mean axis1
            panels = [("CT", ct[:, a1c, :], "gray"),
                      ("flux (MIP)", flux.max(1), "magma"),
                      ("centerline_soft (MIP)", soft.max(1), "magma"),
                      ("centerline_binary (MIP)", binr.max(1), "magma")]
            for c, (title, img, cmap) in enumerate(panels):
                ax = axes[r, c]
                # nearest interpolation: do not let bilinear upscaling alias the
                # smooth narrow band into a false staircase.
                ax.imshow(img, origin="lower", aspect="auto", cmap=cmap,
                          interpolation="nearest",
                          vmin=(-1000 if c == 0 else None), vmax=(800 if c == 0 else None))
                ax.plot(d, c0l, color="#2ad1c9", lw=1.0, ls="-")   # analytic (smooth) center
                ax.plot(d, c0f, color="#2a78d6", lw=1.0, ls="--")  # flux ridge centroid (smooth)
                if r == 0:
                    ax.set_title(title, fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{k[:8]}\nsteer={np.round(ba,2).tolist()}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Analytic centerline (blue dashed) vs flux — raw record frame; "
                     "axis0 vs depth (flux/centerline shown as MIP over axis1)",
                     fontsize=12, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(OUT / "centerline_vs_flux.png", dpi=140)
        print(f"\nwrote {OUT / 'centerline_vs_flux.png'}")


if __name__ == "__main__":
    main()
