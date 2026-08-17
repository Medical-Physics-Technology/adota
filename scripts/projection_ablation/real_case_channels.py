"""Real-case comparison of the second-channel encodings on dataset records.

Three random records at three different energies and beamlet angles. For each,
five quantities are shown in axial and sagittal views, in the single_beam style
(grayscale CT underlay + colored overlay):

    CT | original flux | centerline (soft line) | fixed-size flux | MC dose

The beam channels (flux / centerline / fixed / dose) are shown as maximum-
intensity projections over the out-of-plane lateral axis, so the full drifting
beam is visible for any steering angle; the CT underlay is the matching MIP.

Run: uv run python scripts/projection_ablation/real_case_channels.py
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
SIGMA_LINE, SIGMA_FIXED, DZ_MM = 1.0, 1.71, 2.0
plt.rcParams.update({"figure.facecolor": "white", "font.size": 11})


def find_json(uid):
    for d in JDIRS:
        p = os.path.join(d, uid + "_metadata.json")
        if os.path.exists(p):
            return p
    return None


def pick_records(f, n_per_band=1):
    """One record per low/mid/high energy band, each with some steering."""
    bands = {"low (70-100)": (70, 100), "mid (140-175)": (140, 175), "high (220-270)": (220, 270)}
    chosen = {}
    for k in list(f.keys()):
        g = f[k]
        e = 70 + float(g.attrs["initial_energy"]) * 200
        ba = np.array(g.attrs["beamlet_angles"], float)
        for name, (lo, hi) in bands.items():
            if name not in chosen and lo <= e <= hi and np.hypot(*ba) > 1.0:
                chosen[name] = (k, e, ba)
        if len(chosen) == len(bands):
            break
    return [chosen[b] for b in bands]


def mip(vol, axis):
    return vol.max(axis=axis)


def overlay(ax, ct_mip, chan_mip, cmap, title=None, ylabel=None):
    ax.imshow(ct_mip, cmap="gray", origin="lower", aspect="auto")
    if chan_mip is not None:
        m = chan_mip.max()
        alpha = np.where(chan_mip > 0.01 * m, 0.72, 0.0)
        ax.imshow(chan_mip, cmap=cmap, alpha=alpha, origin="lower", aspect="auto",
                  vmin=0, vmax=m)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(ls="--", lw=0.4, color="white")
    if title:
        ax.set_title(title, fontsize=13, weight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, weight="bold")


def main():
    with h5py.File(H5, "r") as f:
        recs = pick_records(f)
        cols = ["CT", "Original flux", "Centerline (soft)", "Fixed-size flux", "MC dose"]
        cmaps = [None, "hot", "hot", "hot", "jet"]
        fig, axes = plt.subplots(6, 5, figsize=(19, 16))
        for si, (k, e, ba) in enumerate(recs):
            g = f[k]
            ct = g["ct"][:]; flux = g["flux"][:]; dose = g["dose"][:]
            meta = json.load(open(find_json(k)))
            roi = meta.get("roi_size")
            ds = (roi[0] / ct.shape[0]) if roi else 2.0
            line = beam_line_from_metadata(meta["rays_entrence_point"], ba, ds)
            soft = render_centerline(line, ct.shape, mode="soft", sigma=SIGMA_LINE)
            fixed = render_centerline(line, ct.shape, mode="soft", sigma=SIGMA_FIXED)
            quants = [None, flux, soft, fixed, dose]

            for view, vax in (("axial", 1), ("sagittal", 0)):
                r = si * 2 + (0 if view == "axial" else 1)
                ct_m = mip(ct, vax)
                for c, (name, cm, q) in enumerate(zip(cols, cmaps, quants)):
                    ax = axes[r, c]
                    title = name if r == 0 else None
                    yl = None
                    if c == 0:
                        yl = (f"{k[:8]}  E={e:.0f} MeV\nsteer={np.round(ba,2).tolist()}\n{view}"
                              if view == "axial" else f"{view}")
                    overlay(ax, ct_m, (None if q is None else mip(q, vax)),
                            cm or "gray", title=title, ylabel=yl)
                    if r == 5:
                        ax.set_xlabel("depth  (entrance left -> Bragg peak right)", fontsize=9)
        fig.suptitle("Second-channel encodings on real records — CT | flux | centerline | fixed-size flux | MC dose\n"
                     "(maximum-intensity projections; beam channels overlaid on the CT)",
                     fontsize=15, weight="bold", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(OUT / "real_case_channels.png", dpi=150)
        for k, e, ba in recs:
            print(f"{k[:8]}  E={e:.0f} MeV  steer={np.round(ba,2).tolist()}")
        print(f"wrote {OUT / 'real_case_channels.png'}")


if __name__ == "__main__":
    main()
