"""Compare four direction-encoding options for the second input channel, one record.

  1. baseline flux         -- the stored fast beamlet-shape projection
                              (energy-conditioned: spot width sigma(E), here shown
                              normalized to [0,1] as the model consumes it).
  2. centerline_soft       -- the beam axis as a thin soft line (fixed narrow sigma).
  3. centerline_binary     -- the beam axis as a single nearest voxel per slice.
  4. fixed-size projection -- a Gaussian tube of constant width (median spot sigma),
                              NOT conditioned by energy.

Two views per option: a direction view (axis0 vs depth, MIP over axis1) and a width
view (axial slice at mid-depth), so both the encoded direction and the lateral
extent are visible.

Run: uv run python scripts/projection_ablation/compare_four_options.py
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
SIGMA_LINE = 1.0     # option 2: thin soft line
SIGMA_FIXED = 1.71   # option 4: median spot sigma across the dataset (energy-independent)
UID = "0169e3ec"     # low-energy (75 MeV) high-drift record: wide spot (sigma~2.0)
#                      so the baseline (energy-conditioned) width differs visibly
#                      from the fixed-size option
INK, SURF = "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.size": 10, "text.color": INK})


def find_json(uid):
    for d in JDIRS:
        p = os.path.join(d, uid + "_metadata.json")
        if os.path.exists(p):
            return p
    return None


def norm01(a):
    a = a.astype(np.float64)
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def spot_sigma(flux):
    H, W, D = flux.shape
    ii = np.arange(H)[:, None]; jj = np.arange(W)[None, :]
    sg = []
    for d in range(0, D, 10):
        s = flux[:, :, d]; tot = s.sum()
        if tot > 0:
            c0 = (s * ii).sum() / tot; c1 = (s * jj).sum() / tot
            sg.append(np.sqrt(((s * ((ii - c0) ** 2 + (jj - c1) ** 2)).sum() / tot) / 2))
    return float(np.mean(sg))


def main():
    with h5py.File(H5, "r") as f:
        k = next(x for x in f.keys() if x.startswith(UID))
        g = f[k]
        flux = g["flux"][:]
        ba = np.array(g.attrs["beamlet_angles"], dtype=float)
        e_norm = float(g.attrs["initial_energy"])
    e_mev = 70 + e_norm * (270 - 70)
    sig_real = spot_sigma(flux)
    meta = json.load(open(find_json(k)))
    roi = meta.get("roi_size")
    ds = (roi[0] / flux.shape[0]) if roi else 2.0
    line = beam_line_from_metadata(meta["rays_entrence_point"], ba, ds)

    soft = render_centerline(line, flux.shape, mode="soft", sigma=SIGMA_LINE)
    binr = render_centerline(line, flux.shape, mode="binary")
    fixed = render_centerline(line, flux.shape, mode="soft", sigma=SIGMA_FIXED)

    H, W, D = flux.shape
    d = np.arange(D)
    c0, c1 = line.lateral_center(d)
    dmid = D // 2
    win = 9  # axial crop half-window around the center for visibility
    ci0, ci1 = int(round(c0[dmid])), int(round(c1[dmid]))
    sl = (slice(max(ci0 - win, 0), ci0 + win), slice(max(ci1 - win, 0), ci1 + win))

    cols = [
        (f"1. baseline flux\nenergy-conditioned, sigma(E)={sig_real:.2f} vox", norm01(flux)),
        (f"2. centerline_soft\nthin line, sigma={SIGMA_LINE:.1f} vox", soft),
        ("3. centerline_binary\nnearest voxel (no sub-voxel)", binr),
        (f"4. fixed-size projection\nconstant sigma={SIGMA_FIXED:.2f}, no energy", fixed),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
    for c, (title, vol) in enumerate(cols):
        # direction view: MIP over axis1 (axis0 vs depth)
        ax = axes[0, c]
        ax.imshow(vol.max(1), origin="lower", aspect="auto", cmap="magma",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.plot(d, c0, color="#2ad1c9", lw=1.0, ls="-")
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("DIRECTION\naxis0 vs depth (MIP)", fontsize=9)
        # width view: axial slice at mid-depth (cropped window)
        ax = axes[1, c]
        ax.imshow(vol[sl[0], sl[1], dmid], origin="lower", cmap="magma",
                  interpolation="nearest", vmin=0, vmax=1)
        ax.plot(c1[dmid] - sl[1].start, c0[dmid] - sl[0].start, "+", color="#2ad1c9", ms=10)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f"WIDTH\naxial slice @ depth {dmid}", fontsize=9)

    fig.suptitle(
        f"Four direction-encoding options for the 2nd input channel  "
        f"(record {k[:8]}, steer={np.round(ba,2).tolist()} deg, E={e_mev:.0f} MeV)\n"
        f"cyan = analytic beam center; all channels normalized to [0,1] as the model sees them",
        fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "four_options.png", dpi=145)
    print(f"record {k}  E={e_mev:.0f} MeV  real spot sigma={sig_real:.2f} vox  "
          f"steer={ba.tolist()}")
    print(f"wrote {OUT / 'four_options.png'}")


if __name__ == "__main__":
    main()
