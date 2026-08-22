"""Per-beamlet MC quality-control figure.

For one generated record (cropped CT, cropped MC dose, ADoTA flux), shows the CT
with the dose overlaid and with the flux overlaid in the two lateral planes and
the depth plane, plus depth profiles and a lateral profile at the Bragg peak.
Lets a human confirm: the dose is captured inside the crop, the Bragg peak sits
along the depth axis, and the ADoTA flux channel is laterally aligned with the
dose (i.e. the flux the model will see matches where the beam actually deposits).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.figures.single_beam import save_figure_as_publication_formats


def mc_beamlet_qc_figure(
    cropped_ct: np.ndarray,
    cropped_dose: np.ndarray,
    flux: np.ndarray,
    figure_path: str,
    title: str = "",
    info: Optional[dict] = None,
    dpi: int = 130,
) -> List[Path]:
    """Render the QC figure; return written paths. Arrays share the crop shape."""
    assert cropped_ct.shape == cropped_dose.shape == flux.shape, "crop shapes must match"
    depth = int(np.argmax(cropped_ct.shape))          # longest axis = beam depth (320)
    lat = [a for a in range(3) if a != depth]         # the two lateral axes
    # Bragg-peak depth index (from the lateral-integrated depth-dose)
    dd = cropped_dose.sum(axis=tuple(lat))
    bragg = int(np.argmax(dd))

    def take(vol, axis, idx):
        return np.take(vol, idx, axis=axis)

    ct_kw = dict(cmap="gray", aspect="auto")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=dpi, layout="constrained")

    # views: two lateral mid-planes + the depth plane at the Bragg peak
    views = [
        (lat[0], cropped_ct.shape[lat[0]] // 2, f"lateral axis {lat[0]} (mid)"),
        (lat[1], cropped_ct.shape[lat[1]] // 2, f"lateral axis {lat[1]} (mid)"),
        (depth, bragg, f"depth axis {depth} @ Bragg ({bragg})"),
    ]
    dmax = float(cropped_dose.max()) or 1.0
    fmax = float(flux.max()) or 1.0
    for c, (axis, idx, name) in enumerate(views):
        ct_s = np.rot90(take(cropped_ct, axis, idx))
        # row 0: CT + dose
        ax = axes[0, c]
        ax.imshow(ct_s, **ct_kw)
        ax.imshow(np.rot90(take(cropped_dose, axis, idx)), cmap="inferno", alpha=0.55,
                  vmin=0, vmax=dmax, aspect="auto")
        ax.set_title(f"CT+dose | {name}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        # row 1: CT + flux
        ax = axes[1, c]
        ax.imshow(ct_s, **ct_kw)
        ax.imshow(np.rot90(take(flux, axis, idx)), cmap="viridis", alpha=0.55,
                  vmin=0, vmax=fmax, aspect="auto")
        ax.set_title(f"CT+flux | {name}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # row 2: profiles
    ax = axes[2, 0]
    ax.plot(dd / (dd.max() + 1e-12), color="#e34948", label="dose (depth)")
    fd = flux.sum(axis=tuple(lat))
    ax.plot(fd / (fd.max() + 1e-12), color="#2a78d6",
                                            ls="--", label="flux (depth)")
    ax.axvline(bragg, color="grey", ls=":")
    ax.set_title("depth profiles (norm.)", fontsize=10)
    ax.set_xlabel("depth voxel")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", lw=0.5)

    # lateral profiles at the Bragg peak (dose vs flux alignment)
    dose_bragg = take(cropped_dose, depth, bragg)
    flux_bragg = take(flux, depth, bragg)
    for c, axl in enumerate(lat):
        ax = axes[2, 1 + c]
        pd_ = dose_bragg.sum(axis=1 - c)  # collapse the other lateral axis
        pf_ = flux_bragg.sum(axis=1 - c)
        ax.plot(pd_ / (pd_.max() + 1e-12), color="#e34948", label="dose")
        ax.plot(pf_ / (pf_.max() + 1e-12), color="#2a78d6", ls="--", label="flux")
        ax.set_title(f"lateral profile axis {axl} @ Bragg", fontsize=10)
        ax.set_xlabel("voxel")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", lw=0.5)

    sub = title
    if info:
        sub += "\n" + "  ".join(f"{k}={v}" for k, v in info.items())
    fig.suptitle(sub, fontsize=12, weight="bold")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
