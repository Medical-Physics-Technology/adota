"""Dose-distribution comparison figure for MC runs (shared color scale).

Renders two dose volumes (e.g. adota-vendored runner vs datagenerator engine, or
two seeds) over the CT in three orthogonal planes through the peak-dose voxel,
their difference, and 1-D dose profiles along all three axes (the Bragg fall-off
is visible on whichever axis carries the beam). Comparable dose panels share a
single vmin/vmax so panels are visually comparable, as the reviewer requires.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import save_figure_as_publication_formats


def _overlay(ax, ct_slice, dose_slice, vmax, title, cmap="inferno"):
    ax.imshow(np.rot90(ct_slice), cmap="gray", aspect="auto")
    im = ax.imshow(np.rot90(dose_slice), cmap=cmap, alpha=0.55, vmin=0.0, vmax=vmax,
                   aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def mc_dose_comparison_figure(
    ct: np.ndarray,
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    figure_path: str,
    label_a: str = "A",
    label_b: str = "B",
    dpi: int = 150,
    gamma_pass: Optional[float] = None,
) -> List[Path]:
    """Compare two dose volumes over the CT; return the written paths.

    All arrays are ``(z, y, x)`` on the same grid. Dose panels for A, B share
    vmin=0 and a common vmax; the difference uses a symmetric diverging scale.
    """
    assert ct.shape == dose_a.shape == dose_b.shape, "ct/dose shapes must match"
    # peak voxel of A defines the slice location.
    zc, yc, xc = np.unravel_index(int(np.argmax(dose_a)), dose_a.shape)
    vmax = float(max(dose_a.max(), dose_b.max())) or 1.0
    diff = dose_a - dose_b
    dmax = float(np.abs(diff).max()) or 1.0

    planes = [
        ("axial (z={})".format(zc), lambda v: v[zc, :, :]),
        ("coronal (y={})".format(yc), lambda v: v[:, yc, :]),
        ("sagittal (x={})".format(xc), lambda v: v[:, :, xc]),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(14, 16), dpi=dpi, layout="constrained")
    im_dose = None
    im_diff = None
    for c, (pname, sl) in enumerate(planes):
        im_dose = _overlay(axes[0, c], sl(ct), sl(dose_a), vmax, f"{label_a} | {pname}")
        _overlay(axes[1, c], sl(ct), sl(dose_b), vmax, f"{label_b} | {pname}")
        ax = axes[2, c]
        im_diff = ax.imshow(np.rot90(sl(diff)), cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                            aspect="auto")
        ax.set_title(f"{label_a}-{label_b} | {pname}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # shared colorbars
    fig.colorbar(im_dose, ax=axes[0:2, :].ravel().tolist(), fraction=0.02, pad=0.01,
                 label="dose (shared scale)")
    fig.colorbar(im_diff, ax=axes[2, :].ravel().tolist(), fraction=0.02, pad=0.01,
                 label="dose difference")

    # 1-D profiles through the peak voxel along each axis (beam fall-off visible).
    prof = [
        ("along z", dose_a[:, yc, xc], dose_b[:, yc, xc]),
        ("along y", dose_a[zc, :, xc], dose_b[zc, :, xc]),
        ("along x", dose_a[zc, yc, :], dose_b[zc, yc, :]),
    ]
    for c, (name, pa, pb) in enumerate(prof):
        ax = axes[3, c]
        ax.plot(pa, color="#2a78d6", lw=1.6, label=label_a)
        ax.plot(pb, color="#e34948", lw=1.4, ls="--", label=label_b)
        ax.set_title(f"profile {name} (through peak)", fontsize=11)
        ax.set_xlabel("voxel", fontsize=9)
        ax.set_ylabel("dose", fontsize=9)
        ax.grid(True, ls=":", lw=0.5)
        ax.legend(fontsize=9)

    title = f"MC dose comparison: {label_a} vs {label_b}"
    if gamma_pass is not None:
        title += f"   (gamma pass {gamma_pass:.1f}%)"
    fig.suptitle(title, fontsize=14, weight="bold")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
