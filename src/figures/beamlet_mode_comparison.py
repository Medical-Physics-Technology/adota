"""Side-by-side GPR panels comparing MCsquare beamlet mode with the sequential path.

One row of angle-grid panels on a shared colour scale, so the three questions the
beamlet-mode switch raises can be read off a single figure:

* is the beamlet-mode ground truth the same as the sequential one (MC vs MC), and
* does ADoTA score the same against each of them.

Unlike the paper's robustness panels these carry titles: the figure is a
diagnostic comparison, not a LaTeX grid where the caption names the panels.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import save_figure_as_publication_formats

_FS_TITLE, _FS_LABEL, _FS_TICK, _FS_CBAR = 17, 15, 12, 14


def beamlet_mode_comparison_figure(
    panels: Mapping[str, np.ndarray],
    thetas: Sequence[float],
    figure_path: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    cbar_label: str = "Gamma pass rate [%]",
    marker_size: float = 260.0,
    dpi: int = 300,
) -> List[Path]:
    """Render ``{title: (n, n) GPR grid}`` as one row of panels with a shared bar.

    ``panels`` keeps insertion order, which is the left-to-right panel order.
    Cells that hold no value (a QA-dropped or unsimulated angle) are left blank
    rather than drawn, as in :func:`~src.figures.angle_robustness_grid`.
    """
    if not panels:
        raise ValueError("no panels to render")
    if not np.isfinite([vmin, vmax]).all():
        raise ValueError(f"non-finite colour scale ({vmin}, {vmax})")
    t = np.asarray(thetas, dtype=float)
    TX, TY = np.meshgrid(t, t, indexing="ij")
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n + 1.4, 5.6), dpi=dpi, squeeze=False)
    scatter = None
    for ax, (title, grid) in zip(axes[0], panels.items()):
        g = np.asarray(grid, dtype=float).ravel()
        finite = np.isfinite(g)
        scatter = ax.scatter(TX.ravel()[finite], TY.ravel()[finite], c=g[finite],
                             cmap=cmap, vmin=vmin, vmax=vmax, s=marker_size,
                             edgecolors="none")
        ax.set_title(title, fontsize=_FS_TITLE)
        ax.set_xlabel("Beamlet angle X [degrees]", fontsize=_FS_LABEL)
        ax.set_ylabel("Beamlet angle Y [degrees]", fontsize=_FS_LABEL)
        lo, hi = float(t.min()), float(t.max())
        ticks = [v for v in (-2, -1, 0, 1, 2) if lo - 1e-6 <= v <= hi + 1e-6] or [lo, 0, hi]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(axis="both", labelsize=_FS_TICK)
        ax.set_aspect("equal")
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
        pad = 0.12 * (hi - lo + 1e-6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    cb = fig.colorbar(scatter, ax=axes[0].tolist(), fraction=0.030, pad=0.02)
    cb.set_label(cbar_label, fontsize=_FS_CBAR)
    cb.ax.tick_params(labelsize=_FS_TICK)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
