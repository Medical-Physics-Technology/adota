"""18x18 beamlet-angle gamma-pass-rate grid panels (reviewer R3).

One panel = a scatter of gamma pass rate over (Beamlet angle X, Beamlet angle Y).
Panels are saved separately for LaTeX, and a shared color scale (computed by the
caller across comparable panels) makes them directly comparable. No plot titles
(the LaTeX caption carries them); A4-legible fonts; fully labelled axes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.figures.single_beam import save_figure_as_publication_formats

_FS_LABEL, _FS_TICK, _FS_CBAR = 24, 19, 20


def angle_robustness_panel(
    gpr_grid: np.ndarray,
    thetas_x: Sequence[float],
    thetas_y: Sequence[float],
    figure_path: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    with_colorbar: bool = False,
    cbar_label: str = "",
    marker_size: float = 180.0,
    dpi: int = 300,
) -> List[Path]:
    """Render one GPR grid panel (``gpr_grid[ix, iy]`` at ``thetas_x[ix], thetas_y[iy]``)."""
    tx = np.asarray(thetas_x, dtype=float)
    ty = np.asarray(thetas_y, dtype=float)
    TX, TY = np.meshgrid(tx, ty, indexing="ij")
    fig_w = 8.6 if with_colorbar else 7.0
    fig, ax = plt.subplots(figsize=(fig_w, 7.0), dpi=dpi)
    # Only plot cells with a GPR value: missing/QA-skipped beamlets (NaN) are left
    # blank rather than drawn, so a partial grid renders cleanly.
    g = np.asarray(gpr_grid, dtype=float).ravel()
    txr, tyr = TX.ravel(), TY.ravel()
    finite = np.isfinite(g)
    sc = ax.scatter(txr[finite], tyr[finite], c=g[finite],
                    cmap=cmap, vmin=vmin, vmax=vmax, s=marker_size, edgecolors="none")
    if not finite.any():  # nothing to color; keep a mappable for the colorbar
        sc = ax.scatter([], [], c=[], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Beamlet angle X [degrees]", fontsize=_FS_LABEL)
    ax.set_ylabel("Beamlet angle Y [degrees]", fontsize=_FS_LABEL)
    lo, hi = float(tx.min()), float(tx.max())
    ticks = [t for t in (-2, -1, 0, 1, 2) if lo - 1e-6 <= t <= hi + 1e-6] or [lo, 0, hi]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(axis="both", labelsize=_FS_TICK)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    pad = 0.15 * (hi - lo + 1e-6)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    if with_colorbar:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=_FS_CBAR)
        cb.ax.tick_params(labelsize=_FS_TICK)
    fig.tight_layout()
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def standalone_colorbar(
    figure_path: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    label: str = "",
    orientation: str = "vertical",
    dpi: int = 300,
) -> List[Path]:
    """Render just the shared colorbar as its own file (for a LaTeX panel grid)."""
    if orientation == "vertical":
        fig, ax = plt.subplots(figsize=(1.7, 7.0), dpi=dpi)
    else:
        fig, ax = plt.subplots(figsize=(7.0, 1.7), dpi=dpi)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=plt.get_cmap(cmap), norm=norm,
                                   orientation=orientation)
    cb.set_label(label, fontsize=_FS_CBAR + 1)
    cb.ax.tick_params(labelsize=_FS_TICK)
    fig.tight_layout()
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
