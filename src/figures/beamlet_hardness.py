"""Figures of the per-engine hardness analysis (EXP-0014,
``scripts/active_learning/beamlet_hardness.py``).

- :func:`annotated_heatmap`: a labelled matrix (metrics by engines, or the cross
  transfer of fitted scores), rows grouped by metric cluster.
- :func:`cluster_importance_figure`: the slide figure, grouped horizontal bars of
  cluster importance per target with their intervals.

The single-beamlet examples reuse :func:`src.figures.hpo.finalist_beamlet_figure`.
Colours are those of the HPO figures (:data:`src.figures.hpo.MIXER_COLORS`);
saving goes through :func:`save_figure_as_publication_formats` and colorbars
through :func:`aligned_colorbar`.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures.al_curves import COLORS
from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats
from src.figures.hpo import MIXER_COLORS

CONSENSUS_COLOR = "#4d4d4d"
TARGET_COLORS = {"consensus": CONSENSUS_COLOR, "none": MIXER_COLORS["none"],
                 "transformer": MIXER_COLORS["transformer"], "mlstm": MIXER_COLORS["mlstm"],
                 "pencil_beam": COLORS[3], "published": COLORS[4]}
"""Per engine family: the HPO mixer colours, orange for the pencil beam, purple
for the published ADoTA, dark grey for the consensus of all engines."""


def annotated_heatmap(values: pd.DataFrame, *, figure_path: str, title: str, value_label: str,
                      caption: str = "", groups: Optional[Sequence[str]] = None, vmax: Optional[float] = None,
                      cmap: str = "RdBu_r", fmt: str = "{:.2f}", lo: Optional[pd.DataFrame] = None,
                      hi: Optional[pd.DataFrame] = None, cell_in: float = 0.52) -> List[Path]:
    """One annotated matrix on a symmetric diverging scale.

    Args:
        values: Rows by columns.
        groups: Per row, a group label (its metric cluster id), prefixed to the
            row label; a rule separates consecutive groups.
        lo, hi: Optional interval bounds of the same shape; a cell whose interval
            excludes zero is written in bold.
    """
    data = values.to_numpy(dtype=float)
    span = vmax or max(float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 1.0, 0.05)
    n_rows, n_cols = data.shape
    width = max(4.0, cell_in * n_cols + 3.6)
    height = max(2.6, 0.24 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(width, height), dpi=300, constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=-span, vmax=span, aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(values.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    names = list(values.index) if groups is None else [f"{g}  {r}" for g, r in zip(groups, values.index)]
    ax.set_yticklabels(names, fontsize=6.5)
    significant = None
    if lo is not None and hi is not None:
        significant = (lo.to_numpy(dtype=float) > 0) | (hi.to_numpy(dtype=float) < 0)
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isfinite(data[i, j]):
                strong = abs(data[i, j]) / span > 0.6
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=5.5,
                        color="white" if strong else "#222222",
                        fontweight="bold" if significant is not None and significant[i, j] else "normal")
    if groups is not None:
        groups = list(groups)
        for i in range(1, n_rows):
            if groups[i] != groups[i - 1]:
                ax.axhline(i - 0.5, color="black", linewidth=0.6)
    ax.set_title(title, loc="left", fontweight="semibold", fontsize=9)
    if caption:
        ax.set_xlabel(textwrap.fill(caption, width=int(16 * width)), fontsize=6.5)
    aligned_colorbar(fig, image, ax, value_label, label_coords=(4.2, 0.5), label_fontsize=7, tick_fontsize=6)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def cluster_importance_figure(frame: pd.DataFrame, series: Sequence[str], colors: Mapping[str, str], *,
                              figure_path: str, title: str, caption: str = "",
                              xlabel: str = "Importance: drop in out-of-fold within-cell Spearman") -> List[Path]:
    """Grouped horizontal bars: one group per cluster (rows of ``frame`` in
    plotting order, top first), one bar per ``series`` with its interval.

    ``frame`` has a ``cluster_name`` column and, per series ``s``, the columns
    ``s``, ``s:lo`` and ``s:hi``.
    """
    n, k = len(frame), len(series)
    height = 0.8 / k
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * n + 1.5), dpi=300, constrained_layout=True)
    for j, s in enumerate(series):
        y = np.arange(n) + (j - (k - 1) / 2) * height
        value = frame[s].to_numpy(dtype=float)
        err = np.vstack([value - frame[f"{s}:lo"].to_numpy(dtype=float),
                         frame[f"{s}:hi"].to_numpy(dtype=float) - value])
        ax.barh(y, value, height=height * 0.92, color=colors[s], label=s, zorder=2)
        ax.errorbar(value, y, xerr=np.clip(err, 0, None), fmt="none", ecolor="#222222", elinewidth=0.7, capsize=1.6,
                    zorder=3)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(frame["cluster_name"], fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.7)
    ax.grid(True, axis="x", color="#d9d9d9", linewidth=0.5, zorder=0)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.set_title(title, loc="left", fontweight="semibold", fontsize=9.5)
    if caption:
        fig.text(0.0, -0.01, textwrap.fill(caption, width=135), ha="left", va="top", fontsize=6.5, color="#333333")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
