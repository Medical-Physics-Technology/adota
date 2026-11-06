"""Agreement figures of an ensemble of ADoTA models (EXP-0014,
``scripts/active_learning/ensemble_agreement.py``).

- :func:`agreement_matrix_figure`: the members-by-members matrix of one pair
  statistic (the median within-cell Spearman of the errors), every cell
  annotated, the pairs that differ only in their seed outlined.
- :func:`noise_floor_figure`: the shared target against either Monte Carlo
  reference, each engine's error against the floor, and the score against the floor.

House style of :mod:`src.figures.al_curves`: a left-aligned semibold title,
saving through :func:`save_figure_as_publication_formats`, the colorbar through
:func:`aligned_colorbar`.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from src.figures.al_curves import COLORS, style_axis
from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats


def agreement_matrix_figure(matrix: pd.DataFrame, *, figure_path: str, value_label: str,
                            title: str, caption: str = "",
                            outlined: Sequence[Tuple[str, str]] = (),
                            vmin: float = 0.0, vmax: float = 1.0) -> List[Path]:
    """One heatmap of a symmetric pair statistic.

    Args:
        matrix: Members by members, NaN on the diagonal
            (:func:`src.active_learning.ensemble.agreement.agreement_matrix`).
        figure_path: Output stem; SVG, PDF and PNG are written beside it.
        value_label: Colorbar label.
        outlined: Pairs ``(a, b)`` drawn with a black frame (the seed-only pairs).
        caption: Placed under the matrix, e.g. the metric and the cell definition.
    """
    labels = list(matrix.index)
    values = matrix.to_numpy(dtype=float)
    size = 0.62 * len(labels) + 2.8
    fig, ax = plt.subplots(figsize=(size + 1.0, size), dpi=300, constrained_layout=True)
    image = ax.imshow(np.ma.masked_invalid(values), cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    span = vmax - vmin if vmax > vmin else 1.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(values[i, j]):
                shade = (values[i, j] - vmin) / span
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="#222222" if shade > 0.6 else "white")
    position = {label: k for k, label in enumerate(labels)}
    for a, b in outlined:
        for i, j in ((position[a], position[b]), (position[b], position[a])):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black",
                                   linewidth=1.6))
    ax.set_title(title, loc="left", fontweight="semibold")
    if caption:
        ax.set_xlabel(textwrap.fill(caption, width=int(14 * size)), fontsize=8)
    aligned_colorbar(fig, image, ax, value_label, label_coords=(3.6, 0.5), label_fontsize=9,
                     tick_fontsize=8)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def noise_floor_figure(beamlets: pd.DataFrame, engines: pd.DataFrame, *, figure_path: str, title: str,
                       caption: str = "") -> List[Path]:
    """Three panels of the paired noise set.

    Args:
        beamlets: One row per beamlet: ``anatomy``, ``target_1e6``, ``target_1e7``,
            ``score``, ``floor_rde``.
        engines: One row per engine: ``engine``, ``median_rde_1e6``, ``median_rde_1e7``.
        figure_path: Output stem; SVG, PDF and PNG are written beside it.
        caption: Placed under the panels (say the metric and the population).
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), dpi=300, constrained_layout=True)
    anatomies = sorted(beamlets["anatomy"].unique())
    for i, anatomy in enumerate(anatomies):
        part = beamlets[beamlets["anatomy"] == anatomy]
        style = {"s": 7, "alpha": 0.55, "color": COLORS[i % len(COLORS)], "linewidths": 0}
        axes[0].scatter(part["target_1e7"], part["target_1e6"], label=anatomy, **style)
        axes[2].scatter(part["score"], part["floor_rde"], **style)
    span = [beamlets[["target_1e6", "target_1e7"]].min().min(), beamlets[["target_1e6", "target_1e7"]].max().max()]
    axes[0].plot(span, span, color="#444444", linewidth=0.9, linestyle="--")
    style_axis(axes[0], xlabel="Shared target against 1e7", ylabel="Shared target against 1e6",
               title="Does noise change the target?")
    axes[0].legend(frameon=False, fontsize=8, markerscale=2)

    table = engines.sort_values("median_rde_1e7")
    y = np.arange(len(table))
    axes[1].barh(y + 0.19, table["median_rde_1e6"], height=0.36, color=COLORS[1], label="against 1e6")
    axes[1].barh(y - 0.19, table["median_rde_1e7"], height=0.36, color=COLORS[0], label="against 1e7")
    axes[1].axvline(beamlets["floor_rde"].median(), color="#222222", linewidth=1.1, linestyle="--",
                    label="floor: 1e6 against 1e7")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(table["engine"], fontsize=7)
    style_axis(axes[1], xlabel="Median RDE [%]", ylabel="", title="Engine error and the floor")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].set_yscale("log")
    style_axis(axes[2], xlabel="Difficulty score", ylabel="Floor RDE [%]", title="Does the score track noise?")
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    if caption:
        fig.supxlabel(textwrap.fill(caption, width=170), fontsize=8)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
