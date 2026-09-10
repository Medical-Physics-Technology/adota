"""Learning-curve and selection figures of the retrospective active-learning
benchmark (``scripts/al_compare.py``).

Four figures, all multi-run, one line per strategy:

- :func:`training_curves_figure`: training and validation loss against the
  cumulative epoch, with the cycle boundaries as labelled vertical rules;
- :func:`quality_curves_figure`: gamma pass rate, its two tails, MAPE and dR80
  against either the training set size or the cumulative epoch, markers at the
  cycle boundaries;
- :func:`selection_fingerprint_figure`: per cycle, the share of the selected
  batch per score decile (or energy bin) against the pool's share, one heatmap
  row block per strategy.

The house style follows the ablation summary of the training runs: a left-aligned
semibold title per panel, light grids, no top or right spines. Saving goes through
:func:`save_figure_as_publication_formats`; colorbars through :func:`aligned_colorbar`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats

COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)))

QUALITY_PANELS: Tuple[Tuple[str, str, float], ...] = (
    ("gpr_mean", "Mean gamma pass rate [%]", 100.0),
    ("gpr_p05", "5th percentile gamma pass rate [%]", 100.0),
    ("gpr_frac_below_95", "Records below 95% pass rate [%]", 100.0),
    ("mape_pct_mean", "Mean MAPE [%]", 1.0),
    ("abs_dr80_median_mm", "Median |dR80| [mm]", 1.0),
    ("abs_dr80_p95_mm", "95th percentile |dR80| [mm]", 1.0),
)


def _style(label: str, index: int) -> Dict:
    return {"color": COLORS[index % len(COLORS)],
            "linestyle": LINESTYLES[index % len(LINESTYLES)], "label": label}


def style_axis(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#b5b5b5", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_cycle_rules(ax, boundaries: pd.DataFrame, label: bool = True) -> None:
    """Vertical rules at the cycle boundaries carrying the cycle index and the
    training set size at that cycle."""
    top = ax.get_ylim()[1]
    for row in boundaries.itertuples(index=False):
        ax.axvline(row.cumulative_epoch, color="#666666", linewidth=0.8, linestyle=":", zorder=1)
        if label:
            ax.text(row.cumulative_epoch, top, f" c{int(row.cycle)}\n n={int(row.n_train):,}",
                    fontsize=7.5, color="#444444", ha="left", va="top")


def training_curves_figure(curves: Dict[str, pd.DataFrame], boundaries: pd.DataFrame,
                           figure_path: str, title: str = "Training curves") -> List[Path]:
    """F1: training and validation loss against the cumulative epoch.

    Args:
        curves: ``label -> frame`` with ``cumulative_epoch``, ``train_loss``, ``val_loss``.
        boundaries: ``cycle``, ``cumulative_epoch``, ``n_train`` at every cycle end.
        figure_path: Output stem; SVG, PDF and PNG are written beside each other.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, constrained_layout=True)
    for index, (label, frame) in enumerate(curves.items()):
        style = _style(label, index)
        axes[0].plot(frame["cumulative_epoch"], frame["train_loss"], linewidth=1.2, **style)
        axes[1].plot(frame["cumulative_epoch"], frame["val_loss"], linewidth=1.2, **style)
    for ax, ylabel, panel in zip(axes, ("Training loss", "Validation loss"),
                                 ("Training loss", "Validation loss (full V)")):
        style_axis(ax, xlabel="Cumulative epoch", ylabel=ylabel, title=panel)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(left=0)
        draw_cycle_rules(ax, boundaries)
    axes[1].legend(frameon=True, framealpha=0.95, fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def quality_curves_figure(curves: Dict[str, pd.DataFrame], *, x: str, x_label: str,
                          figure_path: str, gamma_caption: str,
                          panels: Sequence[Tuple[str, str, float]] = QUALITY_PANELS,
                          prefix: str = "sub_", boundaries: Optional[pd.DataFrame] = None,
                          title: str = "Validation quality") -> List[Path]:
    """F2 and F3: the metric panels against ``x`` (``n_train`` or ``cumulative_epoch``).

    Rows flagged ``cycle_boundary`` get a marker; the caption names the gamma
    criteria, as every gamma figure must.
    """
    n = len(panels)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), dpi=300,
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (metric, ylabel, factor) in zip(axes, panels):
        column = f"{prefix}{metric}"
        for index, (label, frame) in enumerate(curves.items()):
            if column not in frame:
                continue
            style = _style(label, index)
            ax.plot(frame[x], frame[column] * factor, linewidth=1.3, alpha=0.9, **style)
            marks = frame[frame["cycle_boundary"]] if "cycle_boundary" in frame else frame.iloc[:0]
            ax.scatter(marks[x], marks[column] * factor, color=style["color"], s=28,
                       edgecolor="white", linewidth=0.8, zorder=5)
        style_axis(ax, xlabel=x_label, ylabel=ylabel, title=ylabel)
        if boundaries is not None and x == "cumulative_epoch":
            draw_cycle_rules(ax, boundaries, label=False)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(frameon=True, framealpha=0.95, fontsize=9)
    fig.suptitle(f"{title} against {x_label.lower()}", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, gamma_caption, ha="center", va="top", fontsize=8.5, color="#444444")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def selection_fingerprint_figure(table: pd.DataFrame, *, figure_path: str, key_label: str,
                                 title: str = "Selection fingerprint") -> List[Path]:
    """F4: per strategy, a heatmap of the selected share per category and cycle,
    with the pool's share of the first scored cycle as the reference row.

    Args:
        table: The long table of :func:`src.active_learning.retrospective.compare.fingerprint_table`
            (``label``, ``cycle``, ``category``, ``selected_share``, ``pool_share``).
        figure_path: Output stem.
        key_label: What the categories are (``"score decile"`` or ``"energy bin [MeV]"``).
    """
    labels = list(dict.fromkeys(table["label"]))
    categories = list(dict.fromkeys(sorted(table["category"], key=_category_key)))
    vmax = float(max(table["selected_share"].max(), table["pool_share"].max()))
    fig, axes = plt.subplots(len(labels), 1, figsize=(1.1 * len(categories) + 3.5,
                                                      2.2 * len(labels) + 1.0), dpi=300,
                             constrained_layout=True, squeeze=False)
    for ax, label in zip(axes[:, 0], labels):
        part = table[table["label"] == label]
        cycles = sorted(part["cycle"].unique())
        first = part[part["cycle"] == cycles[0]].set_index("category")["pool_share"]
        grid = np.zeros((len(cycles) + 1, len(categories)))
        grid[0] = [float(first.get(c, 0.0)) for c in categories]
        for i, cycle in enumerate(cycles, start=1):
            shares = part[part["cycle"] == cycle].set_index("category")["selected_share"]
            grid[i] = [float(shares.get(c, 0.0)) for c in categories]
        image = ax.imshow(grid * 100.0, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax * 100.0)
        ax.set_yticks(range(len(cycles) + 1))
        ax.set_yticklabels([f"pool (c{cycles[0]})"] + [f"selected c{c}" for c in cycles],
                           fontsize=8)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
        ax.set_title(label, loc="left", fontweight="semibold")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j] * 100.0
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if value > 0.55 * vmax * 100.0 else "#222222")
        aligned_colorbar(fig, image, ax, "share [%]", label_coords=(3.6, 0.5),
                         label_fontsize=9, tick_fontsize=8)
    axes[-1, 0].set_xlabel(key_label)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def _category_key(value) -> Tuple[int, float, str]:
    text = str(value)
    try:
        return (0, float(text.split("-")[0]), text)
    except ValueError:
        return (1, 0.0, text)
