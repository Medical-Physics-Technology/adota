"""Timing and speed-up panels for the gamma-index backends.

One figure answers the question the two gamma benchmarks were built to answer:
how much the torch backend saves, and how that saving depends on the size of
the problem. The beamlet grid (160 x 30 x 30 = 144,000 voxels) and the plan grid
(67-100 million voxels) differ by close to three orders of magnitude, so they
are drawn side by side rather than averaged into one number.

Left panel: wall time for one case, on a logarithmic axis, grouped by criterion.
Right panel: speed-up over ``pymedphys.gamma``, with unity marked, so a bar
below the line is a slow-down and reads as one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import save_figure_as_publication_formats

_FS_TITLE, _FS_LABEL, _FS_TICK, _FS_LEGEND = 15, 13, 11, 11

# One colour per rung, shared by both panels and both scales, so a bar keeps its
# identity across the figure.
RUNG_COLOURS: Dict[str, str] = {
    "pymedphys cpu": "#4c4c4c",
    "torch cpu float64": "#7fa8d8",
    "torch cuda float64": "#e08214",
    "torch cuda float32": "#b2182b",
}


def _grouped_bars(
    ax,
    criteria: Sequence[str],
    series: Dict[str, Sequence[float]],
    ylabel: str,
    title: str,
    log: bool,
) -> None:
    """Draw one grouped bar panel, one group per criterion."""
    n_series = len(series)
    width = 0.8 / max(1, n_series)
    positions = np.arange(len(criteria), dtype=float)
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (n_series - 1) / 2) * width
        heights = np.asarray(values, dtype=float)
        ax.bar(
            positions + offset,
            heights,
            width=width,
            label=label,
            color=RUNG_COLOURS.get(label, f"C{index}"),
            edgecolor="black",
            linewidth=0.4,
        )
    if log:
        ax.set_yscale("log")
        # A panel spanning less than about a decade shows a single labelled
        # decade tick and is then unreadable, so the minor ticks are labelled
        # too. `LogFormatterSciNotation` on the minor locator prints them as
        # plain numbers where they fit and omits them where they would collide.
        ax.yaxis.set_minor_formatter(
            matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(4, 1))
        )
        ax.tick_params(axis="y", which="minor", labelsize=_FS_TICK - 2)
    ax.set_xticks(positions)
    ax.set_xticklabels(criteria, fontsize=_FS_TICK)
    ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
    ax.set_title(title, fontsize=_FS_TITLE)
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def gamma_backend_performance_figure(
    criteria: Sequence[str],
    beamlet_times_s: Dict[str, Sequence[float]],
    plan_times_s: Dict[str, Sequence[float]],
    beamlet_speedups: Dict[str, Sequence[float]],
    plan_speedups: Dict[str, Sequence[float]],
    figure_path: str,
    dpi: int = 300,
) -> List[Path]:
    """Render the four-panel timing and speed-up comparison.

    Args:
        criteria: Criterion labels, shared by every series and both scales.
        beamlet_times_s: ``{rung label: seconds per beamlet}``, one value per
            criterion.
        plan_times_s: ``{rung label: seconds per plan}``, same ordering.
        beamlet_speedups: ``{rung label: speed-up over pymedphys}`` at beamlet
            scale.
        plan_speedups: The same at plan scale.
        figure_path: Output path; the extension is replaced by svg, pdf and png.
        dpi: Raster resolution of the PNG.

    Returns:
        The paths written.

    Raises:
        ValueError: If no criteria are given.
    """
    if not len(criteria):
        raise ValueError("no criteria to plot")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), dpi=dpi)
    _grouped_bars(
        axes[0, 0], criteria, beamlet_times_s,
        "Time per beamlet [s]", "(a) One beamlet, 144,000 voxels", log=True,
    )
    _grouped_bars(
        axes[0, 1], criteria, plan_times_s,
        "Time per plan [s]", "(b) One plan, 67-100 million voxels", log=True,
    )
    _grouped_bars(
        axes[1, 0], criteria, beamlet_speedups,
        "Speed-up over pymedphys", "(c) Beamlet speed-up", log=True,
    )
    _grouped_bars(
        axes[1, 1], criteria, plan_speedups,
        "Speed-up over pymedphys", "(d) Plan speed-up", log=True,
    )
    # Unity is the line that separates a speed-up from a slow-down, so it has to
    # stay visibly inside the axis even when every bar is far above it.
    for ax in axes[1]:
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        bottom, top = ax.get_ylim()
        ax.set_ylim(min(bottom, 0.75), max(top, 1.4))
        ax.set_xlabel("Gamma criterion", fontsize=_FS_LABEL)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        fontsize=_FS_LEGEND,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    return save_figure_as_publication_formats(fig, figure_path)
