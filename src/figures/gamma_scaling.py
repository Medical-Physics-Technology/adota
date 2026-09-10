"""Runtime and speed-up against problem size, for the gamma scaling experiment.

Four panels on logarithmic axes. The left column plots against the total voxel
count of the crop, the right column against the number of reference points that
cleared the dose cutoff and were therefore searched. The two are shown side by
side because the second is what the search actually costs, and a crop around
the high-dose region can grow by an order of magnitude in voxels while its
evaluated population barely moves.

Top row: median wall time per backend with the interquartile range as error
bars. Bottom row: paired speed-up of each GPU precision over PyMedPhys, with
unity marked. One marker shape per plan, one colour per backend, shared with
:mod:`src.figures.gamma_backend_performance`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.figures.axes_utils import save_figure_as_publication_formats
from src.figures.gamma_backend_performance import RUNG_COLOURS

_FS_TITLE, _FS_LABEL, _FS_TICK, _FS_LEGEND = 15, 13, 11, 10

SERIES = {1: "pymedphys cpu", 3: "torch cuda float64", 4: "torch cuda float32"}
SERIES_TEX = {1: "PyMedPhys, CPU", 3: "PyTorch, GPU, float64", 4: "PyTorch, GPU, float32"}
MARKERS = ("o", "s", "^", "D")


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=_FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
    ax.set_title(title, fontsize=_FS_TITLE)
    ax.tick_params(labelsize=_FS_TICK)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)


def gamma_scaling_figure(rows: Sequence[Dict], figure_path: str, dpi: int = 300) -> List[Path]:
    """Render the scaling panels from :func:`summarise_scaling` rows.

    Args:
        rows: One row per (plan, crop) carrying ``voxels``, ``n_evaluated``,
            ``rung{k}_median_s``/``_q1_s``/``_q3_s`` and ``rung{k}_speedup``.
        figure_path: Output path; the extension is replaced by svg, pdf, png.
        dpi: Raster resolution of the PNG.

    Returns:
        The paths written.

    Raises:
        ValueError: If no rows are given.
    """
    if not rows:
        raise ValueError("no scaling rows to plot")
    plans = sorted({row["plan"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), dpi=dpi)

    for plan_index, plan in enumerate(plans):
        subset = sorted((r for r in rows if r["plan"] == plan), key=lambda r: r["voxels"])
        marker = MARKERS[plan_index % len(MARKERS)]
        short = plan.split("_")[0]
        for x_key, column in (("voxels", 0), ("n_evaluated", 1)):
            x = [r[x_key] for r in subset]
            for rung, label in SERIES.items():
                medians = [r[f"rung{rung}_median_s"] for r in subset]
                lower = [m - r[f"rung{rung}_q1_s"] for m, r in zip(medians, subset)]
                upper = [r[f"rung{rung}_q3_s"] - m for m, r in zip(medians, subset)]
                axes[0, column].errorbar(
                    x, medians, yerr=[lower, upper], marker=marker, color=RUNG_COLOURS[label], capsize=3,
                    linewidth=1.2, markersize=6, label=f"{SERIES_TEX[rung]}, {short}",
                )
            for rung in (3, 4):
                axes[1, column].plot(
                    x, [r[f"rung{rung}_speedup"] for r in subset], marker=marker, color=RUNG_COLOURS[SERIES[rung]],
                    linewidth=1.2, markersize=6, label=f"{SERIES_TEX[rung]}, {short}",
                )

    _style(axes[0, 0], "Voxels in crop", "Median time [s]", "(a) Time against crop size")
    _style(axes[0, 1], "Reference points above cutoff", "Median time [s]", "(b) Time against evaluated points")
    _style(axes[1, 0], "Voxels in crop", "Paired speed-up over PyMedPhys", "(c) Speed-up against crop size")
    _style(axes[1, 1], "Reference points above cutoff", "Paired speed-up over PyMedPhys",
           "(d) Speed-up against evaluated points")
    for ax in axes[1]:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        bottom, top = ax.get_ylim()
        ax.set_ylim(min(bottom, 0.75), max(top, 1.4))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=_FS_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    # Leave the bottom strip to the legend so it cannot overprint the axis labels.
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save_figure_as_publication_formats(fig, figure_path)
