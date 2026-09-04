"""Shared matplotlib helpers for the publication figures.

- :func:`identify_axes` stamps each mosaic pane with its key, for laying a
  figure out before the real content exists.
- :func:`aligned_colorbar` attaches a colorbar whose height matches the image
  axis it belongs to, which ``fig.colorbar`` does not guarantee for imshow axes.
- :func:`save_figure_as_publication_formats` writes one figure as PNG, PDF and
  SVG next to each other and returns the paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def identify_axes(ax_dict: dict[str, plt.Axes], fontsize: int = 48) -> None:
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def aligned_colorbar(
    fig,
    ct_ax,
    ax,
    label: str,
    label_coords: tuple = (4.7, 0.5),
    label_fontsize: int = 18,
    tick_fontsize: int = 15,
    size: str = "5%",
):
    """Function to create an aligned colorbar for the given axes.

    Args:
        fig (_type_): Figure object to which the colorbar will be added.
        ct_ax (_type_): Axes object for which the colorbar is aligned.
        ax (_type_): Axes object to which the colorbar is aligned.
        label (str): Label for the colorbar.

    Returns:
        _type_: _description_
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.05)
    colorbar = fig.colorbar(ct_ax, cax=cax, orientation="vertical")
    colorbar.set_label(label, fontsize=label_fontsize, labelpad=10)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_label_coords(label_coords[0], label_coords[1])
    colorbar.ax.yaxis.set_tick_params(labelsize=tick_fontsize)
    return colorbar


def save_figure_as_publication_formats(fig, figure_path: str) -> list[Path]:
    output_path = Path(figure_path)
    output_paths = [
        output_path.with_suffix(f".{extension}") for extension in ("svg", "pdf", "png")
    ]
    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    return output_paths
