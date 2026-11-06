"""The difficulty score against dose engines on saved beamlet grids
(``scripts/active_learning/difficulty_engine_grids.py``).

- :func:`difficulty_story_figure`: one slide. Per example beamlet, the input (CT, flux,
  energy), the score it gives, and every engine's dose in two planes with its gamma
  pass rate and MAPE.
- :func:`difficulty_trend_figure`: one metric per score decile, one line per engine.

Reuses :func:`src.figures.pencil_beam._bev_planes` for the planes, the house palette,
:func:`src.figures.al_curves.style_axis` and :func:`aligned_colorbar`, and saves through
:func:`save_figure_as_publication_formats`. The dose overlay follows
:func:`src.figures.hpo.finalist_beamlet_figure`: faded in over the CT, with the Monte
Carlo isodose as a white contour in every row, so a shift is visible against it.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from src.figures.al_curves import COLORS, LINESTYLES, style_axis
from src.figures.axes_utils import PUBLICATION_FORMATS, aligned_colorbar, save_figure_as_publication_formats
from src.figures.pencil_beam import _bev_planes

PLANE_TITLES = ("Axial", "Longitudinal")
MARKERS = ("o", "s", "^", "D", "v")


@dataclass(frozen=True)
class StoryCase:
    """One example beamlet of the story figure. Arrays are ``(z, y, x)``, beam along x."""

    title: str                        # "Low difficulty"
    subtitle: str                     # the score, the anatomy, the energy
    ct: np.ndarray
    flux: np.ndarray
    reference: np.ndarray             # the Monte Carlo dose
    doses: Mapping[str, np.ndarray]   # engine label -> dose
    notes: Mapping[str, str]          # engine label -> its metrics


def _depth_window(reference: np.ndarray, floor: float = 0.01, margin_mm: int = 15) -> int:
    """How many depth voxels show the beamlet: to its last laterally integrated dose
    above ``floor`` of the maximum, plus ``margin_mm`` (1 mm voxels)."""
    idd = reference.sum(axis=(0, 1))
    above = np.flatnonzero(idd > floor * idd.max())
    return min(reference.shape[2], int(above[-1]) + 1 + margin_mm) if len(above) else reference.shape[2]


def _draw_ct(ax, ct_plane: np.ndarray, extent, title: str) -> None:
    ax.imshow(ct_plane, cmap="gray", vmin=-1000, vmax=1500, aspect="auto", extent=extent)
    ax.set_title(title, fontsize=8, pad=2)


def _draw_flux(ax, flux_plane: np.ndarray, extent, top: float):
    """The flux as a heatmap, as a fraction of the crop's maximum (the model's normalisation)."""
    return ax.imshow(flux_plane / top, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto", extent=extent,
                     interpolation="bilinear")


def _colorbar(fig, image, slot, label: str, ticks=None):
    """A colorbar as tall as ``slot``, with its label along the bar, right of the ticks."""
    host = fig.add_subplot(slot.subgridspec(1, 2, width_ratios=[0.17, 0.83])[0])
    host.axis("off")
    bar = aligned_colorbar(fig, image, host, label, label_coords=(17.0, 0.5), label_fontsize=8, tick_fontsize=7,
                           size="100%")
    if ticks is None:
        bar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        bar.set_ticks(ticks)
    bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    bar.ax.tick_params(length=2, pad=1.5)
    bar.outline.set_linewidth(0.5)
    return bar


def _draw_dose(ax, ct_plane, dose_plane, reference_plane, extent, vmax: float):
    ax.imshow(ct_plane, cmap="gray", vmin=-1000, vmax=1500, aspect="auto", extent=extent)
    fade = np.clip(dose_plane / (0.05 * vmax), 0.0, 1.0) * 0.85
    image = ax.imshow(dose_plane, cmap="turbo", vmin=0.0, vmax=vmax, aspect="auto", extent=extent, alpha=fade,
                      interpolation="bilinear")
    ax.contour(reference_plane, levels=[0.5 * vmax], colors="white", linewidths=0.6,
               extent=(extent[0], extent[1], extent[3], extent[2]))
    return image


def difficulty_story_figure(cases: Sequence[StoryCase], engines: Sequence[str], *, figure_path: str, title: str,
                            caption: str = "", reference_label: str = "Monte Carlo", dose_unit: str = "Gy per MU",
                            width_in: float = 13.33, height_in: float = 7.5,
                            formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """Input, then estimated difficulty, then every engine's dose and metrics.

    Rows: the CT, the flux heatmap, the reference dose, one per engine. Each case has
    its own dose scale, drawn once beside all its dose rows.

    Args:
        cases: The example beamlets, one column pair each, chosen from their scores alone.
        engines: Engine labels, one row each, in display order; keys of ``doses`` and ``notes``.
        figure_path: Output stem; each of ``formats`` is written beside it.
    """
    n_input = 2
    n_rows = n_input + 1 + len(engines)
    widths = [0.55]
    for _ in cases:
        widths += [1.0, 1.0, 0.42]
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)
    grid = fig.add_gridspec(n_rows, len(widths), width_ratios=widths, left=0.012, right=0.985, top=0.825, bottom=0.08,
                            wspace=0.03, hspace=0.1)
    row_labels = ["CT", "Flux", reference_label, *engines]
    for r, label in enumerate(row_labels):
        ax = fig.add_subplot(grid[r, 0])
        ax.axis("off")
        ax.text(0.98, 0.5, label, ha="right", va="center", fontsize=9, fontweight="semibold")
    for k, case in enumerate(cases):
        c0 = 1 + 3 * k
        scale_top = float(case.reference.max()) or 1.0
        flux_top = float(case.flux.max()) or 1.0
        iz, iy, _ = np.unravel_index(int(np.argmax(case.reference)), case.reference.shape)
        depth = _depth_window(case.reference)
        ct_planes = [p[:, :depth] for p in _bev_planes(case.ct, iz, iy)]
        flux_planes = [p[:, :depth] for p in _bev_planes(case.flux, iz, iy)]
        ref_planes = [p[:, :depth] for p in _bev_planes(case.reference, iz, iy)]
        extent = (0, depth, case.ct.shape[1], 0)
        flux_image = dose_image = None
        for c in range(2):
            ax = fig.add_subplot(grid[0, c0 + c])
            _draw_ct(ax, ct_planes[c], extent, PLANE_TITLES[c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax = fig.add_subplot(grid[1, c0 + c])
            flux_image = _draw_flux(ax, flux_planes[c], extent, flux_top)
            ax.set_xticks([])
            ax.set_yticks([])
        for r, label in enumerate([reference_label, *engines], start=n_input):
            volume = case.reference if r == n_input else case.doses[label]
            planes = [p[:, :depth] for p in _bev_planes(volume, iz, iy)]
            for c in range(2):
                ax = fig.add_subplot(grid[r, c0 + c])
                dose_image = _draw_dose(ax, ct_planes[c], planes[c], ref_planes[c], extent, scale_top)
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0 and label in case.notes:
                    ax.text(0.015, 0.95, case.notes[label], transform=ax.transAxes, ha="left", va="top", fontsize=7,
                            color="white", bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=1.2))
        _colorbar(fig, flux_image, grid[1, c0 + 2], "Flux", ticks=[0.0, 0.5, 1.0])
        _colorbar(fig, dose_image, grid[n_input:, c0 + 2], f"Dose [{dose_unit}]")
        left = grid[0, c0].get_position(fig).x0
        right = grid[0, c0 + 1].get_position(fig).x1
        fig.text(0.5 * (left + right), 0.905, case.title, ha="center", va="bottom", fontsize=11, fontweight="semibold")
        fig.text(0.5 * (left + right), 0.9, f"{case.subtitle}\nDepth shown: {depth} mm", ha="center", va="top",
                 fontsize=8, linespacing=1.1)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=13, fontweight="bold")
    if caption:
        fig.text(0.012, 0.012, textwrap.fill(caption, width=215), fontsize=7.5, ha="left", va="bottom")
    paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
    plt.close(fig)
    return paths


def difficulty_trend_figure(trend: pd.DataFrame, metric: str, engines: Sequence[str], *, figure_path: str,
                            ylabel: str, title: str, caption: str = "",
                            formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """Median of one metric per within-field score decile, one line per engine.

    Args:
        trend: :func:`src.active_learning.ensemble.engine_grids.decile_table` with engine labels.
        metric: The ``metric`` value of ``trend`` to draw.
        engines: Engine labels, in display order; each keeps its color across figures.
        figure_path: Output stem; each of ``formats`` is written beside it.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=300, constrained_layout=True)
    part = trend[trend["metric"] == metric]
    for i, engine in enumerate(engines):
        line = part[part["engine"] == engine].sort_values("field_decile")
        ax.plot(line["field_decile"], line["median"], color=COLORS[i], linestyle=LINESTYLES[i], linewidth=2.0,
                marker=MARKERS[i], markersize=5, label=engine)
    style_axis(ax, xlabel="Difficulty decile within field (1 = easiest, 10 = hardest)", ylabel=ylabel, title=title)
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=True, framealpha=0.9, fontsize=8)
    if caption:
        fig.supxlabel(textwrap.fill(caption, width=105), fontsize=7.5)
    paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
    plt.close(fig)
    return paths
