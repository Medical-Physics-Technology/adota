"""Figures for the anti-diagonal transect (appendix A.6).

Two separate files, so each can be placed independently in LaTeX:

* :func:`angle_map_figure` -- (a) the angular gamma map with the sampled
  anti-diagonal marked.
* :func:`diagonal_transect_figure` -- (b)-(f), stacked on a shared position axis:

    (b) flux-weighted HU along the beamlet, with the MC and ADoTA R80 overlaid
    (c) MC integrated depth dose, normalised per position
    (d) ADoTA - MC depth dose, in % of that position's MC maximum
    (e) gamma pass rate per position
    (f) |R80(ADoTA) - R80(MC)| per position

Both are laid out with ``fig.subplot_mosaic`` so the panes and their colorbars are
placed by name rather than by index arithmetic, and both are written as vector
PDF/SVG plus a 300 dpi PNG.

Colour: grayscale for HU (the CT convention, and achromatic ramps are the safest
possible under any colour-vision deficiency), ``viridis`` for the sequential dose
(matching the grid panels elsewhere in the paper), and ``RdBu_r``, a ColorBrewer
colourblind-safe diverging pair with a neutral midpoint, centred on zero for the
difference. Line colours come from the Okabe-Ito palette.

The gamma pass rate [%] and the range error [mm] are drawn as two stacked panels
rather than one twin-axis plot, so neither quantity is read off a scale that does
not belong to it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import save_figure_as_publication_formats
from src.mc_generation.diagonal_transect import DiagonalTransect

# Okabe-Ito, colourblind-safe.
C_MC = "#E69F00"       # orange   - Monte Carlo
C_AD = "#0072B2"       # blue     - ADoTA
C_SURF = "#009E73"     # green    - patient surface
C_GAMMA = "#0072B2"
C_DR80 = "#D55E00"

# Sized for A4 print: the transect figure is 7.1 in wide, so at full text width on
# A4 (~6.7 in) it scales by ~0.94 and these land near their nominal point sizes.
# Base point sizes, and the factor they are multiplied by. The manuscript
# includes this figure at roughly 0.6 of the width it is authored at, so a
# nominal size renders about 0.6x on the page; FONT_SCALE keeps the smallest
# label above the 6 pt floor the typesetter checks for.
FONT_SCALE = 1.3


@dataclass(frozen=True)
class Fonts:
    """Point sizes used across the figure, scalable as a set."""

    label: float = 9.5
    tick: float = 8.5
    # Legend and bracket labels. Kept level with the tick size rather than below
    # it because they carry the only mathtext ("$R_{80}$"), whose subscripts are
    # drawn at 0.7x and would otherwise be the smallest glyphs in the figure.
    note: float = 9.0
    cbar_label: float = 10.0
    cbar_tick: float = 9.0
    cbar_min: float = 9.0      # floor for the auto-shrunk colorbar labels

    def scaled(self, factor: float) -> "Fonts":
        return Fonts(*(factor * v for v in (self.label, self.tick, self.note,
                                            self.cbar_label, self.cbar_tick,
                                            self.cbar_min)))
# HU window: lung reads black, soft tissue mid-grey, rib/bone white.
HU_VMIN, HU_VMAX = -1000.0, 300.0
DIFF_LIMIT = 20.0      # [%] symmetric limit of the diverging difference map
DPI = 300


def _fitted_cbar_fontsize(fig, cb, label: str, fonts: Fonts) -> float:
    """Largest font (<= ``base``) at which ``label`` fits the colorbar's height.

    A rotated colorbar label runs along the height of its axes, so a two-line
    label on a short panel silently overruns into the neighbouring panel. Measure
    the rendered extent and step the size down until it fits (never below
    ``fonts.cbar_min``).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    available = cb.ax.get_window_extent(renderer).height
    for fs in np.arange(fonts.cbar_label, fonts.cbar_min - 1e-6, -0.25):
        cb.set_label(label, fontsize=float(fs), labelpad=3)
        fig.canvas.draw()
        if cb.ax.yaxis.label.get_window_extent(renderer).height <= available:
            return float(fs)
    return fonts.cbar_min


def _align_axis_labels(fig, axes, side: str = "left") -> None:
    """Put a column of rotated y labels on one vertical line.

    Left to matplotlib, each label is pushed out by its own tick labels, and
    "-1000" is far wider than "100", so a stack of panels ends up with ragged
    label positions. Measure the drawn labels, take the outermost edge, and pin
    every label to it in figure coordinates so they share a clean margin.

    Args:
        fig: The figure (must be drawable; this triggers a draw).
        axes: Axes whose y labels should line up, top to bottom.
        side: ``"left"`` for panel y labels, ``"right"`` for colorbar labels.
    """
    axes = list(axes)
    if len(axes) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [ax.yaxis.label.get_window_extent(renderer) for ax in axes]
    if side == "left":
        edge = min(b.x0 for b in boxes)
        centres = [edge + 0.5 * b.width for b in boxes]
    else:
        edge = max(b.x1 for b in boxes)
        centres = [edge - 0.5 * b.width for b in boxes]
    for ax, x_disp in zip(axes, centres):
        bb = ax.get_window_extent(renderer)
        x, y = inv.transform((x_disp, bb.y0 + 0.5 * bb.height))
        ax.yaxis.set_label_coords(x, y, transform=fig.transFigure)


def _style_cbars(fig, pairs, fonts: Fonts) -> None:
    """Label a set of colorbars at one shared, auto-fitted font size.

    Each label is fitted to its own bar first; the smallest fitting size is then
    applied to all of them, so the labels stay uniform *and* none overruns.
    """
    sizes = []
    for cb, label in pairs:
        cb.ax.tick_params(labelsize=fonts.cbar_tick, length=2.0, pad=1.5)
        sizes.append(_fitted_cbar_fontsize(fig, cb, label, fonts))
    fs = min(sizes)
    for cb, label in pairs:
        cb.set_label(label, fontsize=fs, labelpad=3)


def _pcolor(ax, t: DiagonalTransect, values: np.ndarray, depth_lim, **kw):
    """Draw a (position x depth) map with one column per diagonal position."""
    step = float(t.depth_mm[1] - t.depth_mm[0])
    x_edges = np.arange(t.n + 1) - 0.5
    y_edges = np.concatenate([t.depth_mm - step / 2.0, [t.depth_mm[-1] + step / 2.0]])
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values).T,
                         shading="flat", rasterized=True, **kw)
    ax.set_ylim(depth_lim[1], depth_lim[0])   # depth increases downwards
    ax.set_xlim(-0.5, t.n - 0.5)
    return mesh


def _mark_transition(ax, transition: Optional[float]) -> None:
    """Divider between the two regimes; haloed so it reads on light and dark maps."""
    if transition is None:
        return
    ax.axvline(transition, color="w", lw=1.5, alpha=0.3, zorder=9)
    ax.axvline(transition, color="0.10", lw=0.9, ls=(0, (4, 2.5)), zorder=10)


def angle_map_figure(
    t: DiagonalTransect,
    angle_grid: np.ndarray,
    figure_path: str,
    theta_range: Sequence[float] = (-2.0, 2.0),
    cmap: str = "viridis",
    low_percentile: float = 2.0,
    width_in: float = 3.8,
    height_in: float = 3.1,
    dpi: int = DPI,
    font_scale: float = FONT_SCALE,
) -> List[Path]:
    """(a) The full angular gamma map with the sampled anti-diagonal marked."""
    f = Fonts().scaled(font_scale)
    grid = np.asarray(angle_grid, dtype=float)
    n = grid.shape[0]
    thetas = np.linspace(float(theta_range[0]), float(theta_range[1]), n)
    step = thetas[1] - thetas[0]
    edges = np.concatenate([thetas - step / 2.0, [thetas[-1] + step / 2.0]])
    finite = grid[np.isfinite(grid)]
    vmin = float(np.percentile(finite, low_percentile)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 100.0

    fig, axd = plt.subplot_mosaic(
        [["map", "cbar"]], figsize=(width_in, height_in), dpi=dpi,
        gridspec_kw=dict(width_ratios=[1.0, 0.05], wspace=0.06),
    )
    ax, cax = axd["map"], axd["cbar"]
    m = ax.pcolormesh(edges, edges, np.ma.masked_invalid(grid).T, cmap=cmap,
                      vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    ax.plot(thetas, thetas[::-1], color="w", lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot(thetas, thetas[::-1], color="k", lw=1.1, solid_capstyle="round", zorder=4)
    ax.scatter(thetas, thetas[::-1], s=5.0, color="k", zorder=5, linewidths=0)
    ax.annotate("", xy=(thetas[-1], thetas[0]), xytext=(thetas[-4], thetas[3]),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.1,
                                shrinkA=0, shrinkB=0), zorder=6)
    plate = dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.85)
    ax.text(thetas[0], thetas[-1], "00", fontsize=f.note, ha="left", va="top",
            zorder=7, bbox=plate)
    ax.text(thetas[-1], thetas[0], "17", fontsize=f.note, ha="right", va="bottom",
            zorder=7, bbox=plate)
    ax.set_xlabel(r"$\theta_x$ [degrees]", fontsize=f.label, labelpad=1.5)
    ax.set_ylabel(r"$\theta_y$ [degrees]", fontsize=f.label, labelpad=1.5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(labelsize=f.tick, length=2.2, pad=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    _style_cbars(fig, [(fig.colorbar(m, cax=cax), f"{t.gamma_label} [%]")], f)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def diagonal_transect_figure(
    t: DiagonalTransect,
    figure_path: str,
    depth_lim: Optional[Tuple[float, float]] = None,
    transition: Optional[float] = 10.5,
    regimes: Sequence[Tuple[int, int, str]] = (),
    annotate_positions: Sequence[int] = (0, 6, 9, 17),
    width_in: float = 7.1,
    height_in: float = 8.6,
    dpi: int = DPI,
    font_scale: float = FONT_SCALE,
) -> List[Path]:
    """(b)-(f) stacked on the shared diagonal-position axis.

    Args:
        t: The extracted transect.
        figure_path: Output path without suffix.
        depth_lim: ``(min, max)`` depth [mm] to show; derived from the data if None.
        transition: x position of the regime divider (between two columns), or None.
        regimes: ``(first, last, label)`` brackets drawn above panel (b).
        annotate_positions: Positions whose ``(theta_x, theta_y)`` pair is spelled
            out, on one line, on a secondary axis below panel (f). Keep them far
            enough apart that the one-line pairs do not touch at the print size.
        font_scale: Multiplier on every point size (see :data:`FONT_SCALE`).
    """
    f = Fonts().scaled(font_scale)
    if depth_lim is None:
        lo = np.nanmin(t.entry_mm) - 15.0
        hi = np.nanmax(np.concatenate([t.r80_mc, t.r80_ad])) + 35.0
        depth_lim = (max(float(t.depth_mm[0]), float(lo)),
                     min(float(t.depth_mm[-1]), float(hi)))

    fig, axd = plt.subplot_mosaic(
        [["regimes", "."],
         ["hu", "cb_hu"],
         ["mc", "cb_mc"],
         ["diff", "cb_diff"],
         ["gamma", "."],
         ["range", "."]],
        figsize=(width_in, height_in), dpi=dpi, empty_sentinel=".",
        gridspec_kw=dict(width_ratios=[1.0, 0.026],
                         height_ratios=[0.42, 1.95, 1.72, 1.72, 1.00, 0.88],
                         hspace=0.16, wspace=0.015),
    )
    ax_reg, ax_hu = axd["regimes"], axd["hu"]
    ax_mc, ax_df = axd["mc"], axd["diff"]
    ax_g, ax_r = axd["gamma"], axd["range"]

    # ---- regime brackets ---------------------------------------------------
    ax_reg.set_xlim(-0.5, t.n - 0.5)
    ax_reg.set_ylim(0, 1)
    ax_reg.axis("off")
    for first, last, label in regimes:
        ax_reg.plot([first - 0.4, last + 0.4], [0.88, 0.88], color="0.25", lw=0.9,
                    solid_capstyle="butt", clip_on=False)
        for x in (first - 0.4, last + 0.4):
            ax_reg.plot([x, x], [0.62, 0.88], color="0.25", lw=0.9, clip_on=False)
        ax_reg.text((first + last) / 2.0, 0.52, label, fontsize=f.note,
                    ha="center", va="top", color="0.15")

    # ---- (b) HU + ranges ---------------------------------------------------
    m_hu = _pcolor(ax_hu, t, t.hu, depth_lim, cmap="gray", vmin=HU_VMIN, vmax=HU_VMAX)
    ax_hu.plot(t.positions, t.entry_mm, color=C_SURF, lw=1.1, ls=(0, (1.6, 1.6)),
               label="patient surface", zorder=6)
    ax_hu.plot(t.positions, t.r80_mc, color=C_MC, lw=1.4, marker="o", ms=2.6,
               label="$R_{80}$ Monte Carlo", zorder=7)
    ax_hu.plot(t.positions, t.r80_ad, color=C_AD, lw=1.4, ls="--", marker="s", ms=2.4,
               label="$R_{80}$ ADoTA", zorder=8)
    ax_hu.set_ylabel("Depth\n[mm]", fontsize=f.label)
    leg = ax_hu.legend(fontsize=f.note, loc="lower center", ncol=3, framealpha=0.92,
                       borderpad=0.28, handlelength=1.8, columnspacing=1.4,
                       bbox_to_anchor=(0.5, -0.012))
    leg.get_frame().set_linewidth(0.4)
    leg.set_zorder(13)

    # ---- (c) MC depth dose -------------------------------------------------
    norm_mc = 100.0 * t.ddd_mc / np.nanmax(t.ddd_mc, axis=1, keepdims=True)
    m_mc = _pcolor(ax_mc, t, norm_mc, depth_lim, cmap="viridis", vmin=0.0, vmax=100.0)
    ax_mc.plot(t.positions, t.r80_mc, color="w", lw=0.9, ls=(0, (2.5, 2.0)), zorder=6)
    ax_mc.set_ylabel("Depth\n[mm]", fontsize=f.label)

    # ---- (d) ADoTA - MC ----------------------------------------------------
    diff = 100.0 * (t.ddd_ad - t.ddd_mc) / np.nanmax(t.ddd_mc, axis=1, keepdims=True)
    m_df = _pcolor(ax_df, t, diff, depth_lim, cmap="RdBu_r",
                   vmin=-DIFF_LIMIT, vmax=DIFF_LIMIT)
    ax_df.set_ylabel("Depth\n[mm]", fontsize=f.label)

    # ---- (e) gamma pass rate ----------------------------------------------
    ax_g.plot(t.positions, t.gamma, color=C_GAMMA, lw=1.4, marker="o", ms=3.2)
    ax_g.set_ylabel("Γ pass rate\n[%]", fontsize=f.label)
    ax_g.grid(True, ls=":", lw=0.4, color="0.75")
    ax_g.set_axisbelow(True)

    # ---- (f) distal range error -------------------------------------------
    ax_r.plot(t.positions, np.abs(t.r80_ad - t.r80_mc), color=C_DR80, lw=1.4,
              marker="^", ms=3.4)
    ax_r.set_ylabel("$|\\Delta R_{80}|$\n[mm]", fontsize=f.label)
    ax_r.grid(True, ls=":", lw=0.4, color="0.75")
    ax_r.set_axisbelow(True)

    # Depth ticks every 50 mm on all three maps, so the shorter panels do not fall
    # back to a two-tick axis at the larger print font.
    depth_ticks = [d for d in range(0, int(t.depth_mm[-1]) + 1, 50)
                   if depth_lim[0] <= d <= depth_lim[1]]
    for ax in (ax_hu, ax_mc, ax_df):
        ax.set_yticks(depth_ticks)
    ax_r.set_yticks([0, 2, 4])

    # ---- shared x formatting ----------------------------------------------
    for ax in (ax_hu, ax_mc, ax_df, ax_g, ax_r):
        ax.set_xlim(-0.5, t.n - 0.5)
        ax.set_xticks(t.positions)
        _mark_transition(ax, transition)
        ax.tick_params(labelsize=f.tick, length=2.2, pad=1.5)
    for ax in (ax_hu, ax_mc, ax_df, ax_g):
        ax.set_xticklabels([])
    ax_r.set_xticklabels([f"{k:02d}" for k in t.positions], fontsize=f.tick)
    _mark_transition(ax_reg, transition)

    # The (theta_x, theta_y) pairs go on their own axis below the index ticks, one
    # line each, so a pair is never broken across lines or squeezed into a single
    # column's width.
    picks = [k for k in annotate_positions if 0 <= k < t.n and np.isfinite(t.theta_x[k])]
    if picks:
        sec = ax_r.secondary_xaxis(-0.34)
        sec.set_xticks(picks)
        sec.set_xticklabels(
            [f"({t.theta_x[k]:+.1f}°, {t.theta_y[k]:+.1f}°)" for k in picks],
            fontsize=f.tick)
        sec.tick_params(axis="x", length=0, pad=1.0)
        sec.spines["bottom"].set_visible(False)
    ax_r.set_xlabel("Position along the sampled anti-diagonal "
                    r"(index; $\theta_x$, $\theta_y$)", fontsize=f.label,
                    labelpad=22 if picks else 3)

    # ---- colorbars (labels auto-fitted to the bar height) ------------------
    _style_cbars(fig, [
        (fig.colorbar(m_hu, cax=axd["cb_hu"], extend="both"),
         "CT number\n[HU]"),
        (fig.colorbar(m_mc, cax=axd["cb_mc"]),
         "MC dose\n[% of max]"),
        (fig.colorbar(m_df, cax=axd["cb_diff"], extend="both"),
         "ADoTA $-$ MC\n[% of max]"),
    ], f)
    _align_axis_labels(fig, [axd["cb_hu"], axd["cb_mc"], axd["cb_diff"]], side="right")

    for k in t.missing:
        for ax in (ax_hu, ax_mc, ax_df):
            ax.add_patch(plt.Rectangle((k - 0.5, depth_lim[0]), 1.0,
                                       depth_lim[1] - depth_lim[0], facecolor="0.88",
                                       edgecolor="none", zorder=2))
            ax.text(k, np.mean(depth_lim), "no data", rotation=90, fontsize=f.note,
                    ha="center", va="center", color="0.35", zorder=3)

    _align_axis_labels(fig, [ax_hu, ax_mc, ax_df, ax_g, ax_r], side="left")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
