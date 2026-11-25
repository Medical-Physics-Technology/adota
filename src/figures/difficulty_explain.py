"""Slide figures that explain a beamlet's difficulty score.

Four figures, all for a supervisor-meeting slide (readable fonts, units on every axis, one colour per
feature group across every figure):

- :func:`contribution_figure`: per-group contributions to the score, one panel per scorer and example;
- :func:`contrast_figure`: per-group difference between a harder and an easier beamlet, ensemble bars with
  the members' differences as markers;
- :func:`decile_profile_figure`: per-group contributions and the mean score across the deciles of the
  reference score, one panel per beamlet subset;
- :func:`beam_path_figure`: the beamlet's CT, analytic dose and depth profiles along the beam axis, with the
  depth intervals the explanation refers to marked.

Reuses the house style of :mod:`src.figures.al_curves` (``COLORS``, ``style_axis``), the CT window and dose
overlay of :mod:`src.figures.difficulty_engine_grids`, and saves through
:func:`src.figures.axes_utils.save_figure_as_publication_formats`; colorbars via ``aligned_colorbar``.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.acquisition.explain.profiles import BeamAxisProfile, BeamletInputs
from src.figures.al_curves import COLORS, style_axis
from src.figures.axes_utils import PUBLICATION_FORMATS, aligned_colorbar, save_figure_as_publication_formats

GROUP_PALETTE: Tuple[str, ...] = COLORS + (
    "#17becf", "#e377c2", "#bcbd22", "#7f7f7f", "#393b79", "#8c6d31",
)
MEMBER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
CT_WINDOW = (-1000.0, 1500.0)
TICK_SIZE, LABEL_SIZE, TITLE_SIZE = 8.5, 9.5, 10.5
_RC = {"font.size": 9, "axes.titlesize": TITLE_SIZE, "axes.labelsize": LABEL_SIZE, "xtick.labelsize": TICK_SIZE,
       "ytick.labelsize": TICK_SIZE, "legend.fontsize": TICK_SIZE}


def group_colors(groups: Sequence[str]) -> Dict[str, str]:
    """Stable colour per group: sorted unique names mapped onto :data:`GROUP_PALETTE` in order.

    The same group set gives the same colours in every figure; a set larger than the palette cycles.
    """
    names = sorted(set(groups))
    return {name: GROUP_PALETTE[i % len(GROUP_PALETTE)] for i, name in enumerate(names)}


def _wrap(text: str, width: int = 28) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False)


def _group_legend(fig, groups: Sequence[str], colors: Mapping[str, str], ncol: int, y: float = 0.0) -> None:
    handles = [Patch(facecolor=colors[g], label=g) for g in groups]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
               fontsize=TICK_SIZE, title="Feature group", title_fontsize=LABEL_SIZE)


def contribution_figure(contrib: pd.DataFrame, totals: pd.DataFrame, *, examples: Sequence[str],
                        example_labels: Mapping[str, str], scorers: Sequence[str], colors: Mapping[str, str],
                        figure_path: str, title: str, score_unit: str = "score units",
                        formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """Grid of per-group contributions: rows are scorers (first row, the ensemble, taller), columns examples.

    Every panel has the same group order (mean |contribution| of the first scorer across the examples, largest
    on top), a vertical line at 0 and x limits shared within its row. The group names are the y tick labels of
    the first row only (wrapped); the member rows are read through the colours and the legend below.

    Args:
        contrib: Columns ``beamlet_id, scorer, group, contribution``.
        totals: Columns ``scorer, beamlet_id, score, baseline``, one row per scorer and beamlet.
        examples: Beamlet ids, one column each.
        scorers: Scorer labels, one row each; the first is drawn taller.
    """
    sel = contrib[contrib["beamlet_id"].isin(examples)]
    first = sel[sel["scorer"] == scorers[0]]
    order = (first.assign(a=first["contribution"].abs()).groupby("group")["a"].mean()
             .sort_values(ascending=False).index.tolist())
    for g in sorted(set(sel["group"]) - set(order)):
        order.append(g)
    n_rows, n_cols = len(scorers), len(examples)
    heights = [max(2.6, 0.42 * len(order))] + [1.15] * (n_rows - 1)  # room for two-line group labels
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 0.9 + sum(heights) + 1.0), squeeze=False,
                                 gridspec_kw={"height_ratios": heights, "wspace": 0.08, "hspace": 0.55,
                                              "left": 0.2, "right": 0.985, "top": 0.88, "bottom": 0.12})
        ypos = np.arange(len(order))[::-1]
        for r, scorer in enumerate(scorers):
            rows = sel[sel["scorer"] == scorer]
            vals = {(b, g): float(v) for b, g, v in zip(rows["beamlet_id"], rows["group"], rows["contribution"])}
            allv = list(vals.values()) + [0.0]
            pad = 0.1 * (max(allv) - min(allv) or 1.0)
            xlim = (min(allv) - pad, max(allv) + pad)
            for c, ex in enumerate(examples):
                ax = axes[r, c]
                x = [vals.get((ex, g), 0.0) for g in order]
                ax.barh(ypos, x, color=[colors[g] for g in order], height=0.72)
                ax.axvline(0.0, color="black", linewidth=0.9)
                ax.set_xlim(*xlim)
                ax.set_ylim(-0.6, len(order) - 0.4)
                style_axis(ax, xlabel="", ylabel="", title="")
                ax.grid(False, axis="y")
                ax.set_title(scorer, loc="left", fontsize=9 if r else 9.5, fontweight="semibold", pad=3)
                ax.set_yticks(ypos)
                ax.set_yticklabels([_wrap(g) for g in order] if (c == 0 and r == 0) else [], fontsize=TICK_SIZE)
                if c > 0 or r > 0:
                    ax.tick_params(axis="y", length=0)
                tot = totals[(totals["scorer"] == scorer) & (totals["beamlet_id"] == ex)]
                if len(tot):
                    delta = float(tot["score"].iloc[0] - tot["baseline"].iloc[0])
                    ax.set_title(f"score - baseline = {delta:+.2f}", loc="right", fontsize=TICK_SIZE, pad=3)
                if r == 0:
                    ax.text(0.5, 1.2, example_labels.get(ex, ex), transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=11, fontweight="bold")
                if r == n_rows - 1:
                    ax.set_xlabel(f"Contribution [{score_unit}]")
                else:
                    ax.tick_params(axis="x", labelbottom=True)
        fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=13, fontweight="bold")
        _group_legend(fig, order, colors, ncol=min(4, len(order)), y=0.045)
        paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
        plt.close(fig)
    return paths


def contrast_figure(contrast: pd.DataFrame, *, pairs: Sequence[Tuple[str, str]],
                    pair_labels: Mapping[Tuple[str, str], str], colors: Mapping[str, str], figure_path: str,
                    title: str, ensemble: str = "ensemble", top_k: int = 3,
                    formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """One panel per ``(harder_id, easier_id)`` pair: the ensemble's per-group difference as bars.

    Bars are sorted with the largest positive difference on top and coloured by group; every other scorer's
    difference is a black marker (one shape per member, legend once) on the same row. The ``top_k`` groups of
    the ensemble have a bold y label.

    Args:
        contrast: Columns ``scorer, item, harder, easier, difference, rank``; ``harder`` and ``easier`` hold
            the beamlet ids of the pair, ``item`` the group name.
    """
    members = [s for s in dict.fromkeys(contrast["scorer"]) if s != ensemble]
    markers = {m: MEMBER_MARKERS[i % len(MEMBER_MARKERS)] for i, m in enumerate(members)}
    n = len(pairs)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.6), squeeze=False,
                                 gridspec_kw={"wspace": 0.75, "left": 0.2 / n + 0.02, "right": 0.985,
                                              "top": 0.84, "bottom": 0.27})
        for ax, pair in zip(axes[0], pairs):
            sub = contrast[(contrast["harder"] == pair[0]) & (contrast["easier"] == pair[1])]
            ens = sub[sub["scorer"] == ensemble].sort_values("difference", ascending=False)
            items = ens["item"].tolist()
            ypos = np.arange(len(items))[::-1]
            ax.barh(ypos, ens["difference"].to_numpy(), color=[colors[i] for i in items], height=0.7, zorder=2)
            for m in members:
                d = sub[sub["scorer"] == m].set_index("item")["difference"]
                keep = [(y, float(d[i])) for y, i in zip(ypos, items) if i in d.index]
                if keep:
                    ax.scatter([k[1] for k in keep], [k[0] for k in keep], marker=markers[m], s=34, color="black",
                               edgecolor="white", linewidth=0.5, zorder=3)
            ax.axvline(0.0, color="black", linewidth=0.9, zorder=1)
            style_axis(ax, xlabel="Contribution difference (harder minus easier) [score units]", ylabel="",
                       title=pair_labels.get(pair, f"{pair[0]} vs {pair[1]}"))
            ax.grid(False, axis="y")
            ax.set_yticks(ypos)
            ax.set_yticklabels([_wrap(i, 30) for i in items], fontsize=TICK_SIZE)
            for k, lab in enumerate(ax.get_yticklabels()):
                if k < top_k:
                    lab.set_fontweight("bold")
            ax.set_ylim(-0.6, len(items) - 0.4)
        handles = [plt.Line2D([], [], marker=markers[m], linestyle="", color="black", markeredgecolor="white",
                              markersize=7, label=m) for m in members]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=min(3, len(handles)),
                   frameon=False, fontsize=TICK_SIZE, title="Ensemble members (markers), ensemble (bars)",
                   title_fontsize=LABEL_SIZE)
        fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=13, fontweight="bold")
        paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
        plt.close(fig)
    return paths


def decile_profile_figure(profile: pd.DataFrame, *, panels: Sequence[str], panel_titles: Mapping[str, str],
                          colors: Mapping[str, str], figure_path: str, title: str, min_n: int = 20,
                          score_unit: str = "score units",
                          formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """One panel per entry of ``panels``: per-group contributions across the deciles of the reference score.

    Each decile is a diverging stacked bar (positive group means up from 0, negative ones down, one colour per
    group, the same stacking order everywhere: the order of ``colors``) with the mean score minus baseline as a
    black line on top. The score is linear, so the line is exactly the sum of the bars (positive stack plus
    negative stack). Deciles with fewer than ``min_n`` beamlets show no bars and no marker, only a grey note.
    All panels share y limits.

    Args:
        profile: Columns ``panel, decile, group, mean_contribution, n_beamlets, n_patients, mean_total``.
        panels: Panel keys in drawing order (two columns).
    """
    groups = [g for g in colors if g in set(profile["group"])]
    n_dec = int(profile["decile"].max())
    deciles = np.arange(1, n_dec + 1)
    stacks, lows, highs = {}, [], []
    for panel in panels:
        sub = profile[profile["panel"] == panel]
        for d in deciles:
            rows = sub[sub["decile"] == d]
            n = int(rows["n_beamlets"].max()) if len(rows) else 0
            if n < min_n:
                stacks[(panel, d)] = None
                continue
            vals = rows.groupby("group")["mean_contribution"].sum()
            total = float(rows["mean_total"].iloc[0])
            stacks[(panel, d)] = (n, [(g, float(vals.get(g, 0.0))) for g in groups], total)
            v = [float(vals.get(g, 0.0)) for g in groups]
            lows.append(min(sum(x for x in v if x < 0), total))
            highs.append(max(sum(x for x in v if x > 0), total))
    lo, hi = (min(lows), max(highs)) if lows else (-1.0, 1.0)
    pad = 0.08 * ((hi - lo) or 1.0)
    ncols = min(2, len(panels))
    nrows = -(-len(panels) // ncols)
    fig_h = 3.9 * nrows + 2.3
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, fig_h), squeeze=False,
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.22, "left": 0.08, "right": 0.985,
                                              "top": 0.9, "bottom": 1.55 / fig_h})
        flat = list(axes.ravel())
        for ax in flat[len(panels):]:
            ax.remove()
        for ax, panel in zip(flat, panels):
            ticks = []
            xs, ys = [], []
            for d in deciles:
                st = stacks[(panel, d)]
                n_all = profile[(profile["panel"] == panel) & (profile["decile"] == d)]["n_beamlets"]
                ticks.append(f"D{d}\nn={int(n_all.max()) if len(n_all) else 0}")
                if st is None:
                    ax.text(d, 0.0, f"n < {min_n}", color="grey", fontsize=TICK_SIZE - 1.5, ha="center",
                            va="center", rotation=90)
                    continue
                pos = neg = 0.0
                for g, v in st[1]:
                    base = pos if v >= 0 else neg
                    ax.bar(d, v, bottom=base, width=0.75, color=colors[g], zorder=2)
                    if v >= 0:
                        pos += v
                    else:
                        neg += v
                xs.append(d)
                ys.append(st[2])
            ax.plot(xs, ys, color="black", marker="o", markersize=4.5, linewidth=1.3, zorder=4)
            ax.axhline(0.0, color="black", linewidth=0.9, zorder=3)
            style_axis(ax, xlabel="Score decile of the reference (D1 easiest)",
                       ylabel=f"Mean contribution [{score_unit}]", title=panel_titles.get(panel, panel))
            ax.set_xticks(deciles)
            ax.set_xticklabels(ticks, fontsize=TICK_SIZE - 1)
            ax.set_xlim(0.4, n_dec + 0.6)
            ax.set_ylim(lo - pad, hi + pad)
        _group_legend(fig, groups, colors, ncol=min(4, len(groups)), y=1.05 / fig_h)
        fig.legend(handles=[plt.Line2D([], [], color="black", marker="o", markersize=4.5,
                                       label="mean score minus baseline")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=TICK_SIZE)
        fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=13, fontweight="bold")
        paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
        plt.close(fig)
    return paths


@dataclass(frozen=True)
class DepthMark:
    """A depth interval (or a point if start == end) to mark on the beam-path figure."""

    label: str
    start_mm: float
    end_mm: float
    color: str


def _depth_window(profile: BeamAxisProfile) -> Tuple[float, float]:
    """Depth range of the dose region: to the deeper of the 1% fall-off and the BP zone end, plus 15 mm."""
    idd = np.asarray(profile.idd_analytic)
    above = np.flatnonzero(idd >= 0.01)
    falloff = float(profile.depth_mm[above[-1]]) if above.size else float(profile.depth_mm[-1])
    end = max(falloff, float(profile.bp_zone_mm[1])) + 15.0
    return -0.5 * profile.dz_mm, min(end, float(profile.depth_mm[-1]) + 0.5 * profile.dz_mm)


def _plane_image(ax, ct_plane, dose_plane, dz: float, peak_lat: int, dose_vmax: float = 1.0):
    """CT in grey with the dose faded in and its 50% isodose; ``*_plane`` are ``(lateral, depth)``."""
    n_lat, n_depth = ct_plane.shape
    x0, x1 = -0.5 * dz, (n_depth - 0.5) * dz
    top, bottom = (-0.5 - peak_lat) * dz, (n_lat - 0.5 - peak_lat) * dz
    extent = (x0, x1, bottom, top)
    ax.imshow(ct_plane, cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], aspect="equal", extent=extent)
    fade = np.clip(dose_plane / (0.05 * dose_vmax), 0.0, 1.0) * 0.85
    ax.imshow(dose_plane, cmap="turbo", vmin=0.0, vmax=dose_vmax, aspect="equal", extent=extent, alpha=fade,
              interpolation="bilinear")
    ax.contour(dose_plane, levels=[0.5 * dose_vmax], colors="white", linewidths=0.8,
               extent=(x0, x1, top, bottom))
    return ax.images[0]


def beam_path_figure(inputs: BeamletInputs, profile: BeamAxisProfile, *, title: str, marks: Sequence[DepthMark],
                     figure_path: str, formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """The beamlet on its beam axis: two image planes on the left, three depth profiles on the right.

    Left: ``Axial`` (depth x ``W``, fixed z) and ``Longitudinal`` (depth x ``H``, fixed y) planes through the
    analytic peak, the CT in grey (-1000 to 1500 HU), the analytic dose faded in with its 50% isodose, the beam
    axis (dashed), the Bragg-peak zone (band), the peak (marker) and the edge-metric sphere (circle).
    Right: HU along the axis and flux-weighted, cumulative WEPL, and the laterally integrated dose.
    ``marks`` are drawn on all panels.
    """
    dz = float(profile.dz_mm)
    pd_, ph, pw = (int(i) for i in profile.peak_index)
    peak_mm = pd_ * dz
    xlim = _depth_window(profile)
    dose, ct = np.asarray(profile.analytic_dose), np.asarray(inputs.ct_hu)
    # The model grid is the (z, y, x) beam's-eye crop permuted to (x, z, y): H is z, W is y. The story
    # figure's planes (pencil_beam._bev_planes) are "Axial" = fixed z and "Longitudinal" = fixed y.
    planes = {"Axial": (ct[:, ph, :].T, dose[:, ph, :].T, pw),
              "Longitudinal": (ct[:, :, pw].T, dose[:, :, pw].T, ph)}
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13.0, 6.9))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.06, right=0.985, top=0.9, bottom=0.2,
                                 wspace=0.22)
        left = outer[0].subgridspec(2, 2, width_ratios=[1.0, 0.035], wspace=0.04, hspace=0.3)
        right = outer[1].subgridspec(3, 1, hspace=0.4)
        img_axes, first_image = [], None
        for i, (name, (ct_p, dose_p, lat)) in enumerate(planes.items()):
            ax = fig.add_subplot(left[i, 0])
            image = _plane_image(ax, ct_p, dose_p, dz, lat)
            first_image = image if first_image is None else first_image
            ax.axhline(0.0, color="white", linestyle="--", linewidth=1.0)
            ax.plot([peak_mm], [0.0], marker="*", color="white", markeredgecolor="black", markersize=11, zorder=5)
            t = np.linspace(0, 2 * np.pi, 200)
            r = profile.sphere_radius_mm
            ax.plot(peak_mm + r * np.cos(t), r * np.sin(t), color="cyan", linewidth=1.1)
            ax.set_xlim(*xlim)
            ax.set_ylim(min(ax.get_ylim()), max(ax.get_ylim()))
            ax.set_title(name, loc="left", fontweight="semibold", pad=3)
            ax.set_ylabel("Lateral position [mm]")
            ax.set_anchor("S" if i == 0 else "N")
            img_axes.append(ax)
        img_axes[0].tick_params(labelbottom=False)
        img_axes[1].set_xlabel("Depth [mm]")
        host = fig.add_subplot(left[:, 1])
        host.axis("off")
        aligned_colorbar(fig, first_image, host, "HU", label_coords=(-2.6, 0.5), label_fontsize=LABEL_SIZE,
                         tick_fontsize=TICK_SIZE, size="100%")
        line_axes = [fig.add_subplot(right[k, 0]) for k in range(3)]
        d = profile.depth_mm
        line_axes[0].plot(d, profile.hu_axis, color=COLORS[0], label="HU on the beam axis")
        line_axes[0].plot(d, profile.hu_flux_weighted, color=COLORS[3], label="Flux-weighted mean HU")
        style_axis(line_axes[0], xlabel="", ylabel="HU", title="Tissue along the beam")
        line_axes[1].plot(d, profile.wepl_axis_mm, color=COLORS[0], label="Cumulative WEPL on the axis")
        line_axes[1].plot(d, d, color="black", linestyle="--", linewidth=1.0, label="Water (WEPL = depth)")
        style_axis(line_axes[1], xlabel="", ylabel="WEPL (mm)", title="Water-equivalent path length")
        line_axes[2].plot(d, profile.idd_analytic, color=COLORS[1], label="Analytic")
        if profile.idd_mc is not None:
            line_axes[2].plot(d, profile.idd_mc, color="black", linestyle="--", label="Monte Carlo")
        style_axis(line_axes[2], xlabel="Depth [mm]", ylabel="Relative IDD", title="Laterally integrated dose")
        for k, ax in enumerate(line_axes):
            ax.set_xlim(*xlim)
            if k < 2:
                ax.tick_params(labelbottom=False)
            ax.legend(loc="best", frameon=True, framealpha=0.85, fontsize=TICK_SIZE)
        zone = profile.bp_zone_mm
        handles = [Patch(facecolor="#bbbbbb", alpha=0.5, label=f"Bragg-peak zone ({zone[0]:.0f} to {zone[1]:.0f} mm)")]
        for ax in img_axes + line_axes:
            if ax in line_axes:
                ax.axvspan(zone[0], zone[1], color="#bbbbbb", alpha=0.35, zorder=0)
            else:  # edges only, so the tissue inside the zone stays visible on the CT
                for edge in zone:
                    ax.axvline(edge, color="white", linestyle=":", linewidth=1.2, zorder=3)
            for m in marks:
                if m.end_mm > m.start_mm:
                    ax.axvspan(m.start_mm, m.end_mm, color=m.color, alpha=0.25, zorder=0)
                else:
                    ax.axvline(m.start_mm, color=m.color, linewidth=1.6, zorder=1)
        for m in marks:
            span = f" ({m.start_mm:.0f} to {m.end_mm:.0f} mm)" if m.end_mm > m.start_mm else f" ({m.start_mm:.0f} mm)"
            handles.append(Patch(facecolor=m.color, alpha=0.5, label=m.label + span))
        handles += [plt.Line2D([], [], marker="*", linestyle="", color="white", markeredgecolor="black",
                               markersize=11, label="Analytic peak"),
                    plt.Line2D([], [], color="cyan",
                               label=f"Edge-metric sphere (r = {profile.sphere_radius_mm:.0f} mm)")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
                   fontsize=TICK_SIZE)
        fig.suptitle(title, x=0.012, y=0.985,
                     ha="left", fontsize=13, fontweight="bold")
        paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
        plt.close(fig)
    return paths


__all__ = ["DepthMark", "GROUP_PALETTE", "beam_path_figure", "contrast_figure", "contribution_figure",
           "decile_profile_figure", "group_colors"]
