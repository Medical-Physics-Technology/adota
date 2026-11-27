"""Slide figures that describe a beamlet dataset per anatomical site: what the CTs contain, which energies,
where the spots sit on the isocentre plane and from which gantry angles the fields come.

- :func:`site_hu_histograms_figure`: Hounsfield-unit histogram per site plus an overlay of all sites;
- :func:`energy_by_site_figure`: energy spectrum per site with min / median / max printed from the summary;
- :func:`spot_positions_figure`: beamlets per spot lattice position on the isocentre plane, per site;
- :func:`gantry_polar_figure`: beamlets per gantry sector, one polar panel per site on one radial scale.

Every number drawn or printed comes from the frames passed in (the orchestrator writes them as CSVs beside the
figures); nothing is recomputed from the raw data here.

Reuses the summary style of :mod:`src.figures.ct_vs_error` (``_RC_SUMMARY``, ``SITE_MARKERS``, the ink, muted
and grid colours, the left-aligned bold suptitle), the site colours ``ANATOMY_COLORS`` of
:mod:`src.figures.validation_set`, the polar convention of :func:`src.figures.beamlet_pool.gantry_frame_figure`
(0 deg at the top, clockwise), colorbars through :func:`src.figures.axes_utils.aligned_colorbar` and saving
through :func:`src.figures.axes_utils.save_figure_as_publication_formats`. Each public function is a thin
save-and-close wrapper around a private builder that returns the open figure, so tests can inspect it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats
from src.figures.ct_vs_error import _RC_SUMMARY, GRID, INK, MUTED, SITE_MARKERS
from src.figures.validation_set import ANATOMY_COLORS

SITE_ORDER = ("thoracic", "abdominal", "pelvic")
"""Site order in every figure; sites outside it follow alphabetically."""
HU_REFERENCES = ((-1000.0, "air"), (0.0, "water"), (1000.0, "bone (approx.)"))
SPOT_CMAP = "Blues"
_RC = {**_RC_SUMMARY, "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
       "legend.fontsize": 11}
_TITLE_SIZE = 15
_LOG_BOTTOM, _LOG_FLOOR = 0.5, 0.05


def _sites(*frames: pd.DataFrame) -> List[str]:
    """Sites present in any frame, in :data:`SITE_ORDER` first, then the rest alphabetically."""
    present = set()
    for frame in frames:
        present.update(str(s) for s in frame["site"].dropna().unique())
    return [s for s in SITE_ORDER if s in present] + sorted(present - set(SITE_ORDER))


def _color(site: str) -> str:
    return ANATOMY_COLORS.get(site, MUTED)


def _suptitle(fig, title: str, y: float = 0.98) -> None:
    fig.suptitle(title, x=0.015, y=y, ha="left", va="top", fontsize=_TITLE_SIZE, fontweight="bold", color=INK)


def _save(fig, figure_path, formats: Sequence[str]) -> List[Path]:
    try:
        Path(figure_path).parent.mkdir(parents=True, exist_ok=True)
        return save_figure_as_publication_formats(fig, str(figure_path), formats=tuple(formats))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# HU histograms
# ---------------------------------------------------------------------------------------------------------------

def hu_bin_grid(bin_low_hu: Sequence[float], upper_hu: float) -> np.ndarray:
    """Gapless bin edges from the lower edges in the frame: the regular step is the smallest gap between
    them, and the last upper edge is capped at ``upper_hu`` (the last bin of [3066, 3071] is 5 HU wide)."""
    lows = np.unique(np.round(np.asarray(bin_low_hu, dtype=float), 6))
    if lows.size == 0:
        return np.array([upper_hu - 10.0, upper_hu])
    step = float(np.min(np.diff(lows))) if lows.size > 1 else 10.0
    n = int(round((lows[-1] - lows[0]) / step)) + 1
    grid = lows[0] + step * np.arange(n)
    return np.append(grid, min(grid[-1] + step, max(upper_hu, grid[-1] + 1e-6)))


def _site_counts(hist: pd.DataFrame, site: str, edges: np.ndarray) -> np.ndarray:
    part = hist[hist["site"] == site]
    lows = np.round(part["bin_low_hu"].astype(float), 6)
    counts = part.groupby(lows)["count"].sum()
    return counts.reindex(np.round(edges[:-1], 6), fill_value=0).to_numpy(dtype=float)


def _hu_references(ax, x_range: Tuple[float, float], label: bool) -> None:
    for hu, name in HU_REFERENCES:
        if not x_range[0] <= hu <= x_range[1]:
            continue
        ax.axvline(hu, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        if label:
            ax.text(hu, 1.0, f" {name}", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=10,
                    color=MUTED)


def _build_hu_histograms(hist: pd.DataFrame, *, title: str, x_range: Tuple[float, float]):
    sites = _sites(hist)
    edges = hu_bin_grid(hist["bin_low_hu"], 3071.0)
    counts = {s: _site_counts(hist, s, edges) for s in sites}
    peak = max([float(c.max()) for c in counts.values() if c.size] + [1.0])
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), sharex=True, sharey=True)
        panels = [(ax, [s], s) for ax, s in zip(axes.ravel(), sites[:3])]
        panels.append((axes.ravel()[3], sites, "all sites"))
        for ax in axes.ravel()[len(panels):]:
            ax.set_visible(False)
        for ax, panel_sites, name in panels:
            for site in panel_sites:
                # Zero bins drop to a floor below the visible range, so the step stays one continuous line.
                floored = np.maximum(counts[site], _LOG_FLOOR)
                ax.stairs(floored, edges, color=_color(site), linewidth=1.6 if len(panel_sites) == 1 else 1.4,
                          label=site, baseline=None)
            ax.set_yscale("log")
            ax.set_ylim(_LOG_BOTTOM, peak * 3.0)
            ax.set_xlim(*x_range)
            ax.grid(color=GRID, linewidth=0.8)
            _hu_references(ax, x_range, label=True)
            ax.set_title(name, loc="left", color=_color(name) if name in ANATOMY_COLORS else INK)
        handles = [Line2D([], [], color=_color(s), linewidth=2, label=s) for s in sites]
        panels[-1][0].legend(handles=handles, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
        for ax in axes[1, :]:
            ax.set_xlabel("Hounsfield units [HU]")
        for ax in axes[:, 0]:
            ax.set_ylabel("voxels per bin")
        _suptitle(fig, title)
        fig.subplots_adjust(left=0.08, right=0.985, top=0.88, bottom=0.09, hspace=0.3, wspace=0.06)
    return fig


def site_hu_histograms_figure(hist: pd.DataFrame, *, title: str, figure_path, formats=("png", "pdf"),
                              x_range=(-1024.0, 3071.0)) -> List[Path]:
    """Voxel counts per HU bin, one panel per site plus an overlay of all sites, log y.

    Args:
        hist: Columns ``site, bin_low_hu, count`` (10 HU bins from -1024; the last bin is [3066, 3071]).
            Absent bins count as 0; zero bins drop below the bottom of the log axis.
        x_range: Horizontal limits only; reference lines outside it are not drawn.

    Returns:
        The written paths, one per entry of ``formats``.
    """
    fig = _build_hu_histograms(hist, title=title, x_range=(float(x_range[0]), float(x_range[1])))
    return _save(fig, figure_path, formats)


# ---------------------------------------------------------------------------------------------------------------
# Energy per site
# ---------------------------------------------------------------------------------------------------------------

def energy_summary_text(row: pd.Series) -> str:
    """``min X / median Y / max Z MeV`` from one row of the summary table."""
    return (f"min {float(row['energy_min_mev']):.1f} / median {float(row['energy_median_mev']):.1f} / "
            f"max {float(row['energy_max_mev']):.1f} MeV")


def _horizontal_violin(ax, values: np.ndarray, y: float):
    """One horizontal violin body; ``orientation`` exists from matplotlib 3.10, ``vert`` before it."""
    kwargs = dict(positions=[y], widths=0.8, showextrema=False, points=200)
    try:
        return ax.violinplot(values, orientation="horizontal", **kwargs)["bodies"][0]
    except TypeError:
        return ax.violinplot(values, vert=False, **kwargs)["bodies"][0]


def _build_energy_by_site(metadata: pd.DataFrame, stats: pd.DataFrame, *, title: str):
    sites = _sites(metadata, stats)
    by_site = {str(s): r for s, r in stats.drop_duplicates("site", keep="first").set_index("site").iterrows()}
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12.8, 1.35 * len(sites) + 2.2))
        ypos = {s: len(sites) - 1 - i for i, s in enumerate(sites)}
        for site in sites:
            values = metadata.loc[metadata["site"] == site, "energy_mev"].astype(float).dropna().to_numpy()
            y = ypos[site]
            if values.size > 1 and np.ptp(values) > 0:
                body = _horizontal_violin(ax, values, y)
                body.set_facecolor(_color(site))
                body.set_edgecolor(_color(site))
                body.set_alpha(0.55)
            elif values.size:
                ax.plot(values, [y] * values.size, "|", color=_color(site), markersize=20)
            if site in by_site:
                row = by_site[site]
                ax.plot([row["energy_min_mev"], row["energy_max_mev"]], [y, y], color=INK, linewidth=1.2, zorder=3)
                ax.plot(row["energy_median_mev"], y, SITE_MARKERS.get(site, "o"), color="white",
                        markeredgecolor=INK, markersize=8, zorder=4)
                ax.text(1.02, y, energy_summary_text(row), transform=ax.get_yaxis_transform(), ha="left",
                        va="center", fontsize=11.5, color=INK)
        ax.set_yticks([ypos[s] for s in sites])
        ax.set_yticklabels([f"{s}\n{int((metadata['site'] == s).sum()):,} beamlets" for s in sites], fontsize=12)
        for tick, site in zip(ax.get_yticklabels(), sites):
            tick.set_color(_color(site))
        ax.set_ylim(-0.6, len(sites) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlabel("beamlet energy [MeV]")
        ax.text(1.02, len(sites) - 0.45, "from the summary table", transform=ax.get_yaxis_transform(), ha="left",
                va="bottom", fontsize=10, color=MUTED)
        _suptitle(fig, title)
        fig.subplots_adjust(left=0.13, right=0.66, top=1.0 - 0.7 / fig.get_figheight(),
                            bottom=0.75 / fig.get_figheight())
    return fig


def energy_by_site_figure(metadata: pd.DataFrame, stats: pd.DataFrame, *, title: str, figure_path,
                          formats=("png", "pdf")) -> List[Path]:
    """Energy spectrum per site (horizontal violin in the site colour) with the summary printed beside it.

    Args:
        metadata: One row per beamlet with at least ``site, energy_mev``.
        stats: ``site, energy_min_mev, energy_median_mev, energy_max_mev``; the printed numbers, the range bar
            and the median marker come from here, not from ``metadata``.
    """
    return _save(_build_energy_by_site(metadata, stats, title=title), figure_path, formats)


# ---------------------------------------------------------------------------------------------------------------
# Spot positions on the isocentre plane
# ---------------------------------------------------------------------------------------------------------------

def lattice_edges(positions: Sequence[float]) -> np.ndarray:
    """Cell edges of a lattice: midpoints between the sorted unique positions, extended by half the first and
    the last pitch at the two ends. A single position gets a 1 mm cell."""
    u = np.unique(np.round(np.asarray(positions, dtype=float), 3))
    if u.size == 1:
        return np.array([u[0] - 0.5, u[0] + 0.5])
    mid = (u[:-1] + u[1:]) / 2.0
    return np.concatenate([[u[0] - (u[1] - u[0]) / 2.0], mid, [u[-1] + (u[-1] - u[-2]) / 2.0]])


def spot_grid(cells: pd.DataFrame, site: str, value: str = "beamlets") -> Tuple[np.ndarray, np.ndarray,
                                                                                   np.ndarray]:
    """``(x_edges, y_edges, grid)`` of one site; ``grid[iy, ix]`` sums ``value`` per lattice position and is 0
    where a position inside the range is unoccupied."""
    part = cells[cells["site"] == site]
    x = np.round(part["spot_x_mm"].astype(float).to_numpy(), 3)
    y = np.round(part["spot_y_mm"].astype(float).to_numpy(), 3)
    xs, ys = np.unique(x), np.unique(y)
    grid = np.zeros((ys.size, xs.size))
    np.add.at(grid, (np.searchsorted(ys, y), np.searchsorted(xs, x)), part[value].astype(float).to_numpy())
    return lattice_edges(xs), lattice_edges(ys), grid


def _build_spot_positions(cells: pd.DataFrame, *, title: str, value_label: str):
    sites = _sites(cells)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(sites), figsize=(max(4.3 * len(sites), 6.0), 5.0), squeeze=False)
        for ax, site in zip(axes[0], sites):
            x_edges, y_edges, grid = spot_grid(cells, site)
            mesh = ax.pcolormesh(x_edges, y_edges, grid, cmap=SPOT_CMAP, vmin=0.0,
                                 vmax=max(float(grid.max()), 1.0), edgecolors="white", linewidth=0.3)
            ax.set_aspect("equal")
            ax.set_xlim(x_edges[0], x_edges[-1])
            ax.set_ylim(y_edges[0], y_edges[-1])
            ax.set_title(f"{site}: {int(round(grid.sum())):,} {value_label}", loc="left", color=_color(site))
            ax.set_xlabel("spot X in the isocentre plane [mm]")
            ax.spines[["top", "right"]].set_visible(True)
            bar = aligned_colorbar(fig, mesh, ax, value_label, label_coords=(5.0, 0.5), label_fontsize=11,
                                   tick_fontsize=10, size="6%")
            bar.outline.set_edgecolor(MUTED)
        axes[0, 0].set_ylabel("spot Y [mm]")
        _suptitle(fig, title)
        fig.subplots_adjust(left=0.07, right=0.93, top=0.87, bottom=0.13, wspace=0.58)
    return fig


def spot_positions_figure(cells: pd.DataFrame, *, title: str, figure_path, formats=("png", "pdf"),
                          value_label: str = "beamlets") -> List[Path]:
    """Beamlets per spot lattice position on the isocentre plane, one panel per site.

    Args:
        cells: ``site, spot_x_mm, spot_y_mm, beamlets``, one row per occupied lattice position per site.
            Each site's cells span its own [min, max] plus half a pitch (see :func:`lattice_edges`); unoccupied
            positions inside that range are drawn as 0.
        value_label: Colorbar label and title unit.
    """
    return _save(_build_spot_positions(cells, title=title, value_label=value_label), figure_path, formats)


# ---------------------------------------------------------------------------------------------------------------
# Gantry angles
# ---------------------------------------------------------------------------------------------------------------

def _build_gantry_polar(sectors: pd.DataFrame, *, title: str, bin_deg: float, r_label_deg: float = 22.5):
    sites = _sites(sectors)
    rmax = max(float(sectors["beamlets"].astype(float).max()) if len(sectors) else 0.0, 1.0) * 1.08
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, len(sites), figsize=(max(4.2 * len(sites), 6.0), 5.2), squeeze=False,
                                 subplot_kw={"projection": "polar"})
        for ax, site in zip(axes[0], sites):
            part = sectors[sectors["site"] == site]
            per_bin = part.groupby(part["gantry_bin_low_deg"].astype(float))["beamlets"].sum()
            ax.bar(np.deg2rad(per_bin.index.to_numpy()), per_bin.to_numpy(dtype=float), width=np.deg2rad(bin_deg),
                   align="edge", color=_color(site), edgecolor="white", linewidth=0.6, zorder=3)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0.0, rmax)
            ax.set_rlabel_position(r_label_deg)
            ax.yaxis.set_major_locator(MaxNLocator(3, prune="lower"))
            ax.yaxis.set_zorder(4)    # radial ticks and rings above the bars, so they stay readable
            ax.set_thetagrids(np.arange(0, 360, 45), [f"{a} deg" for a in range(0, 360, 45)])
            ax.tick_params(axis="x", pad=2, labelsize=10.5, colors=INK)
            ax.tick_params(axis="y", labelsize=10, colors=MUTED)
            ax.grid(color=GRID, linewidth=0.8)
            ax.spines["polar"].set_edgecolor(MUTED)
            ax.set_title(f"{site}: {int(per_bin.sum()):,} beamlets", loc="left", color=_color(site), pad=22)
            for label in ax.get_yticklabels():
                label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5})
        _suptitle(fig, title)
        fig.text(0.015, 0.985 - 0.4 / fig.get_figheight(), f"radius: beamlets per {bin_deg:g} deg gantry sector,"
                 " one scale for every site", ha="left", va="top", fontsize=11, color=MUTED)
        fig.subplots_adjust(left=0.04, right=0.97, top=0.76, bottom=0.06, wspace=0.45)
    return fig


def gantry_polar_figure(sectors: pd.DataFrame, *, title: str, figure_path, bin_deg: float = 15.0,
                        formats=("png", "pdf")) -> List[Path]:
    """Beamlets per gantry sector, one polar panel per site, 0 deg at the top and clockwise.

    Args:
        sectors: ``site, gantry_bin_low_deg, beamlets`` (the gantry histogram weighted by beamlets per field).
            All panels share one radial limit, the largest sector of any site, so the sites compare.
    """
    return _save(_build_gantry_polar(sectors, title=title, bin_deg=float(bin_deg)), figure_path, formats)


__all__ = [
    "SITE_ORDER", "energy_by_site_figure", "energy_summary_text", "gantry_polar_figure", "hu_bin_grid",
    "lattice_edges", "site_hu_histograms_figure", "spot_grid", "spot_positions_figure",
]
