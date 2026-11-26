"""Grids relating input CT features to the per-beamlet dose error of one engine, per anatomical site.

Two views of the same question (does an input CT feature go with a worse dose?), one row per feature and one
column per error measure:

- :func:`feature_vs_error_grid`: x = error, y = feature; each site's line is the median (and quartile band) of
  the feature in bins of the error;
- :func:`error_vs_feature_grid`: x = feature, y = error; each site's line is the median (and quartile band) of
  the error in feature deciles.

Both draw a grey log-density hexbin of every beamlet underneath, so the site lines can be read against the
whole population. Axis limits come from quantiles of the data (``clip_quantiles``); beamlets outside are not
drawn but stay in every table. Errors listed in ``reversed_errors`` (higher is better, such as the gamma pass
rate) are drawn with the axis inverted so that worse is always to the right (x) or up (y). Panel titles print the
Spearman statistics from ``stats`` (already oriented so positive means worse).

For slides, two summary figures ask one question each with at most one number per panel:
:func:`correlation_overview` (which features go with a worse error, per engine) and
:func:`feature_shape_grid` (how the error grows with a few features, per site). The grids above are the
detailed appendix.

Reuses ``ANATOMY_COLORS`` of :mod:`src.figures.validation_set`, ``style_axis`` of :mod:`src.figures.al_curves`,
and saves through :func:`src.figures.axes_utils.save_figure_as_publication_formats`.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.figures.al_curves import style_axis
from src.figures.axes_utils import save_figure_as_publication_formats
from src.figures.validation_set import ANATOMY_COLORS

BAND_ALPHA = 0.18
HEX_ALPHA = 0.55
TITLE_SIZE = 7.5
_RC = {"font.size": 9, "axes.titlesize": TITLE_SIZE, "axes.labelsize": 9, "xtick.labelsize": 8,
       "ytick.labelsize": 8, "legend.fontsize": 9}


def _is_reversed(error: str, reversed_errors: Sequence[str]) -> bool:
    """True when ``error`` is higher-is-better, so its axis is inverted to put the worse values at the end."""
    return error in set(reversed_errors)


def _fmt(row: pd.Series) -> str:
    return f"{row['spearman']:+.2f} [{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"


def _title_lines(stats: pd.DataFrame, error: str, feature: str, sites: Sequence[str]) -> List[str]:
    """Two title lines for one panel: pooled and energy-controlled rho, then one item per site.

    A scope absent from ``stats`` or without a finite value (e.g. energy controlled for itself) is
    skipped. A line with no items is omitted.
    """
    sub = stats[(stats["error"] == error) & (stats["feature"] == feature)]
    by_scope = {str(r["scope"]): r for _, r in sub.iterrows() if np.isfinite(float(r["spearman"]))}
    first = []
    if "pooled" in by_scope:
        first.append(f"pooled rho {_fmt(by_scope['pooled'])}")
    if "energy_partial" in by_scope:
        first.append(f"energy-controlled {_fmt(by_scope['energy_partial'])}")
    second = [f"{site[:4]} {_fmt(by_scope[site])}" for site in sites if site in by_scope]
    return [line for line in ("; ".join(first), "  ".join(second)) if line]


def _axis_label(label: str, reversed_axis: bool, on_x: bool) -> str:
    """An error axis label with its reading direction when the axis is inverted."""
    if not reversed_axis:
        return label
    return f"{label}\n(axis reversed: worse to the right)" if on_x else f"{label}\n(axis reversed: worse upward)"


def _limits(values: pd.Series, clip_quantiles: Tuple[float, float]) -> Tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(v, clip_quantiles[0]), np.quantile(v, clip_quantiles[1])
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    return float(lo), float(hi)


def _site_lines(ax, table: pd.DataFrame, *, error: str, feature: str, sites: Sequence[str], x: str, y: str,
                lo: str, hi: str) -> None:
    sub = table[(table["error"] == error) & (table["feature"] == feature)]
    for site in sites:
        rows = sub[sub["site"] == site].sort_values("bin")
        if rows.empty:
            continue
        color = ANATOMY_COLORS[site]
        ax.fill_between(rows[x], rows[lo], rows[hi], color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
        ax.plot(rows[x], rows[y], "-o", color=color, markersize=3.5, linewidth=1.6, zorder=3)


def _legend_axis(ax, sites: Sequence[str], caption: str, width_chars: int) -> None:
    ax.axis("off")
    handles = [Line2D([0], [0], color=ANATOMY_COLORS[s], marker="o", markersize=4, linewidth=1.6, label=s)
               for s in sites]
    handles.append(Patch(facecolor="#9a9a9a", edgecolor="none", label="all beamlets (log density)"))
    ax.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    if caption:
        ax.text(0.0, 0.45, textwrap.fill(caption, width_chars), transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#333333")


def _grid(points: pd.DataFrame, table: pd.DataFrame, stats: pd.DataFrame, *, features: Sequence[str],
          errors: Sequence[str], feature_labels: Mapping[str, str], error_labels: Mapping[str, str],
          reversed_errors: Sequence[str], sites: Sequence[str], title: str, figure_path: str, caption: str,
          gridsize: int, clip_quantiles: Tuple[float, float], formats: Sequence[str],
          error_on_x: bool) -> List[Path]:
    n_rows, n_cols = len(features), len(errors)
    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(13, 2.5 * n_rows + 1.6), constrained_layout=True)
        gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[1.0] * n_rows + [0.38 if caption else 0.14])
        for i, feature in enumerate(features):
            for j, error in enumerate(errors):
                ax = fig.add_subplot(gs[i, j])
                e_lim = _limits(points[error], clip_quantiles)
                f_lim = _limits(points[feature], clip_quantiles)
                x_col, y_col = (error, feature) if error_on_x else (feature, error)
                x_lim, y_lim = (e_lim, f_lim) if error_on_x else (f_lim, e_lim)
                ax.hexbin(points[x_col], points[y_col], gridsize=gridsize, bins="log", cmap="Greys", mincnt=1,
                          alpha=HEX_ALPHA, extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), linewidths=0, zorder=1)
                if error_on_x:
                    _site_lines(ax, table, error=error, feature=feature, sites=sites, x="error_median",
                                y="feature_median", lo="feature_q25", hi="feature_q75")
                else:
                    _site_lines(ax, table, error=error, feature=feature, sites=sites, x="feature_median",
                                y="error_median", lo="error_q25", hi="error_q75")
                ax.set_xlim(*x_lim)
                ax.set_ylim(*y_lim)
                if _is_reversed(error, reversed_errors):
                    (ax.invert_xaxis if error_on_x else ax.invert_yaxis)()
                err_label = _axis_label(error_labels[error], _is_reversed(error, reversed_errors), error_on_x)
                x_label = err_label if error_on_x else feature_labels[feature]
                y_label = feature_labels[feature] if error_on_x else err_label
                style_axis(ax, xlabel=x_label if (i == n_rows - 1 or not error_on_x) else "",
                           ylabel=y_label if j == 0 or error_on_x else "",
                           title="\n".join(_title_lines(stats, error, feature, sites)))
                ax.set_title("\n".join(_title_lines(stats, error, feature, sites)), loc="left",
                             fontsize=TITLE_SIZE, fontweight="normal")
                if i == 0:
                    ax.annotate(err_label.replace("\n", " "), xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 26),
                                textcoords="offset points", ha="center", va="bottom", fontsize=10.5,
                                fontweight="bold")
        _legend_axis(fig.add_subplot(gs[n_rows, :]), sites, caption, 190)
        fig.suptitle(title, x=0.012, y=1.04, ha="left", fontsize=13, fontweight="bold")
        paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
        plt.close(fig)
    return paths


def feature_vs_error_grid(points: pd.DataFrame, ebins: pd.DataFrame, stats: pd.DataFrame, *,
                          features: Sequence[str], errors: Sequence[str],
                          feature_labels: Mapping[str, str], error_labels: Mapping[str, str],
                          reversed_errors: Sequence[str], sites: Sequence[str], title: str, figure_path: str,
                          caption: str = "", gridsize: int = 40,
                          clip_quantiles: Tuple[float, float] = (0.005, 0.995),
                          formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """x = error, y = feature: per-site median and quartile band of the feature in bins of the error.

    ``ebins`` has columns ``error, feature, site, bin, error_median, n, feature_median, feature_q25,
    feature_q75``. Returns the written paths, one per entry of ``formats``.
    """
    return _grid(points, ebins, stats, features=features, errors=errors, feature_labels=feature_labels,
                 error_labels=error_labels, reversed_errors=reversed_errors, sites=sites, title=title,
                 figure_path=figure_path, caption=caption, gridsize=gridsize, clip_quantiles=clip_quantiles,
                 formats=formats, error_on_x=True)


def error_vs_feature_grid(points: pd.DataFrame, fbins: pd.DataFrame, stats: pd.DataFrame, *,
                          features: Sequence[str], errors: Sequence[str],
                          feature_labels: Mapping[str, str], error_labels: Mapping[str, str],
                          reversed_errors: Sequence[str], sites: Sequence[str], title: str, figure_path: str,
                          caption: str = "", gridsize: int = 40,
                          clip_quantiles: Tuple[float, float] = (0.005, 0.995),
                          formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """x = feature, y = error: per-site median and quartile band of the error in feature deciles.

    ``fbins`` has columns ``error, feature, site, bin, feature_median, n, error_median, error_q25, error_q75``.
    Returns the written paths, one per entry of ``formats``.
    """
    return _grid(points, fbins, stats, features=features, errors=errors, feature_labels=feature_labels,
                 error_labels=error_labels, reversed_errors=reversed_errors, sites=sites, title=title,
                 figure_path=figure_path, caption=caption, gridsize=gridsize, clip_quantiles=clip_quantiles,
                 formats=formats, error_on_x=False)


# ---------------------------------------------------------------------------------------------------------------
# Summary figures for slides: one question per figure, at most one number per panel.
# ---------------------------------------------------------------------------------------------------------------

SITE_MARKERS = {"thoracic": "o", "abdominal": "s", "pelvic": "^"}
"""Second encoding of the site, so identity never rests on colour alone."""
INK, MUTED, GRID, REF_BAND = "#222222", "#6b6b6b", "#e6e6e6", "#f4f4f4"
_RC_SUMMARY = {"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
               "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
               "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK}


def _site_handles(sites: Sequence[str], line: bool) -> List[Line2D]:
    return [Line2D([], [], color=ANATOMY_COLORS[s], marker=SITE_MARKERS.get(s, "o"),
                   linestyle="-" if line else "", linewidth=2, label=s) for s in sites]


def correlation_overview(stats: Mapping[str, pd.DataFrame], *, features: Sequence[str],
                         reference_features: Sequence[str], feature_names: Mapping[str, str], error: str,
                         sites: Sequence[str], xlabel: str, title: str, figure_path: str,
                         xlim: Tuple[float, float] = (-0.75, 0.9),
                         formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """Dot plot of each feature's Spearman correlation with one error, one panel per engine.

    Black dot and bar: the pooled correlation and its 95% interval; small coloured markers: each site. The
    ``reference_features`` (depth, energy) sit at the bottom on a grey band.

    Args:
        stats: ``{engine label: correlations}`` with columns ``error, feature, scope, spearman, ci_low, ci_high``
            (``scope`` ``pooled`` and each site), oriented so that positive means a worse error.
        features: Rows from the top, then ``reference_features``.
    """
    order = list(features) + list(reference_features)
    ypos = {f: len(order) - 1 - i for i, f in enumerate(order)}
    with plt.rc_context(_RC_SUMMARY):
        fig, axes = plt.subplots(1, len(stats), figsize=(5.5 * len(stats) + 2.0, 0.5 * len(order) + 2.4),
                                 sharey=True, squeeze=False)
        for ax, (engine, table) in zip(axes[0], stats.items()):
            s = table[table["error"] == error].set_index(["feature", "scope"])
            ax.axhspan(-0.5, len(reference_features) - 0.5, color=REF_BAND, zorder=0)
            ax.axvline(0.0, color=MUTED, linewidth=1, zorder=1)
            for f in order:
                p = s.loc[(f, "pooled")]
                ax.plot([p["ci_low"], p["ci_high"]], [ypos[f]] * 2, color=INK, linewidth=2,
                        solid_capstyle="round", zorder=3)
                ax.plot(p["spearman"], ypos[f], "o", color=INK, markersize=7, zorder=4)
                for k, site in enumerate(sites):
                    if (f, site) in s.index:
                        ax.plot(s.loc[(f, site), "spearman"], ypos[f] + 0.22 * (1 - k), SITE_MARKERS.get(site, "o"),
                                color=ANATOMY_COLORS[site], markersize=5, markeredgewidth=0, zorder=2)
            ax.set_xlim(*xlim)
            ax.set_title(engine, loc="left")
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_xlabel(xlabel)
            ax.tick_params(axis="y", length=0)
        axes[0, 0].set_yticks([ypos[f] for f in order])
        axes[0, 0].set_yticklabels([feature_names.get(f, f) for f in order])
        axes[0, 0].text(xlim[0] + 0.01, len(reference_features) - 0.62, "reference", fontsize=9, color=MUTED,
                        va="top")
        handles = [Line2D([], [], color=INK, marker="o", linewidth=2, label="all sites (95% interval)")]
        handles += _site_handles(sites, line=False)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.55, 0.0))
        fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
        fig.subplots_adjust(left=0.26, right=0.98, top=0.88, bottom=0.2, wspace=0.08)
        paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
        plt.close(fig)
    return paths


def feature_shape_grid(curves: Mapping[str, pd.DataFrame], rho: Mapping[str, Mapping[str, float]], *,
                       features: Sequence[str], feature_names: Mapping[str, Tuple[str, str]], sites: Sequence[str],
                       ylabel: str, title: str, figure_path: str, min_bins: int = 3,
                       formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """Small multiples of an error against a few features: rows engines, columns features, one line per site.

    Args:
        curves: ``{engine label: table}`` with columns ``feature, site, bin, feature_median, value``: the
            error summary (``value``) per feature bin of one site.
        rho: ``{engine label: {feature: pooled Spearman}}``, printed once per panel.
        feature_names: ``{feature: (plain name, unit)}``.
        min_bins: A site with fewer bins for a feature (a feature that is almost always zero there) is not drawn.
    """
    engines = list(curves)
    with plt.rc_context(_RC_SUMMARY):
        fig, axes = plt.subplots(len(engines), len(features), figsize=(3.3 * len(features) + 0.6,
                                                                        3.2 * len(engines)),
                                 sharey="row", squeeze=False)
        for r, engine in enumerate(engines):
            table = curves[engine]
            for c, f in enumerate(features):
                ax = axes[r, c]
                for site in sites:
                    b = table[(table["feature"] == f) & (table["site"] == site)].sort_values("bin")
                    if len(b) < min_bins:
                        continue
                    ax.plot(b["feature_median"], b["value"], color=ANATOMY_COLORS[site], linewidth=2,
                            marker=SITE_MARKERS.get(site, "o"), markersize=5, markeredgewidth=0)
                ax.grid(color=GRID, linewidth=0.8)
                if f in rho.get(engine, {}):
                    ax.text(0.03, 0.95, f"rho = {rho[engine][f]:+.2f}", transform=ax.transAxes, va="top",
                            fontsize=10, color=INK)
                name, unit = feature_names.get(f, (f, ""))
                if r == 0:
                    ax.set_title(name, loc="left", fontsize=11)
                ax.set_xlabel(f"[{unit}]" if unit else "", color=MUTED, fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{engine}\n{ylabel}")
        fig.legend(handles=_site_handles(sites, line=True), loc="lower center", ncol=len(sites), frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.85, bottom=0.17, hspace=0.55, wspace=0.12)
        paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
        plt.close(fig)
    return paths
