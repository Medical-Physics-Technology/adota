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

    A scope absent from ``stats`` is skipped. A line with no items is omitted.
    """
    sub = stats[(stats["error"] == error) & (stats["feature"] == feature)]
    by_scope = {str(r["scope"]): r for _, r in sub.iterrows()}
    first = []
    if "pooled" in by_scope:
        first.append(f"pooled rho {_fmt(by_scope['pooled'])}")
    if "energy_partial" in by_scope:
        first.append(f"energy-controlled {_fmt(by_scope['energy_partial'])}")
    second = [f"{site[:4]} {_fmt(by_scope[site])}" for site in sites if site in by_scope]
    return [line for line in ("; ".join(first), "  ".join(second)) if line]


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
                x_label = error_labels[error] if error_on_x else feature_labels[feature]
                y_label = feature_labels[feature] if error_on_x else error_labels[error]
                style_axis(ax, xlabel=x_label if (i == n_rows - 1 or not error_on_x) else "",
                           ylabel=y_label if j == 0 or error_on_x else "",
                           title="\n".join(_title_lines(stats, error, feature, sites)))
                ax.set_title("\n".join(_title_lines(stats, error, feature, sites)), loc="left",
                             fontsize=TITLE_SIZE, fontweight="normal")
                if i == 0:
                    ax.annotate(error_labels[error], xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 26),
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
