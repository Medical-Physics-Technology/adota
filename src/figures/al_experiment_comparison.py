"""Figures comparing two retrospective active-learning experiments run by run
(``scripts/al_compare_experiments.py``).

Experiment A and experiment B share splits, cycle-0 checkpoint and selections
and differ in one training setting, so every figure puts A and B side by side:

- :func:`schedule_and_loss_figure`: the learning rate and the validation loss
  (median across runs, min-max band) against the cumulative epoch;
- :func:`paired_quality_figure`: per metric, the mean across runs per cycle (top)
  and the per-pair difference B minus A per cycle (bottom);
- :func:`strategy_endpoint_figure`: every run at one cycle, grouped by strategy,
  A and B next to each other, one marker per seed;
- :func:`stability_figure`: diverged runs and the boundary validation loss over
  the cycle's best, per cycle.

The panel style (:func:`style_axis`, :func:`draw_cycle_rules`) and the palette
come from :mod:`src.figures.al_curves`, at a larger type size for slides, with
the legends below the panels so they never cover a point. Saving goes through
:func:`save_figure_as_publication_formats`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter

from src.figures.al_curves import COLORS, draw_cycle_rules, style_axis
from src.figures.axes_utils import save_figure_as_publication_formats

EXPERIMENT_COLORS = (COLORS[0], COLORS[3])
STRATEGY_COLORS = {"random": "#7f7f7f", "score_topk": COLORS[1], "score_topk_mixed": COLORS[2]}
STRATEGY_NAMES = {"random": "random", "score_topk": "score top-k",
                  "score_topk_mixed": "score top-k mixed"}
SEED_MARKERS = ("o", "s", "^", "D", "v", "P")
SLIDE_RC = {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 11,
            "ytick.labelsize": 11, "legend.fontsize": 11}

Panel = Tuple[str, str, str, float]
"""(column, panel title, axis label, factor): the column of the cycle table, what
the panel shows, its unit, and the factor to that unit (100 for a fraction in percent)."""


def gamma_text(label: str) -> str:
    """``2pct_2mm_cutoff10pct`` -> ``2%/2mm, 10% dose cutoff``; anything else unchanged."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)pct_(\d+(?:\.\d+)?)mm_cutoff(\d+(?:\.\d+)?)pct", label)
    if not match:
        return label.replace("_", " ")
    dose, dta, cutoff = match.groups()
    return f"{dose}%/{dta}mm, {cutoff}% dose cutoff"


def _strategy_color(strategy: str) -> str:
    return STRATEGY_COLORS.get(strategy, COLORS[4])


def _experiment_handles(experiments: Sequence[str], legend: Dict[str, str], **style) -> List:
    return [Line2D([], [], color=c, label=legend[n], **style)
            for c, n in zip(EXPERIMENT_COLORS, experiments)]


def _finish(fig, figure_path: str, handles: Sequence, caption: Optional[str],
            ncol: Optional[int] = None) -> List[Path]:
    fig.canvas.draw()
    below = min(ax.get_tightbbox().transformed(fig.transFigure.inverted()).y0 for ax in fig.axes)
    if handles:
        box = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, below - 0.005),
                         ncol=ncol or len(handles), frameon=False)
        fig.canvas.draw()
        below = box.get_window_extent().transformed(fig.transFigure.inverted()).y0
    if caption:
        fig.text(0.5, below - 0.005, caption, ha="center", va="top", fontsize=10, color="#444444")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def schedule_and_loss_figure(curves: pd.DataFrame, boundaries: pd.DataFrame,
                             experiments: Sequence[str], legend: Dict[str, str],
                             figure_path: str, title: str, caption: Optional[str] = None,
                             shared_until: Optional[int] = None) -> List[Path]:
    """Learning rate (top) and validation loss (bottom) against the cumulative epoch.

    Args:
        curves: :func:`src.active_learning.retrospective.experiment_compare.loss_curves`
            of both experiments stacked (``experiment``, ``cumulative_epoch``, ``lr``,
            ``val_loss``).
        boundaries: ``cycle``, ``cumulative_epoch``, ``n_train`` at every cycle end.
        experiments: The two experiment names, A first.
        legend: Experiment name to its legend entry.
        shared_until: Last cumulative epoch both experiments share (the reused
            cycle 0); shaded grey, and the loss axis is scaled to what follows it.
    """
    with plt.rc_context(SLIDE_RC):
        fig, (ax_lr, ax_val) = plt.subplots(2, 1, figsize=(13, 7.8), dpi=300, sharex=True,
                                            gridspec_kw={"height_ratios": [1, 2]},
                                            layout="constrained")
        for color, name in zip(EXPERIMENT_COLORS, experiments):
            part = curves[curves["experiment"] == name]
            lr = part.groupby("cumulative_epoch")["lr"].median()
            ax_lr.plot(lr.index, lr.values, color=color, linewidth=1.8)
            val = part.groupby("cumulative_epoch")["val_loss"].agg(["median", "min", "max"])
            ax_val.fill_between(val.index, val["min"], val["max"], color=color, alpha=0.18,
                                linewidth=0)
            ax_val.plot(val.index, val["median"], color=color, linewidth=1.6)
        style_axis(ax_lr, xlabel="", ylabel="Learning rate", title="Learning-rate schedule")
        style_axis(ax_val, xlabel="Cumulative epoch", ylabel="Validation loss (log)",
                   title="Validation loss on the full validation set: median across runs, band = min to max")
        lrs = np.unique(np.round(curves["lr"].dropna().to_numpy(dtype=float), 12))
        ax_lr.set_yscale("log")
        ticks = [v for v in (5e-5, 1e-4, 2e-4, 5e-4, 1e-3) if lrs.min() * 0.9 <= v <= lrs.max() * 1.1]
        ax_lr.yaxis.set_major_locator(FixedLocator(ticks))
        ax_lr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-")))
        ax_lr.yaxis.set_minor_formatter(NullFormatter())
        ax_lr.set_ylim(lrs.min() * 0.7, lrs.max() * 2.2)
        ax_val.set_yscale("log")
        after = curves[curves["cumulative_epoch"] > (shared_until if shared_until is not None else -1)]
        if not after.empty:
            ax_val.set_ylim(after["val_loss"].min() * 0.8, after["val_loss"].max() * 1.3)
        for ax in (ax_lr, ax_val):
            ax.set_xlim(left=0)
            if shared_until is not None:
                ax.axvspan(0, shared_until, color="#e6e6e6", zorder=0, linewidth=0)
        if shared_until is not None:
            ax_val.text(shared_until / 2, 0.97, "cycle 0\n(shared)", transform=ax_val.get_xaxis_transform(),
                        ha="center", va="top", fontsize=10, color="#555555")
        draw_cycle_rules(ax_lr, boundaries)
        draw_cycle_rules(ax_val, boundaries, label=False)
        fig.suptitle(title, fontsize=15, fontweight="bold")
        return _finish(fig, figure_path, _experiment_handles(experiments, legend, linewidth=2.2),
                       caption)


def paired_quality_figure(table: pd.DataFrame, paired: pd.DataFrame, panels: Sequence[Panel],
                          experiments: Sequence[str], legend: Dict[str, str], figure_path: str,
                          title: str, caption: Optional[str] = None,
                          lower_is_better: Sequence[str] = ()) -> List[Path]:
    """Top row: per panel metric, mean across runs (min-max band) per cycle, one
    line per experiment. Bottom row: B minus A per pair and cycle, a dot per pair
    coloured by strategy, the mean as a bar, and above each cycle on how many
    pairs B was the better one.

    Args:
        table: Both experiments' :func:`cycle_table` stacked, cycles >= 1.
        paired: :func:`paired_deltas` of the same metrics.
    """
    cycles = sorted(int(c) for c in paired["cycle"].unique())
    sizes = table.drop_duplicates("cycle").set_index("cycle")["n_train"]
    ticklabels = [f"{c}\n({int(sizes[c]):,})" for c in cycles]
    with plt.rc_context(SLIDE_RC):
        n = len(panels)
        fig, axes = plt.subplots(2, n, figsize=(4.6 * n + 0.6, 8.6), dpi=300, squeeze=False,
                                 gridspec_kw={"height_ratios": [1.2, 1]}, layout="constrained")
        rng = np.random.default_rng(0)
        for j, (column, panel_title, unit, factor) in enumerate(panels):
            top, bottom = axes[0, j], axes[1, j]
            for color, name in zip(EXPERIMENT_COLORS, experiments):
                part = table[(table["experiment"] == name) & table["cycle"].isin(cycles)] \
                    .groupby("cycle")[column].agg(["mean", "min", "max"]).dropna()
                top.fill_between(part.index, part["min"] * factor, part["max"] * factor,
                                 color=color, alpha=0.18, linewidth=0)
                top.plot(part.index, part["mean"] * factor, color=color, linewidth=1.8,
                         marker="o", markersize=5)
            style_axis(top, xlabel="Cycle (training records)", ylabel=unit, title=panel_title)
            top.set_xticks(cycles, ticklabels)
            delta = f"delta_{column}"
            for cycle in cycles:
                part = paired[paired["cycle"] == cycle]
                x = cycle + rng.uniform(-0.18, 0.18, len(part))
                bottom.scatter(x, part[delta] * factor, s=30, zorder=4, edgecolor="white",
                               linewidth=0.6, c=[_strategy_color(s) for s in part["strategy"]])
                bottom.hlines(part[delta].mean() * factor, cycle - 0.3, cycle + 0.3, color="#222222",
                              linewidth=2.2, zorder=5)
                better = (part[delta] < 0) if column in lower_is_better else (part[delta] > 0)
                bottom.annotate(f"{int(better.sum())}/{len(part)}", (cycle, 1.0),
                                xycoords=("data", "axes fraction"), ha="center", va="bottom",
                                fontsize=10, color="#444444")
            bottom.axhline(0.0, color="#444444", linewidth=0.9)
            style_axis(bottom, xlabel="Cycle", ylabel=f"{experiments[1]} - {experiments[0]} [pts]",
                       title="")
            bottom.set_xticks(cycles)
        handles = _experiment_handles(experiments, legend, linewidth=2.2, marker="o")
        handles += [Line2D([], [], marker="o", linestyle="", color=_strategy_color(s),
                           label=f"pair: {STRATEGY_NAMES.get(s, s)}")
                    for s in dict.fromkeys(paired["strategy"])]
        handles.append(Line2D([], [], color="#222222", linewidth=2.2, label="mean of pairs"))
        fig.suptitle(title, fontsize=15, fontweight="bold")
        return _finish(fig, figure_path, handles, caption, ncol=3)


def strategy_endpoint_figure(table: pd.DataFrame, cycle: int, panels: Sequence[Panel],
                             experiments: Sequence[str], legend: Dict[str, str],
                             figure_path: str, title: str,
                             caption: Optional[str] = None) -> List[Path]:
    """Every run at ``cycle``: x groups the strategies, A left and B right inside
    a group, one marker per seed and a bar at the mean. The panels share the y axis."""
    part = table[table["cycle"] == cycle]
    strategies = [s for s in STRATEGY_NAMES if s in set(part["strategy"])] + \
        sorted(set(part["strategy"]) - set(STRATEGY_NAMES))
    seeds = sorted(part["seed"].dropna().unique())
    with plt.rc_context(SLIDE_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.8), dpi=300,
                                 squeeze=False, sharey=True, layout="constrained")
        for ax, (column, panel_title, unit, factor) in zip(axes[0], panels):
            for i, strategy in enumerate(strategies):
                for k, (color, name) in enumerate(zip(EXPERIMENT_COLORS, experiments)):
                    runs = part[(part["strategy"] == strategy) & (part["experiment"] == name)]
                    x0 = i + (k - 0.5) * 0.36
                    for s, seed in enumerate(seeds):
                        value = runs.loc[runs["seed"] == seed, column]
                        if value.empty:
                            continue
                        ax.scatter(x0 + (s - (len(seeds) - 1) / 2) * 0.06, value.iloc[0] * factor,
                                   marker=SEED_MARKERS[s % len(SEED_MARKERS)], s=50, color=color,
                                   edgecolor="white", linewidth=0.6, zorder=4)
                    ax.hlines(runs[column].mean() * factor, x0 - 0.14, x0 + 0.14, color=color,
                              linewidth=2.4, zorder=5)
            ax.set_xticks(range(len(strategies)), [STRATEGY_NAMES.get(s, s) for s in strategies])
            style_axis(ax, xlabel="", ylabel=unit if ax is axes[0, 0] else "", title=panel_title)
            ax.set_xlim(-0.6, len(strategies) - 0.4)
        handles = _experiment_handles(experiments, legend, linewidth=2.4)
        handles += [Line2D([], [], marker=SEED_MARKERS[s % len(SEED_MARKERS)], linestyle="",
                           color="#555555", label=f"seed {int(seed)}") for s, seed in enumerate(seeds)]
        fig.suptitle(title, fontsize=15, fontweight="bold")
        return _finish(fig, figure_path, handles, caption)


def stability_figure(table: pd.DataFrame, experiments: Sequence[str], legend: Dict[str, str],
                     figure_path: str, title: str, caption: Optional[str] = None) -> List[Path]:
    """Left: diverged runs per cycle, one bar per experiment, the count written on
    it. Right: the boundary validation loss over the cycle's minimum, a dot per
    run, a bar at the median."""
    scored = table[table["cycle"] >= 1]
    cycles = sorted(scored["cycle"].unique())
    n_runs = int(scored.groupby("experiment")["label"].nunique().max())
    with plt.rc_context(SLIDE_RC):
        fig, (ax_div, ax_noise) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=300,
                                               layout="constrained")
        rng = np.random.default_rng(0)
        width = 0.36
        totals = {}
        for k, (color, name) in enumerate(zip(EXPERIMENT_COLORS, experiments)):
            part = scored[scored["experiment"] == name]
            counts = part.groupby("cycle")["diverged"].sum().reindex(cycles, fill_value=0)
            totals[name] = int(counts.sum())
            offset = (k - 0.5) * width
            ax_div.bar(np.array(cycles) + offset, counts.values, width=width, color=color)
            for cycle, count in counts.items():
                ax_div.text(cycle + offset, count + 0.1, str(int(count)), ha="center", va="bottom",
                            fontsize=10, color=color, fontweight="semibold")
            for cycle in cycles:
                values = part.loc[part["cycle"] == cycle, "val_last_over_min"].dropna()
                x = cycle + offset + rng.uniform(-0.1, 0.1, len(values))
                ax_noise.scatter(x, values, s=26, color=color, alpha=0.85, edgecolor="white",
                                 linewidth=0.5, zorder=4)
                ax_noise.hlines(values.median(), cycle + offset - 0.15, cycle + offset + 0.15,
                                color=color, linewidth=2.4, zorder=5)
        style_axis(ax_div, xlabel="Cycle", ylabel=f"Diverged runs (of {n_runs})",
                   title="Training-loss divergences")
        ax_div.set_xticks(cycles)
        ax_div.set_ylim(0, n_runs + 0.5)
        style_axis(ax_noise, xlabel="Cycle", ylabel="Boundary / best validation loss in the cycle",
                   title="Snapshot noise at the cycle boundary")
        ax_noise.axhline(1.0, color="#444444", linewidth=0.9)
        ax_noise.set_xticks(cycles)
        handles = [Patch(color=c, label=f"{legend[n]} ({totals[n]} diverged run-cycles)")
                   for c, n in zip(EXPERIMENT_COLORS, experiments)]
        fig.suptitle(title, fontsize=15, fontweight="bold")
        return _finish(fig, figure_path, handles, caption)
