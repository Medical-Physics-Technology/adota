"""Figures of the per-mixer hyperparameter optimisation (EXP-0014,
``scripts/active_learning/hpo_figures.py``).

- :func:`hpo_learning_curves_figure`: training and validation loss per epoch,
  one column per sequence mixer, with the successive-halving rungs as labelled
  rules and the three tiers of a search (cut at the first rung, cut at the
  second, reached the top) drawn at decreasing weight.
- :func:`finalist_beamlet_figure`: one validation beamlet through every
  finalist: the ground truth and each model in the axial and coronal planes
  through the ground-truth Bragg peak, then all depth-dose curves and all
  lateral profiles overlaid.

The error agreement of the finalists is not here: it is
:func:`src.figures.ensemble_agreement.agreement_matrix_figure`, run through
``scripts/active_learning/ensemble_agreement.py`` with the finalists' error CSVs.

House style of :mod:`src.figures.al_curves` (:func:`style_axis`,
:func:`draw_cycle_rules`); saving through
:func:`save_figure_as_publication_formats`. The beamlet figure is laid out in
inches at journal page width, so its one colorbar is placed in that layout,
spanning both image columns, rather than through :func:`aligned_colorbar`,
which sizes a bar to a single axis.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

from src.figures.al_curves import COLORS, LINESTYLES, draw_cycle_rules, style_axis
from src.figures.axes_utils import save_figure_as_publication_formats
from src.utils.dose_grid_utils import estimate_bragg_peak

MIXER_TITLES = {"transformer": "Transformer", "mlstm": "mLSTM", "none": "No sequence mixer (U-Net)"}
MIXER_COLORS = {"transformer": COLORS[0], "mlstm": COLORS[1], "none": COLORS[2]}
CUT_COLOR = "#9a9a9a"


# --------------------------------------------------------------------------- learning curves

def search_tier(last_epoch: int, budgets: Sequence[int]) -> int:
    """0: cut at the first rung, ..., ``len(budgets) - 1``: reached the top rung
    (a trial that diverged inside a rung belongs to that rung)."""
    for tier, budget in enumerate(budgets[:-1]):
        if last_epoch <= budget:
            return tier
    return len(budgets) - 1


def hpo_learning_curves_figure(curves: pd.DataFrame, trials: pd.DataFrame, budgets: Sequence[int],
                               figure_path: str, title: str, caption: str = "") -> List[Path]:
    """Training (top row) and validation loss (bottom row) per epoch, one column
    per mixer.

    Args:
        curves: ``trial``, ``mixer``, ``epoch`` (completed epochs, from 1),
            ``train_loss``, ``val_loss``.
        trials: ``trial``, ``mixer``, ``failed`` (bool), ``label`` (the role,
            empty for none) and ``selected_epoch`` (the epoch of the checkpoint the
            selection used, NaN for none).
        budgets: Cumulative epochs of the rungs, e.g. ``[3, 9, 27]``.
        figure_path: Output stem; SVG, PDF and PNG are written beside it.
    """
    mixers = [m for m in MIXER_TITLES if m in set(trials["mixer"])]
    fig, axes = plt.subplots(2, len(mixers), figsize=(5.4 * len(mixers), 8.6), dpi=300,
                             sharex=True, sharey="row", constrained_layout=True, squeeze=False)
    info = trials.set_index("trial")
    # A rule between the last epoch of a rung and the first of the next.
    rules = pd.DataFrame({"cumulative_epoch": [b + 0.5 for b in budgets[:-1]], "budget": list(budgets[:-1])})
    for col, mixer in enumerate(mixers):
        part = curves[curves["mixer"] == mixer]
        last = part.groupby("trial")["epoch"].max()
        tiers = {t: search_tier(int(e), budgets) for t, e in last.items()}
        top = sorted(t for t, k in tiers.items() if k == len(budgets) - 1)
        handles: List[Line2D] = []
        for row, column in enumerate(("train_loss", "val_loss")):
            ax = axes[row, col]
            for trial, frame in part.groupby("trial"):
                tier = tiers[trial]
                if tier == len(budgets) - 1:
                    continue
                color, width, alpha = ((CUT_COLOR, 0.6, 0.5) if tier == 0
                                       else (MIXER_COLORS[mixer], 0.9, 0.45))
                ax.plot(frame["epoch"], frame[column], color=color, linewidth=width, alpha=alpha, zorder=2)
            for index, trial in enumerate(top):
                frame = part[part["trial"] == trial]
                label = trial + (f" ({info.loc[trial, 'label']})" if info.loc[trial, "label"] else "")
                line, = ax.plot(frame["epoch"], frame[column], color=COLORS[(index + 3) % len(COLORS)],
                                linestyle=LINESTYLES[index % len(LINESTYLES)], linewidth=1.8,
                                label=label + (", diverged" if info.loc[trial, "failed"] else ""), zorder=4)
                if row == 0:
                    handles.append(line)
            for trial in part["trial"].unique():
                frame = part[part["trial"] == trial]
                if bool(info.loc[trial, "failed"]):
                    ax.plot(frame["epoch"].iloc[-1], frame[column].iloc[-1], marker="X", color="black",
                            markersize=8, zorder=6)
                chosen = info.loc[trial, "selected_epoch"]
                if row == 1 and pd.notna(chosen):
                    hit = frame[frame["epoch"] == int(chosen)]
                    if len(hit):
                        ax.plot(hit["epoch"], hit[column], marker="*", color="gold", markeredgecolor="black",
                                markersize=13, zorder=7)
                        ax.annotate(str(info.loc[trial, "label"]), (hit["epoch"].iloc[0], hit[column].iloc[0]),
                                    xytext=(6, -10), textcoords="offset points", fontsize=8, fontweight="bold")
            style_axis(ax, xlabel="Epoch" if row == 1 else "",
                       ylabel=("Training loss" if row == 0 else "Validation loss") if col == 0 else "",
                       title=f"{MIXER_TITLES[mixer]}: {'training' if row == 0 else 'validation'} loss")
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(0.5, max(budgets) + 0.5)
        for row in range(2):
            draw_cycle_rules(axes[row, col], rules, label=True,
                             text=lambda r: f" halving after\n {int(r.budget)} epochs")
        handles += [Line2D([], [], color=MIXER_COLORS[mixer], linewidth=0.9, alpha=0.6,
                           label=f"cut after {budgets[-2]} epochs"),
                    Line2D([], [], color=CUT_COLOR, linewidth=0.6, label=f"cut after {budgets[0]} epochs"),
                    Line2D([], [], marker="*", color="gold", markeredgecolor="black", linestyle="",
                           markersize=11, label="checkpoint used for selection"),
                    Line2D([], [], marker="X", color="black", linestyle="", markersize=7,
                           label="non-finite loss (trial stopped)")]
        axes[1, col].legend(handles=handles, fontsize=7.5, frameon=True, framealpha=0.95, loc="upper right")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    if caption:
        fig.supxlabel(textwrap.fill(caption, width=int(65 * len(mixers))), fontsize=8.5)
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- one beamlet, all finalists

def beamlet_planes(volume: np.ndarray, peak: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """The (depth, lateral) planes of a ``(depth, y, x)`` volume through ``peak``:
    axial (fixed y) and coronal (fixed x). These are the planes
    :func:`src.figures.single_beam.publication_figure` calls axial and sagittal;
    for the lateral beams of the dataset the second one is the coronal plane."""
    return volume[:, peak[1], :], volume[:, :, peak[2]]


def depth_dose(dose: np.ndarray, voxel_mm: float) -> np.ndarray:
    """Integrated depth dose: the lateral sum at each depth times the voxel area."""
    return dose.sum(axis=(1, 2)) * voxel_mm ** 2


def lateral_profile(dose: np.ndarray, peak: Tuple[int, int, int]) -> np.ndarray:
    """Dose along x at the depth and y of ``peak`` (the axial plane's lateral axis)."""
    return dose[peak[0], peak[1], :]


def shown_depth(ground_truth: np.ndarray, margin_mm: float, voxel_mm: float, floor: float = 0.01) -> int:
    """Voxels of depth to show: up to where the ground-truth depth dose falls below
    ``floor`` of its maximum for good, plus ``margin_mm``."""
    idd = depth_dose(ground_truth, voxel_mm)
    above = np.nonzero(idd >= floor * idd.max())[0]
    end = int(above[-1]) + 1 if above.size else ground_truth.shape[0]
    return int(min(ground_truth.shape[0], end + round(margin_mm / voxel_mm)))


MIXER_SHADES = {"transformer": ("#1d4e89", "#4f86c6", "#7aa5da"),
                "mlstm": ("#b2182b", "#e0646f", "#f2a7ad"),
                "none": ("#1b7f4c", "#4fb784", "#9ad8b6")}
SHADE_STYLES = ("-", (0, (4.5, 1.6)), (0, (1.2, 1.2)))
GT_STYLE = {"color": "black", "linewidth": 1.8, "linestyle": "-"}


def model_styles(mixers: Mapping[str, str]) -> Dict[str, Dict]:
    """``label -> line style``: one hue per sequence mixer, models of the same
    mixer told apart by shade and dash, in the order given."""
    seen: Dict[str, int] = {}
    out = {}
    for label, mixer in mixers.items():
        k = seen.get(mixer, 0)
        seen[mixer] = k + 1
        shades = MIXER_SHADES.get(mixer, COLORS)
        out[label] = {"color": shades[k % len(shades)], "linestyle": SHADE_STYLES[k % len(SHADE_STYLES)],
                      "linewidth": 1.25}
    return out


def _axes_in(fig, size: Tuple[float, float], left: float, top: float, width: float, height: float):
    """Axes placed in inches from the figure's top-left corner."""
    w, h = size
    return fig.add_axes([left / w, 1.0 - (top + height) / h, width / w, height / h])


def finalist_beamlet_figure(ct: np.ndarray, ground_truth: np.ndarray, predictions: Mapping[str, np.ndarray], *,
                            figure_path: str, title: str, subtitle: str = "",
                            notes: Optional[Mapping[str, str]] = None,
                            styles: Optional[Mapping[str, Mapping]] = None, caption: str = "",
                            voxel_mm: float = 2.0, overlay_floor_pct: float = 5.0, contour_pct: float = 50.0,
                            margin_mm: float = 30.0,
                            width_in: float = 7.2) -> Tuple[List[Path], Dict[str, pd.DataFrame]]:
    """Ground truth and every model of ``predictions`` in two planes, then the
    depth-dose curves and lateral profiles of all of them, at journal page width.

    One row per model: its label on the left in its line color, the axial and
    coronal planes through the ground-truth Bragg peak over the CT, and its
    ``notes`` on the right. Dose is a percentage of the ground-truth maximum on
    one color scale, faded in above ``overlay_floor_pct``; the white contour is
    the ground truth's ``contour_pct`` isodose in every row, and the dotted line
    the ground-truth peak depth, where the lateral profile is taken. The depth-dose
    panel sits under the axial column on the same depth axis.

    Args:
        ct: ``(depth, y, x)`` CT channel of the model input (display only).
        ground_truth, predictions: ``(depth, y, x)`` dose in one unit.
        notes: Per row label (``"Ground truth"`` or a prediction label), a short,
            possibly multi-line text for the right-hand column.
        styles: Per prediction label, a line style (:func:`model_styles`).
        figure_path: Output stem; SVG, PDF and PNG are written beside it.

    Returns:
        The written paths, and the plotted curves (``idd``, ``profile``) as frames
        for the CSV beside the figure.
    """
    notes = dict(notes or {})
    rows = {"Ground truth": ground_truth, **predictions}
    styles = {"Ground truth": GT_STYLE, **(dict(styles) if styles else
                                           {k: {"color": COLORS[i % len(COLORS)], "linewidth": 1.25,
                                                "linestyle": LINESTYLES[i % len(LINESTYLES)]}
                                            for i, k in enumerate(predictions)})}
    scale = float(ground_truth.max())
    peak = estimate_bragg_peak(ground_truth)
    depth = shown_depth(ground_truth, margin_mm, voxel_mm)
    depth_extent, lateral_extent = depth * voxel_mm, ground_truth.shape[2] * voxel_mm
    peak_mm = (peak[0] + 0.5) * voxel_mm
    vmax = max(100.0, max(float(d[:depth].max()) for d in rows.values()) / scale * 100.0)
    ct_planes = beamlet_planes(ct, peak)
    gt_planes = [p[:depth].T / scale * 100.0 for p in beamlet_planes(ground_truth, peak)]
    lo, hi = np.percentile(ct[:depth], [1, 99])
    extent = (0, depth_extent, -lateral_extent / 2, lateral_extent / 2)

    # Layout in inches: gutter | axial | coronal | notes; the panels keep 1 mm = 1 mm.
    n = len(rows)
    gutter, gap, notes_w, right = 1.30, 0.10, 1.00, 0.05
    pw = (width_in - gutter - 2 * gap - notes_w - right) / 2
    ph = pw * lateral_extent / depth_extent
    row_gap, bar_top, img_top = 0.07, 0.62, 1.12
    img_bottom = img_top + n * ph + (n - 1) * row_gap
    prof_top, prof_h = img_bottom + 0.62, 1.55
    legend_top = prof_top + prof_h + 0.45
    caption_top = legend_top + 0.40
    size = (width_in, caption_top + (0.55 if caption else 0.05))

    with plt.rc_context({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5}):
        fig = plt.figure(figsize=size, dpi=300)
        image = None
        for r, (label, dose) in enumerate(rows.items()):
            top = img_top + r * (ph + row_gap)
            for c, (plane, ct_plane) in enumerate(zip(beamlet_planes(dose, peak), ct_planes)):
                ax = _axes_in(fig, size, gutter + c * (pw + gap), top, pw, ph)
                ax.imshow(ct_plane[:depth].T, cmap="gray", vmin=lo, vmax=hi, extent=extent, origin="lower",
                          aspect="auto", interpolation="bilinear")
                relative = plane[:depth].T / scale * 100.0
                fade = np.clip((relative - overlay_floor_pct) / overlay_floor_pct, 0.0, 1.0) * 0.9
                image = ax.imshow(relative, cmap="turbo", vmin=0.0, vmax=vmax, extent=extent, origin="lower",
                                  aspect="auto", interpolation="bilinear", alpha=fade)
                ax.contour(gt_planes[c], levels=[contour_pct], colors="white", linewidths=0.7, extent=extent,
                           origin="lower")
                ax.axvline(peak_mm, color="white", linestyle=(0, (1, 1.5)), linewidth=0.7, alpha=0.9)
                ax.set_xlim(0, depth_extent)
                ax.set_yticks([-20, 0, 20])
                for spine in ax.spines.values():
                    spine.set_color("#555555")
                if r == 0:
                    ax.set_title(("Axial", "Coronal")[c] + " plane", loc="left", fontweight="semibold", pad=3)
                if r < n - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("Depth [mm]")
                if c == 1:
                    ax.tick_params(labelleft=False)
            # Row label: role and trial in the model's line color, with a swatch of its line.
            centre = top + ph / 2
            head, _, tail = label.partition(": ")
            color = styles[label]["color"]
            fig.text((gutter - 0.33) / size[0], 1 - (centre - 0.07) / size[1], head, ha="right", va="center",
                     fontweight="bold", color=color, fontsize=7.5)
            if tail:
                fig.text((gutter - 0.33) / size[0], 1 - (centre + 0.07) / size[1], tail, ha="right",
                         va="center", color=color, fontsize=6.5)
            swatch_y = 1 - (centre + (0.20 if tail else 0.12)) / size[1]
            fig.add_artist(Line2D([(gutter - 0.80) / size[0], (gutter - 0.35) / size[0]], [swatch_y, swatch_y],
                                  transform=fig.transFigure, **{**styles[label], "linewidth": 1.4}))
            if label in notes:
                fig.text((gutter + 2 * pw + 2 * gap) / size[0], 1 - centre / size[1], notes[label], ha="left",
                         va="center", fontsize=6.5, color="#222222", linespacing=1.25)

        # The one lateral-axis label, above the tick column instead of beside every row.
        fig.text((gutter - 0.04) / size[0], 1 - (img_top - 0.06) / size[1], "lateral\n[mm]", ha="right",
                 va="bottom", fontsize=6.5, color="#444444", linespacing=1.1)
        cax = _axes_in(fig, size, gutter + 0.5 * pw, bar_top, pw + gap, 0.07)
        bar = fig.colorbar(image, cax=cax, orientation="horizontal")
        bar.set_label("Dose [% of the ground-truth maximum]", labelpad=2)
        bar.ax.xaxis.set_label_position("top")
        bar.set_ticks([0, 25, 50, 75, 100])
        bar.outline.set_linewidth(0.5)

        depth_mm = (np.arange(ground_truth.shape[0]) + 0.5) * voxel_mm
        lateral_mm = (np.arange(ground_truth.shape[2]) + 0.5 - ground_truth.shape[2] / 2) * voxel_mm
        idd_gt = depth_dose(ground_truth, voxel_mm)
        idd = pd.DataFrame({"depth_mm": depth_mm,
                            **{k: depth_dose(v, voxel_mm) / idd_gt.max() * 100 for k, v in rows.items()}})
        profile = pd.DataFrame({"lateral_mm": lateral_mm,
                                **{k: lateral_profile(v, peak) / scale * 100 for k, v in rows.items()}})
        ax_idd = _axes_in(fig, size, gutter, prof_top, pw, prof_h)
        lat_left = gutter + pw + gap + 0.50
        ax_lat = _axes_in(fig, size, lat_left, prof_top, width_in - right - 0.05 - lat_left, prof_h)
        handles = []
        for label in reversed(list(rows)):  # the ground truth on top
            style = styles[label]
            ax_idd.plot(idd["depth_mm"], idd[label], **style, zorder=3 if label == "Ground truth" else 2)
            line, = ax_lat.plot(profile["lateral_mm"], profile[label], **style,
                                zorder=3 if label == "Ground truth" else 2)
            handles.append((label, line))
        for ax, xlabel, ylabel, panel in (
                (ax_idd, "Depth [mm]", "Integrated depth dose [%]", "Integrated depth dose"),
                (ax_lat, "Lateral position [mm]", "Dose [%]", f"Lateral profile at {peak_mm:.0f} mm depth")):
            style_axis(ax, xlabel=xlabel, ylabel=ylabel, title=panel)
            ax.grid(True, color="#d9d9d9", linewidth=0.5)
            ax.set_ylim(bottom=0)
        ax_idd.set_xlim(0, depth_extent)
        ax_idd.axvline(peak_mm, color="#555555", linestyle=(0, (1, 1.5)), linewidth=0.7)
        ax_lat.set_xlim(lateral_mm[0], lateral_mm[-1])
        ordered = [h for label in rows for lab, h in handles if lab == label]
        fig.legend(ordered, list(rows), loc="upper center", ncol=4, frameon=False, handlelength=2.6,
                   columnspacing=1.4, bbox_to_anchor=(0.5, 1 - legend_top / size[1]))

        fig.text(0.01, 1 - 0.10 / size[1], title, ha="left", va="top", fontsize=10, fontweight="bold")
        if subtitle:
            fig.text(0.01, 1 - 0.32 / size[1], subtitle, ha="left", va="top", fontsize=7, color="#444444")
        if caption:
            fig.text(0.01, 1 - caption_top / size[1], textwrap.fill(caption, width=150), ha="left", va="top",
                     fontsize=6.5, color="#333333")
        paths = save_figure_as_publication_formats(fig, figure_path)
        plt.close(fig)
    return paths, {"idd": idd, "profile": profile}
