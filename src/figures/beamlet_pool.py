"""Figures of the labelled beamlet pool (``al_build_beamlet_pool.py stats``).

- :func:`anatomy_figure`: beamlets per anatomy and split, what was lost per
  anatomy, and the energy and range spectra;
- :func:`patient_figure`: beamlets per patient against the quota, the Lorenz
  curve of the pool over patients, patients per collection, and beamlets per
  patient against the CT's z extent;
- :func:`gantry_frame_figure`: the beamlets in the IEC 61217 gantry coordinate
  system, field angles around the patient and spot positions on the isocentre
  plane, overall and per anatomy, as density relative to a uniform spread.

Tables come from :mod:`src.active_learning.beamlet_pool.stats`. House style of
:mod:`src.figures.validation_set`; saving through
:func:`save_figure_as_publication_formats`, colorbars through :func:`aligned_colorbar`.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from src.figures.al_curves import COLORS, style_axis
from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats
from src.figures.validation_set import ANATOMY_COLORS

SPLITS = ("train", "validation", "test")
SPLIT_COLORS = {"train": COLORS[0], "validation": COLORS[3], "test": COLORS[2]}


def _anatomy_color(anatomy: str, i: int = 0) -> str:
    return ANATOMY_COLORS.get(anatomy, COLORS[i % len(COLORS)])


def _finish(fig, title: str, caption: str, figure_path: str) -> List[Path]:
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
    if caption:
        fig.text(0.01, -0.01, textwrap.fill(caption, 140), ha="left", va="top", fontsize=6.5, color="#333333")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def _small(ax) -> None:
    ax.title.set_fontsize(8)
    ax.tick_params(labelsize=6.5)
    ax.xaxis.label.set_fontsize(7)
    ax.yaxis.label.set_fontsize(7)


def anatomy_figure(index: pd.DataFrame, *, figure_path: str, title: str, caption: str = "",
                   energy_bin_mev: float = 5.0, r80_bin_mm: float = 10.0) -> List[Path]:
    """(a) stored beamlets per anatomy and split, (b) fate of the planned beamlets
    per anatomy, (c) energy and (d) MC R80 per anatomy."""
    kept = index[index["status"] == "stored"]
    anatomies = sorted(kept["anatomy"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), dpi=300, constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    counts = pd.crosstab(kept["anatomy"], kept["split"]).reindex(index=anatomies, columns=SPLITS, fill_value=0)
    patients = kept.groupby("anatomy")["patient_id"].nunique().reindex(anatomies)
    bottom = np.zeros(len(anatomies))
    for split in SPLITS:
        ax_a.bar(anatomies, counts[split], bottom=bottom, color=SPLIT_COLORS[split], label=split)
        bottom += counts[split].to_numpy()
    for x, (total, n) in enumerate(zip(bottom, patients)):
        ax_a.text(x, total, f"{int(total):,}\n{n} patients", ha="center", va="bottom", fontsize=6)
    ax_a.set_ylim(0, bottom.max() * 1.25)
    style_axis(ax_a, xlabel="", ylabel="Stored beamlets", title="(a) Beamlets per anatomy and split")
    ax_a.legend(fontsize=6.5, frameon=False, loc="upper left")

    lost = index[index["status"] != "stored"]
    planned = index.groupby("anatomy").size().reindex(anatomies)
    reasons = sorted(lost["status"].unique())
    shares = pd.crosstab(lost["anatomy"], lost["status"]).reindex(index=anatomies, columns=reasons, fill_value=0) \
        .div(planned, axis=0) * 100
    bottom = np.zeros(len(anatomies))
    for j, reason in enumerate(reasons):
        label = {"missing": "generator QA (dose in crop < 95%)"}.get(reason, reason.replace("rejected:", ""))
        ax_b.bar(anatomies, shares[reason], bottom=bottom, color=COLORS[(j + 4) % len(COLORS)], label=label)
        bottom += shares[reason].to_numpy()
    for x, value in enumerate(bottom):
        ax_b.text(x, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=6)
    ax_b.set_ylim(0, bottom.max() * 1.35)
    style_axis(ax_b, xlabel="", ylabel="Planned beamlets lost [%]", title="(b) Losses per anatomy")
    ax_b.legend(fontsize=6, frameon=False, loc="upper left")

    for ax, column, step, xlabel, panel in [(ax_c, "energy_mev", energy_bin_mev, "Energy [MeV]", "(c) Energy"),
                                            (ax_d, "r80_mm", r80_bin_mm, "MC R80 in the crop [mm]",
                                             "(d) Range")]:
        values = kept[column].astype(float)
        edges = np.arange(np.floor(values.min() / step) * step, values.max() + step, step)
        for i, anatomy in enumerate(anatomies):
            part = kept.loc[kept["anatomy"] == anatomy, column].astype(float)
            ax.hist(part, bins=edges, histtype="step", density=True, color=_anatomy_color(anatomy, i), linewidth=1.2,
                    label=f"{anatomy} (median {part.median():.0f})")
        style_axis(ax, xlabel=xlabel, ylabel="Density", title=f"{panel} per anatomy")
        ax.legend(fontsize=6, frameon=False)
    for ax in axes.ravel():
        _small(ax)
    return _finish(fig, title, caption, figure_path)


def patient_figure(per_patient: pd.DataFrame, *, figure_path: str, title: str, caption: str = "") -> List[Path]:
    """(a) beamlets per patient per split with the quota, (b) Lorenz curves,
    (c) patients per collection and split, (d) beamlets per patient against the
    CT's z extent."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), dpi=300, constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes
    edges = np.arange(per_patient["stored"].min() - 2.5, per_patient["stored"].max() + 2.5, 2.0)
    for split in SPLITS:
        part = per_patient[per_patient["split"] == split]
        if part.empty:
            continue
        ax_a.hist(part["stored"], bins=edges, histtype="step", color=SPLIT_COLORS[split], linewidth=1.2,
                  label=f"{split} ({len(part)} patients)")
        if "quota" in part and part["quota"].notna().any():
            ax_a.axvline(float(part["quota"].median()), color=SPLIT_COLORS[split], linestyle="--", linewidth=0.9)
    ax_a.text(0.99, 0.97, "dashed: quota", transform=ax_a.transAxes, ha="right", va="top", fontsize=6,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    style_axis(ax_a, xlabel="Stored beamlets per patient", ylabel="Patients",
               title="(a) Beamlets per patient")
    ax_a.legend(fontsize=6, frameon=False, loc="upper left")

    ax_b.plot([0, 1], [0, 1], color="#888888", linewidth=0.8, linestyle=":")
    for split in (*SPLITS, "all"):
        part = per_patient if split == "all" else per_patient[per_patient["split"] == split]
        x = np.sort(part["stored"].to_numpy(dtype=float))
        if x.size == 0:
            continue
        cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
        g = 1.0 - (cum[:-1] + cum[1:]).sum() / x.size     # 1 - 2 x area under the Lorenz curve
        ax_b.plot(np.linspace(0, 1, cum.size), cum, color=SPLIT_COLORS.get(split, "#333333"),
                  linewidth=1.6 if split == "all" else 1.0, label=f"{split} (Gini {g:.3f})")
    style_axis(ax_b, xlabel="Share of patients (fewest beamlets first)", ylabel="Share of beamlets",
               title="(b) Lorenz curves")
    ax_b.legend(fontsize=6, frameon=False, loc="upper left")

    table = per_patient.groupby(["collection", "split"]).size().unstack(fill_value=0) \
        .reindex(columns=SPLITS, fill_value=0)
    bottom = np.zeros(len(table))
    for split in SPLITS:
        ax_c.barh(table.index, table[split], left=bottom, color=SPLIT_COLORS[split], label=split)
        bottom += table[split].to_numpy()
    for y, total in enumerate(bottom):
        ax_c.text(total, y, f" {int(total)}", va="center", fontsize=6)
    ax_c.set_xlim(0, bottom.max() * 1.15)
    style_axis(ax_c, xlabel="Patients", ylabel="", title="(c) Patients per collection")
    ax_c.legend(fontsize=6, frameon=False, loc="lower right")

    if "z_extent_mm" in per_patient:
        for i, (anatomy, part) in enumerate(per_patient.groupby("anatomy", sort=True)):
            ax_d.scatter(part["z_extent_mm"], part["stored"], s=5, alpha=0.6, color=_anatomy_color(anatomy, i),
                         label=anatomy, linewidths=0)
        ax_d.legend(fontsize=6, frameon=False, markerscale=2)
    style_axis(ax_d, xlabel="CT z extent [mm]", ylabel="Stored beamlets", title="(d) Beamlets vs scan length")
    for ax in axes.ravel():
        _small(ax)
    return _finish(fig, title, caption, figure_path)


def _relative_density(frame: pd.DataFrame, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    grid = frame.groupby(["y_g_mm", "x_g_mm"]).size().unstack(fill_value=0) \
        .reindex(index=ys, columns=xs, fill_value=0).to_numpy(dtype=float)
    return grid / grid.mean() if grid.mean() > 0 else grid


def gantry_frame_figure(frame: pd.DataFrame, *, figure_path: str, title: str, caption: str = "",
                        gantry_bin_deg: float = 15.0, energy_bin_mev: float = 10.0) -> List[Path]:
    """(a) field angles around the patient, (b) gantry angle against energy,
    (c) spot positions on the isocentre plane, (d-f) the same per anatomy."""
    anatomies = sorted(frame["anatomy"].unique())
    fig = plt.figure(figsize=(7.2, 5.8), dpi=300, constrained_layout=True)
    grid = fig.add_gridspec(2, max(3, len(anatomies)))
    ax_a = fig.add_subplot(grid[0, 0], projection="polar")
    ax_b, ax_c = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])

    edges = np.deg2rad(np.arange(0.0, 360.0 + gantry_bin_deg, gantry_bin_deg))
    bottom = np.zeros(edges.size - 1)
    for i, anatomy in enumerate(anatomies):
        counts, _ = np.histogram(np.deg2rad(frame.loc[frame["anatomy"] == anatomy, "gantry_deg"]), bins=edges)
        ax_a.bar(edges[:-1], counts, width=np.diff(edges), bottom=bottom, align="edge",
                 color=_anatomy_color(anatomy, i), edgecolor="white", linewidth=0.3, label=anatomy)
        bottom += counts
    ax_a.set_theta_zero_location("N")
    ax_a.set_theta_direction(-1)
    ax_a.set_title("(a) Gantry angle", loc="left", fontweight="semibold")
    ax_a.tick_params(labelsize=5.5)
    ax_a.title.set_fontsize(8)
    ax_a.legend(fontsize=5.5, frameon=False, loc="lower left", bbox_to_anchor=(-0.25, -0.32))

    g_edges = np.arange(0.0, 360.0 + gantry_bin_deg, gantry_bin_deg)
    e_lo = np.floor(frame["energy_mev"].min() / energy_bin_mev) * energy_bin_mev
    e_edges = np.arange(e_lo, frame["energy_mev"].max() + energy_bin_mev, energy_bin_mev)
    hist, _, _ = np.histogram2d(frame["gantry_deg"], frame["energy_mev"], bins=(g_edges, e_edges))
    image = ax_b.imshow(hist.T, origin="lower", aspect="auto", cmap="magma",
                        extent=(g_edges[0], g_edges[-1], e_edges[0], e_edges[-1]))
    ax_b.set_xticks(np.arange(0, 361, 90))
    style_axis(ax_b, xlabel="Gantry angle [deg]", ylabel="Energy [MeV]", title="(b) Gantry x energy")
    ax_b.grid(False)
    aligned_colorbar(fig, image, ax_b, "", label_coords=(3.6, 0.5), label_fontsize=6, tick_fontsize=5.5)

    xs, ys = np.sort(frame["x_g_mm"].unique()), np.sort(frame["y_g_mm"].unique())
    half_x, half_y = (xs[1] - xs[0]) / 2 if xs.size > 1 else 1.0, (ys[1] - ys[0]) / 2 if ys.size > 1 else 1.0
    extent = (xs[0] - half_x, xs[-1] + half_x, ys[0] - half_y, ys[-1] + half_y)
    densities = [_relative_density(frame, xs, ys)] + \
        [_relative_density(frame[frame["anatomy"] == a], xs, ys) for a in anatomies]
    spread = max(float(np.nanmax(np.abs(d - 1.0))) for d in densities)
    norm = TwoSlopeNorm(vmin=max(0.0, 1.0 - spread), vcenter=1.0, vmax=1.0 + spread)
    spot_axes = [ax_c] + [fig.add_subplot(grid[1, i]) for i in range(len(anatomies))]
    panels = ["(c) All anatomies"] + [f"({'def'[i] if i < 3 else 'g'}) {a}" for i, a in enumerate(anatomies)]
    parts = [frame] + [frame[frame["anatomy"] == a] for a in anatomies]
    for k, (ax, density, panel, part) in enumerate(zip(spot_axes, densities, panels, parts)):
        image = ax.imshow(density, origin="lower", cmap="RdBu_r", norm=norm, aspect="equal", extent=extent)
        style_axis(ax, xlabel="X_g (lateral) [mm]", ylabel="Y_g (gantry axis) [mm]" if k == 1 else "",
                   title=f"{panel}: {len(part):,}")
        ax.grid(False)
        if k in (0, len(spot_axes) - 1):        # one shared scale: a bar at the end of each row
            aligned_colorbar(fig, image, ax, "density / uniform", label_coords=(4.6, 0.5), label_fontsize=6,
                             tick_fontsize=5.5)
    for ax in [ax_b, *spot_axes]:
        _small(ax)
    return _finish(fig, title, caption, figure_path)


def spot_extent_mm(frame: pd.DataFrame) -> Optional[tuple]:
    """``(max |X_g|, max |Y_g|)`` of the spot lattice, for captions."""
    if frame.empty:
        return None
    return float(frame["x_g_mm"].abs().max()), float(frame["y_g_mm"].abs().max())
