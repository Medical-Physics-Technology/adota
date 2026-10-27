"""Figures of the frozen validation set (``scripts/al_validation_set_stats.py``).

- :func:`hu_histograms_figure`: the Hounsfield-unit distribution inside the patient,
  per anatomy, at three levels (whole CT, beamlet crops, beam path), over the
  tissue classes of :data:`src.figures.ct_visualizations.HU_LUT`.
- :func:`tissue_fractions_figure`: the same distributions as stacked tissue-class
  fractions, per CT and per anatomy.
- :func:`composition_figure`: what the set contains (anatomy x energy, patients,
  field angles, steering lattice) and the candidates' validity per energy.
- :func:`metric_distributions_figure`: the input-only physics metrics per anatomy.

House style of :mod:`src.figures.al_curves`; saving through
:func:`save_figure_as_publication_formats`, colorbars through :func:`aligned_colorbar`.
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

from src.figures.al_curves import COLORS, style_axis
from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats
from src.figures.ct_visualizations import HU_LUT

ANATOMY_COLORS = {"thoracic": COLORS[0], "abdominal": COLORS[3], "pelvic": COLORS[2],
                  "head_neck": COLORS[4], "brain": COLORS[5]}
LEVEL_TEXT = {
    "ct_body": "Whole patient (every voxel inside the contour)",
    "crop_body": "Beamlet crops (60 x 60 x 320 mm, inside the patient)",
    "beam_path": "Beam path (flux-weighted, up to the distal edge)",
}


def _color(anatomy: str, i: int = 0) -> str:
    return ANATOMY_COLORS.get(anatomy, COLORS[i % len(COLORS)])


def _tissue_bands(ax, x_max: float, labels: bool) -> None:
    for name, lo, hi, color in HU_LUT:
        if lo >= x_max:
            continue
        ax.axvspan(lo, min(hi, x_max), color=color, alpha=0.35, zorder=0, linewidth=0)
        if labels:
            ax.text((lo + min(hi, x_max)) / 2.0, 0.97, name, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=5.5, rotation=90, color="#333333")


def density(part: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Probability density per HU of a long histogram frame (summed over its rows)."""
    counts = part.groupby("bin_lo")["weight"].sum().reindex(edges[:-1], fill_value=0.0).to_numpy()
    total = counts.sum()
    return counts / total / np.diff(edges) if total > 0 else counts


def hu_histograms_figure(hist: pd.DataFrame, edges: np.ndarray, *, figure_path: str, title: str,
                         caption: str = "", x_max: float = 2000.0,
                         levels: Sequence[str] = ("ct_body", "crop_body", "beam_path"),
                         ) -> Tuple[List[Path], pd.DataFrame]:
    """One panel per level; per anatomy the pooled density (thick) and each CT
    (thin). At the whole-CT level the pool CTs are drawn dashed for comparison.

    Returns the paths and the plotted densities (for the CSV beside the figure).
    """
    centres = (edges[:-1] + edges[1:]) / 2.0
    keep = centres <= x_max
    levels = [lv for lv in levels if (hist["kind"] == lv).any()]
    fig, axes = plt.subplots(len(levels), 1, figsize=(7.2, 2.4 * len(levels) + 1.0), sharex=True, dpi=300,
                             constrained_layout=True, squeeze=False)
    curves: Dict[str, np.ndarray] = {"hu": centres[keep]}
    for r, (ax, level) in enumerate(zip(axes[:, 0], levels)):
        _tissue_bands(ax, x_max, labels=(r == 0))
        part = hist[hist["kind"] == level]
        for i, (anatomy, a_part) in enumerate(part.groupby("anatomy", sort=True)):
            color = _color(anatomy, i)
            for role, style in (("validation", "-"), ("pool", "--")):
                r_part = a_part[a_part["role"] == role] if "role" in a_part else a_part
                if r_part.empty:
                    continue
                for _, ct_part in r_part.groupby("patient_id"):
                    ax.plot(centres[keep], density(ct_part, edges)[keep], color=color, linewidth=0.5,
                            alpha=0.3, linestyle=style)
                d = density(r_part, edges)
                n_ct = r_part["patient_id"].nunique()
                ax.plot(centres[keep], d[keep], color=color, linewidth=1.6, linestyle=style,
                        label=f"{anatomy}, {role} ({n_ct} CTs)")
                curves[f"{level}:{anatomy}:{role}"] = d[keep]
        ax.set_yscale("log")
        ax.set_ylim(1e-7, None)
        style_axis(ax, xlabel="Hounsfield units" if r == len(levels) - 1 else "",
                   ylabel="Density [1/HU]", title=f"({'abc'[r]}) " + LEVEL_TEXT.get(level, level))
        ax.title.set_fontsize(7.5)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, frameon=True, framealpha=0.85, edgecolor="none", loc="upper right")
        ax.set_xlim(edges[0], x_max)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
    if caption:
        fig.text(0.01, -0.01, textwrap.fill(caption, 140), ha="left", va="top", fontsize=6.5, color="#333333")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths, pd.DataFrame(curves)


def tissue_fractions_figure(per_ct: pd.DataFrame, per_anatomy: pd.DataFrame, *, figure_path: str, title: str,
                            caption: str = "") -> List[Path]:
    """Stacked tissue-class fractions: one bar per CT (whole patient, validation and
    pool), then one bar per anatomy and level for the validation set.

    Both frames carry the ``frac_<tissue>`` columns of
    :func:`src.active_learning.validation_stats.histogram_summary`; ``per_ct`` also
    ``patient_id``, ``anatomy`` and ``role``, ``per_anatomy`` ``anatomy`` and ``kind``.
    """
    from src.active_learning.validation_stats import tissue_key

    tissues = [(name, f"frac_{tissue_key(name)}", color) for name, _, _, color in HU_LUT]
    ct = per_ct.sort_values(["anatomy", "role", "patient_id"], ascending=[True, False, True])
    an = per_anatomy.sort_values(["anatomy", "kind"])
    labels = [f"{r.patient_id} ({r.role[0]})" for r in ct.itertuples()] + \
             [f"{r.anatomy}: {r.kind}" for r in an.itertuples()]
    frame = pd.concat([ct, an], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.17 * len(frame) + 1.6), dpi=300, constrained_layout=True)
    left = np.zeros(len(frame))
    y = np.arange(len(frame))
    for name, col, color in tissues:
        values = frame[col].fillna(0.0).to_numpy()
        ax.barh(y, values, left=left, color=color, edgecolor="white", linewidth=0.3, label=name)
        left += values
    ax.axhline(len(ct) - 0.5, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.5)
    for tick, anatomy in zip(ax.get_yticklabels(), frame["anatomy"]):
        tick.set_color(_color(anatomy))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    style_axis(ax, xlabel="Fraction of voxels (beam path: of flux weight)", ylabel="", title=title)
    ax.legend(ncol=7, fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.title.set_position((0.0, 1.08))
    if caption:
        fig.text(0.01, -0.02, textwrap.fill(caption, 140), ha="left", va="top", fontsize=6.5, color="#333333")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def composition_figure(selected: pd.DataFrame, validity: Optional[pd.DataFrame], *, figure_path: str,
                       title: str, caption: str = "", gantry_bin_deg: float = 30.0) -> List[Path]:
    """What the set contains: (a) beamlets per anatomy and energy, (b) per CT and
    energy, (c) field angles, (d) candidates' validity per energy, (e) the steering
    lattice per anatomy."""
    anatomies = sorted(selected["anatomy"].unique())
    energies = sorted(selected["energy_mev"].unique())
    fig = plt.figure(figsize=(7.2, 8.6), dpi=300, constrained_layout=True)
    grid = fig.add_gridspec(3, max(2, len(anatomies)))
    ax_a, ax_d = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_b = fig.add_subplot(grid[1, :])
    width = 0.8 / len(anatomies)
    for i, anatomy in enumerate(anatomies):
        counts = selected[selected["anatomy"] == anatomy].groupby("energy_mev").size().reindex(energies, fill_value=0)
        ax_a.bar(np.arange(len(energies)) + (i - (len(anatomies) - 1) / 2) * width, counts, width=width,
                 color=_color(anatomy, i), label=anatomy)
        if validity is not None and anatomy in validity.index.get_level_values(0):
            v = validity.loc[anatomy]["valid_fraction"].reindex(energies)
            ax_d.plot(energies, v * 100, marker="o", color=_color(anatomy, i), label=anatomy)
    ax_a.set_xticks(np.arange(len(energies)))
    ax_a.set_xticklabels([f"{e:g}" for e in energies])
    style_axis(ax_a, xlabel="Energy [MeV]", ylabel="Beamlets", title="(a) Beamlets per energy")
    ax_a.legend(fontsize=6.5, frameon=False)
    style_axis(ax_d, xlabel="Energy [MeV]", ylabel="Valid candidates [%]",
               title="(d) Candidates passing the validity gate")
    ax_d.set_ylim(0, 105)
    ax_d.legend(fontsize=6.5, frameon=False)

    per_ct = selected.groupby(["anatomy", "patient_id", "energy_mev"]).size().unstack(fill_value=0) \
        .reindex(columns=energies, fill_value=0)
    bottom = np.zeros(len(per_ct))
    cmap = plt.get_cmap("viridis", len(energies))
    for j, e in enumerate(energies):
        ax_b.bar(np.arange(len(per_ct)), per_ct[e], bottom=bottom, color=cmap(j), label=f"{e:g} MeV")
        bottom += per_ct[e].to_numpy()
    ax_b.set_xticks(np.arange(len(per_ct)))
    ax_b.set_xticklabels([p for _, p in per_ct.index], rotation=35, ha="right", fontsize=6)
    for tick, (anatomy, _) in zip(ax_b.get_xticklabels(), per_ct.index):
        tick.set_color(_color(anatomy))
    style_axis(ax_b, xlabel="", ylabel="Beamlets", title="(b) Beamlets per CT and energy (labels coloured by anatomy)")
    ax_b.set_ylim(0, float(per_ct.sum(axis=1).max()) * 1.18)
    ax_b.legend(ncol=len(energies), fontsize=6, frameon=False, loc="upper center")

    for i, anatomy in enumerate(anatomies):
        part = selected[selected["anatomy"] == anatomy]
        ax = fig.add_subplot(grid[2, i])
        occ = part.groupby(["theta_y_deg", "theta_x_deg"]).size().unstack(fill_value=0)
        image = ax.imshow(occ.to_numpy(), origin="lower", cmap="magma", aspect="equal",
                          extent=(occ.columns.min(), occ.columns.max(), occ.index.min(), occ.index.max()))
        n_fields = part.groupby(["patient_id", "gantry_deg"]).ngroups
        gantries = np.sort(part["gantry_deg"].unique())
        style_axis(ax, xlabel="theta_x [deg]", ylabel="theta_y [deg]",
                   title=f"({'ce'[i] if i < 2 else 'f'}) {anatomy}: steering, {n_fields} fields")
        ax.title.set_fontsize(8)
        ax.grid(False)
        ax.text(0.0, -0.30, "gantry angles: " + textwrap.fill(", ".join(f"{g:.0f}" for g in gantries), 55),
                transform=ax.transAxes, fontsize=5.5, va="top", color="#333333")
        aligned_colorbar(fig, image, ax, "beamlets", label_coords=(4.5, 0.5), label_fontsize=6, tick_fontsize=5.5)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
    if caption:
        fig.text(0.01, -0.01, textwrap.fill(caption, 140), ha="left", va="top", fontsize=6.5, color="#333333")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def metric_distributions_figure(selected: pd.DataFrame, metrics: Mapping[str, str], *, figure_path: str,
                                title: str, caption: str = "", n_cols: int = 3, bins: int = 40) -> List[Path]:
    """One histogram per metric (``{column: axis label}``), anatomies overlaid."""
    names = [m for m in metrics if m in selected.columns]
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 1.9 * n_rows + 0.8), dpi=300,
                             constrained_layout=True, squeeze=False)
    for ax, m in zip(axes.ravel(), names):
        values = selected[m].astype(float)
        edges = np.linspace(np.nanpercentile(values, 0.5), np.nanpercentile(values, 99.5), bins + 1)
        for i, (anatomy, part) in enumerate(selected.groupby("anatomy", sort=True)):
            ax.hist(part[m].astype(float).clip(edges[0], edges[-1]), bins=edges, histtype="step", density=True,
                    color=_color(anatomy, i), linewidth=1.2, label=anatomy)
        style_axis(ax, xlabel=metrics[m], ylabel="", title=m)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(5.5)
        ax.yaxis.get_offset_text().set_fontsize(5.5)
        ax.title.set_fontsize(7.5)
        ax.tick_params(labelsize=6)
        ax.xaxis.label.set_fontsize(6.5)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=10)
    if caption:
        fig.text(0.01, -0.01, textwrap.fill(caption, 140), ha="left", va="top", fontsize=6.5, color="#333333")
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
