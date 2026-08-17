"""Per-beamlet ADoTA-vs-MCsquare analysis figure.

Metric-vs-energy scatter panels (GPR 2%/2mm, GPR 3%/3mm, MAPE, R80 difference),
each point coloured and sized by the beamlet's MU fraction, with the 150 MeV
thoracic-training-limit reference line. Answers whether ADoTA degrades above
150 MeV and whether the degrading spots carry meaningful MU.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures.single_beam import save_figure_as_publication_formats

_PANELS = [
    ("gpr_2pct_2mm", "GPR 2%/2mm [%]", None),
    ("gpr_3pct_3mm", "GPR 3%/3mm [%]", None),
    ("mape_pct", "MAPE (high-dose) [%]", None),
    ("r80_diff_mm", "R80 diff (ADoTA - MC) [mm]", 0.0),
]


def beamlet_analysis_figure(
    df: pd.DataFrame,
    figure_path: str,
    primaries: float | None = None,
    energy_split_mev: float = 150.0,
    dpi: int = 300,
) -> List[Path]:
    """Render the 2x2 metric-vs-energy scatter; return the written paths."""
    mu_frac = df["mu_fraction"].to_numpy(dtype=float)
    # size in points^2: scale MU fraction to a visible range (min 12, max ~320)
    fmax = float(mu_frac.max()) if mu_frac.size and mu_frac.max() > 0 else 1.0
    sizes = 12.0 + 308.0 * (mu_frac / fmax)
    energy = df["energy_mev"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=dpi)
    sc = None
    for ax, (col, ylabel, hline) in zip(axes.ravel(), _PANELS):
        if col not in df:
            ax.set_visible(False)
            continue
        y = df[col].to_numpy(dtype=float)
        ok = np.isfinite(y)
        sc = ax.scatter(energy[ok], y[ok], s=sizes[ok], c=mu_frac[ok],
                        cmap="viridis", alpha=0.8, edgecolors="black", linewidths=0.4)
        ax.axvline(energy_split_mev, color="red", ls="--", lw=1.6)
        ax.text(energy_split_mev, ax.get_ylim()[1], f" {energy_split_mev:.0f} MeV",
                color="red", va="top", ha="left", fontsize=11)
        if hline is not None:
            ax.axhline(hline, color="grey", ls=":", lw=1.0)
        ax.set_xlabel("Beamlet energy [MeV]", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, ls=":", lw=0.5)
        ax.tick_params(labelsize=11)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        cbar.set_label("MU fraction (point size ∝ MU fraction)", fontsize=12)

    n = len(df)
    prim = f"{primaries:.0e} primaries/beamlet" if primaries else "per beamlet"
    fig.suptitle(
        f"ADoTA vs MCsquare per-beamlet agreement ({n} spots, {prim})",
        fontsize=16, weight="bold",
    )
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths


def gpr_per_layer_figure(
    df: pd.DataFrame,
    figure_path: str,
    gpr_col: str,
    dose_pct: float,
    dist_mm: float,
    cutoff_pct: float,
    primaries: float | None = None,
    energy_split_mev: float = 150.0,
    dpi: int = 300,
) -> List[Path]:
    """Layer-mean gamma pass rate vs plan energy layer (single publication panel).

    One point per plan energy layer (grouped by beam + control-point); y is the
    plain mean of the per-spot GPR in that layer -- a single marker, no error
    bars and no connecting line. The x-axis is ticked at the exact per-layer
    energies. The training-limit line and OOD styling are only drawn when layers
    actually exceed ``energy_split_mev`` (so prostate plans, whose energies stay
    below the split, show no red line). No MU weighting. Fonts set for A4 print.
    """
    # A4 print fonts -- large per publication guidelines (readable on their own)
    fs_tick, fs_label, fs_legend = 22, 26, 20

    g = df.groupby(["beam_idx", "layer_idx"], as_index=False).agg(
        energy_mev=("energy_mev", "first"),
        gpr_mean=(gpr_col, "mean"),
        n_spots=(gpr_col, "size"),
    ).sort_values("energy_mev")
    x = g["energy_mev"].to_numpy(dtype=float)
    y = g["gpr_mean"].to_numpy(dtype=float)
    ood = x >= energy_split_mev

    # per-layer energies as x-ticks, merged to whole MeV so the two beams'
    # near-identical layers (e.g. 150.0 / 150.1) collapse to one clean tick
    xticks = np.unique(np.round(x, 0))
    # scale figure width with the layer count so every tick keeps a constant,
    # generous horizontal budget -- lets the x-tick font stay large even when dense
    width = float(np.clip(0.46 * len(xticks) + 3.0, 16.0, 46.0))

    fig, ax = plt.subplots(figsize=(width, 7.5), dpi=dpi)
    # in-distribution markers (single point per layer, no line, no error bars)
    ax.scatter(x[~ood], y[~ood], s=80, color="#2a78d6", edgecolors="black",
               linewidths=0.6, zorder=3, label="in-distribution")
    # OOD styling only when layers actually exceed the training limit
    if ood.any():
        xmax = float(x.max()) + max(1.0, 0.03 * (float(x.max()) - float(x.min())))
        ax.axvspan(energy_split_mev, xmax, color="#e34948", alpha=0.10, zorder=0)
        ax.scatter(x[ood], y[ood], s=110, color="#e34948", marker="D",
                   edgecolors="black", linewidths=0.6, zorder=4,
                   label=f"OOD (≥ {energy_split_mev:.0f} MeV)")
        ax.axvline(energy_split_mev, color="#e34948", ls="--", lw=2.0, zorder=2)
        ax.text(energy_split_mev, ax.get_ylim()[1], f" {energy_split_mev:.0f} MeV "
                "training limit", color="#e34948", va="top", ha="left", fontsize=fs_legend)
        ax.legend(fontsize=fs_legend, loc="lower left", framealpha=0.9)

    ax.set_xlabel("Plan energy layer [MeV]", fontsize=fs_label)
    ax.set_ylabel(f"Layer-mean GPR {dose_pct:g}%/{dist_mm:g}mm [%]", fontsize=fs_label)
    ax.grid(True, ls=":", lw=0.6)
    # width scales with tick count, so the labels stay at the full print size
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:g}" for t in xticks], rotation=90, fontsize=fs_tick)
    ax.tick_params(axis="y", which="major", labelsize=fs_tick)
    fig.tight_layout()
    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return paths
