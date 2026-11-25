"""Slide panels of one beamlet: the Monte Carlo reference and every dose engine on it.

One landscape (16:9) figure per beamlet, for a meeting slide:

- row 1: the dose of each engine (Monte Carlo first) over the CT, on the plane through
  the Monte Carlo Bragg peak (maximum of its depth dose), one shared colour scale;
- row 2: each engine minus the Monte Carlo, in % of the Monte Carlo maximum (symmetric,
  shared scale), and a table of the per-beamlet errors;
- row 3: the laterally integrated depth dose of every engine with the HU on the beam
  axis, and the lateral profile through the peak.

Arrays are on the model grid ``(D, H, W)``: depth first, ``H`` = z, ``W`` = y, so the
plane shown is the axial one (fixed z), as in :func:`src.figures.difficulty_explain.beam_path_figure`.
Saved through :func:`save_figure_as_publication_formats`, colour bars through
:func:`aligned_colorbar`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures.axes_utils import PUBLICATION_FORMATS, aligned_colorbar, save_figure_as_publication_formats

ENGINE_STYLE: Mapping[str, dict] = {
    "MC": {"color": "black", "linestyle": "-", "linewidth": 1.8},
    "ADoTA": {"color": "#d62728", "linestyle": "--", "linewidth": 1.6},
    "PBA": {"color": "#1f77b4", "linestyle": "-.", "linewidth": 1.6},
}
METRIC_ROWS = (("gpr_2pct_2mm_10pct", "GPR 2%/2mm [%]", "{:.1f}"),
               ("gpr_3pct_3mm_10pct", "GPR 3%/3mm [%]", "{:.1f}"),
               ("mape_pct", "MAPE [%]", "{:.1f}"), ("rde_pct", "RDE [%]", "{:.3f}"),
               ("dr80_mm", "dR80 [mm]", "{:+.1f}"))
"""Error columns of :func:`src.engine_drivers.scoring.score_doses` shown in the table."""


def _style(name: str, i: int) -> dict:
    return dict(ENGINE_STYLE.get(name, {"color": f"C{i}", "linestyle": ":", "linewidth": 1.5}))


def _window(idd: np.ndarray, dz: float, pad_mm: float = 20.0) -> tuple:
    """Depth window: from 0 to 20 mm past the last depth above 1% of the IDD maximum."""
    above = np.nonzero(idd > 0.01 * idd.max())[0]
    end = (above[-1] + 1) * dz + pad_mm if above.size else len(idd) * dz
    return 0.0, min(len(idd) * dz, end)


def engine_panel_figure(ct_hu: np.ndarray, doses: Dict[str, np.ndarray], errors: pd.DataFrame, *, title: str,
                        figure_path: str, dz_mm: float = 2.0, dose_scale: float = 1.0,
                        dose_unit: str = "Gy per MU", reference: str = "MC",
                        formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """One beamlet, every engine against ``reference``.

    Args:
        ct_hu: CT in HU, ``(D, H, W)``.
        doses: ``{name: dose (D, H, W)}``, the reference first; on MCsquare's beamlet scale, shown
            as ``dose * dose_scale`` in ``dose_unit``.
        errors: One row per non-reference engine, indexed by its name, with the columns of
            :data:`METRIC_ROWS`.
    """
    doses = {k: np.asarray(v, dtype=np.float64) * dose_scale for k, v in doses.items()}
    names = list(doses)
    others = [n for n in names if n != reference]
    ref = doses[reference]
    # The Bragg peak is the maximum of the depth dose, not of the volume: at high energy the hottest
    # voxel sits on the narrow entrance axis. Lateral position: the hottest voxel at that depth.
    pd_ = int(np.argmax(ref.sum(axis=(1, 2))))
    ph, pw = np.unravel_index(int(np.argmax(ref[pd_])), ref.shape[1:])
    depth = (np.arange(ref.shape[0]) + 0.5) * dz_mm
    lateral = (np.arange(ref.shape[2]) + 0.5 - (pw + 0.5)) * dz_mm
    idd = {n: d.sum(axis=(1, 2)) * dz_mm ** 2 for n, d in doses.items()}
    x0, x1 = _window(idd[reference], dz_mm)
    extent = [0, ref.shape[0] * dz_mm, ref.shape[2] * dz_mm / 2, -ref.shape[2] * dz_mm / 2]
    vmax = max(float(d.max()) for d in doses.values()) or 1.0
    peak = float(ref.max()) or 1.0
    diffs = {n: 100.0 * (doses[n] - ref) / peak for n in others}
    dmax = max([float(np.abs(d[:, ph, :]).max()) for d in diffs.values()] + [1.0])
    ncol = max(len(names), len(others) + 1)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(3, ncol, height_ratios=[1.0, 1.0, 1.25], hspace=0.55, wspace=0.45,
                            left=0.05, right=0.95, top=0.9, bottom=0.07)
    bottom = grid[2, :].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.32)
    for c, n in enumerate(names):
        ax = fig.add_subplot(grid[0, c])
        ax.imshow(ct_hu[:, ph, :].T, cmap="gray", vmin=-1000, vmax=1500, aspect="auto", extent=extent)
        im = ax.imshow(np.ma.masked_less(doses[n][:, ph, :].T, 0.01 * vmax), cmap="jet", alpha=0.65, vmin=0,
                       vmax=vmax, aspect="auto", extent=extent)
        ax.set_xlim(x0, x1)
        ax.set_title(n, fontsize=12, fontweight="semibold", loc="left")
        ax.set_ylabel("Lateral [mm]")
        if c == len(names) - 1:
            aligned_colorbar(fig, im, ax, f"Dose [{dose_unit}]", label_fontsize=9, tick_fontsize=8,
                             label_coords=(5.2, 0.5))
    for c, n in enumerate(others):
        ax = fig.add_subplot(grid[1, c])
        im = ax.imshow(diffs[n][:, ph, :].T, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="auto", extent=extent)
        ax.set_xlim(x0, x1)
        ax.set_title(f"{n} - {reference}", fontsize=12, fontweight="semibold", loc="left")
        ax.set_ylabel("Lateral [mm]")
        ax.set_xlabel("Depth [mm]")
        if c == len(others) - 1:
            aligned_colorbar(fig, im, ax, f"Difference [% of {reference} max]", label_fontsize=9,
                             tick_fontsize=8, label_coords=(4.0, 0.5))
    ax_tab = fig.add_subplot(grid[1, len(others):])
    ax_tab.axis("off")
    cells = [[fmt.format(float(errors.at[n, col])) if col in errors.columns and np.isfinite(errors.at[n, col])
              else "n/a" for n in others] for col, _, fmt in METRIC_ROWS]
    table = ax_tab.table(cellText=cells, rowLabels=[lab for _, lab, _ in METRIC_ROWS], colLabels=others,
                         loc="center", cellLoc="center", colWidths=[0.28] * len(others))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax_tab.set_title(f"Errors against {reference} (2 mm grid, global gamma, 10% cutoff)", fontsize=10, loc="left")

    ax_idd = fig.add_subplot(bottom[0, 0])
    for i, n in enumerate(names):
        ax_idd.plot(depth, idd[n], label=n, **_style(n, i))
    ax_idd.set_xlim(x0, x1)
    ax_idd.set_xlabel("Depth [mm]")
    ax_idd.set_ylabel(f"IDD [{dose_unit} mm$^2$]")
    ax_idd.grid(alpha=0.3)
    ax_hu = ax_idd.twinx()
    ax_hu.plot(depth, ct_hu[:, ph, pw], color="0.55", linewidth=1.0, label="HU on the beam axis")
    ax_hu.set_ylabel("HU", color="0.4")
    ax_hu.set_ylim(-1100, max(1600.0, float(ct_hu[:, ph, pw].max()) + 100))
    ax_hu.tick_params(axis="y", colors="0.4")
    lines = ax_idd.get_lines() + ax_hu.get_lines()
    ax_idd.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=9, ncol=2)
    ax_idd.set_title("Laterally integrated depth dose and the tissue on the beam axis", fontsize=11, loc="left")

    ax_lat = fig.add_subplot(bottom[0, 1])
    for i, n in enumerate(names):
        ax_lat.plot(lateral, doses[n][pd_, ph, :], **_style(n, i))
    ax_lat.set_xlabel("Lateral [mm]")
    ax_lat.set_ylabel(f"Dose [{dose_unit}]")
    ax_lat.set_title(f"Lateral profile at the {reference} Bragg peak ({(pd_ + 0.5) * dz_mm:.0f} mm)",
                     fontsize=11, loc="left")
    ax_lat.grid(alpha=0.3)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=14, fontweight="bold")
    paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
    plt.close(fig)
    return paths


__all__ = ["ENGINE_STYLE", "METRIC_ROWS", "engine_panel_figure"]
