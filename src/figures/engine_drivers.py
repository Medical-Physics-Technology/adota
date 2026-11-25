"""Slide panels of one beamlet: the Monte Carlo reference and every dose engine on it.

The layout follows the paper's single-beamlet figure
(:func:`src.figures.single_beam.publication_figure`): an axial and a sagittal column,
one row per engine (Monte Carlo first), the dose over the CT through the Monte Carlo
Bragg peak on one shared colour scale. Below them:

- the laterally integrated depth dose of every engine, with the HU on the beam axis;
- the lateral profile at the Monte Carlo Bragg peak in each of the two planes, and a
  table of the per-beamlet errors.

``differences=True`` adds one row per engine of its difference to the Monte Carlo
(% of the Monte Carlo maximum).

Arrays are on the model grid ``(D, H, W)``. As in the paper figure, the axial view is
the plane of fixed ``H`` and the sagittal view the plane of fixed ``W``; images are
``np.rot90`` of those planes, so depth runs along x. Saved through
:func:`save_figure_as_publication_formats`, colour bars through :func:`aligned_colorbar`.
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
    "MC": {"color": "black", "linestyle": "-", "linewidth": 2.0},
    "ADoTA": {"color": "#d62728", "linestyle": "--", "linewidth": 1.8},
    "PBA": {"color": "#1f77b4", "linestyle": "-.", "linewidth": 1.8},
}
ROW_LABELS: Mapping[str, str] = {"MC": "MCSquare", "ADoTA": "ADoTA", "PBA": "Pencil beam"}
METRIC_ROWS = (("gpr_2pct_2mm_10pct", "GPR 2%/2mm [%]", "{:.1f}"),
               ("gpr_3pct_3mm_10pct", "GPR 3%/3mm [%]", "{:.1f}"),
               ("mape_pct", "MAPE [%]", "{:.1f}"), ("rde_pct", "RDE [%]", "{:.3f}"),
               ("dr80_mm", "dR80 [mm]", "{:+.1f}"))
"""Error columns of :func:`src.engine_drivers.scoring.score_doses` shown in the table."""

TICK, LABEL, TITLE = 13, 15, 17


def _style(name: str, i: int) -> dict:
    return dict(ENGINE_STYLE.get(name, {"color": f"C{i}", "linestyle": ":", "linewidth": 1.6}))


def _image(ax, ct_plane: np.ndarray, layer: np.ndarray, dz: float, **kwargs):
    """``(D, lateral)`` planes as in the paper figure: rot90, depth along x, mm on both axes."""
    extent = [0, ct_plane.shape[0] * dz, 0, ct_plane.shape[1] * dz]
    ax.imshow(np.rot90(ct_plane), cmap="gray", vmin=-1000, vmax=1500, extent=extent, aspect="auto")
    return ax.imshow(np.rot90(layer), extent=extent, aspect="auto", **kwargs)


def engine_panel_figure(ct_hu: np.ndarray, doses: Dict[str, np.ndarray], errors: pd.DataFrame, *, title: str,
                        figure_path: str, dz_mm: float = 2.0, dose_scale: float = 1.0,
                        dose_unit: str = "Gy per MU", reference: str = "MC", differences: bool = False,
                        formats: Sequence[str] = PUBLICATION_FORMATS) -> List[Path]:
    """One beamlet, every engine against ``reference``.

    Args:
        ct_hu: CT in HU, ``(D, H, W)``.
        doses: ``{name: dose (D, H, W)}``, the reference first; on MCsquare's beamlet scale, shown
            as ``dose * dose_scale`` in ``dose_unit``.
        errors: One row per non-reference engine, indexed by its name, with the columns of
            :data:`METRIC_ROWS`.
        differences: Add a row per engine of its difference to the reference.
    """
    doses = {k: np.asarray(v, dtype=np.float64) * dose_scale for k, v in doses.items()}
    names = list(doses)
    others = [n for n in names if n != reference]
    ref = doses[reference]
    # The Bragg peak is the maximum of the depth dose, not of the volume: at high energy the hottest
    # voxel sits on the narrow entrance axis. Lateral position: the hottest voxel at that depth.
    pd_ = int(np.argmax(ref.sum(axis=(1, 2))))
    ph, pw = np.unravel_index(int(np.argmax(ref[pd_])), ref.shape[1:])
    peak_mm = (pd_ + 0.5) * dz_mm
    planes = {"Axial view": lambda a: a[:, ph, :], "Sagittal view": lambda a: a[:, :, pw]}
    vmax = max(float(d.max()) for d in doses.values()) or 1.0
    peak = float(ref.max()) or 1.0
    rows = [(n, "dose") for n in names] + ([(n, "diff") for n in others] if differences else [])
    dmax = max([float(np.abs(doses[n] - ref).max()) / peak * 100.0 for n in others] + [1.0])

    n_img = len(rows)
    fig = plt.figure(figsize=(18, 2.7 * n_img + 7.6))
    grid = fig.add_gridspec(n_img + 2, 2, height_ratios=[1.0] * n_img + [1.5, 1.4], hspace=0.42, wspace=0.12,
                            left=0.09, right=0.92, top=1 - 1.15 / fig.get_figheight(), bottom=0.05)
    for r, (name, kind) in enumerate(rows):
        for c, (plane_name, cut) in enumerate(planes.items()):
            ax = fig.add_subplot(grid[r, c])
            ct_plane = cut(ct_hu)
            if kind == "dose":
                layer = cut(doses[name])
                alpha = np.where(np.rot90(layer) > 0.01 * vmax, 0.7, 0.0)
                im = _image(ax, ct_plane, layer, dz_mm, cmap="jet", vmin=0.0, vmax=vmax, alpha=alpha)
                label, bar = f"{ROW_LABELS.get(name, name)}\n[mm]", f"Dose [{dose_unit}]"
            else:
                layer = cut((doses[name] - ref) / peak * 100.0)
                im = _image(ax, ct_plane, layer, dz_mm, cmap="seismic", vmin=-dmax, vmax=dmax)
                label, bar = f"{ROW_LABELS.get(name, name)} - MC\n[mm]", f"Diff. [% of {reference} max]"
            ax.axvline(peak_mm, color="red", linewidth=1.5)
            ax.grid(linestyle="--", linewidth=0.5, color="white")
            ax.tick_params(labelsize=TICK)
            if r == 0:
                ax.set_title(plane_name, fontsize=TITLE, pad=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=LABEL)
            else:
                ax.set_yticklabels([])
                aligned_colorbar(fig, im, ax, bar, label_fontsize=12, tick_fontsize=11, label_coords=(4.6, 0.5))
            if r == n_img - 1:
                ax.set_xlabel("Depth [mm]", fontsize=LABEL)
            else:
                ax.set_xticklabels([])

    depth = (np.arange(ref.shape[0]) + 0.5) * dz_mm
    idd = {n: d.sum(axis=(1, 2)) * dz_mm ** 2 for n, d in doses.items()}
    ax_idd = fig.add_subplot(grid[n_img, :])
    for i, n in enumerate(names):
        ax_idd.plot(depth, idd[n], label=ROW_LABELS.get(n, n), **_style(n, i))
    ax_idd.axvline(peak_mm, color="red", linestyle="--", linewidth=1.2,
                   label=f"{ROW_LABELS.get(reference, reference)} Bragg peak ({peak_mm:.0f} mm)")
    ax_idd.set_xlim(0, ref.shape[0] * dz_mm)
    ax_idd.set_xlabel("Depth [mm]", fontsize=LABEL)
    ax_idd.set_ylabel(f"IDD [{dose_unit} mm$^2$]", fontsize=LABEL)
    ax_idd.tick_params(labelsize=TICK)
    ax_idd.grid(linestyle="--", linewidth=0.5)
    ax_hu = ax_idd.twinx()
    ax_hu.plot(depth, ct_hu[:, ph, pw], color="0.55", linewidth=1.0, label="HU on the beam axis")
    ax_hu.set_ylabel("HU", fontsize=LABEL, color="0.4")
    ax_hu.set_ylim(-1100, max(1600.0, float(ct_hu[:, ph, pw].max()) + 100))
    ax_hu.tick_params(labelsize=TICK, colors="0.4")
    lines = ax_idd.get_lines() + ax_hu.get_lines()
    ax_idd.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=12, ncol=2)

    bottom = grid[n_img + 1, :].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.95], wspace=0.3)
    for c, (plane_name, axis_of) in enumerate((("Axial", lambda a: a[pd_, ph, :]),
                                                ("Sagittal", lambda a: a[pd_, :, pw]))):
        ax = fig.add_subplot(bottom[0, c])
        prof = axis_of(ref)
        lateral = (np.arange(prof.size) + 0.5 - (np.argmax(prof) + 0.5)) * dz_mm
        for i, n in enumerate(names):
            ax.plot(lateral, axis_of(doses[n]), label=ROW_LABELS.get(n, n), **_style(n, i))
        ax.set_title(f"{plane_name} lateral profile at {peak_mm:.0f} mm", fontsize=14)
        ax.set_xlabel("Lateral [mm]", fontsize=LABEL)
        if c == 0:
            ax.set_ylabel(f"Dose [{dose_unit}]", fontsize=LABEL)
            ax.legend(fontsize=11)
        ax.tick_params(labelsize=TICK)
        ax.grid(linestyle="--", linewidth=0.5)
    ax_tab = fig.add_subplot(bottom[0, 2])
    ax_tab.axis("off")
    cells = [[fmt.format(float(errors.at[n, col])) if col in errors.columns and np.isfinite(errors.at[n, col])
              else "n/a" for n in others] for col, _, fmt in METRIC_ROWS]
    table = ax_tab.table(cellText=cells, rowLabels=[lab for _, lab, _ in METRIC_ROWS],
                         colLabels=[ROW_LABELS.get(n, n) for n in others], loc="center", cellLoc="center",
                         colWidths=[0.32] * len(others))
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.7)
    ax_tab.set_title(f"Errors against {ROW_LABELS.get(reference, reference)}\n"
                     "(2 mm grid, global gamma, 10% cutoff)", fontsize=12)
    fig.suptitle(title, x=0.012, y=1 - 0.2 / fig.get_figheight(), ha="left", fontsize=18, fontweight="bold")
    paths = save_figure_as_publication_formats(fig, figure_path, formats=formats)
    plt.close(fig)
    return paths


__all__ = ["ENGINE_STYLE", "METRIC_ROWS", "ROW_LABELS", "engine_panel_figure"]
