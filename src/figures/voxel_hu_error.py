"""Dose error per voxel against its HU: rows = dose bands, columns = sites.

Reuses the summary style of :mod:`src.figures.ct_vs_error` (``_RC_SUMMARY``, ink and grid
colours), the engine colours of :mod:`src.figures.engine_drivers` (``ENGINE_STYLE``) and
saves through :func:`src.figures.axes_utils.save_figure_as_publication_formats`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.figures.axes_utils import save_figure_as_publication_formats
from src.figures.ct_vs_error import _RC_SUMMARY, GRID, INK, MUTED
from src.figures.engine_drivers import ENGINE_STYLE

__all__ = ["hu_error_grid"]

_STYLE_KEY = {"ADoTA": "ADoTA", "Pencil beam": "PBA"}
_CLASS_EDGES = (-500.0, 300.0)


def _style(engine: str) -> dict:
    style = dict(ENGINE_STYLE.get(_STYLE_KEY.get(engine, engine), {"color": "C2"}))
    style["linestyle"] = "-"
    style["linewidth"] = 2.0
    return style


def hu_error_grid(table: pd.DataFrame, *, value: str, ylabel: str, title: str, figure_path,
                  bands: Sequence[str], sites: Sequence[str] = ("thoracic", "abdominal", "pelvic"),
                  engines: Sequence[str] = ("ADoTA", "Pencil beam"), bin_width: float = 50.0,
                  min_patients: int = 5, zero_line: bool = False, x_max: Optional[float] = 2000.0,
                  note: str = "", formats: Sequence[str] = ("png", "pdf")) -> List[Path]:
    """One panel per dose band (row) and site (column): ``<value>_mean`` per HU bin for each
    engine with its ``<value>_ci_low/high`` band, and below it the voxel count per bin.

    ``table`` is the pooled CSV (columns engine, site, band, bin_low_hu, voxels, patients and
    the value columns). Cells seen in fewer than ``min_patients`` patients are not drawn.
    The x axis is the bin centre in HU.
    """
    counts_col = "local_voxels" if value == "local" else "voxels"
    data = table[table["patients"] >= min_patients].copy()
    data["hu"] = data["bin_low_hu"] + bin_width / 2.0
    x_lo = -1050.0
    x_hi = min(float(data["bin_low_hu"].max() + bin_width), x_max) if x_max else float(data["hu"].max())
    heights = [3, 1] * len(bands)
    with plt.rc_context(_RC_SUMMARY):
        fig = plt.figure(figsize=(13.0, 2.9 * len(bands) + 1.4))
        grid = fig.add_gridspec(2 * len(bands), len(sites), height_ratios=heights, hspace=0.12, wspace=0.12)
        first = None
        for r, band in enumerate(bands):
            row_axes = []
            for c, site in enumerate(sites):
                ax = fig.add_subplot(grid[2 * r, c], sharex=first, sharey=row_axes[0] if row_axes else None)
                strip = fig.add_subplot(grid[2 * r + 1, c], sharex=ax)
                first = first or ax
                row_axes.append(ax)
                part = data[(data["site"] == site) & (data["band"] == band)]
                for engine in engines:
                    eng = part[part["engine"] == engine].sort_values("hu")
                    if eng.empty:
                        continue
                    style = _style(engine)
                    ax.fill_between(eng["hu"], eng[f"{value}_ci_low"], eng[f"{value}_ci_high"],
                                    color=style["color"], alpha=0.18, linewidth=0, step="mid")
                    ax.plot(eng["hu"], eng[f"{value}_mean"], drawstyle="steps-mid", label=engine, **style)
                ref = part[part["engine"] == engines[0]].sort_values("hu")
                strip.bar(ref["hu"], ref[counts_col], width=bin_width, color="#bdbdbd", linewidth=0)
                strip.set_yscale("log")
                for axis in (ax, strip):
                    for edge in _CLASS_EDGES:
                        axis.axvline(edge, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)
                    axis.grid(axis="y", color=GRID, linewidth=0.8)
                if zero_line:
                    ax.axhline(0.0, color=INK, linewidth=0.9)
                ax.tick_params(labelbottom=False)
                strip.tick_params(axis="y", labelsize=8)
                if c > 0:
                    ax.tick_params(labelleft=False)
                    strip.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(f"dose {band}\nof max\n{ylabel}", fontsize=10)
                    strip.set_ylabel("voxels", fontsize=8, color=MUTED)
                if r == 0:
                    ax.set_title(site, loc="left")
                if r < len(bands) - 1:
                    strip.tick_params(labelbottom=False)
                else:
                    strip.set_xlabel("voxel HU (bin centre)")
            row_axes[0].set_xlim(x_lo, x_hi)
            row_axes[0].set_xticks([t for t in (-1000, -500, 0, 500, 1000, 1500, 2000, 2500, 3000) if t < x_hi - 150])
        for x, label in ((-775.0, "lung / air"), (-100.0, "soft tissue"), (650.0, "bone")):
            first.text(x, 0.98, label, transform=first.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                       color=MUTED)
        handles = [plt.Line2D([], [], **_style(e), label=e) for e in engines]
        fig.legend(handles=handles, loc="upper right", ncol=len(engines), frameon=False, bbox_to_anchor=(0.99, 1.0))
        fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
        if note:
            fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=9, color=MUTED)
        height = fig.get_figheight()
        fig.subplots_adjust(left=0.1, right=0.99, top=1.0 - 0.95 / height, bottom=(1.0 if note else 0.7) / height)
        paths = save_figure_as_publication_formats(fig, str(Path(figure_path)), formats=tuple(formats))
        plt.close(fig)
    return paths
