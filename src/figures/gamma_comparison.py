"""Plan-level gamma-map figure (one column per gamma criterion).

Renders the gamma index maps from :func:`src.metrics.plan_gamma.plan_gamma` on the
CT, sliced around the isocenter, in three orthogonal views. The layout mirrors
:mod:`src.figures.plan_comparison`: a :func:`matplotlib.pyplot.subplot_mosaic`
grid with a dedicated shared-colorbar cell, height ratios derived from the data
aspect so ``aspect="auto"`` fills each cell undistorted.

Grid shape: **3 rows** (axial, sagittal, coronal) x **N columns** (one per gamma
criterion). Each cell shows the CT in grayscale with the gamma map overlaid; voxels
that were not evaluated (sub-cutoff, gamma == 0) are transparent, gamma ``<= 1``
(pass) is green and ``> 1`` (fail) is red on a ``RdYlGn_r`` scale, with the gamma
== 1 iso-contour drawn. Column titles carry the criterion and its pass rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.figures.single_beam import save_figure_as_publication_formats

__all__ = ["plan_gamma_figure"]


def _style_panel(ax, xlabels: bool, ylabels: bool) -> None:
    """Hide ticks on an image panel (gamma maps carry no spatial ticks)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(labelbottom=xlabels, labelleft=ylabels)


def _draw_gamma(ax, ct_slice, gamma_slice, vmax, cmap):
    """CT grayscale + alpha-masked gamma overlay; return the overlay image."""
    ax.imshow(ct_slice, cmap="gray", origin="lower", aspect="auto")
    # Not-evaluated voxels (gamma == 0) stay transparent; evaluated voxels opaque.
    alpha = np.where(gamma_slice > 0, 0.75, 0.0)
    im = ax.imshow(
        gamma_slice,
        cmap=cmap,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
        alpha=alpha,
    )
    # Pass/fail boundary at gamma == 1.
    if np.any(gamma_slice > 1.0) and np.any((gamma_slice > 0) & (gamma_slice <= 1.0)):
        ax.contour(
            gamma_slice, levels=[1.0], colors="black", linewidths=0.6, origin="lower"
        )
    return im


def plan_gamma_figure(
    ct: np.ndarray,
    gamma_results: Sequence[dict],
    slice_zyx: Tuple[int, int, int],
    figure_path: str,
    ct_window: Tuple[float, float] = (-200.0, 400.0),
    max_gamma: float = 2.0,
    cmap: str = "RdYlGn_r",
    dpi: int = 300,
    write_caption: bool = True,
) -> List[Path]:
    """Render the gamma maps (3 views x N criteria) around the isocenter.

    Args:
        ct: CT volume ``(z, y, x)`` in HU.
        gamma_results: List of dicts from :func:`src.metrics.plan_gamma.plan_gamma`
            (each needs ``"gamma_map"``, ``"label"``, ``"pass_rate_pct"``).
        slice_zyx: ``(z, y, x)`` slice centre (the isocenter voxel).
        figure_path: Output path stem (``.svg``/``.pdf``/``.png`` written).
        ct_window: ``(vmin, vmax)`` HU window for the CT background.
        max_gamma: Upper limit of the gamma colour scale.
        cmap: Diverging colormap (low = pass). Default ``RdYlGn_r``.
        dpi: Output resolution.
        write_caption: Write a ``*_caption.txt`` sidecar with the proposed caption.

    Returns:
        The written paths (figures, then the caption file if written).

    Raises:
        ValueError: If ``gamma_results`` is empty or a map shape mismatches the CT.
    """
    if not gamma_results:
        raise ValueError("gamma_results is empty; nothing to plot.")
    for res in gamma_results:
        if res["gamma_map"].shape != ct.shape:
            raise ValueError(
                f"gamma_map shape {res['gamma_map'].shape} != ct shape {ct.shape} "
                f"for criterion {res['label']}."
            )

    n_z, n_y, n_x = ct.shape
    zc, yc, xc = (int(slice_zyx[0]), int(slice_zyx[1]), int(slice_zyx[2]))
    zc = int(np.clip(zc, 0, n_z - 1))
    yc = int(np.clip(yc, 0, n_y - 1))
    xc = int(np.clip(xc, 0, n_x - 1))
    ct_w = np.clip(ct, *ct_window)
    n_cols = len(gamma_results)

    # One row per orthogonal view: (name, slicer).
    rows = [
        ("axial", lambda v: v[zc, :, :]),
        ("sagittal", lambda v: v[:, :, xc]),
        ("coronal", lambda v: v[:, yc, :]),
    ]
    # Height ratios from the data aspect so aspect="auto" fills cells undistorted.
    height_ratios = [n_y / n_x, n_z / n_y, n_z / n_x]
    width_ratios = [1.0] * n_cols + [0.06, 0.05]

    col_keys = [f"c{j}" for j in range(n_cols)]
    mosaic = [
        [f"{name[0]}_{key}" for key in col_keys] + [".", "cbar"]
        for name, _ in rows
    ]

    fig = plt.figure(layout="constrained", figsize=(3.3 * n_cols + 2.0, 11), dpi=dpi)
    ax = fig.subplot_mosaic(
        mosaic, width_ratios=width_ratios, height_ratios=height_ratios
    )

    gamma_im = None
    for i, (name, slicer) in enumerate(rows):
        ct_s = slicer(ct_w)
        bottom = i == len(rows) - 1
        for j, res in enumerate(gamma_results):
            key = f"{name[0]}_{col_keys[j]}"
            gamma_s = slicer(res["gamma_map"])
            gamma_im = _draw_gamma(ax[key], ct_s, gamma_s, max_gamma, cmap)
            _style_panel(ax[key], xlabels=bottom, ylabels=(j == 0))
            if j == 0:
                ax[key].set_ylabel(name, fontsize=14, weight="bold")
            if i == 0:
                ax[key].set_title(
                    f"{res['label']}\nGPR={res['pass_rate_pct']:.1f}%",
                    fontsize=13,
                    weight="bold",
                )

    cb = fig.colorbar(gamma_im, cax=ax["cbar"])
    cb.set_label("Gamma index", fontsize=13)
    cb.ax.axhline(1.0, color="black", linewidth=1.0)
    cb.ax.tick_params(labelsize=10)

    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)

    if write_caption:
        per_crit = "; ".join(
            f"{res['label']}: {res['pass_rate_pct']:.1f}%" for res in gamma_results
        )
        caption = (
            "Gamma index maps of the ADoTA-predicted plan dose against the MCsquare "
            "reference. Columns: gamma criteria (dose difference / distance-to-"
            "agreement / lower dose cutoff). Rows: axial, sagittal and coronal slices "
            f"through the isocenter voxel (z, y, x) = ({zc}, {yc}, {xc}). The gamma "
            "index is overlaid on the CT; values <= 1 (green) pass and > 1 (red) fail, "
            "with the gamma = 1 boundary contoured; sub-cutoff voxels are transparent. "
            f"Gamma pass rates: {per_crit}.\n"
        )
        caption_path = Path(str(figure_path) + "_caption.txt")
        caption_path.write_text(caption)
        paths.append(caption_path)

    return paths
