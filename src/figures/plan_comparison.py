"""Plan-level dose comparison figures (ADoTA vs MCsquare).

Renders two full-grid dose maps (e.g. the accumulated ``Dose_ADoTA.mhd`` and the
MCsquare ``Dose.mhd``) on the CT in three orthogonal views plus a depth profile.
Layout is a :func:`matplotlib.pyplot.subplot_mosaic` grid with **dedicated
colorbar cells** so the two global colorbars (a shared dose scale and a
percentage difference scale) are cleanly placed and aligned. The dose panels
share one scale; the difference is shown as a percentage of the reference peak.

The figure carries no metric text; a sidecar ``*_caption.txt`` with the proposed
caption (including the quantitative agreement) is written next to it. Reuses the
save helper from :mod:`src.figures.single_beam`; ``publication_figure`` is
untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from src.figures.single_beam import save_figure_as_publication_formats

__all__ = ["dose_comparison_metrics", "plan_dose_comparison"]


def dose_comparison_metrics(
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """Summary metrics comparing ``dose_a`` to reference ``dose_b``.

    Args:
        dose_a: Dose map under test (e.g. ADoTA), ``(z, y, x)``.
        dose_b: Reference dose map (e.g. MCsquare), same shape.
        threshold: Fraction of the reference max defining the high-dose mask the
            error metrics are computed over.

    Returns:
        Dict with peak/integral ratios and RMSE / mean-absolute-percent error over
        the high-dose region (relative to the reference peak).
    """
    b_max = float(dose_b.max())
    mask = dose_b > threshold * b_max if b_max > 0 else np.zeros_like(dose_b, bool)
    diff = dose_a - dose_b
    rmse = float(np.sqrt(np.mean(diff[mask] ** 2))) if mask.any() else float("nan")
    mape = (
        float(np.mean(np.abs(diff[mask])) / b_max * 100.0)
        if mask.any() and b_max > 0
        else float("nan")
    )
    return {
        "peak_ratio": float(dose_a.max() / b_max) if b_max > 0 else float("nan"),
        "integral_ratio": float(dose_a.sum() / max(float(dose_b.sum()), 1e-30)),
        "rmse_high_dose": rmse,
        "mape_high_dose_pct": mape,
        "threshold": threshold,
    }


def _style_panel(ax, grid: bool, grid_color: str, xlabels: bool, ylabels: bool) -> None:
    """Apply (or clear) shared ticks + grid on an image panel."""
    if not grid:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.grid(True, linestyle=":", linewidth=0.5, color=grid_color, alpha=0.6)
    ax.tick_params(labelsize=9, labelbottom=xlabels, labelleft=ylabels)


def _overlay(ax, ct_slice, dose_slice, vmax, threshold):
    """Draw a CT grayscale slice with an alpha-masked dose overlay (filled cell)."""
    ax.imshow(ct_slice, cmap="gray", origin="lower", aspect="auto")
    alpha = np.where(dose_slice > threshold * vmax, 0.6, 0.0)
    return ax.imshow(
        dose_slice, cmap="jet", origin="lower", aspect="auto", vmin=0, vmax=vmax, alpha=alpha
    )


def plan_dose_comparison(
    ct: np.ndarray,
    dose_a: np.ndarray,
    dose_b: np.ndarray,
    figure_path: str,
    labels: Tuple[str, str] = ("ADoTA", "MCsquare"),
    slice_zyx: Optional[Tuple[int, int, int]] = None,
    ct_window: Tuple[float, float] = (-200.0, 400.0),
    dose_threshold: float = 0.05,
    dose_unit: str = "Gy",
    grid: bool = True,
    diff_clip_pct: float = 20.0,
    dpi: int = 300,
    write_caption: bool = True,
) -> list[Path]:
    """Compare two plan dose maps on the CT (axial/coronal/sagittal + profile).

    Args:
        ct: CT volume ``(z, y, x)`` in HU.
        dose_a: First dose map (e.g. ADoTA), same shape as ``ct``.
        dose_b: Reference dose map (e.g. MCsquare), same shape.
        figure_path: Output path stem (``.svg``/``.pdf``/``.png`` written).
        labels: ``(name_a, name_b)`` for titles / caption.
        slice_zyx: ``(z, y, x)`` index for the three views; defaults to the
            reference (``dose_b``) peak voxel.
        ct_window: ``(vmin, vmax)`` HU window for the CT background.
        dose_threshold: Fraction of the shared max below which the dose overlay is
            transparent.
        dose_unit: Unit label for the dose colorbar (doses are expected pre-scaled).
        grid: Show shared ticks + grid on the image panels.
        diff_clip_pct: Symmetric clip (percent of reference peak) for the
            difference colorbar.
        dpi: Output resolution.
        write_caption: Write a ``*_caption.txt`` sidecar with the proposed caption.

    Returns:
        The written paths (figures, then the caption file if written).
    """
    if not (ct.shape == dose_a.shape == dose_b.shape):
        raise ValueError(
            f"ct/dose_a/dose_b must share shape, got {ct.shape}, "
            f"{dose_a.shape}, {dose_b.shape}."
        )

    n_z, n_y, n_x = ct.shape
    zc, yc, xc = (
        slice_zyx
        if slice_zyx is not None
        else tuple(int(i) for i in np.unravel_index(int(dose_b.argmax()), dose_b.shape))
    )
    ct_w = np.clip(ct, *ct_window)
    vmax = max(float(dose_a.max()), float(dose_b.max()), 1e-12)
    ref_peak = max(float(dose_b.max()), 1e-30)
    metrics = dose_comparison_metrics(dose_a, dose_b, threshold=max(dose_threshold, 0.1))

    # One row per orthogonal view: (name, slicer, grid colour for the diff panel).
    rows = [
        ("axial", lambda v: v[zc, :, :]),
        ("coronal", lambda v: v[:, yc, :]),
        ("sagittal", lambda v: v[:, :, xc]),
    ]
    # Height ratios from the data aspect so aspect="auto" fills cells undistorted.
    height_ratios = [n_y / n_x, n_z / n_x, n_z / n_y, 0.55 * (n_y / n_x)]
    width_ratios = [1.0, 1.0, 1.0, 0.10, 0.05, 0.22, 0.05]

    mosaic = [
        ["ax_a", "ax_b", "ax_d", ".", "cdose", ".", "cdiff"],
        ["co_a", "co_b", "co_d", ".", "cdose", ".", "cdiff"],
        ["sa_a", "sa_b", "sa_d", ".", "cdose", ".", "cdiff"],
        ["prof", "prof", "prof", ".", ".", ".", "."],
    ]
    keys = [("ax_a", "ax_b", "ax_d"), ("co_a", "co_b", "co_d"), ("sa_a", "sa_b", "sa_d")]

    fig = plt.figure(layout="constrained", figsize=(17, 16), dpi=dpi)
    ax = fig.subplot_mosaic(mosaic, width_ratios=width_ratios, height_ratios=height_ratios)

    dose_im = None
    diff_im = None
    for i, ((name, slicer), (ka, kb, kd)) in enumerate(zip(rows, keys)):
        bottom = i == len(rows) - 1
        ct_s, a_s, b_s = slicer(ct_w), slicer(dose_a), slicer(dose_b)

        dose_im = _overlay(ax[ka], ct_s, a_s, vmax, dose_threshold)
        _style_panel(ax[ka], grid, "white", xlabels=bottom, ylabels=True)
        ax[ka].set_ylabel(name, fontsize=15, weight="bold")

        _overlay(ax[kb], ct_s, b_s, vmax, dose_threshold)
        _style_panel(ax[kb], grid, "white", xlabels=bottom, ylabels=False)

        diff_pct = (a_s - b_s) / ref_peak * 100.0
        diff_im = ax[kd].imshow(
            diff_pct, cmap="seismic", origin="lower", aspect="auto",
            vmin=-diff_clip_pct, vmax=diff_clip_pct,
        )
        _style_panel(ax[kd], grid, "0.4", xlabels=bottom, ylabels=False)

    ax["ax_a"].set_title(labels[0], fontsize=17, weight="bold")
    ax["ax_b"].set_title(labels[1], fontsize=17, weight="bold")
    ax["ax_d"].set_title(f"{labels[0]} - {labels[1]}", fontsize=17, weight="bold")

    cb_dose = fig.colorbar(dose_im, cax=ax["cdose"])
    cb_dose.set_label(f"Dose [{dose_unit}]", fontsize=15)
    cb_dose.ax.tick_params(labelsize=11)
    cb_diff = fig.colorbar(diff_im, cax=ax["cdiff"])
    cb_diff.set_label(f"{labels[0]} - {labels[1]} [% of {labels[1]} peak]", fontsize=14)
    cb_diff.ax.tick_params(labelsize=11)

    # Integrated depth dose along x, normalized to the reference (dose_b) peak.
    idd_a = dose_a.sum(axis=(0, 1))
    idd_b = dose_b.sum(axis=(0, 1))
    idd_ref = max(float(idd_b.max()), 1e-30)
    axp = ax["prof"]
    axp.plot(idd_a / idd_ref * 100.0, label=labels[0], color="tab:orange", lw=2)
    axp.plot(idd_b / idd_ref * 100.0, label=labels[1], color="tab:blue", lw=2, ls="--")
    axp.axvline(xc, color="red", ls=":", lw=1, label=f"slice x={xc}")
    axp.set_xlabel("Depth x [voxels]", fontsize=13)
    axp.set_ylabel("Normalized IDD [%]", fontsize=13)
    axp.set_title(
        f"Integrated depth dose along x (lateral sum over z, y), "
        f"normalized to {labels[1]} peak",
        fontsize=13,
    )
    axp.grid(True, linestyle=":", linewidth=0.5)
    axp.tick_params(labelsize=11)
    axp.legend(fontsize=11)

    paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)

    if write_caption:
        caption = (
            f"Comparison of the {labels[0]}-predicted plan dose against the "
            f"{labels[1]} reference. Columns: {labels[0]}, {labels[1]} and their "
            f"difference ({labels[0]} - {labels[1]}). Rows: axial, coronal and "
            f"sagittal slices through voxel (z, y, x) = ({zc}, {yc}, {xc}). Dose is "
            f"shown in {dose_unit} on a shared colour scale; the difference is given "
            f"as a percentage of the {labels[1]} peak dose. Bottom: integrated depth "
            f"dose along x (lateral sum over z and y), normalised to the {labels[1]} "
            f"peak. Quantitative agreement over voxels above "
            f"{int(metrics['threshold'] * 100)}% of the peak dose: peak ratio "
            f"{metrics['peak_ratio']:.3f}, integral ratio "
            f"{metrics['integral_ratio']:.3f}, RMSE {metrics['rmse_high_dose']:.3f} "
            f"{dose_unit}, MAPE {metrics['mape_high_dose_pct']:.2f}%.\n"
        )
        caption_path = Path(str(figure_path) + "_caption.txt")
        caption_path.write_text(caption)
        paths.append(caption_path)

    return paths
