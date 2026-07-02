"""Plan-level dose comparison figures (ADoTA vs MCsquare).

Renders two full-grid dose maps (e.g. the accumulated ``Dose_ADoTA.mhd`` and the
MCsquare ``Dose.mhd``) on the CT in three orthogonal views plus a depth profile.
Layout is a :func:`matplotlib.pyplot.subplot_mosaic` grid with **dedicated
colorbar cells** so the two global colorbars (a shared dose scale and an absolute
difference scale) are cleanly placed and aligned. The dose panels share one scale;
the difference is shown in absolute dose (e.g. Gy) on a symmetric, data-driven
colour scale (a robust percentile of the difference, so it adapts to the plan).

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


def _overlay_contour(ax, ct_slice, dose_slice, vmax, levels_frac, label_lines):
    """Draw a CT grayscale slice with filled isodose contours (clinical view).

    The dose is shown as translucent ``contourf`` bands at ``levels_frac``
    fractions of the shared dose peak over the CT (origin matches the ``imshow``
    CT so the two align), optionally with thin labeled isodose lines on top.
    Returns the filled ``QuadContourSet`` so it can drive the shared colorbar.
    """
    ax.imshow(ct_slice, cmap="gray", origin="lower", aspect="auto")
    levels = [f * vmax for f in levels_frac]
    # extend="max" keeps the level set identical across panels (shared colorbar);
    # values above the top isodose fall into the over-range colour.
    cf = ax.contourf(dose_slice, levels=levels, cmap="jet", alpha=0.6, extend="max")
    if label_lines:
        smax = float(dose_slice.max())
        line_levels = [lv for lv in levels if 0.0 < lv < smax]
        if line_levels:
            cl = ax.contour(
                dose_slice, levels=line_levels, colors="k", linewidths=0.5, alpha=0.7
            )
            ax.clabel(cl, fmt=lambda x: f"{x / vmax * 100:.0f}%", fontsize=7, inline=True)
    return cf


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
    diff_percentile: float = 99.0,
    dpi: int = 300,
    write_caption: bool = True,
    dose_render: str = "image",
    isodose_levels_pct: Tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0, 95.0, 100.0),
    isodose_labels: bool = True,
    single_colorbar: bool = False,
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
        diff_percentile: Symmetric colour limit for the difference panel, taken as
            this percentile of ``|dose_a - dose_b|`` over the high-dose region, so a
            single outlier voxel at a gradient edge cannot blow out the scale
            (``100`` = the literal max). Shown in ``dose_unit``, centred on 0.
        dpi: Output resolution.
        write_caption: Write a ``*_caption.txt`` sidecar with the proposed caption.
        dose_render: How to render the two dose columns. ``"image"`` (default) draws
            the alpha-masked filled dose overlay; ``"contour"`` draws filled isodose
            contours (``contourf``) at ``isodose_levels_pct`` of the shared peak --
            the clinical isodose view. The difference column stays an ``imshow``
            heatmap in both modes.
        isodose_levels_pct: Isodose levels as percentages of the shared dose peak
            (used only when ``dose_render="contour"``).
        isodose_labels: Overlay thin labeled isodose lines on the filled contours
            (``dose_render="contour"`` only).
        single_colorbar: When ``True`` a **single** colorbar (the shared dose scale)
            covers the whole figure: the difference panel shows the **absolute error**
            ``|dose_a - dose_b|`` on the same 0->peak dose colormap as the dose panels,
            so a small error sinks to the bottom of the scale and reads as negligible
            *relative to the dose*. The signed difference is not shown (its numeric
            agreement stays in the caption). Default ``False`` (a separate signed,
            data-scaled difference colorbar that instead highlights the error).

    Returns:
        The written paths (figures, then the caption file if written).
    """
    if dose_render not in ("image", "contour"):
        raise ValueError(
            f"dose_render must be 'image' or 'contour', got {dose_render!r}."
        )
    if not (ct.shape == dose_a.shape == dose_b.shape):
        raise ValueError(
            f"ct/dose_a/dose_b must share shape, got {ct.shape}, "
            f"{dose_a.shape}, {dose_b.shape}."
        )
    levels_frac = sorted(p / 100.0 for p in isodose_levels_pct)

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

    # Symmetric, data-driven colour limit for the difference panels: the
    # ``diff_percentile`` of |a - b| over the high-dose region. Robust to single-
    # voxel outliers at gradient edges, symmetric about 0 (white = perfect
    # agreement), and small when ADoTA agrees well -- so the colorbar's own scale
    # communicates the level of agreement.
    _diff = dose_a - dose_b
    _hi = dose_b > metrics["threshold"] * ref_peak
    diff_lim = max(
        float(np.percentile(np.abs(_diff[_hi]), diff_percentile))
        if _hi.any() else float(np.abs(_diff).max()),
        1e-9,
    )

    # One row per orthogonal view: (name, slicer, grid colour for the diff panel).
    rows = [
        ("axial", lambda v: v[zc, :, :]),
        ("coronal", lambda v: v[:, yc, :]),
        ("sagittal", lambda v: v[:, :, xc]),
    ]
    # Height ratios from the data aspect so aspect="auto" fills cells undistorted.
    height_ratios = [n_y / n_x, n_z / n_x, n_z / n_y, 0.55 * (n_y / n_x)]

    # Two colorbars (dose + difference) by default; ``single_colorbar`` drops the
    # difference colorbar column so only the shared dose scale remains (its +/-
    # range moves into the difference panel title instead).
    if single_colorbar:
        width_ratios = [1.0, 1.0, 1.0, 0.10, 0.05]
        mosaic = [
            ["ax_a", "ax_b", "ax_d", ".", "cdose"],
            ["co_a", "co_b", "co_d", ".", "cdose"],
            ["sa_a", "sa_b", "sa_d", ".", "cdose"],
            ["prof", "prof", "prof", ".", "."],
        ]
    else:
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

    def _draw_dose(axx, ct_s, d_s):
        if dose_render == "contour":
            return _overlay_contour(axx, ct_s, d_s, vmax, levels_frac, isodose_labels)
        return _overlay(axx, ct_s, d_s, vmax, dose_threshold)

    dose_im = None
    diff_im = None
    for i, ((name, slicer), (ka, kb, kd)) in enumerate(zip(rows, keys)):
        bottom = i == len(rows) - 1
        ct_s, a_s, b_s = slicer(ct_w), slicer(dose_a), slicer(dose_b)

        dose_im = _draw_dose(ax[ka], ct_s, a_s)
        _style_panel(ax[ka], grid, "white", xlabels=bottom, ylabels=True)
        ax[ka].set_ylabel(name, fontsize=15, weight="bold")

        _draw_dose(ax[kb], ct_s, b_s)
        _style_panel(ax[kb], grid, "white", xlabels=bottom, ylabels=False)

        if single_colorbar:
            # |error| on the SAME dose scale/colormap as the panels, so the single
            # dose colorbar covers all three and a small error sinks to the bottom
            # of the 0->peak scale (visually negligible vs the dose).
            _draw_dose(ax[kd], ct_s, np.abs(a_s - b_s))
            _style_panel(ax[kd], grid, "white", xlabels=bottom, ylabels=False)
        else:
            diff_im = ax[kd].imshow(
                a_s - b_s, cmap="seismic", origin="lower", aspect="auto",
                vmin=-diff_lim, vmax=diff_lim,
            )
            _style_panel(ax[kd], grid, "0.4", xlabels=bottom, ylabels=False)

    ax["ax_a"].set_title(labels[0], fontsize=17, weight="bold")
    ax["ax_b"].set_title(labels[1], fontsize=17, weight="bold")
    # On the shared dose scale the panel is the absolute error |a - b|; otherwise
    # it is the signed difference with its own diverging colorbar.
    diff_title = (
        f"|{labels[0]} - {labels[1]}|" if single_colorbar
        else f"{labels[0]} - {labels[1]}"
    )
    ax["ax_d"].set_title(diff_title, fontsize=17, weight="bold")

    cb_dose = fig.colorbar(dose_im, cax=ax["cdose"])
    cb_dose.set_label(f"Dose [{dose_unit}]", fontsize=19)
    cb_dose.ax.tick_params(labelsize=15)
    if not single_colorbar:
        cb_diff = fig.colorbar(diff_im, cax=ax["cdiff"])
        cb_diff.set_label(f"{labels[0]} - {labels[1]} [{dose_unit}]", fontsize=18)
        cb_diff.ax.tick_params(labelsize=15)

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
        if dose_render == "contour":
            lv = ", ".join(f"{int(round(p))}" for p in sorted(isodose_levels_pct))
            dose_desc = (
                f"shown as filled isodose contours at {lv}% of the shared peak "
                f"({dose_unit}); the difference is given"
            )
        else:
            dose_desc = (
                f"shown in {dose_unit} on a shared colour scale; the difference is given"
            )
        if single_colorbar:
            diff_desc = (
                f"as the absolute error |{labels[0]} - {labels[1]}| on the same "
                f"0-{vmax:.2f} {dose_unit} dose colour scale, so a small error reads "
                f"as negligible relative to the dose"
            )
        else:
            diff_desc = (
                f"in {dose_unit} on a symmetric colour scale clipped at "
                f"+/-{diff_lim:.3f} {dose_unit} (the {int(round(diff_percentile))}th "
                f"percentile of |{labels[0]} - {labels[1]}| over the high-dose region)"
            )
        caption = (
            f"Comparison of the {labels[0]}-predicted plan dose against the "
            f"{labels[1]} reference. Columns: {labels[0]}, {labels[1]} and their "
            f"difference ({labels[0]} - {labels[1]}). Rows: axial, coronal and "
            f"sagittal slices through voxel (z, y, x) = ({zc}, {yc}, {xc}). Dose is "
            f"{dose_desc} {diff_desc}. Bottom: integrated depth "
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
