"""Bragg-peak estimation diagnostic figure.

:func:`plot_bp_estimation_diagnostic` shows the ground-truth dose over the CT in
axial and sagittal views beside the integral depth-dose curve, with one marker
per Bragg-peak estimator. Used to see at a glance which estimator disagrees and
where along the beam it does so.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import aligned_colorbar

# ── Default method colour palette ───────────────────────────────────────────

_DEFAULT_METHOD_COLORS: dict[str, str] = {
    "gt_idd": "#E53935",  # red
    "csda_water": "#1E88E5",  # blue
    "csda_density_corrected": "#43A047",  # green
    "ct_density_gradient": "#8E24AA",  # purple
    "r80_density_corrected": "#FB8C00",  # orange
}

_FALLBACK_COLORS = [
    "#00ACC1",
    "#D81B60",
    "#6D4C41",
    "#546E7A",
    "#FFB300",
]


def plot_bp_estimation_diagnostic(
    ct_hu: np.ndarray,
    gt_dose: np.ndarray,
    sample_id: str,
    energy_mev: float,
    output_path: Path,
    bp_estimates: dict[str, float],
    voxel_spacing_mm: float = 2.0,
    method_colors: dict[str, str] | None = None,
) -> None:
    """Diagnostic figure: GT dose on CT (axial + sagittal) + IDD with BP markers.

    Three-row, single-column layout:

    1. **Axial** – CT grayscale with GT dose overlay at the Bragg-peak
       lateral centre.
    2. **Sagittal** – same, orthogonal plane.
    3. **IDD** – normalised Integrated Depth Dose with vertical lines
       for each method's estimated Bragg-peak depth.

    Args:
        ct_hu: 3-D CT volume in HU ``(D, H, W)``.
        gt_dose: 3-D ground-truth dose ``(D, H, W)``.
        sample_id: Beamlet identifier (used in title / filename).
        energy_mev: Initial beam energy [MeV].
        output_path: Where to save the PNG figure.
        bp_estimates: ``{method_name: bp_depth_mm}``.
        voxel_spacing_mm: Isotropic voxel size [mm].
        method_colors: Optional colour overrides per method name.
    """
    from src.utils.dose_grid_utils import estimate_bragg_peak
    from src.utils.unit_conversions import to_gy

    D, H, W = ct_hu.shape

    # ── Bragg-peak location (3-D index) ─────────────────────────────────
    bp_d, bp_y, bp_x = estimate_bragg_peak(gt_dose)

    # ── Dose in Gy for display ──────────────────────────────────────────
    dose_gy = to_gy(gt_dose)
    dose_max = dose_gy.max()
    if dose_max < 1e-15:
        dose_max = 1.0

    # Alpha mask: show dose only where > 1 % of max
    alpha_threshold = 0.01 * dose_max

    # ── Colour mapping ──────────────────────────────────────────────────
    colors = dict(_DEFAULT_METHOD_COLORS)
    if method_colors is not None:
        colors.update(method_colors)
    # Assign fallback colours to unknown methods
    fb_idx = 0
    for mname in bp_estimates:
        if mname not in colors:
            colors[mname] = _FALLBACK_COLORS[fb_idx % len(_FALLBACK_COLORS)]
            fb_idx += 1

    # ── Build figure ────────────────────────────────────────────────────
    fig, ax_dict = plt.subplot_mosaic(
        "A\nB\nC",
        figsize=(10, 14),
        dpi=200,
        gridspec_kw={
            "hspace": 0.30,
            "height_ratios": [1.0, 1.0, 1.2],
        },
    )

    norm_dose = plt.Normalize(vmin=0, vmax=dose_gy.max())

    # ── Row A: Axial slice at BP lateral centre ─────────────────────────
    ax = ax_dict["A"]
    axial_ct = np.rot90(ct_hu[:, bp_y, :])
    axial_dose = np.rot90(dose_gy[:, bp_y, :])
    axial_alpha = np.where(np.rot90(dose_gy[:, bp_y, :]) > alpha_threshold, 0.7, 0.0)

    ax.imshow(axial_ct, cmap="gray", aspect="auto")
    im = ax.imshow(
        axial_dose, cmap="jet", alpha=axial_alpha, norm=norm_dose, aspect="auto"
    )
    aligned_colorbar(fig, im, ax, "Dose [Gy]", label_coords=(6.0, 0.5))
    ax.set_title("GT dose – axial slice", fontsize=12)

    # mm tick labels on x-axis (depth)
    x_ticks = np.arange(0, D + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels((x_ticks * voxel_spacing_mm).astype(int), fontsize=9)
    ax.set_xlabel("Depth [mm]", fontsize=10)
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels((y_ticks * voxel_spacing_mm).astype(int), fontsize=9)
    ax.set_ylabel("[mm]", fontsize=10)
    ax.grid(linestyle="--", linewidth=0.3, color="white", alpha=0.5)

    # ── Row B: Sagittal slice at BP lateral centre ──────────────────────
    ax = ax_dict["B"]
    sag_ct = np.rot90(ct_hu[:, :, bp_x])
    sag_dose = np.rot90(dose_gy[:, :, bp_x])
    sag_alpha = np.where(np.rot90(dose_gy[:, :, bp_x]) > alpha_threshold, 0.7, 0.0)

    ax.imshow(sag_ct, cmap="gray", aspect="auto")
    im = ax.imshow(sag_dose, cmap="jet", alpha=sag_alpha, norm=norm_dose, aspect="auto")
    aligned_colorbar(fig, im, ax, "Dose [Gy]", label_coords=(6.0, 0.5))
    ax.set_title("GT dose – sagittal slice", fontsize=12)

    x_ticks = np.arange(0, D + 1, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels((x_ticks * voxel_spacing_mm).astype(int), fontsize=9)
    ax.set_xlabel("Depth [mm]", fontsize=10)
    y_ticks = ax.get_yticks()
    ax.set_yticks(y_ticks)
    ax.set_yticklabels((y_ticks * voxel_spacing_mm).astype(int), fontsize=9)
    ax.set_ylabel("[mm]", fontsize=10)
    ax.grid(linestyle="--", linewidth=0.3, color="white", alpha=0.5)

    # ── Row C: IDD with BP markers ──────────────────────────────────────
    ax = ax_dict["C"]
    idd = gt_dose.sum(axis=(1, 2))
    idd_max = idd.max()
    idd_norm = idd / idd_max * 100.0 if idd_max > 0 else idd

    depth_slices = np.arange(D)
    depth_mm = depth_slices * voxel_spacing_mm
    ax.plot(depth_mm, idd_norm, color="blue", linewidth=1.5, label="GT IDD")

    # Vertical lines for each method
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, (mname, bp_mm) in enumerate(bp_estimates.items()):
        if bp_mm is None or np.isnan(bp_mm):
            continue
        ls = linestyles[i % len(linestyles)]
        ax.axvline(
            x=bp_mm,
            color=colors.get(mname, "gray"),
            linestyle=ls,
            linewidth=1.5,
            label=f"{mname} ({bp_mm:.0f} mm)",
        )

    ax.set_xlabel("Depth [mm]", fontsize=11)
    ax.set_ylabel("Normalised IDD [%]", fontsize=11)
    ax.set_title("Integrated Depth Dose + BP estimates", fontsize=12)
    ax.set_xlim(0, (D - 1) * voxel_spacing_mm)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=10)

    fig.suptitle(
        f"BP Estimation Diagnostic – {sample_id}\n" f"Energy: {energy_mev:.1f} MeV",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
