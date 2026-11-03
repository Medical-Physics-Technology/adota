"""The single-beamlet publication layout without a prediction.

:func:`ground_truth_figure` renders one simulated beamlet in the layout of
:func:`src.figures.single_beam.publication_figure` with ``beamlet_shape=True``,
keeping the rows that do not need a model: ground-truth dose over the CT and CT
with the flux channel (axial and sagittal views), the integrated depth dose, and
the beam's-eye-view strips of dose and flux at six depths. Used to inspect Monte
Carlo ground truth as it is generated, before any model has seen it.

Inputs are on the ``(D, H, W)`` 2 mm model grid, like the paper figure, and the
dose is already in its reporting unit (``dose_label``): this figure converts
nothing.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats
from src.utils.dose_grid_utils import estimate_bragg_peak

VOXEL_MM = 2


def ground_truth_figure(
    ct_hu: np.ndarray,
    flux: np.ndarray,
    dose: np.ndarray,
    figure_path: str,
    dose_label: str = "Dose [Gy/MU]",
    title: Optional[str] = None,
):
    """CT, flux and ground-truth dose of one beamlet, all ``(D, H, W)`` at 2 mm."""
    tick_fontsize = 18
    row_label_fontsize = 18
    colorbar_label_fontsize = 22
    colorbar_tick_fontsize = 18
    bev_colorbar_size = "15%"

    bp = estimate_bragg_peak(dose)
    depth_layers = np.linspace(1, min(bp[0] + 8, dose.shape[0] - 1), 6, dtype=int)
    depth_layers[-2] = bp[0]

    mosaic = "AAABBB;111222;TTTTTT;GHJKLM;NOPQRS"
    fig = plt.figure(layout="constrained", figsize=(18, 14), dpi=300)
    ax_dict = fig.subplot_mosaic(mosaic)

    dose_alpha = np.where(dose > 0.01 * dose.max(), 0.7, 0.0)
    flux_alpha = np.where(flux > 0.01 * flux.max(), 0.65, 0.0)
    norm_dose = plt.Normalize(vmin=float(dose.min()), vmax=float(dose.max()))

    def colorbar(image, ax, label, size="5%", label_x=4.7):
        return aligned_colorbar(fig, image, ax, label, label_coords=(label_x, 0.5),
                                label_fontsize=colorbar_label_fontsize, tick_fontsize=colorbar_tick_fontsize,
                                size=size)

    def depth_view(ax, plane, overlay, alpha, cmap, norm=None, row_label=None, markers=True):
        """One axial or sagittal view: CT, an overlay, the depth markers, mm ticks."""
        ax.imshow(np.rot90(plane(ct_hu)), cmap="gray")
        image = ax.imshow(np.rot90(plane(overlay)), cmap=cmap, alpha=np.rot90(plane(alpha)), norm=norm)
        for i, layer in enumerate(depth_layers):
            ax.axvline(x=layer, color="red", linewidth=2)
            if markers:         # the labels, once per column: the row below has depth ticks
                ax.text(layer + 0.5, 32 if i % 2 == 0 else -2, f"{layer * VOXEL_MM} mm", transform=ax.transData,
                        fontsize=16, color="red", va="center", ha="left")
        if row_label is not None:
            ax.set_ylabel(row_label, fontsize=row_label_fontsize)
            ax.yaxis.set_label_coords(-0.13, 0.5)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels((ax.get_yticks() * VOXEL_MM).astype(int))
        else:
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([""] * len(ax.get_yticks()))
        x_ticks = np.arange(0, dose.shape[0], 10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([""] * len(x_ticks))
        ax.set_ylim(dose.shape[1] - 0.5, -0.5)
        ax.grid(linestyle="--", linewidth=0.5, color="white")
        ax.tick_params(labelsize=tick_fontsize)
        return image

    def axial(volume):
        return volume[:, bp[1], :]

    def sagittal(volume):
        return volume[:, :, bp[2]]

    # Dose and flux rows: axial view left, sagittal view right
    ax_dict["A"].set_title("Axial view", fontsize=20, pad=20)
    ax_dict["B"].set_title("Sagittal view", fontsize=20, pad=20)
    depth_view(ax_dict["A"], axial, dose, dose_alpha, "jet", norm_dose, row_label="MCsquare\n[mm]")
    colorbar(depth_view(ax_dict["B"], sagittal, dose, dose_alpha, "jet", norm_dose), ax_dict["B"], dose_label)
    depth_view(ax_dict["1"], axial, flux, flux_alpha, "hot", row_label="CT + Flux\n[mm]", markers=False)
    colorbar(depth_view(ax_dict["2"], sagittal, flux, flux_alpha, "hot", markers=False), ax_dict["2"],
             "Flux [a.u.]")
    for key in ("1", "2"):
        x_ticks = np.arange(0, dose.shape[0], 10)
        ax_dict[key].set_xticklabels(x_ticks * VOXEL_MM, rotation=45)
        ax_dict[key].set_xlabel("Depth [mm]", fontsize=16)

    # Integrated depth dose
    ax = ax_dict["T"]
    idd = dose.sum(axis=(1, 2))
    ax.plot(idd / idd.max() * 100, label="MCsquare", color="blue")
    ax.axvline(x=bp[0], color="red", linestyle="--", label=f"Bragg peak ({bp[0] * VOXEL_MM} mm)")
    ax.set_xticks(np.arange(0, dose.shape[0] + 1, 10))
    ax.set_xticklabels(np.arange(0, (dose.shape[0] + 1) * VOXEL_MM, 10 * VOXEL_MM), rotation=45)
    ax.set_xlabel("Depth [mm]", fontsize=16)
    ax.set_ylabel("Normalized IDD [%]", fontsize=row_label_fontsize)
    ax.yaxis.set_label_coords(-0.055, 0.5)
    ax.legend(fontsize=14)
    ax.set_xlim(0, dose.shape[0] - 1)
    ax.grid(linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=tick_fontsize)

    # Beam's-eye-view strips at the marked depths
    rows = mosaic.split(";")[-2:]
    for row, (overlay, alpha, cmap, norm, label, bar_label) in zip(rows, [
            (dose, dose_alpha, "jet", norm_dose, "MCsquare - BEV", dose_label),
            (flux, flux_alpha, "hot", None, "CT + Flux - BEV", "Flux [a.u.]")]):
        for i, key in enumerate(row):
            ax, layer = ax_dict[key], depth_layers[i]
            ax.imshow(np.rot90(ct_hu[layer]), cmap="gray")
            image = ax.imshow(np.rot90(overlay[layer]), cmap=cmap, alpha=np.rot90(alpha[layer]), norm=norm)
            if i == 0:
                ax.set_ylabel(label, fontsize=row_label_fontsize)
                ax.yaxis.set_label_coords(-0.13, 0.5)
            if i == len(row) - 1:
                colorbar(image, ax, bar_label, size=bev_colorbar_size, label_x=6.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == rows[-1]:
                ax.set_xlabel(f"{layer * VOXEL_MM} mm", fontsize=16)

    if title:
        fig.suptitle(title, fontsize=20)
    output_paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return output_paths
