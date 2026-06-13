from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from src.utils.dose_grid_utils import estimate_bragg_peak

from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy


# Helper function used for visualization in the following examples
def identify_axes(ax_dict: dict[str, plt.Axes], fontsize: int = 48) -> None:
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def aligned_colorbar(
    fig,
    ct_ax,
    ax,
    label: str,
    label_coords: tuple = (4.7, 0.5),
    label_fontsize: int = 18,
    tick_fontsize: int = 15,
    size: str = "5%",
):
    """Function to create an aligned colorbar for the given axes.

    Args:
        fig (_type_): Figure object to which the colorbar will be added.
        ct_ax (_type_): Axes object for which the colorbar is aligned.
        ax (_type_): Axes object to which the colorbar is aligned.
        label (str): Label for the colorbar.

    Returns:
        _type_: _description_
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.05)
    colorbar = fig.colorbar(ct_ax, cax=cax, orientation="vertical")
    colorbar.set_label(label, fontsize=label_fontsize, labelpad=10)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_label_coords(label_coords[0], label_coords[1])
    colorbar.ax.yaxis.set_tick_params(labelsize=tick_fontsize)
    return colorbar


def save_figure_as_publication_formats(fig, figure_path: str) -> list[Path]:
    output_path = Path(figure_path)
    output_paths = [
        output_path.with_suffix(f".{extension}") for extension in ("svg", "pdf", "png")
    ]
    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", dpi=300)
    return output_paths


def compare_two_inputs(
    original_input: np.ndarray,
    rotated_input: np.ndarray,
    original_dose: np.ndarray,
    rotated_dose: np.ndarray,
    initial_energy: float,
    beamlet_angles: tuple[float, float],
    figure_path: str,
    rotation_angles: tuple[float, float] | None = None,
) -> None:
    """Compare original and rotated CT/flux/dose inputs.

    Args:
        original_input: Two-channel array ``(2, D, H, W)`` with CT and flux.
        rotated_input: Rotated two-channel array ``(2, D, H, W)``.
        original_dose: Ground-truth dose array ``(D, H, W)`` before rotation.
        rotated_dose: Ground-truth dose array ``(D, H, W)`` after rotation.
        initial_energy: Beam energy in MeV.
        beamlet_angles: Original beamlet angles ``(ba0, ba1)`` in degrees.
        figure_path: Output figure path.
        rotation_angles: Applied rotations ``(rotation_y, rotation_x)`` in degrees.
    """
    if original_input.shape[0] != 2 or rotated_input.shape[0] != 2:
        raise ValueError("Expected original_input and rotated_input with shape (2, D, H, W).")
    if original_input.shape != rotated_input.shape:
        raise ValueError(
            f"Original and rotated inputs must have the same shape, got "
            f"{original_input.shape} and {rotated_input.shape}."
        )
    if original_dose.shape != original_input.shape[1:]:
        raise ValueError(
            f"original_dose must have shape {original_input.shape[1:]}, got {original_dose.shape}."
        )
    if rotated_dose.shape != original_input.shape[1:]:
        raise ValueError(
            f"rotated_dose must have shape {original_input.shape[1:]}, got {rotated_dose.shape}."
        )

    original_ct = original_input[0]
    original_flux = original_input[1]
    rotated_ct = rotated_input[0]
    rotated_flux = rotated_input[1]
    depth, height, width = original_ct.shape
    center_h = height // 2
    center_w = width // 2
    depth_layers_to_disp = np.linspace(1, max(1, depth - 2), 6, dtype=int)

    flux_max = max(float(np.max(original_flux)), float(np.max(rotated_flux)), 1e-12)
    flux_alpha_threshold = 0.01 * flux_max
    dose_max = max(float(np.max(original_dose)), float(np.max(rotated_dose)), 1e-12)
    dose_alpha_threshold = 0.01 * dose_max

    def _view_slice(volume: np.ndarray, view: str) -> np.ndarray:
        if view == "axial":
            return np.rot90(volume[:, center_h, :])
        if view == "sagittal":
            return np.rot90(volume[:, :, center_w])
        raise ValueError(f"Unsupported view: {view}")

    def _format_view_axis(ax, view: str) -> None:
        for depth_idx in depth_layers_to_disp:
            ax.axvline(x=depth_idx, color="red", linewidth=1.5)
        ax.set_xlabel("Depth [voxels]", fontsize=12)
        ax.set_ylabel("Width [voxels]" if view == "axial" else "Height [voxels]", fontsize=12)
        ax.grid(linestyle="--", linewidth=0.5, color="white")
        ax.tick_params(labelsize=11)

    def _plot_ct_overlay(
        ax,
        ct: np.ndarray,
        overlay: np.ndarray,
        view: str,
        title: str,
        overlay_cmap: str,
        overlay_max: float,
        alpha_threshold: float,
    ):
        ct_slice = _view_slice(ct, view)
        overlay_slice = _view_slice(overlay, view)
        ax.imshow(ct_slice, cmap="gray", aspect="auto")
        overlay_alpha = np.where(overlay_slice > alpha_threshold, 0.65, 0.0)
        overlay_im = ax.imshow(
            overlay_slice,
            cmap=overlay_cmap,
            alpha=overlay_alpha,
            vmin=0,
            vmax=overlay_max,
            aspect="auto",
        )
        ax.set_title(title, fontsize=15, weight="bold")
        _format_view_axis(ax, view)
        return overlay_im

    idd_gt = (
        np.sum(original_dose, axis=(1, 2))
        / max(float(np.max(np.sum(original_dose, axis=(1, 2)))), 1e-12)
        * 100
    )
    idd_pred = (
        np.sum(rotated_dose, axis=(1, 2))
        / max(float(np.max(np.sum(original_dose, axis=(1, 2)))), 1e-12)
        * 100
    )

    fig = plt.figure(layout="constrained", figsize=(14, 18), dpi=300)
    ax_dict = fig.subplot_mosaic("AB;CD;EF;GH;II")

    flux_im = _plot_ct_overlay(
        ax_dict["A"], original_ct, original_flux, "axial", "Original CT + flux | axial", "hot", flux_max, flux_alpha_threshold
    )
    _plot_ct_overlay(
        ax_dict["B"], original_ct, original_flux, "sagittal", "Original CT + flux | sagittal", "hot", flux_max, flux_alpha_threshold
    )
    dose_im = _plot_ct_overlay(
        ax_dict["C"], original_ct, original_dose, "axial", "Original CT + dose | axial", "jet", dose_max, dose_alpha_threshold
    )
    _plot_ct_overlay(
        ax_dict["D"], original_ct, original_dose, "sagittal", "Original CT + dose | sagittal", "jet", dose_max, dose_alpha_threshold
    )
    _plot_ct_overlay(
        ax_dict["E"], rotated_ct, rotated_flux, "axial", "Rotated CT + flux | axial", "hot", flux_max, flux_alpha_threshold
    )
    _plot_ct_overlay(
        ax_dict["F"], rotated_ct, rotated_flux, "sagittal", "Rotated CT + flux | sagittal", "hot", flux_max, flux_alpha_threshold
    )
    _plot_ct_overlay(
        ax_dict["G"], rotated_ct, rotated_dose, "axial", "Rotated CT + dose | axial", "jet", dose_max, dose_alpha_threshold
    )
    _plot_ct_overlay(
        ax_dict["H"], rotated_ct, rotated_dose, "sagittal", "Rotated CT + dose | sagittal", "jet", dose_max, dose_alpha_threshold
    )
    aligned_colorbar(fig, flux_im, ax_dict["B"], "Flux [a.u.]", label_coords=(4.2, 0.5))
    aligned_colorbar(fig, dose_im, ax_dict["D"], "Dose [a.u.]", label_coords=(4.2, 0.5))

    ax = ax_dict["I"]
    ax.plot(idd_gt, label="GT dose IDD before rotation", color="blue", linewidth=2)
    ax.plot(idd_pred, label="GT dose IDD after rotation", color="orange", linestyle="--", linewidth=2)
    ax.set_xlabel("Depth [voxels]", fontsize=13)
    ax.set_ylabel("Normalized IDD [%]", fontsize=13)
    ax.set_xlim(0, depth - 1)
    ax.grid(linestyle="--", linewidth=0.5)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

    rotation_text = ""
    if rotation_angles is not None:
        rotation_text = f" | applied rotations Y={rotation_angles[0]:.3f} deg, X={rotation_angles[1]:.3f} deg"
    fig.suptitle(
        f"Energy: {initial_energy:.2f} MeV | beamlet angles ba0={beamlet_angles[0]:.3f} deg, "
        f"ba1={beamlet_angles[1]:.3f} deg{rotation_text}",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(figure_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def publication_figure(
    ct_input: np.ndarray,  # Change variable - this is 2 channel ct-flux pair.
    initial_energy: float,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    figure_path: str,
    rmse: float,
    mape: float,
    gpr: float,
    gamma_params: dict = {
        "dose_percent_threshold": 1,
        "distance_mm_threshold": 3,
        "lower_percent_dose_cutoff": 0.1,
    },
    lateral_profiles_per_slice: bool = False,
    beamlet_shape: bool = False,
):
    tick_fontsize = 18
    row_label_fontsize = 18
    colorbar_label_fontsize = 22
    colorbar_tick_fontsize = 18
    row_label_x = -0.13
    wide_row_label_x = -0.055
    bev_colorbar_size = "15%"

    def set_row_ylabel(ax, label: str, label_x: float = row_label_x) -> None:
        ax.set_ylabel(label, fontsize=row_label_fontsize)
        ax.yaxis.set_label_coords(label_x, 0.5)

    def publication_colorbar(
        fig,
        ct_ax,
        ax,
        label: str,
        label_coords: tuple = (4.7, 0.5),
        size: str = "5%",
    ):
        return aligned_colorbar(
            fig,
            ct_ax,
            ax,
            label,
            label_coords=label_coords,
            label_fontsize=colorbar_label_fontsize,
            tick_fontsize=colorbar_tick_fontsize,
            size=size,
        )

    # Preprocessing - handled here for simplicity
    bp_idx_gt = estimate_bragg_peak(ground_truth)
    bp_idx_pred = estimate_bragg_peak(prediction)

    if not lateral_profiles_per_slice:
        if beamlet_shape:
            # Insert CT + flux overlay row below GT dose row
            mosaic = "AAABBB;111222;CCCDDD;EEEFFF;TTTTTT;GHJKLM;NOPQRS;UVWXYZ"
        else:
            mosaic = "AAABBB;CCCDDD;EEEFFF;TTTTTT;GHJKLM;NOPQRS;UVWXYZ"
    else:
        # Last row represents lateral profiles per displayed slice in the rows above
        mosaic = "AAABBB;CCCDDD;EEEFFF;TTTTTT;GHIJKL;MNOPQR;STUVWX;lfyzab"

    fig_height = 19 if beamlet_shape else 17
    fig = plt.figure(layout="constrained", figsize=(18, fig_height), dpi=300)
    ax_dict = fig.subplot_mosaic(mosaic)

    alphas = np.zeros_like(ground_truth)
    alphas[ground_truth > np.max(ground_truth) * 0.01] = 0.7

    alphas_pred = np.zeros_like(prediction)
    alphas_pred[prediction > np.max(prediction) * 0.01] = 0.7

    depth_layers_to_disp = np.linspace(1, min(bp_idx_gt[0] + 8, 159), 6, dtype=int)
    print("Depth layers to display: ", depth_layers_to_disp)
    depth_layers_to_disp[-2] = bp_idx_gt[0]

    diff = np.abs(ground_truth - prediction) / np.max(ground_truth) * 100

    y_true_np = to_gy(ground_truth) * 1000 # Convert to Gy / 10^7 particles, which is a more intuitive unit for visualization (and is what we used in the paper). The scaling by 1000 is to convert from Gy to mGy, which is a common unit for dose visualization.
    y_pred_np = to_gy(prediction) * 1000 # Convert to Gy / 10^7 particles, which is a more intuitive unit for visualization (and is what we used in the paper). The scaling by 1000 is to convert from Gy to mGy, which is a common unit for dose visualization.
    x_np = ct_input.copy()

    true_min, true_max = np.min(y_true_np), np.max(y_true_np)
    pred_min, pred_max = np.min(y_pred_np), np.max(y_pred_np)
    diff_min, diff_max = np.min(diff), np.max(diff)
    norm_true = plt.Normalize(vmin=true_min, vmax=true_max)
    norm_pred = plt.Normalize(vmin=pred_min, vmax=pred_max)
    norm_diff = plt.Normalize(vmin=diff_min, vmax=diff_max)

    # Axial view ------
    # Axial - GT
    ax = ax_dict["A"]
    ax.set_title(f"Axial view", fontsize=20, pad=20)
    ax.imshow(np.rot90(x_np[0][:, bp_idx_gt[1], :]), cmap="gray")
    for i in range(len(depth_layers_to_disp)):
        ax.axvline(x=depth_layers_to_disp[i], color="red", linewidth=2)
        # Add the text next to the vertical line, representing the depth in mm
        ax.text(
            depth_layers_to_disp[i] + 0.5,
            32 if i % 2 == 0 else -2,
            f"{depth_layers_to_disp[i] * 2} mm",
            transform=ax.transData,
            fontsize=16,
            color="red",
            va="center",
            ha="left",
        )
    ct_ax = ax.imshow(
        np.rot90(y_true_np[:, bp_idx_gt[1], :]),
        cmap="jet",
        alpha=np.rot90(alphas[:, bp_idx_gt[1], :]),
        norm=norm_true,
    )
    ax.set_xticks([])
    set_row_ylabel(ax, "MCSquare\n[mm]")
    # Replace y axis ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times 2 because of avg pooling
    ax.set_yticklabels(y_tick_labels)
    ax.tick_params(labelsize=tick_fontsize)

    # Add xtick to effectively display the grid, but to not display x-ticks labels
    x_axis_ticks = np.arange(0, y_true_np.shape[0], 10)
    x_axis_ticks_labels = np.arange(0, 2 * x_np[0].shape[0], 20)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([""] * len(x_axis_ticks))

    ax.grid(linestyle="--", linewidth=0.5, color="white")

    # ── Beamlet shape row: CT + Flux overlay ─────────────────────────────
    if beamlet_shape:
        flux = x_np[1]  # (D, H, W) – flux channel

        # Build a flux alpha mask: transparent where flux is negligible
        flux_alpha_threshold = 0.01 * np.max(flux)

        # Axial CT + Flux
        ax = ax_dict["1"]
        ax.imshow(np.rot90(x_np[0][:, bp_idx_gt[1], :]), cmap="gray")
        flux_axial = np.rot90(flux[:, bp_idx_gt[1], :])
        flux_alpha_axial = np.where(flux_axial > flux_alpha_threshold, 0.65, 0.0)
        flux_im = ax.imshow(flux_axial, cmap="hot", alpha=flux_alpha_axial)
        for i in range(len(depth_layers_to_disp)):
            ax.axvline(x=depth_layers_to_disp[i], color="red", linewidth=2)
            ax.text(
                depth_layers_to_disp[i] + 0.5,
                32 if i % 2 == 0 else -2,
                f"{depth_layers_to_disp[i] * 2} mm",
                transform=ax.transData,
                fontsize=16,
                color="red",
                va="center",
                ha="left",
            )
        set_row_ylabel(ax, "CT + Flux\n[mm]")
        y_ticks = ax.get_yticks()
        y_tick_labels = (y_ticks * 2).astype(int)
        ax.set_yticklabels(y_tick_labels)
        ax.tick_params(labelsize=tick_fontsize)
        x_axis_ticks = np.arange(0, x_np[0].shape[0], 10)
        ax.set_xticks(x_axis_ticks)
        ax.set_xticklabels([""] * len(x_axis_ticks))
        ax.grid(linestyle="--", linewidth=0.5, color="white")

        # Sagittal CT + Flux
        ax = ax_dict["2"]
        ax.imshow(np.rot90(x_np[0][:, :, bp_idx_gt[2]]), cmap="gray")
        flux_sag = np.rot90(flux[:, :, bp_idx_gt[2]])
        flux_alpha_sag = np.where(flux_sag > flux_alpha_threshold, 0.65, 0.0)
        flux_im = ax.imshow(flux_sag, cmap="hot", alpha=flux_alpha_sag)
        publication_colorbar(fig, flux_im, ax, "Flux [a.u.]")
        y_ticks = ax.get_yticks()
        ax.set_yticklabels([""] * len(y_ticks))
        x_axis_ticks = np.arange(0, x_np[0].shape[0], 10)
        ax.set_xticks(x_axis_ticks)
        ax.set_xticklabels([""] * len(x_axis_ticks))
        ax.grid(linestyle="--", linewidth=0.5, color="white")
        ax.tick_params(labelsize=tick_fontsize)

    # Axial - Prediction
    ax = ax_dict["C"]
    ax.imshow(np.rot90(x_np[0][:, bp_idx_gt[1], :]), cmap="gray")
    for i in range(len(depth_layers_to_disp)):
        ax.axvline(x=depth_layers_to_disp[i], color="red", linewidth=2)
        # Add the text next to the vertical line, representing the depth in mm
        ax.text(
            depth_layers_to_disp[i] + 0.5,
            32 if i % 2 == 0 else -2,
            f"{depth_layers_to_disp[i] * 2} mm",
            transform=ax.transData,
            fontsize=14,
            color="red",
            va="center",
            ha="left",
        )
    ct_ax = ax.imshow(
        np.rot90(y_pred_np[:, bp_idx_gt[1], :]),
        cmap="jet",
        alpha=np.rot90(alphas_pred[:, bp_idx_gt[1], :]),
        norm=norm_true,
    )
    # aligned_colorbar(fig, ct_ax, ax, '')
    # ax.set_xticks([])
    set_row_ylabel(ax, "ADoTA\n[mm]")
    # Replace y axis ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times
    ax.set_yticklabels(y_tick_labels)

    # Add xtick to effectively display the grid, but to not display x-ticks labels
    x_axis_ticks = np.arange(0, y_true_np.shape[0], 10)
    x_axis_ticks_labels = np.arange(0, 2 * x_np[0].shape[0], 20)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([""] * len(x_axis_ticks))

    ax.grid(linestyle="--", linewidth=0.5, color="white")
    ax.tick_params(labelsize=tick_fontsize)

    # Axial - Difference
    ax = ax_dict["E"]
    ct_ax = ax.imshow(
        np.rot90(diff[:, bp_idx_gt[1], :]), cmap="seismic", vmin=diff_min, vmax=diff_max
    )
    # aligned_colorbar(fig, ct_ax, ax, '')
    set_row_ylabel(ax, "Abs. diff. [%]\n[mm]")
    # Replace y axis ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times 2 because of avg pooling
    ax.set_yticklabels(y_tick_labels)
    ax.set_xticks([])
    ax.tick_params(labelsize=tick_fontsize)

    x_axis_ticks = np.arange(0, y_true_np.shape[0], 10)
    x_axis_ticks_labels = np.arange(0, 2 * x_np[0].shape[0], 20)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels(x_axis_ticks_labels, rotation=45)
    ax.set_xlabel(
        "Depth [mm]", fontsize=14
    )  # if sample_index == number_of_samples_to_disp - 1 else ax.set_xlabel('')
    ax.grid(linestyle="--", linewidth=0.5, color="white")

    # SAGGITAL VIEW
    # Saggital - GT
    ax = ax_dict["B"]
    ax.set_title(f"Sagittal view", fontsize=20, pad=20)
    ax.imshow(np.rot90(x_np[0][:, :, bp_idx_gt[2]]), cmap="gray")
    ct_ax = ax.imshow(
        np.rot90(y_true_np[:, :, bp_idx_gt[2]]),
        cmap="jet",
        alpha=np.rot90(alphas[:, :, bp_idx_gt[2]]),
        norm=norm_true,
    )
    for i in range(len(depth_layers_to_disp)):
        ax.axvline(x=depth_layers_to_disp[i], color="red", linewidth=2)
        # Add the text next to the vertical line, representing the depth in mm
        ax.text(
            depth_layers_to_disp[i] + 0.5,
            32 if i % 2 == 0 else -2,
            f"{depth_layers_to_disp[i] * 2} mm",
            transform=ax.transData,
            fontsize=14,
            color="red",
            va="center",
            ha="left",
        )
    publication_colorbar(fig, ct_ax, ax, "Dose [Gy]")
    # Maintain the grid, but remove ticks
    # Set x and y ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times
    ax.set_yticklabels([""] * len(y_ticks))

    # Add xtick to effectively display the grid, but to not display x-ticks labels
    x_axis_ticks = np.arange(0, y_true_np.shape[0], 10)
    x_axis_ticks_labels = np.arange(0, 2 * x_np[0].shape[0], 20)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([""] * len(x_axis_ticks))

    ax.grid(linestyle="--", linewidth=0.5, color="white")
    ax.tick_params(labelsize=tick_fontsize)

    # Saggital - Prediction
    ax = ax_dict["D"]
    ax.imshow(np.rot90(x_np[0][:, :, bp_idx_gt[2]]), cmap="gray")
    ct_ax = ax.imshow(
        np.rot90(y_pred_np[:, :, bp_idx_gt[2]]),
        cmap="jet",
        alpha=np.rot90(alphas_pred[:, :, bp_idx_gt[2]]),
        norm=norm_true,
    )
    for i in range(len(depth_layers_to_disp)):
        ax.axvline(x=depth_layers_to_disp[i], color="red", linewidth=2)
        # Add the text next to the vertical line, representing the depth in mm
        ax.text(
            depth_layers_to_disp[i] + 0.5,
            32 if i % 2 == 0 else -2,
            f"{depth_layers_to_disp[i] * 2} mm",
            transform=ax.transData,
            fontsize=14,
            color="red",
            va="center",
            ha="left",
        )
    publication_colorbar(fig, ct_ax, ax, "Dose [Gy]")
    ax.tick_params(labelsize=tick_fontsize)
    # Maintain the grid, but remove ticks
    # Set x and y ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times
    ax.set_yticklabels([""] * len(y_ticks))

    # Add xtick to effectively display the grid, but to not display x-ticks labels
    x_axis_ticks = np.arange(0, y_true_np.shape[0], 10)
    x_axis_ticks_labels = np.arange(0, 2 * x_np[0].shape[0], 20)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([""] * len(x_axis_ticks))

    ax.grid(linestyle="--", linewidth=0.5, color="white")
    ax.tick_params(labelsize=tick_fontsize)

    # Saggital - Difference
    ax = ax_dict["F"]
    ct_ax = ax.imshow(
        np.rot90(diff[:, :, bp_idx_gt[2]]), cmap="seismic", vmin=diff_min, vmax=diff_max
    )
    publication_colorbar(fig, ct_ax, ax, "Abs. diff. [%]")
    ax.tick_params(labelsize=tick_fontsize)
    # Set x and y ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times
    ax.set_yticklabels([""] * len(y_ticks))

    x_axis_ticks = np.arange(0, y_true_np.shape[0], 10)
    x_axis_ticks_labels = np.arange(0, 2 * x_np[0].shape[0], 20)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels(x_axis_ticks_labels, rotation=45)
    ax.set_xlabel(
        "Depth [mm]", fontsize=16
    )  # if sample_index == number_of_samples_to_disp - 1 else ax.set_xlabel('')
    ax.grid(linestyle="--", linewidth=0.5, color="white")

    # Last row represents the IDD
    ax = ax_dict["T"]
    idd_gt = (
        np.sum(y_true_np, axis=(1, 2)) / np.max(np.sum(y_true_np, axis=(1, 2))) * 100
    )
    idd_pred = (
        np.sum(y_pred_np, axis=(1, 2)) / np.max(np.sum(y_true_np, axis=(1, 2))) * 100
    )
    ax.plot(idd_gt, label="GT", color="blue")
    ax.plot(idd_pred, label="ADoTA", color="orange", linestyle="--")
    ax.axvline(
        x=bp_idx_gt[0],
        color="red",
        linestyle="--",
        label="Bragg peak GT ({} mm)".format(bp_idx_gt[0] * 2),
    )
    ax.axvline(
        x=bp_idx_pred[0],
        color="green",
        linestyle="--",
        label="Bragg peak ADoTA ({} mm)".format(bp_idx_pred[0] * 2),
    )
    ax.set_xticks(np.arange(0, x_np.shape[1] + 1, 10))
    ax.set_xticklabels(
        np.arange(0, (x_np.shape[1] + 1) * 2, 20), rotation=45
    )  # times 2 because of avg pooling
    ax.set_xlabel("Depth [mm]", fontsize=16)
    set_row_ylabel(ax, "Normalized IDD [%]", label_x=wide_row_label_x)
    ax.legend(fontsize=14)
    ax.set_xlim(0, x_np.shape[1] - 1)
    ax.grid(linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=tick_fontsize)

    # LAST THREE ROWS
    last_three_rows = mosaic.split(";")[-3:]

    for img_idx, ax_label in enumerate(list(last_three_rows[0])):
        ax = ax_dict[ax_label]
        if img_idx == 0:
            set_row_ylabel(ax, "MCSquare - BEV")
        ax.imshow(np.rot90(x_np[0][depth_layers_to_disp[img_idx], :, :]), cmap="gray")
        dose_ax_bev = ax.imshow(
            np.rot90(y_true_np[depth_layers_to_disp[img_idx], :, :]),
            cmap="jet",
            alpha=np.rot90(alphas[depth_layers_to_disp[img_idx], :, :]),
            norm=norm_true,
        )
        if img_idx == len(last_three_rows[0]) - 1:
            publication_colorbar(
                fig, dose_ax_bev, ax, "Dose [Gy]", size=bev_colorbar_size
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(linestyle="--", linewidth=0.5, color="white")

    for img_idx, ax_label in enumerate(list(last_three_rows[1])):
        ax = ax_dict[ax_label]
        if img_idx == 0:
            set_row_ylabel(ax, "ADoTA - BEV")
        ax.imshow(np.rot90(x_np[0][depth_layers_to_disp[img_idx], :, :]), cmap="gray")
        dose_ax_bev = ax.imshow(
            np.rot90(y_pred_np[depth_layers_to_disp[img_idx], :, :]),
            cmap="jet",
            alpha=np.rot90(alphas_pred[depth_layers_to_disp[img_idx], :, :]),
            norm=norm_true,
        )
        if img_idx == len(last_three_rows[1]) - 1:
            publication_colorbar(
                fig, dose_ax_bev, ax, "Dose [Gy]", size=bev_colorbar_size
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(linestyle="--", linewidth=0.5, color="white")
        # ax.set_xlabel(f"{depth_layers_to_disp[img_idx] * 2} mm", fontsize=14)

    for img_idx, ax_label in enumerate(list(last_three_rows[2])):
        ax = ax_dict[ax_label]
        if img_idx == 0:
            set_row_ylabel(ax, "Abs. diff. - BEV")
        diff_ax_bev = ax.imshow(
            np.rot90(diff[depth_layers_to_disp[img_idx], :, :]),
            cmap="seismic",
            vmin=diff_min,
            vmax=diff_max,
        )
        if img_idx == len(last_three_rows[2]) - 1:
            publication_colorbar(
                fig, diff_ax_bev, ax, "Abs. diff. [%]", size=bev_colorbar_size
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(linestyle="--", linewidth=0.5, color="white")
        ax.set_xlabel(f"{depth_layers_to_disp[img_idx] * 2} mm", fontsize=16)

    title = (
        "Initial Energy: {:.2f} MeV\n"
        "MAPE: {:.2f} %, GPR({:.1f}%, {:.1f}mm, {:.1f}%): {:.2f} %"
    ).format(
        initial_energy,
        mape,
        gamma_params["dose_percent_threshold"],
        gamma_params["distance_mm_threshold"],
        gamma_params["lower_percent_dose_cutoff"],
        gpr,
    )
    # TODO: Parametrize the title option. For paper we do not use it!
    # fig.suptitle(
    #     title,
    #     fontsize=16,
    #     y=1.05,
    # )
    print("Publication figure generated.")
    output_paths = save_figure_as_publication_formats(fig, figure_path)
    print("Figures saved to: ", ", ".join(str(path) for path in output_paths))
    plt.close(fig)

    # air_layer = os.path.basename(storage_path).split("_")[-3]
    # fname = "PUB_{}_E{:.2f}_air{}.png".format(model_name, initial_energy, air_layer)
    # fig.savefig(os.path.join(image_storage, fname), bbox_inches='tight', dpi=300)


def beamlet_input_figure(
    ct: np.ndarray,
    flux: np.ndarray,
    figure_path: str,
    initial_energy: float | None = None,
    beamlet_angles: tuple[float, float] | None = None,
    spot_id: str | None = None,
    ct_window: tuple[float, float] | None = None,
) -> list[Path]:
    """Plot a constructed beamlet input (CT crop + flux) for correctness checks.

    Renders a 2x2 mosaic via :meth:`Figure.subplot_mosaic`: the CT crop on the
    top row and the flux projection on the bottom row, each shown in the axial
    and sagittal views. The beam depth (``x``) runs along the horizontal axis
    with the entrance face at the left.

    Both arrays are the extraction outputs in numpy ``(z, y, x)`` order, e.g.
    ``(60, 60, 320)``. ``publication_figure`` is intentionally left untouched;
    this is a separate, simpler view.

    Args:
        ct: CT crop ``(z, y, x)`` in HU.
        flux: Flux projection ``(z, y, x)``, same shape as ``ct``.
        figure_path: Output path (``.svg``/``.pdf``/``.png`` are all written).
        initial_energy: Beam energy in MeV (for the title), optional.
        beamlet_angles: ``(theta_y, theta_z)`` in degrees (for the title), optional.
        spot_id: Spot id (for the title), optional.
        ct_window: ``(vmin, vmax)`` HU window for the CT; defaults to the crop's
            own min/max.

    Returns:
        The list of written figure paths.
    """
    if ct.shape != flux.shape:
        raise ValueError(
            f"ct and flux must have the same shape, got {ct.shape} and {flux.shape}."
        )
    if ct.ndim != 3:
        raise ValueError(f"Expected (z, y, x) arrays, got shape {ct.shape}.")

    n_z, n_y, _ = ct.shape
    mid_z, mid_y = n_z // 2, n_y // 2

    def axial(volume: np.ndarray) -> np.ndarray:
        # (y, x) slice at mid z: lateral-y (vertical) vs depth-x (horizontal).
        return volume[mid_z, :, :]

    def sagittal(volume: np.ndarray) -> np.ndarray:
        # (z, x) slice at mid y: lateral-z (vertical) vs depth-x (horizontal).
        return volume[:, mid_y, :]

    ct_vmin, ct_vmax = ct_window if ct_window is not None else (float(ct.min()), float(ct.max()))
    flux_max = max(float(np.max(flux)), 1e-12)

    fig = plt.figure(layout="constrained", figsize=(14, 7), dpi=200)
    ax_dict = fig.subplot_mosaic("AB;CD")

    ct_kw = dict(cmap="gray", vmin=ct_vmin, vmax=ct_vmax, aspect="auto", origin="lower")
    flux_kw = dict(cmap="hot", vmin=0.0, vmax=flux_max, aspect="auto", origin="lower")

    ct_im = ax_dict["A"].imshow(axial(ct), **ct_kw)
    ax_dict["B"].imshow(sagittal(ct), **ct_kw)
    flux_im = ax_dict["C"].imshow(axial(flux), **flux_kw)
    ax_dict["D"].imshow(sagittal(flux), **flux_kw)

    ax_dict["A"].set_title("Axial (x-y @ mid z)", fontsize=15, weight="bold")
    ax_dict["B"].set_title("Sagittal (x-z @ mid y)", fontsize=15, weight="bold")

    for key in ("A", "C"):
        ax_dict[key].set_ylabel("Lateral y [voxels]", fontsize=12)
    for key in ("B", "D"):
        ax_dict[key].set_ylabel("Lateral z [voxels]", fontsize=12)
    for key in ("C", "D"):
        ax_dict[key].set_xlabel("Depth x [voxels] (0 = entrance)", fontsize=12)
    for key in ("A", "B"):
        ax_dict[key].set_xticklabels([])

    # Row labels on the far left.
    ax_dict["A"].text(
        -0.18, 0.5, "CT [HU]", transform=ax_dict["A"].transAxes,
        rotation=90, va="center", ha="center", fontsize=16, weight="bold",
    )
    ax_dict["C"].text(
        -0.18, 0.5, "Flux [a.u.]", transform=ax_dict["C"].transAxes,
        rotation=90, va="center", ha="center", fontsize=16, weight="bold",
    )

    for key in ("A", "B", "C", "D"):
        ax_dict[key].grid(linestyle="--", linewidth=0.5, color="white")
        ax_dict[key].tick_params(labelsize=11)

    aligned_colorbar(fig, ct_im, ax_dict["B"], "HU", label_coords=(4.2, 0.5))
    aligned_colorbar(fig, flux_im, ax_dict["D"], "Flux [a.u.]", label_coords=(4.2, 0.5))

    title_bits = []
    if spot_id is not None:
        title_bits.append(f"spot {spot_id}")
    if initial_energy is not None:
        title_bits.append(f"E = {initial_energy:.2f} MeV")
    if beamlet_angles is not None:
        title_bits.append(
            f"beamlet angles ({beamlet_angles[0]:.3f}, {beamlet_angles[1]:.3f}) deg"
        )
    if title_bits:
        fig.suptitle(" | ".join(title_bits), fontsize=16, weight="bold")

    output_paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return output_paths
