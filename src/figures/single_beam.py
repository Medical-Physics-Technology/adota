"""The main single-beamlet publication figure.

:func:`publication_figure` renders one beamlet across axial, sagittal and
coronal views: ground-truth dose, prediction and their difference over the CT,
plus a depth-layer strip and optional lateral profiles. It is the figure the
paper uses for qualitative comparison.

Related modules, split out to stay inside the 500-line limit:
:mod:`src.figures.axes_utils` (colorbars, saving),
:mod:`src.figures.input_comparison` (rotation QC),
:mod:`src.figures.beamlet_input` (inputs only).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats
from src.utils.dose_grid_utils import estimate_bragg_peak
from src.utils.unit_conversions import to_gy


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

    # Convert to Gy / 10^7 particles, a more intuitive unit for visualization
    # (and what the paper uses); the 1000 factor converts Gy to mGy.
    y_true_np = to_gy(ground_truth) * 1000
    y_pred_np = to_gy(prediction) * 1000
    x_np = ct_input.copy()

    true_min, true_max = np.min(y_true_np), np.max(y_true_np)
    diff_min, diff_max = np.min(diff), np.max(diff)
    norm_true = plt.Normalize(vmin=true_min, vmax=true_max)

    # Axial view ------
    # Axial - GT
    ax = ax_dict["A"]
    ax.set_title("Axial view", fontsize=20, pad=20)
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
    ax.set_title("Sagittal view", fontsize=20, pad=20)
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

    title = (  # noqa: F841 - used by the commented-out fig.suptitle below
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
