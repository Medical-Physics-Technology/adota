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


def aligned_colorbar(fig, ct_ax, ax, label: str, label_coords: tuple = (4.7, 0.5)):
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
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = fig.colorbar(ct_ax, cax=cax, orientation="vertical")
    colorbar.set_label(label, fontsize=18, labelpad=10)
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.ax.yaxis.set_label_coords(label_coords[0], label_coords[1])
    colorbar.ax.yaxis.set_tick_params(labelsize=15)
    return colorbar


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
):
    # Preprocessing - handled here for simplicity
    bp_idx_gt = estimate_bragg_peak(ground_truth)
    bp_idx_pred = estimate_bragg_peak(prediction)

    if not lateral_profiles_per_slice:
        mosaic = "AAABBB;CCCDDD;EEEFFF;TTTTTT;GHJKLM;NOPQRS;UVWXYZ"
    else:
        # Last row represents lateral profiles per displayed slice in the rows above
        mosaic = "AAABBB;CCCDDD;EEEFFF;TTTTTT;GHIJKL;MNOPQR;STUVWX;lfyzab"

    fig = plt.figure(layout="constrained", figsize=(18, 16), dpi=300)
    ax_dict = fig.subplot_mosaic(mosaic)

    alphas = np.zeros_like(ground_truth)
    alphas[ground_truth > np.max(ground_truth) * 0.01] = 0.7

    alphas_pred = np.zeros_like(prediction)
    alphas_pred[prediction > np.max(prediction) * 0.01] = 0.7

    depth_layers_to_disp = np.linspace(1, min(bp_idx_gt[0] + 8, 159), 6, dtype=int)
    print("Depth layers to display: ", depth_layers_to_disp)
    depth_layers_to_disp[-2] = bp_idx_gt[0]

    diff = np.abs(ground_truth - prediction) / np.max(ground_truth) * 100

    y_true_np = to_gy(ground_truth)
    y_pred_np = to_gy(prediction)
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
            fontsize=14,
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
    ax.set_ylabel("MCSquare\n[mm]", fontsize=16)
    # Replace y axis ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times 2 because of avg pooling
    ax.set_yticklabels(y_tick_labels)
    ax.tick_params(labelsize=16)

    # Add xtick to effectively display the grid, but to not display x-ticks labels
    x_axis_ticks = np.arange(0, y_true_np.shape[0], 10)
    x_axis_ticks_labels = np.arange(0, 2 * x_np[0].shape[0], 20)
    ax.set_xticks(x_axis_ticks)
    ax.set_xticklabels([""] * len(x_axis_ticks))

    ax.grid(linestyle="--", linewidth=0.5, color="white")

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
    ax.set_ylabel("ADoTA\n[mm]", fontsize=16)
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
    ax.tick_params(labelsize=14)

    # Axial - Difference
    ax = ax_dict["E"]
    ct_ax = ax.imshow(
        np.rot90(diff[:, bp_idx_gt[1], :]), cmap="seismic", vmin=diff_min, vmax=diff_max
    )
    # aligned_colorbar(fig, ct_ax, ax, '')
    ax.set_ylabel("Abs. diff. [%]\n[mm]", fontsize=16)
    # Replace y axis ticks to represent physical dimensions
    y_ticks = ax.get_yticks()
    y_tick_labels = (y_ticks * 2).astype(int)  # times 2 because of avg pooling
    ax.set_yticklabels(y_tick_labels)
    ax.set_xticks([])
    ax.tick_params(labelsize=14)

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
    aligned_colorbar(fig, ct_ax, ax, "Dose [Gy]")
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
    ax.tick_params(labelsize=14)

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
    aligned_colorbar(fig, ct_ax, ax, "Dose [Gy]")
    ax.tick_params(labelsize=14)
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
    ax.tick_params(labelsize=14)

    # Saggital - Difference
    ax = ax_dict["F"]
    ct_ax = ax.imshow(
        np.rot90(diff[:, :, bp_idx_gt[2]]), cmap="seismic", vmin=diff_min, vmax=diff_max
    )
    aligned_colorbar(fig, ct_ax, ax, "Abs. diff. [%]")
    ax.tick_params(labelsize=14)
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
    ax.set_ylabel("Normalized IDD [%]", fontsize=16)
    ax.legend(fontsize=14)
    ax.set_xlim(0, x_np.shape[1] - 1)
    ax.grid(linestyle="--", linewidth=0.5)
    ax.tick_params(labelsize=16)

    # LAST THREE ROWS
    last_three_rows = mosaic.split(";")[-3:]

    for img_idx, ax_label in enumerate(list(last_three_rows[0])):
        ax = ax_dict[ax_label]
        if img_idx == 0:
            ax.set_ylabel("MCSquare - BEV", fontsize=16)
        ax.imshow(np.rot90(x_np[0][depth_layers_to_disp[img_idx], :, :]), cmap="gray")
        dose_ax_bev = ax.imshow(
            np.rot90(y_true_np[depth_layers_to_disp[img_idx], :, :]),
            cmap="jet",
            alpha=np.rot90(alphas[depth_layers_to_disp[img_idx], :, :]),
            norm=norm_true,
        )
        if img_idx == len(last_three_rows[0]) - 1:
            aligned_colorbar(fig, dose_ax_bev, ax, "Dose [Gy]", label_coords=(15, 0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(linestyle="--", linewidth=0.5, color="white")

    for img_idx, ax_label in enumerate(list(last_three_rows[1])):
        ax = ax_dict[ax_label]
        if img_idx == 0:
            ax.set_ylabel("ADoTA - BEV", fontsize=16)
        ax.imshow(np.rot90(x_np[0][depth_layers_to_disp[img_idx], :, :]), cmap="gray")
        dose_ax_bev = ax.imshow(
            np.rot90(y_pred_np[depth_layers_to_disp[img_idx], :, :]),
            cmap="jet",
            alpha=np.rot90(alphas_pred[depth_layers_to_disp[img_idx], :, :]),
            norm=norm_true,
        )
        if img_idx == len(last_three_rows[1]) - 1:
            aligned_colorbar(fig, dose_ax_bev, ax, "Dose [Gy]", label_coords=(15, 0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(linestyle="--", linewidth=0.5, color="white")
        # ax.set_xlabel(f"{depth_layers_to_disp[img_idx] * 2} mm", fontsize=14)

    for img_idx, ax_label in enumerate(list(last_three_rows[2])):
        ax = ax_dict[ax_label]
        if img_idx == 0:
            ax.set_ylabel("Abs. diff. - BEV", fontsize=16)
        diff_ax_bev = ax.imshow(
            np.rot90(diff[depth_layers_to_disp[img_idx], :, :]),
            cmap="seismic",
            vmin=diff_min,
            vmax=diff_max,
        )
        if img_idx == len(last_three_rows[2]) - 1:
            aligned_colorbar(
                fig, diff_ax_bev, ax, "Abs. diff. [%]", label_coords=(15, 0.5)
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
    fig.savefig(figure_path, bbox_inches="tight", dpi=300)
    print("Figure saved to: ", figure_path)
    plt.close(fig)

    # air_layer = os.path.basename(storage_path).split("_")[-3]
    # fname = "PUB_{}_E{:.2f}_air{}.png".format(model_name, initial_energy, air_layer)
    # fig.savefig(os.path.join(image_storage, fname), bbox_inches='tight', dpi=300)
