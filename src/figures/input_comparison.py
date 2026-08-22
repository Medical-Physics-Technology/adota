"""Before/after overlays of the model inputs.

:func:`compare_two_inputs` puts the original and the rotated volume side by
side, each as a CT slice with flux or dose overlaid, in axial and sagittal
views. Used to confirm that the beam's-eye-view rotation preserved the geometry
before a batch is handed to the model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import aligned_colorbar


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
        ax_dict["A"],
        original_ct,
        original_flux,
        "axial",
        "Original CT + flux | axial",
        "hot",
        flux_max,
        flux_alpha_threshold,
    )
    _plot_ct_overlay(
        ax_dict["B"],
        original_ct,
        original_flux,
        "sagittal",
        "Original CT + flux | sagittal",
        "hot",
        flux_max,
        flux_alpha_threshold,
    )
    dose_im = _plot_ct_overlay(
        ax_dict["C"],
        original_ct,
        original_dose,
        "axial",
        "Original CT + dose | axial",
        "jet",
        dose_max,
        dose_alpha_threshold,
    )
    _plot_ct_overlay(
        ax_dict["D"],
        original_ct,
        original_dose,
        "sagittal",
        "Original CT + dose | sagittal",
        "jet",
        dose_max,
        dose_alpha_threshold,
    )
    _plot_ct_overlay(
        ax_dict["E"],
        rotated_ct,
        rotated_flux,
        "axial",
        "Rotated CT + flux | axial",
        "hot",
        flux_max,
        flux_alpha_threshold,
    )
    _plot_ct_overlay(
        ax_dict["F"],
        rotated_ct,
        rotated_flux,
        "sagittal",
        "Rotated CT + flux | sagittal",
        "hot",
        flux_max,
        flux_alpha_threshold,
    )
    _plot_ct_overlay(
        ax_dict["G"],
        rotated_ct,
        rotated_dose,
        "axial",
        "Rotated CT + dose | axial",
        "jet",
        dose_max,
        dose_alpha_threshold,
    )
    _plot_ct_overlay(
        ax_dict["H"],
        rotated_ct,
        rotated_dose,
        "sagittal",
        "Rotated CT + dose | sagittal",
        "jet",
        dose_max,
        dose_alpha_threshold,
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
