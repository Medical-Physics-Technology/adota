"""Model-input-only figures for a single beamlet.

:func:`beamlet_input_figure` renders just the two input channels -- the BEV CT
crop and the proton-flux projection -- in axial and sagittal views. Used to
inspect extraction output without needing a prediction or ground truth.
:func:`beamlet_dose_figure` is the same mosaic with the ground-truth dose in
place of the flux, and :func:`entrance_profile_figure` plots the lateral flux
and dose profiles on one depth slice.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats


def _title(initial_energy, beamlet_angles, spot_id) -> str:
    bits = []
    if spot_id is not None:
        bits.append(f"spot {spot_id}")
    if initial_energy is not None:
        bits.append(f"E = {initial_energy:.2f} MeV")
    if beamlet_angles is not None:
        bits.append(f"beamlet angles ({beamlet_angles[0]:.3f}, {beamlet_angles[1]:.3f}) deg")
    return " | ".join(bits)


def _ct_channel_mosaic(
    ct: np.ndarray,
    channel: np.ndarray,
    figure_path: str,
    channel_label: str,
    channel_cmap: str,
    initial_energy: float | None,
    beamlet_angles: tuple[float, float] | None,
    spot_id: str | None,
    ct_window: tuple[float, float] | None,
) -> list[Path]:
    """The 2x2 CT-over-channel mosaic shared by the input and dose figures."""
    if ct.shape != channel.shape:
        raise ValueError(
            f"ct and {channel_label} must have the same shape, got {ct.shape} and {channel.shape}."
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
    channel_max = max(float(np.max(channel)), 1e-12)

    fig = plt.figure(layout="constrained", figsize=(14, 7), dpi=200)
    ax_dict = fig.subplot_mosaic("AB;CD")

    ct_kw = dict(cmap="gray", vmin=ct_vmin, vmax=ct_vmax, aspect="auto", origin="lower")
    channel_kw = dict(cmap=channel_cmap, vmin=0.0, vmax=channel_max, aspect="auto", origin="lower")

    ct_im = ax_dict["A"].imshow(axial(ct), **ct_kw)
    ax_dict["B"].imshow(sagittal(ct), **ct_kw)
    channel_im = ax_dict["C"].imshow(axial(channel), **channel_kw)
    ax_dict["D"].imshow(sagittal(channel), **channel_kw)

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
        -0.18, 0.5, channel_label, transform=ax_dict["C"].transAxes,
        rotation=90, va="center", ha="center", fontsize=16, weight="bold",
    )

    for key in ("A", "B", "C", "D"):
        ax_dict[key].grid(linestyle="--", linewidth=0.5, color="white")
        ax_dict[key].tick_params(labelsize=11)

    aligned_colorbar(fig, ct_im, ax_dict["B"], "HU", label_coords=(4.2, 0.5))
    aligned_colorbar(fig, channel_im, ax_dict["D"], channel_label, label_coords=(4.2, 0.5))

    title = _title(initial_energy, beamlet_angles, spot_id)
    if title:
        fig.suptitle(title, fontsize=16, weight="bold")

    output_paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)
    return output_paths


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
    return _ct_channel_mosaic(ct, flux, figure_path, "Flux [a.u.]", "hot",
                              initial_energy, beamlet_angles, spot_id, ct_window)


def beamlet_dose_figure(
    ct: np.ndarray,
    dose: np.ndarray,
    figure_path: str,
    initial_energy: float | None = None,
    beamlet_angles: tuple[float, float] | None = None,
    spot_id: str | None = None,
    ct_window: tuple[float, float] | None = None,
    dose_label: str = "Dose [a.u.]",
) -> list[Path]:
    """The :func:`beamlet_input_figure` mosaic with the ground-truth dose in place
    of the flux, so a record's target can be read next to its input.

    Args:
        ct: CT crop ``(z, y, x)`` in HU.
        dose: Dose grid ``(z, y, x)``, same shape as ``ct``.
        figure_path: Output path (``.svg``/``.pdf``/``.png`` are all written).
        initial_energy: Beam energy in MeV (for the title), optional.
        beamlet_angles: ``(theta_y, theta_z)`` in degrees (for the title), optional.
        spot_id: Spot id (for the title), optional.
        ct_window: ``(vmin, vmax)`` HU window for the CT; defaults to the crop's
            own min/max.
        dose_label: Row and colorbar label for the dose (say the unit if known).

    Returns:
        The list of written figure paths.
    """
    return _ct_channel_mosaic(ct, dose, figure_path, dose_label, "magma",
                              initial_energy, beamlet_angles, spot_id, ct_window)


def entrance_profile_figure(
    flux: np.ndarray,
    dose: np.ndarray,
    figure_path: str,
    depth_index: int = 0,
    initial_energy: float | None = None,
    beamlet_angles: tuple[float, float] | None = None,
    spot_id: str | None = None,
) -> tuple[list[Path], Path]:
    """Lateral flux and dose profiles on one depth slice, through the flux peak.

    The slice ``[:, :, depth_index]`` of both grids (``(z, y, x)`` order, so the
    default ``0`` is the entrance face) is cut along ``z`` and along ``y`` through
    the voxel where the flux peaks on that slice. Each curve is normalised to its
    own maximum on the slice so the two widths can be compared; the raw values
    are written to ``<figure_path>_profiles.csv`` beside the figure.

    Args:
        flux: Flux projection ``(z, y, x)``.
        dose: Dose grid ``(z, y, x)``, same shape as ``flux``.
        figure_path: Output path (``.svg``/``.pdf``/``.png`` are all written).
        depth_index: Depth slice to profile; ``0`` is the entrance.
        initial_energy: Beam energy in MeV (for the title), optional.
        beamlet_angles: ``(theta_y, theta_z)`` in degrees (for the title), optional.
        spot_id: Spot id (for the title), optional.

    Returns:
        ``(figure paths, csv path)``.
    """
    if flux.shape != dose.shape or flux.ndim != 3:
        raise ValueError(f"flux and dose must be equal-shaped (z, y, x), got {flux.shape} and {dose.shape}.")
    flux_slice = np.asarray(flux[:, :, depth_index], dtype=np.float64)
    dose_slice = np.asarray(dose[:, :, depth_index], dtype=np.float64)
    peak_z, peak_y = np.unravel_index(int(np.argmax(flux_slice)), flux_slice.shape)
    flux_max, dose_max = float(flux_slice.max()), float(dose_slice.max())

    cuts = {
        "z": (np.arange(flux.shape[0]), flux_slice[:, peak_y], dose_slice[:, peak_y], f"y = {peak_y}"),
        "y": (np.arange(flux.shape[1]), flux_slice[peak_z, :], dose_slice[peak_z, :], f"z = {peak_z}"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200, layout="constrained")
    for ax, (axis_name, (pos, f_line, d_line, fixed)) in zip(axes, cuts.items()):
        ax.plot(pos, f_line / max(flux_max, 1e-12), color="tab:red", lw=2,
                label=f"flux (max {flux_max:.3g})")
        ax.plot(pos, d_line / max(dose_max, 1e-12), color="tab:blue", lw=2, ls="--",
                label=f"dose (max {dose_max:.3g})")
        ax.set_xlabel(f"Lateral {axis_name} [voxels]", fontsize=12)
        ax.set_ylabel("Normalised to slice maximum", fontsize=12)
        ax.set_title(f"Profile along {axis_name} at {fixed}, depth {depth_index}", fontsize=14, weight="bold")
        ax.grid(linestyle="--", linewidth=0.5)
        ax.legend(fontsize=11)
        ax.tick_params(labelsize=11)

    title = _title(initial_energy, beamlet_angles, spot_id)
    if title:
        fig.suptitle(title, fontsize=16, weight="bold")

    output_paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)

    csv_path = Path(f"{figure_path}_profiles.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["axis", "position", "flux", "dose", "depth_index", "peak_z", "peak_y"])
        for axis_name, (pos, f_line, d_line, _) in cuts.items():
            for p, f_val, d_val in zip(pos, f_line, d_line):
                writer.writerow([axis_name, int(p), f_val, d_val, depth_index, int(peak_z), int(peak_y)])
    return output_paths, csv_path


def entrance_profile_overlay_figure(
    records: "list[tuple[str, np.ndarray, np.ndarray]]",
    figure_path: str,
    depth_index: int = 0,
    title: str | None = None,
) -> tuple[list[Path], Path]:
    """Entrance-slice profiles of several records on one pair of axes.

    Each record is ``(label, flux, dose)`` in ``(z, y, x)`` order; the cut runs
    through that record's own flux peak on the slice, and every curve is
    normalised to its own slice maximum. Meant for a handful of records that share
    a geometry (same spot, different energies and beamlet angles), so their offsets
    and widths can be read against each other. The raw values go to
    ``<figure_path>_profiles.csv``.

    Returns:
        ``(figure paths, csv path)``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200, layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    csv_rows = []
    for k, (label, flux, dose) in enumerate(records):
        flux_slice = np.asarray(flux[:, :, depth_index], dtype=np.float64)
        dose_slice = np.asarray(dose[:, :, depth_index], dtype=np.float64)
        peak_z, peak_y = np.unravel_index(int(np.argmax(flux_slice)), flux_slice.shape)
        f_max, d_max = max(float(flux_slice.max()), 1e-12), max(float(dose_slice.max()), 1e-12)
        cuts = {"z": (flux_slice[:, peak_y], dose_slice[:, peak_y]),
                "y": (flux_slice[peak_z, :], dose_slice[peak_z, :])}
        for ax, (axis_name, (f_line, d_line)) in zip(axes, cuts.items()):
            pos = np.arange(f_line.size)
            ax.plot(pos, f_line / f_max, color=colors[k % len(colors)], lw=2, label=f"{label} flux")
            ax.plot(pos, d_line / d_max, color=colors[k % len(colors)], lw=2, ls="--", label=f"{label} dose")
            for p, f_val, d_val in zip(pos, f_line, d_line):
                csv_rows.append([label, axis_name, int(p), f_val, d_val, depth_index, int(peak_z), int(peak_y)])
    for ax, axis_name in zip(axes, ("z", "y")):
        ax.set_xlabel(f"Lateral {axis_name} [voxels]", fontsize=12)
        ax.set_ylabel("Normalised to slice maximum", fontsize=12)
        ax.set_title(f"Profile along {axis_name} through each flux peak, depth {depth_index}",
                     fontsize=14, weight="bold")
        ax.grid(linestyle="--", linewidth=0.5)
        ax.legend(fontsize=9, ncol=2)
        ax.tick_params(labelsize=11)
    if title:
        fig.suptitle(title, fontsize=16, weight="bold")
    output_paths = save_figure_as_publication_formats(fig, figure_path)
    plt.close(fig)

    csv_path = Path(f"{figure_path}_profiles.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "axis", "position", "flux", "dose", "depth_index", "peak_z", "peak_y"])
        writer.writerows(csv_rows)
    return output_paths, csv_path
