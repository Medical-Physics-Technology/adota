"""Model-input-only figure for a single beamlet.

:func:`beamlet_input_figure` renders just the two input channels -- the BEV CT
crop and the proton-flux projection -- in axial and sagittal views. Used to
inspect extraction output without needing a prediction or ground truth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figures.axes_utils import aligned_colorbar, save_figure_as_publication_formats


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
