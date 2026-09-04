"""Per-field visual sanity check.

For each field, renders the most-weighted spot: the rotated CT axial slice
through the isocenter with the isocenter and +x beam axis marked, beside that
spot's crop with its flux projection overlaid. Confirms at a glance that the
beam enters at x=0 and that the flux traces the ray.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import SimpleITK as sitk

from src.beamlets.bdl import BeamDataLibrary, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import flux_projection, flux_spatial_spread

logger = logging.getLogger(__name__)

def _save_field_overlay(
    overlays_dir: Path,
    beam: int,
    rotated_ct: sitk.Image,
    field_spots: List[dict],
    isocenter_physical: tuple,
    bdl: BeamDataLibrary,
    roi_size: tuple,
) -> None:
    """Save a visual sanity-check PNG for a field's most-weighted spot.

    Left: the rotated CT axial slice through the isocenter with the isocenter and
    the +x beam axis marked. Right: that spot's crop (depth x vs lateral y) with
    the flux projection overlaid, to confirm the beam enters at x=0 and the flux
    traces the ray.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Representative spot = the most-weighted one in the field.
    record = max(field_spots, key=lambda r: r["simulation_log"]["relative_weight"])
    sim_log = record["simulation_log"]
    spot_position = sim_log["bixelgrid_shifts_xy"][0]
    energy = sim_log["energy"][0]
    d_nozzle, d_smx, d_smy = bdl.distances

    cropped_ct, entrance, crp, _oob = extract_beamlet_roi(
        rotated_ct, d_nozzle, d_smx, d_smy, spot_position, isocenter_physical, roi_size
    )
    beamlet_angles = spot_position_to_angles(
        spot_position[0], spot_position[1], d_smx, d_smy
    )
    sigmas = flux_spatial_spread(bdl, energy)
    re_proj = [entrance[1], entrance[2], entrance[0]]
    flux = flux_projection(re_proj, beamlet_angles, sigmas, cropped_ct.shape)

    iso_idx = rotated_ct.TransformPhysicalPointToIndex(
        [float(c) for c in isocenter_physical]
    )
    ct_arr = sitk.GetArrayFromImage(rotated_ct)  # (z, y, x)
    iz, iy, ix = iso_idx[2], iso_idx[1], iso_idx[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    # Left: rotated CT axial slice through the isocenter.
    axes[0].imshow(ct_arr[iz, :, :], cmap="gray", origin="lower")
    axes[0].scatter([ix], [iy], color="red", s=30, label="isocenter")
    axes[0].axhline(iy, color="cyan", lw=0.6, ls="--", label="beam axis (+x)")
    axes[0].set_title(
        f"beam {beam}: rotated CT axial @ iso z={iz}\n"
        f"gantry(adj)={sim_log['gantry_angle']:.0f} deg"
    )
    axes[0].set_xlabel("x (depth ->)")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right", fontsize=8)

    # Right: the spot's crop, depth (x) vs lateral (y) at mid-z, flux overlaid.
    mid_z = cropped_ct.shape[0] // 2
    axes[1].imshow(cropped_ct[mid_z, :, :], cmap="gray", origin="lower", aspect="auto")
    flux_slice = flux[mid_z, :, :]
    if float(flux_slice.max()) > 0:
        axes[1].contour(
            flux_slice,
            levels=[flux_slice.max() * lvl for lvl in (0.1, 0.5, 0.9)],
            colors="orange",
            linewidths=0.8,
        )
    axes[1].set_title(f"spot {record['id']}: CT crop + flux (mid-z)\nE={energy:.1f} MeV")
    axes[1].set_xlabel("x (depth, 0 = entrance)")
    axes[1].set_ylabel("y (lateral)")

    fig.tight_layout()
    fig.savefig(overlays_dir / f"beam{beam:02d}_{record['id']}.png", bbox_inches="tight")
    plt.close(fig)
