"""Sub-volume (ROI) extraction for beamlet inputs.

Port of datagenerator's ``cropp_around_spatial_point_with_np_indexes`` and
``get_roi_with_indexes_ct_only`` with one correctness fix: out-of-bounds regions
are **air-padded** (``-1024`` HU) and flagged, rather than producing a
wrong-shaped crop or wrapping around near the grid edge (decision 5). The dead
``ranges_of_bbox_dc`` / unused-``intersect`` calls from the source are dropped.

For an in-bounds spot the crop and entrance point are numerically identical to
the notebook; the returned ``crp`` is the centre voxel ``np_indexes = (z, y, x)``
exactly as the notebook saved it under ``crp_numpy_ct``.

ROI layout (matching the trained model and the datagenerator convention):

* ``roi_size = (H, W, D) = (60, 60, 320)``,
* the returned crop has numpy shape ``(z, y, x) = (H, W, D)``,
* the lateral ``H x W`` window is centred on the spot's ``(z, y)`` voxel, and
* the depth ``D`` runs along ``x`` taken as ``x in [0, D)`` from the ``x = 0``
  entrance face.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.beamlets import AIR_HU, ROI_SIZE
from src.beamlets.geometry import beamlet_ray, intersect_line_with_cube

logger = logging.getLogger(__name__)

__all__ = ["clip_axis_window", "crop_around_spatial_point", "extract_beamlet_roi"]


def clip_axis_window(start: int, length: int, size: int) -> Tuple[int, int, int, int]:
    """Clip a ``[start, start+length)`` window to ``[0, size)``.

    Shared by cropping (slicing the grid into a fixed-size crop) and accumulation
    (depositing a crop back into the grid), so both use identical in-bounds
    overlap maths.

    Returns ``(src_lo, src_hi, dst_lo, dst_hi)`` so that
    ``out[dst_lo:dst_hi] = arr[src_lo:src_hi]``; an empty overlap yields
    ``dst_lo == dst_hi``.
    """
    src_lo = max(start, 0)
    src_hi = min(start + length, size)
    if src_hi < src_lo:
        src_hi = src_lo
    dst_lo = src_lo - start
    dst_hi = src_hi - start
    return src_lo, src_hi, dst_lo, dst_hi


def crop_around_spatial_point(
    image: sitk.Image,
    spatial_point: Sequence[float],
    roi_size: Tuple[int, int, int] = ROI_SIZE,
    air_value: float = float(AIR_HU),
    ct_array: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[int], Tuple[int, int, int], bool]:
    """Crop an ``(H, W, D)`` ROI around a physical point, air-padding OOB regions.

    Args:
        image: The image to crop (rotated CT). Used for the physical-to-index
            transform regardless of ``ct_array``.
        spatial_point: Physical point ``(x, y, z)`` to centre the lateral window
            on (the spot's isocenter-plane point).
        roi_size: ``(H, W, D)`` ROI size.
        air_value: Fill value for out-of-bounds voxels.
        ct_array: Optional pre-computed ``sitk.GetArrayFromImage(image)`` result.
            When supplied it is used directly instead of re-copying the full grid
            (the per-field optimization); the output is bit-identical. When
            ``None`` the array is fetched from ``image`` as before.

    Returns:
        Tuple ``(crop, corner, np_indexes, oob)`` where ``crop`` has numpy shape
        ``(H, W, D)`` (``z, y, x``); ``corner = [x_start, y_start, z_start]`` is
        the crop's lower corner in image index space (used internally for the
        entrance-point frame); ``np_indexes`` is the centre voxel in ``(z, y, x)``
        order (the notebook's ``crp_numpy_ct``); and ``oob`` is ``True`` if any
        of the window fell outside the grid.
    """
    assert len(roi_size) == 3, f"roi_size must be (H, W, D), got {roi_size}"
    assert image.GetDimension() == 3, "ct image must be 3D"

    height, width, depth = roi_size
    indexes = image.TransformPhysicalPointToIndex([float(c) for c in spatial_point])
    ix, iy, iz = indexes
    np_indexes = indexes[::-1]  # (iz, iy, ix)

    arr = ct_array if ct_array is not None else sitk.GetArrayFromImage(image)  # (z, y, x)
    nz, ny, nx = arr.shape

    crop = np.full((height, width, depth), air_value, dtype=arr.dtype)

    # z axis (array axis 0) centred on iz; y axis (axis 1) centred on iy;
    # x axis (axis 2) taken from the 0 face for the full depth.
    z_src_lo, z_src_hi, z_dst_lo, z_dst_hi = clip_axis_window(iz - height // 2, height, nz)
    y_src_lo, y_src_hi, y_dst_lo, y_dst_hi = clip_axis_window(iy - width // 2, width, ny)
    x_src_lo, x_src_hi, x_dst_lo, x_dst_hi = clip_axis_window(0, depth, nx)

    crop[z_dst_lo:z_dst_hi, y_dst_lo:y_dst_hi, x_dst_lo:x_dst_hi] = arr[
        z_src_lo:z_src_hi, y_src_lo:y_src_hi, x_src_lo:x_src_hi
    ]

    oob = (
        (z_dst_hi - z_dst_lo) != height
        or (y_dst_hi - y_dst_lo) != width
        or (x_dst_hi - x_dst_lo) != depth
    )

    # Lower corner in image index space, (x, y, z) order (datagenerator layout):
    # [0, indexes[1] - H//2, indexes[2] - W//2].
    corner = [0, iy - height // 2, iz - width // 2]
    return crop, corner, np_indexes, oob


def extract_beamlet_roi(
    ct_scan: sitk.Image,
    d_nozzle: float,
    d_smx: float,
    d_smy: float,
    spot_position: Sequence[float],
    isocenter_physical: Sequence[float],
    roi_size: Tuple[int, int, int] = ROI_SIZE,
    ct_array: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], bool]:
    """Extract a spot's BEV CT crop and its beamlet entrance point.

    Port of ``get_roi_with_indexes_ct_only``: traces the spot ray with
    ``beamlet_ray`` (fixed ``g_ang = -90``), crops around the ray's
    isocenter-plane point, and expresses the grid-entrance point in crop-local
    voxel coordinates for the flux projection.

    Args:
        ct_scan: The rotated CT image.
        d_nozzle: Nozzle-to-isocenter distance.
        d_smx: Scanning-magnet-x-to-isocenter distance.
        d_smy: Scanning-magnet-y-to-isocenter distance.
        spot_position: Spot bixelgrid shift ``(x, y)``.
        isocenter_physical: Physical isocenter ``(x, y, z)``.
        roi_size: ``(H, W, D)`` ROI size.
        ct_array: Optional pre-computed ``sitk.GetArrayFromImage(ct_scan)``,
            forwarded to :func:`crop_around_spatial_point` so a field's grid is
            copied once rather than once per spot. Bit-identical when supplied.

    Returns:
        Tuple ``(cropped_ct, entrance_point, crp, oob)`` where ``entrance_point``
        is the beam's grid-entrance in crop-local voxel ``(x, y, z)`` coordinates
        and ``crp`` is the centre voxel ``(z, y, x)`` (the notebook's
        ``crp_numpy_ct``).
    """
    dc_nozzle_s, dc_iso_s = beamlet_ray(
        spot_position if len(spot_position) == 2 else spot_position[0],
        d_nozzle,
        d_smx,
        d_smy,
        -90,  # fixed: the CT is pre-rotated so the beam axis lies along x
        0,
        isocenter_physical,
    )

    cropped_ct, corner, np_indexes, oob = crop_around_spatial_point(
        ct_scan, dc_iso_s, roi_size, ct_array=ct_array
    )

    spacing = np.array(ct_scan.GetSpacing())
    origin = np.array(ct_scan.TransformIndexToPhysicalPoint([0, 0, 0]))
    max_coords = ct_scan.TransformIndexToPhysicalPoint(ct_scan.GetSize())
    ranges = [(o, m) for o, m in zip(origin, max_coords)]

    entrance_domain = intersect_line_with_cube(ranges, dc_nozzle_s, dc_iso_s)
    entrance_physical = entrance_domain[0][1:]

    # Crop lower corner in physical coords: x at the grid origin (crop x starts
    # at index 0), y/z at the crop's index start.
    shifted_origin = np.asarray(
        [
            origin[0],
            corner[1] * spacing[1] + origin[1],
            corner[2] * spacing[2] + origin[2],
        ]
    )
    entrance_point = (entrance_physical - shifted_origin) / spacing
    return cropped_ct, entrance_point, np_indexes, oob
