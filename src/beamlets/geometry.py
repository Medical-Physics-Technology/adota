"""Ray geometry for beamlet extraction.

Faithful port of the trusted functions from datagenerator's
``geometry/geometry_spatial_operations.py``:

* :func:`rotate_vector`  -- Rodrigues rotation of a 3-vector,
* :func:`beamlet_ray`    -- nozzle and isocenter points of a spot's ray in the
  DICOM frame,
* :func:`check_if_point_in_cube` / :func:`intersect_line_with_cube` -- ray/box
  intersection used to find where a beamlet enters the grid.

The maths is unchanged; only typing, logging and docstrings are cleaned up.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "rotate_vector",
    "beamlet_ray",
    "check_if_point_in_cube",
    "intersect_line_with_cube",
]


def rotate_vector(v: np.ndarray, theta: float, k: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` around unit vector ``k`` by ``theta`` radians.

    Uses Rodrigues' rotation formula
    (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).

    Args:
        v: Vector to rotate, shape ``(3,)``.
        theta: Rotation angle in radians.
        k: Unit vector to rotate around, shape ``(3,)``.

    Returns:
        The rotated vector, shape ``(3,)``.
    """
    return (
        v * np.cos(theta)
        + np.cross(k, v) * np.sin(theta)
        + k * np.dot(k, v) * (1 - np.cos(theta))
    )


def beamlet_ray(
    iso_plane_coords: Sequence[float],
    d_nozzle: float,
    d_smx: float,
    d_smy: float,
    g_ang: float,
    t_ang: float,
    isocenter: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a beamlet's nozzle and isocenter points in the DICOM frame.

    Args:
        iso_plane_coords: Spot position ``(x, y)`` in the isocenter plane
            (the bixelgrid shift).
        d_nozzle: Nozzle-to-isocenter distance.
        d_smx: Scanning-magnet-x-to-isocenter distance.
        d_smy: Scanning-magnet-y-to-isocenter distance.
        g_ang: Gantry angle in degrees.
        t_ang: Table (couch) angle in degrees. Currently must be ``0``.
        isocenter: Isocenter physical position ``(x, y, z)`` in the DICOM frame.

    Returns:
        Tuple ``(dc_nozzle_s, dc_iso_s)`` of physical points in the DICOM frame:
        the spot's nozzle-side point and its isocenter-plane point.
    """
    # Project the spot onto the nozzle plane in the isocenter-plane frame.
    x_nozzle = iso_plane_coords[0] * (d_smx - d_nozzle) / d_smx
    y_nozzle = iso_plane_coords[1] * (d_smy - d_nozzle) / d_smy

    # Gantry coordinates (origin at the isocenter, before the bixelgrid shift).
    gc_iso_s = np.array([*iso_plane_coords, 0], dtype=np.float64)
    gc_nozzle_s = np.array([x_nozzle, y_nozzle, d_nozzle], dtype=np.float64)

    # Rotate around the gantry y-axis for the gantry angle (machine frame).
    k_gantry = np.array([0, 1, 0], dtype=np.float32)
    g_ang_rad = np.deg2rad(g_ang)
    gc_iso_s_g = rotate_vector(gc_iso_s, g_ang_rad, k_gantry)
    gc_nozzle_s_g = rotate_vector(gc_nozzle_s, g_ang_rad, k_gantry)

    # Rotate around the gantry z-axis for the table angle (machine frame).
    k_table = np.array([0, 0, 1], dtype=np.float32)
    t_ang_rad = np.deg2rad(t_ang)
    gc_iso_s_t = rotate_vector(gc_iso_s_g, t_ang_rad, k_table)
    gc_nozzle_s_t = rotate_vector(gc_nozzle_s_g, t_ang_rad, k_table)

    # Rotate from the gantry frame to the DICOM frame (90 deg around x).
    k_dicom = np.array([1, 0, 0], dtype=np.float32)
    dc_iso_s_0 = rotate_vector(gc_iso_s_t, np.pi / 2, k_dicom)
    dc_nozzle_s_0 = rotate_vector(gc_nozzle_s_t, np.pi / 2, k_dicom)

    # Translate by the isocenter to land in the DICOM frame.
    iso = np.asarray(isocenter, dtype=np.float32)
    dc_nozzle_s = iso + dc_nozzle_s_0
    dc_iso_s = iso + dc_iso_s_0
    return dc_nozzle_s, dc_iso_s


def check_if_point_in_cube(
    ranges: Sequence[Tuple[float, float]], point: Sequence[float]
) -> bool:
    """Return whether ``point`` lies inside the axis-aligned box ``ranges``.

    Args:
        ranges: ``[(x_min, x_max), (y_min, y_max), (z_min, z_max)]``.
        point: Point ``(x, y, z)`` to test.

    Returns:
        ``True`` if the point is inside the box (within a small epsilon).
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = ranges
    # Epsilon buffer to absorb floating-point comparison error on the faces.
    eps = 1.0e-5
    return (
        x_min - eps <= point[0] <= x_max + eps
        and y_min - eps <= point[1] <= y_max + eps
        and z_min - eps <= point[2] <= z_max + eps
    )


def intersect_line_with_cube(
    ranges: Sequence[Tuple[float, float]],
    start_point: np.ndarray,
    end_point: np.ndarray,
) -> np.ndarray:
    """Find where the line through two points crosses an axis-aligned box.

    Args:
        ranges: ``[(x_min, x_max), (y_min, y_max), (z_min, z_max)]``.
        start_point: First point on the line ``(x, y, z)``.
        end_point: Second point on the line ``(x, y, z)``.

    Returns:
        Array of shape ``(n, 4)``: each row is ``(distance_along_ray, x, y, z)``,
        sorted by ascending distance. If ``start_point`` is inside the box, the
        first row's coordinates are replaced by ``start_point``.
    """
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = ranges

    # One reference point on each of the six faces (paired per normal).
    cube_pts = np.array(
        [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_min, y_min, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
        ]
    )

    # Unit direction along the ray.
    direction = (end_point - start_point) / np.linalg.norm(start_point - end_point)
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    intersect_pts = np.ones((6, 4)) * np.inf
    pt_idx = 0
    for n_idx in range(3):  # one loop per cube normal
        normal = normals[n_idx, :]
        for f_idx in range(2):  # two faces per normal
            n_face = normal * (-1) ** (f_idx + 1)
            p_0 = cube_pts[2 * n_idx + f_idx, :]
            if np.dot(direction, n_face) != 0.0:
                d = np.dot((p_0 - start_point), n_face) / np.dot(direction, n_face)
                p_i = start_point + d * direction
                if check_if_point_in_cube(ranges, p_i):
                    intersect_pts[pt_idx, 0] = d
                    intersect_pts[pt_idx, 1:] = p_i
                    pt_idx += 1

    # Drop unused rows, deduplicate, sort by distance along the ray.
    intersect_pts = intersect_pts[~np.all(intersect_pts == np.inf, axis=1)]
    intersect_pts = np.unique(intersect_pts, axis=0)
    intersect_pts = intersect_pts[intersect_pts[:, 0].argsort()]

    if check_if_point_in_cube(ranges, start_point):
        intersect_pts[0, 1:] = start_point

    return intersect_pts
