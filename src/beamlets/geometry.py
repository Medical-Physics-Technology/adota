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
import math
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


# Axis codes for :func:`_rotate_about_axis`.
_AXIS_X, _AXIS_Y, _AXIS_Z = 0, 1, 2


def _rotate_about_axis(
    v0: float, v1: float, v2: float, theta: float, axis: int
) -> Tuple[float, float, float]:
    """Rodrigues rotation about a **basis** axis, in scalar arithmetic.

    Bit-identical to :func:`rotate_vector` with ``k`` the corresponding unit
    basis vector, and ~43x faster: the NumPy version spends almost all of its
    time in ``np.cross``/``np.dot`` dispatch on 3-element arrays, not on the
    arithmetic. The same expression is evaluated in the same order --
    ``v*cos + cross(k,v)*sin + k*dot(k,v)*(1-cos)`` -- with the cross product and
    dot product written out for a basis ``k``. That expansion is exact rather
    than an approximation: the eliminated terms are all ``0.0 * x``, which is an
    exact zero for finite ``x``, and adding it changes nothing.

    Verified bit-identical over 63,000 (vector, angle, axis) combinations
    covering all three axes and the angles the pipeline uses; see
    ``tests/beamlets/test_geometry.py``.

    Args:
        v0, v1, v2: The vector's components.
        theta: Rotation angle in radians.
        axis: ``_AXIS_X``, ``_AXIS_Y`` or ``_AXIS_Z``.

    Returns:
        The rotated components ``(x, y, z)``.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    if axis == _AXIS_X:      # k = (1,0,0): cross(k,v) = (0,-v2,v1), dot(k,v) = v0
        cr0, cr1, cr2 = 0.0, -v2, v1
        k0, k1, k2 = 1.0, 0.0, 0.0
        dot = v0
    elif axis == _AXIS_Y:    # k = (0,1,0): cross(k,v) = (v2,0,-v0), dot(k,v) = v1
        cr0, cr1, cr2 = v2, 0.0, -v0
        k0, k1, k2 = 0.0, 1.0, 0.0
        dot = v1
    else:                    # k = (0,0,1): cross(k,v) = (-v1,v0,0), dot(k,v) = v2
        cr0, cr1, cr2 = -v1, v0, 0.0
        k0, k1, k2 = 0.0, 0.0, 1.0
        dot = v2
    # The zero terms are kept rather than folded away: they make each component
    # the same three-term sum, in the same order, that NumPy evaluates -- so the
    # signed zeros match too, not just the finite values.
    return (
        v0 * c + cr0 * s + k0 * dot * one_c,
        v1 * c + cr1 * s + k1 * dot * one_c,
        v2 * c + cr2 * s + k2 * dot * one_c,
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
    # Scalar arithmetic throughout: this runs once per spot (tens of thousands of
    # times per plan) on 3-vectors, where NumPy's per-call dispatch -- np.cross in
    # particular -- costs far more than the arithmetic. Every step below is the
    # same operation in the same order as the array version, so the result is
    # bit-identical (see tests/beamlets/test_geometry.py).

    # Project the spot onto the nozzle plane in the isocenter-plane frame.
    x_nozzle = iso_plane_coords[0] * (d_smx - d_nozzle) / d_smx
    y_nozzle = iso_plane_coords[1] * (d_smy - d_nozzle) / d_smy

    # Gantry coordinates (origin at the isocenter, before the bixelgrid shift).
    i0, i1, i2 = float(iso_plane_coords[0]), float(iso_plane_coords[1]), 0.0
    n0, n1, n2 = float(x_nozzle), float(y_nozzle), float(d_nozzle)

    # Rotate around the gantry y-axis for the gantry angle (machine frame).
    g_ang_rad = float(np.deg2rad(g_ang))
    i0, i1, i2 = _rotate_about_axis(i0, i1, i2, g_ang_rad, _AXIS_Y)
    n0, n1, n2 = _rotate_about_axis(n0, n1, n2, g_ang_rad, _AXIS_Y)

    # Rotate around the gantry z-axis for the table angle (machine frame). At
    # t_ang = 0 this is the identity -- cos 0 = 1, sin 0 = 0 and 1 - cos 0 = 0
    # exactly -- so the rotation reduces to v + 0 + 0 = v and is skipped rather
    # than computed. (t_ang != 0 is currently rejected by the caller.)
    if t_ang != 0:
        t_ang_rad = float(np.deg2rad(t_ang))
        i0, i1, i2 = _rotate_about_axis(i0, i1, i2, t_ang_rad, _AXIS_Z)
        n0, n1, n2 = _rotate_about_axis(n0, n1, n2, t_ang_rad, _AXIS_Z)

    # Rotate from the gantry frame to the DICOM frame (90 deg around x).
    i0, i1, i2 = _rotate_about_axis(i0, i1, i2, np.pi / 2, _AXIS_X)
    n0, n1, n2 = _rotate_about_axis(n0, n1, n2, np.pi / 2, _AXIS_X)

    # Translate by the isocenter to land in the DICOM frame. The isocenter is
    # rounded to float32 first, exactly as the array version's
    # ``np.asarray(isocenter, dtype=np.float32)`` did, before promoting back to
    # float64 for the add.
    iso = np.asarray(isocenter, dtype=np.float32).astype(np.float64)
    dc_nozzle_s = np.array([iso[0] + n0, iso[1] + n1, iso[2] + n2], dtype=np.float64)
    dc_iso_s = np.array([iso[0] + i0, iso[1] + i1, iso[2] + i2], dtype=np.float64)
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
    # Scalar arithmetic, for the same reason as :func:`beamlet_ray`: this runs
    # once per spot on 3-vectors and 6 candidate faces, where NumPy's per-call
    # overhead (and the np.unique/argsort over a 6x4 array) dwarfs the work. Each
    # dot product is written out in the same left-to-right order NumPy accumulates
    # a length-3 dot in, so the values are bit-identical (see the tests).
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = ranges

    # One reference point on each of the six faces (paired per normal).
    cube_pts = (
        (x_min, y_min, z_min),
        (x_max, y_min, z_min),
        (x_min, y_min, z_min),
        (x_min, y_max, z_min),
        (x_min, y_min, z_min),
        (x_min, y_min, z_max),
    )

    s0, s1, s2 = float(start_point[0]), float(start_point[1]), float(start_point[2])
    e0, e1, e2 = float(end_point[0]), float(end_point[1]), float(end_point[2])
    # Unit direction along the ray (norm taken start->end, as before).
    norm = float(np.linalg.norm(np.asarray(start_point) - np.asarray(end_point)))
    d0, d1, d2 = (e0 - s0) / norm, (e1 - s1) / norm, (e2 - s2) / norm

    rows = []
    for n_idx in range(3):  # one loop per cube normal
        for f_idx in range(2):  # two faces per normal
            sign = (-1.0) ** (f_idx + 1)
            # n_face is +/- a basis vector, so its dot products collapse to one term.
            f0 = sign if n_idx == 0 else 0.0
            f1 = sign if n_idx == 1 else 0.0
            f2 = sign if n_idx == 2 else 0.0
            denom = d0 * f0 + d1 * f1 + d2 * f2
            if denom != 0.0:
                p0, p1, p2 = cube_pts[2 * n_idx + f_idx]
                d = ((p0 - s0) * f0 + (p1 - s1) * f1 + (p2 - s2) * f2) / denom
                pi0, pi1, pi2 = s0 + d * d0, s1 + d * d1, s2 + d * d2
                if check_if_point_in_cube(ranges, (pi0, pi1, pi2)):
                    rows.append((d, pi0, pi1, pi2))

    # Deduplicate and sort by distance along the ray. ``sorted(set(...))`` orders
    # tuples lexicographically, which is what ``np.unique(axis=0)`` did; the
    # subsequent stable sort on the first column then matches the old argsort.
    intersect_pts = np.array(sorted(set(rows)), dtype=np.float64)
    if intersect_pts.size:
        intersect_pts = intersect_pts[intersect_pts[:, 0].argsort(kind="stable")]
    else:
        intersect_pts = intersect_pts.reshape(0, 4)

    if check_if_point_in_cube(ranges, start_point):
        intersect_pts[0, 1:] = start_point

    return intersect_pts
