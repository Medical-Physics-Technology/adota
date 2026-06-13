"""Unit + A/B parity tests for :mod:`src.beamlets.geometry`."""

from __future__ import annotations

import numpy as np

from src.beamlets.geometry import (
    beamlet_ray,
    check_if_point_in_cube,
    intersect_line_with_cube,
    rotate_vector,
)


def test_rotate_vector_preserves_norm() -> None:
    """Rotation preserves vector length."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(3)
    k = np.array([0.0, 0.0, 1.0])
    rotated = rotate_vector(v, np.deg2rad(37.0), k)
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(v))


def test_rotate_vector_zero_angle_is_identity() -> None:
    """A zero-angle rotation returns the input."""
    v = np.array([1.0, 2.0, 3.0])
    rotated = rotate_vector(v, 0.0, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(rotated, v)


def test_rotate_vector_90_around_z() -> None:
    """Rotating x-hat by 90 deg around z gives y-hat."""
    rotated = rotate_vector(
        np.array([1.0, 0.0, 0.0]), np.pi / 2, np.array([0.0, 0.0, 1.0])
    )
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)


def test_beamlet_ray_zero_shift_iso_point_is_isocenter() -> None:
    """A spot at the bixel origin has its iso-plane point at the isocenter."""
    iso = (10.0, -5.0, 3.0)
    _, dc_iso_s = beamlet_ray((0.0, 0.0), 420.0, 2014.9, 2584.1, -90, 0, iso)
    np.testing.assert_allclose(dc_iso_s, iso, atol=1e-4)


def test_beamlet_ray_zero_shift_nozzle_enters_from_low_x() -> None:
    """With g_ang=-90 the nozzle sits at iso_x - d_nozzle (beam enters +x)."""
    iso = np.array([100.0, 20.0, -7.0])
    d_nozzle = 420.0
    dc_nozzle_s, _ = beamlet_ray((0.0, 0.0), d_nozzle, 2014.9, 2584.1, -90, 0, iso)
    np.testing.assert_allclose(dc_nozzle_s, iso + np.array([-d_nozzle, 0, 0]), atol=1e-3)


def test_check_if_point_in_cube() -> None:
    """Inside, on-face, and outside points are classified correctly."""
    ranges = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    assert check_if_point_in_cube(ranges, [5.0, 5.0, 5.0])
    assert check_if_point_in_cube(ranges, [0.0, 10.0, 5.0])  # on a face
    assert not check_if_point_in_cube(ranges, [-1.0, 5.0, 5.0])


def test_intersect_line_with_cube_axis_aligned() -> None:
    """A line along x through a box yields the two x-face crossings."""
    ranges = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    start = np.array([-5.0, 5.0, 5.0])
    end = np.array([15.0, 5.0, 5.0])
    pts = intersect_line_with_cube(ranges, start, end)
    assert pts.shape[1] == 4
    # Entry at x=0, exit at x=10, both at y=z=5.
    np.testing.assert_allclose(pts[0, 1:], [0.0, 5.0, 5.0], atol=1e-9)
    np.testing.assert_allclose(pts[-1, 1:], [10.0, 5.0, 5.0], atol=1e-9)


def test_intersect_line_start_inside_uses_start_point() -> None:
    """When the start point is inside, the first row is the start point."""
    ranges = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    start = np.array([5.0, 5.0, 5.0])
    end = np.array([15.0, 5.0, 5.0])
    pts = intersect_line_with_cube(ranges, start, end)
    np.testing.assert_allclose(pts[0, 1:], start, atol=1e-9)


# --- A/B parity vs datagenerator -------------------------------------------


def test_parity_beamlet_ray(datagenerator_geometry) -> None:
    """beamlet_ray matches datagenerator for random spots/angles/isocenters."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        spot = tuple(rng.uniform(-50, 50, size=2))
        iso = tuple(rng.uniform(-200, 200, size=3))
        g_ang = float(rng.uniform(-180, 180))
        ours = beamlet_ray(spot, 420.0, 2014.9, 2584.1, g_ang, 0, iso)
        theirs = datagenerator_geometry.beamlet_ray(
            spot, 420.0, 2014.9, 2584.1, g_ang, 0, iso
        )
        np.testing.assert_allclose(ours[0], theirs[0], atol=1e-4)
        np.testing.assert_allclose(ours[1], theirs[1], atol=1e-4)


def test_parity_intersect_line_with_cube(datagenerator_geometry) -> None:
    """intersect_line_with_cube matches datagenerator on a synthetic box."""
    ranges = [(-30.0, 40.0), (-10.0, 60.0), (5.0, 90.0)]
    rng = np.random.default_rng(7)
    for _ in range(20):
        start = rng.uniform(-100, 100, size=3)
        end = rng.uniform(-100, 100, size=3)
        if np.allclose(start, end):
            continue
        ours = intersect_line_with_cube(ranges, start, end)
        theirs = datagenerator_geometry.intersect_line_with_cube(ranges, start, end)
        assert ours.shape == theirs.shape
        np.testing.assert_allclose(ours, theirs, atol=1e-6)
