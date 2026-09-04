"""Unit + A/B parity tests for :mod:`src.beamlets.geometry`."""

from __future__ import annotations

import numpy as np
import pytest

from src.beamlets import geometry
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


# --- Scalar fast paths ---------------------------------------------------------
# `beamlet_ray` and `intersect_line_with_cube` run once per spot (tens of
# thousands of times per plan) on 3-vectors, where NumPy's per-call dispatch --
# `np.cross` above all -- costs far more than the arithmetic. Both were rewritten
# in scalar arithmetic that evaluates the same expressions in the same order, so
# the results must be BIT-identical, not merely close. These tests pin that down;
# a tolerance here would defeat their purpose.

def _rodrigues_numpy(v, theta, k):
    """The original array formulation, kept here as the reference."""
    return (
        v * np.cos(theta)
        + np.cross(k, v) * np.sin(theta)
        + k * np.dot(k, v) * (1 - np.cos(theta))
    )


@pytest.mark.parametrize("axis, k", [
    (geometry._AXIS_X, np.array([1, 0, 0], dtype=np.float32)),
    (geometry._AXIS_Y, np.array([0, 1, 0], dtype=np.float32)),
    (geometry._AXIS_Z, np.array([0, 0, 1], dtype=np.float32)),
])
@pytest.mark.parametrize("deg", [0.0, 90.0, -90.0, 180.0, 37.5, -123.25])
def test_scalar_rodrigues_is_bit_identical_to_the_array_formula(axis, k, deg) -> None:
    theta = np.deg2rad(deg)
    rng = np.random.default_rng(int(abs(deg)) + axis)
    for v in rng.uniform(-400.0, 400.0, size=(200, 3)):
        ref = _rodrigues_numpy(v, theta, k)
        got = geometry._rotate_about_axis(float(v[0]), float(v[1]), float(v[2]), theta, axis)
        assert tuple(ref) == got, (v, deg, axis)


def test_scalar_rodrigues_matches_the_public_rotate_vector() -> None:
    """The public helper is unchanged; the scalar twin must agree with it."""
    k = np.array([0, 1, 0], dtype=np.float32)
    v = np.array([12.5, -7.25, 3.125])
    theta = np.deg2rad(-90.0)
    assert tuple(rotate_vector(v, theta, k)) == geometry._rotate_about_axis(
        v[0], v[1], v[2], theta, geometry._AXIS_Y
    )


@pytest.mark.parametrize("g_ang", [-90.0, 0.0, 90.0, 180.0, 45.0])
def test_beamlet_ray_zero_table_angle_equals_an_explicit_identity_rotation(g_ang) -> None:
    """Skipping the t_ang=0 rotation is exact, not an approximation.

    At theta=0, cos=1, sin=0 and 1-cos=0 exactly, so Rodrigues reduces to the
    identity. Rotating explicitly by 0 must give the same bits as not rotating.
    """
    iso = (10.0, -20.0, 30.0)
    for pos in [(0.0, 0.0), (12.5, -8.25), (-30.0, 44.75)]:
        skipped = beamlet_ray(pos, 500.0, 2000.0, 2500.0, g_ang, 0, iso)
        # An explicit zero-angle rotation of the same intermediate vectors.
        k = np.array([0, 0, 1], dtype=np.float32)
        for vec in skipped:
            assert np.array_equal(rotate_vector(vec, 0.0, k), vec)


def test_beamlet_ray_returns_float64_arrays_of_length_three() -> None:
    """The scalar rewrite must not change the returned type or shape."""
    n, i = beamlet_ray((3.0, -4.0), 500.0, 2000.0, 2500.0, -90, 0, (1.0, 2.0, 3.0))
    for out in (n, i):
        assert isinstance(out, np.ndarray) and out.shape == (3,)
        assert out.dtype == np.float64


def test_intersect_returns_empty_four_column_array_when_the_ray_misses() -> None:
    """A ray that never enters the box yields a well-shaped empty result."""
    ranges = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    start = np.array([100.0, 100.0, 100.0])
    end = np.array([100.0, 100.0, 200.0])
    out = intersect_line_with_cube(ranges, start, end)
    assert out.shape == (0, 4)
