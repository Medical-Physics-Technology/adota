"""Parity tests for the torch gamma kernel against ``pymedphys.gamma``.

The plan corpus exercises 3D global gamma with ``local_gamma=False``, which is
every criterion the benchmark reports. The options it does *not* exercise --
``local_gamma``, ``skip_once_passed``, ``random_subset``, and the 1D/2D cases --
are covered here against pymedphys directly, on grids small enough for the unit
suite.

Everything runs on the torch CPU device: the kernel is device-agnostic, and a
CPU-only machine must still be able to check that it is correct.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pymedphys import gamma as pymedphys_gamma

from src.metrics.gamma_torch import calculate_coordinates_shell, gamma_torch
from tests.utils.deps import requires_pymedphys_gamma

# float64 in, float64 out: pymedphys stores its per-shell minimum dose difference
# in an array shaped like the reference dose, so a float32 dose quietly quantises
# that minimum and the two implementations then differ at ~1e-7 in gamma. Feeding
# float64 doses removes that and leaves only float64 rounding.
PARITY_TOLERANCE = 1e-10

BASE_PARAMS = dict(
    lower_percent_dose_cutoff=20,
    interp_fraction=5,
    max_gamma=2,
)


def _doses(shape, seed=11, noise=0.3):
    """A reference dose and a perturbed evaluation dose, both float64."""
    rng = np.random.default_rng(seed)
    reference = rng.random(shape) * 10.0
    return reference, reference + rng.normal(0.0, noise, shape)


def _axes(shape, spacing):
    return tuple(np.arange(size) * step for size, step in zip(shape, spacing))


def _assert_matches_pymedphys(shape, spacing, dose_percent, distance_mm, **params):
    """Both implementations agree on gamma and on which voxels are NaN."""
    axes = _axes(shape, spacing)
    reference, evaluation = _doses(shape)

    expected = pymedphys_gamma(
        axes, reference, axes, evaluation, dose_percent, distance_mm, **params
    )
    actual = gamma_torch(
        axes,
        reference,
        axes,
        evaluation,
        dose_percent,
        distance_mm,
        device="cpu",
        dtype=torch.float64,
        **params,
    )

    expected_nan, actual_nan = np.isnan(expected), np.isnan(actual)
    assert np.array_equal(expected_nan, actual_nan), (
        f"{int((expected_nan ^ actual_nan).sum())} voxels disagree about whether "
        "gamma was evaluated"
    )
    evaluated = ~expected_nan
    assert evaluated.any(), "the test case evaluated no voxels at all"
    np.testing.assert_allclose(
        actual[evaluated], expected[evaluated], rtol=0, atol=PARITY_TOLERANCE
    )


@requires_pymedphys_gamma
@pytest.mark.parametrize(
    "shape,spacing",
    [
        ((10, 11, 12), (1.0, 1.0, 1.0)),
        ((9, 10, 8), (1.5, 2.0, 1.0)),  # anisotropic spacing
        ((30, 32), (1.0, 1.0)),
        ((120,), (1.0,)),
    ],
    ids=["3d-isotropic", "3d-anisotropic", "2d", "1d"],
)
def test_matches_pymedphys_across_dimensions(shape, spacing):
    _assert_matches_pymedphys(shape, spacing, 2, 2, **BASE_PARAMS)


@requires_pymedphys_gamma
def test_matches_pymedphys_local_gamma():
    _assert_matches_pymedphys(
        (10, 10, 10), (1.0, 1.0, 1.0), 2, 2, local_gamma=True, **BASE_PARAMS
    )


@requires_pymedphys_gamma
def test_matches_pymedphys_local_gamma_without_a_cutoff():
    """With no cutoff, local gamma divides by zero-dose voxels.

    pymedphys lets that produce ``inf`` (or ``nan`` for 0/0) under a suppressed
    numpy warning. The behaviour is matched rather than guarded away, so a caller
    who passes ``lower_percent_dose_cutoff=0`` gets the same answer from both.
    """
    axes = _axes((8, 8, 8), (1.0, 1.0, 1.0))
    reference, evaluation = _doses((8, 8, 8))
    reference[:2] = 0.0  # exact zeros for the local normalisation to divide by

    params = dict(
        lower_percent_dose_cutoff=0, interp_fraction=5, max_gamma=2, local_gamma=True
    )
    expected = pymedphys_gamma(axes, reference, axes, evaluation, 2, 2, **params)
    actual = gamma_torch(
        axes, reference, axes, evaluation, 2, 2,
        device="cpu", dtype=torch.float64, **params,
    )
    assert np.array_equal(np.isnan(expected), np.isnan(actual))
    evaluated = ~np.isnan(expected)
    np.testing.assert_allclose(
        actual[evaluated], expected[evaluated], rtol=0, atol=PARITY_TOLERANCE
    )


@requires_pymedphys_gamma
def test_matches_pymedphys_skip_once_passed():
    _assert_matches_pymedphys(
        (10, 10, 10), (1.0, 1.0, 1.0), 3, 3, skip_once_passed=True, **BASE_PARAMS
    )


@requires_pymedphys_gamma
def test_matches_pymedphys_without_max_gamma():
    """``max_gamma=None`` means an unbounded search, so the loop must terminate."""
    _assert_matches_pymedphys(
        (8, 8, 8), (1.0, 1.0, 1.0), 3, 3,
        lower_percent_dose_cutoff=20, interp_fraction=4,
    )


@requires_pymedphys_gamma
def test_matches_pymedphys_with_explicit_global_normalisation():
    _assert_matches_pymedphys(
        (10, 10, 10), (1.0, 1.0, 1.0), 2, 2,
        global_normalisation=8.0, **BASE_PARAMS,
    )


@requires_pymedphys_gamma
def test_random_subset_evaluates_that_many_points():
    """``random_subset`` restricts the evaluated voxels, leaving the rest NaN.

    The subset is drawn from numpy's global RNG, as pymedphys does, so seeding
    numpy makes the choice reproducible. The two implementations are not asked to
    pick the *same* subset here -- only to honour its size.
    """
    axes = _axes((10, 10, 10), (1.0, 1.0, 1.0))
    reference, evaluation = _doses((10, 10, 10))

    np.random.seed(0)
    full = gamma_torch(
        axes, reference, axes, evaluation, 2, 2,
        device="cpu", dtype=torch.float64, **BASE_PARAMS,
    )
    subset_size = 50
    np.random.seed(0)
    subset = gamma_torch(
        axes, reference, axes, evaluation, 2, 2,
        device="cpu", dtype=torch.float64, random_subset=subset_size, **BASE_PARAMS,
    )

    assert np.count_nonzero(~np.isnan(full)) > subset_size
    assert np.count_nonzero(~np.isnan(subset)) == subset_size
    # Every evaluated subset voxel must carry the value the full run gave it.
    evaluated = ~np.isnan(subset)
    np.testing.assert_allclose(
        subset[evaluated], full[evaluated], rtol=0, atol=PARITY_TOLERANCE
    )


@requires_pymedphys_gamma
def test_float32_stays_close_to_float64():
    """float32 is the cheap default; it must not move gamma meaningfully."""
    axes = _axes((10, 10, 10), (1.0, 1.0, 1.0))
    reference, evaluation = _doses((10, 10, 10))
    kwargs = dict(device="cpu", **BASE_PARAMS)

    wide = gamma_torch(axes, reference, axes, evaluation, 2, 2,
                       dtype=torch.float64, **kwargs)
    narrow = gamma_torch(axes, reference, axes, evaluation, 2, 2,
                         dtype=torch.float32, **kwargs)
    assert np.array_equal(np.isnan(wide), np.isnan(narrow))
    evaluated = ~np.isnan(wide)
    np.testing.assert_allclose(
        narrow[evaluated], wide[evaluated], rtol=0, atol=1e-4
    )


def test_out_of_bounds_voxels_fall_back_to_the_distance_term():
    """Beyond the evaluation grid, gamma is whatever the search distance costs.

    Out-of-bounds interpolation fills ``+inf``, which loses every minimum, so a
    reference voxel outside the evaluation grid can only be matched by reaching
    back inside it. Both doses are a uniform 5 Gy here, so the dose term is zero
    and gamma is purely ``r / distance_mm_threshold`` for the first search radius
    that reaches the grid. Past ``distance_mm_threshold * max_gamma`` nothing is
    reachable and the voxel is reported as NaN rather than as a number derived
    from a fabricated dose.
    """
    reference_axes = _axes((3, 3, 12), (1.0, 1.0, 1.0))
    evaluation_axes = _axes((3, 3, 6), (1.0, 1.0, 1.0))
    reference = np.full((3, 3, 12), 5.0)
    evaluation = np.full((3, 3, 6), 5.0)

    result = gamma_torch(
        reference_axes, reference, evaluation_axes, evaluation, 2, 2,
        device="cpu", dtype=torch.float64, lower_percent_dose_cutoff=10,
        interp_fraction=5, max_gamma=2,
    )
    # The evaluation grid stops at index 5. Radii step by 2/5 = 0.4 mm and snap
    # to 2 mm, so index 6 (1 mm out) is first reached at r = 1.2 -> gamma 0.6,
    # index 7 at the forced r = 2 -> 1.0, index 8 at 3.2 -> 1.6, index 9 at the
    # last radius 4 -> 2.0. Index 10 onwards is out of reach.
    np.testing.assert_allclose(result[:, :, :6], 0.0, atol=PARITY_TOLERANCE)
    np.testing.assert_allclose(
        result[0, 0, 6:10], [0.6, 1.0, 1.6, 2.0], rtol=0, atol=1e-12
    )
    assert np.isnan(result[:, :, 10:]).all()


def test_sequence_thresholds_raise_not_implemented():
    axes = _axes((6, 6, 6), (1.0, 1.0, 1.0))
    dose = np.ones((6, 6, 6))
    with pytest.raises(NotImplementedError, match="dose_percent_threshold"):
        gamma_torch(axes, dose, axes, dose, [1, 2], 2, device="cpu")
    with pytest.raises(NotImplementedError, match="distance_mm_threshold"):
        gamma_torch(axes, dose, axes, dose, 2, [1, 2], device="cpu")


def test_non_uniform_axes_raise_a_named_error():
    """The floor-based voxel lookup assumes uniformity, so it is checked."""
    axes = (np.array([0.0, 1.0, 2.5, 3.0]), np.arange(4.0), np.arange(4.0))
    dose = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="uniformly spaced"):
        gamma_torch(axes, dose, axes, dose, 2, 2, device="cpu")


def test_descending_axes_raise_a_named_error():
    axes = (np.arange(4.0)[::-1].copy(), np.arange(4.0), np.arange(4.0))
    dose = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="monotonically"):
        gamma_torch(axes, dose, axes, dose, 2, 2, device="cpu")


def test_mismatched_axes_and_dose_shape_raise():
    axes = _axes((4, 4, 4), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="axes_reference"):
        gamma_torch(axes, np.ones((4, 4, 5)), axes, np.ones((4, 4, 4)), 2, 2,
                    device="cpu")


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_zero_radius_shell_is_the_origin(ndim):
    """At r = 0 the search must sample the reference point itself, exactly once."""
    shell = calculate_coordinates_shell(0.0, ndim, 0.5)
    assert len(shell) == ndim
    for offsets in shell:
        assert offsets.shape == (1,)
        assert offsets[0] == 0.0


def test_shell_spacing_never_exceeds_the_step():
    """No gap on the sphere may be wider than the requested step size."""
    step = 0.6
    shell = calculate_coordinates_shell(3.0, 3, step)
    points = np.stack(shell, axis=1)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 3.0, atol=1e-9)
    # Every point has a neighbour within the step, so the surface has no hole.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min(axis=1).max() <= step + 1e-9


def test_stats_dict_is_populated():
    axes = _axes((8, 8, 8), (1.0, 1.0, 1.0))
    reference, evaluation = _doses((8, 8, 8))
    stats: dict = {}
    gamma_torch(
        axes, reference, axes, evaluation, 2, 2,
        device="cpu", dtype=torch.float64, stats=stats, **BASE_PARAMS,
    )
    assert stats["iterations"] > 1
    assert stats["shell_points"] > stats["iterations"]
    assert stats["interp_samples"] > 0
