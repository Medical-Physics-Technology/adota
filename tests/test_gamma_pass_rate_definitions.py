"""Tests for the pass-rate arithmetic in :mod:`src.metrics.gamma_pass_rate`.

The gamma array is hand-built, NaN where the kernel would not have evaluated a
voxel, so these run without pymedphys and pin the two definitions directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.gamma_pass_rate import _gamma_pass_rate


def test_unevaluated_voxels_are_not_counted_as_passing() -> None:
    # Three unevaluated voxels, then gamma 0.5, 1.0 (pass), 1.5 (fail).
    gamma = np.array([np.nan, np.nan, np.nan, 0.5, 1.0, 1.5])
    zeroed, rates = _gamma_pass_rate(gamma.copy())
    # [1] is passing over evaluated: 2 of 3. The previous definition returned
    # 5/6, because nan_to_num turned each NaN into a 0 that satisfies <= 1.
    assert rates[1] == pytest.approx(2 / 3)
    # [0] is the historical definition, unchanged: exact-zero voxels dropped.
    assert rates[0] == pytest.approx(2 / 3)
    np.testing.assert_array_equal(zeroed, [0.0, 0.0, 0.0, 0.5, 1.0, 1.5])


def test_definitions_differ_only_on_exact_zero_gamma() -> None:
    # An evaluated voxel with gamma exactly 0 (a perfect match) passes under
    # [1] but is excluded from both counts of [0].
    gamma = np.array([np.nan, 0.0, 0.5, 1.5])
    _, rates = _gamma_pass_rate(gamma)
    assert rates[1] == pytest.approx(2 / 3)
    assert rates[0] == pytest.approx(1 / 2)


def test_nothing_evaluated_keeps_historical_zero_division() -> None:
    # Unchanged behaviour of [0]: with no voxel above 0 it divides by zero, and
    # every caller wraps the call and records NaN. Pinned so a later change to
    # [0] is a deliberate one.
    gamma = np.full((2, 3), np.nan)
    with pytest.raises(ZeroDivisionError):
        _gamma_pass_rate(gamma)


def test_3d_input_is_flattened_consistently() -> None:
    gamma = np.full((4, 4, 4), np.nan)
    gamma[1:3, 1:3, 1:3] = 0.9          # 8 passing
    gamma[0, 0, 0] = 1.2                # 1 failing
    _, rates = _gamma_pass_rate(gamma)
    assert rates[1] == pytest.approx(8 / 9)
    assert rates[0] == pytest.approx(8 / 9)
