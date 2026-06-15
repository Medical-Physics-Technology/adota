"""Tests for :mod:`src.metrics.plan_metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.plan_metrics import high_dose_mask, plan_dose_metrics


def _ref(shape=(8, 10, 12)):
    ref = np.zeros(shape, dtype=np.float32)
    ref[4, 5, 3:9] = 10.0  # a high-dose track; the rest is out-of-field zeros
    ref[4, 5, 6] = 12.0  # a hotter voxel (p99 vs max differ)
    return ref


def test_high_dose_mask_threshold_is_frac_of_percentile() -> None:
    ref = _ref()
    p99 = float(np.percentile(ref, 99))
    mask = high_dose_mask(ref, frac=0.1, percentile=99.0)
    # Equivalent to a direct threshold at 10% of the 99th percentile.
    np.testing.assert_array_equal(mask, ref > 0.1 * p99)
    # The high-dose track is inside; the zero background is out.
    assert mask[4, 5, 6]
    assert not mask[0, 0, 0]


def test_perfect_match_zero_errors() -> None:
    ref = _ref()
    m = plan_dose_metrics(ref.copy(), ref)
    assert m["mape_pct"] == pytest.approx(0.0)
    assert m["rmse_gy"] == pytest.approx(0.0)
    assert m["relative_dose_error_pct"] == pytest.approx(0.0)
    assert m["n_mask_voxels"] > 0


def test_mape_and_rmse_restricted_to_mask() -> None:
    ref = _ref()
    eval_dose = ref * 1.1  # uniform 10% high inside the field (zeros stay zero)
    m = plan_dose_metrics(eval_dose, ref)
    # MAPE over the mask is exactly 10% (out-of-field zeros are excluded).
    assert m["mape_pct"] == pytest.approx(10.0, abs=1e-3)
    # RMSE over the mask = mean over masked voxels of (0.1*ref)^2, not diluted by zeros.
    mask = ref > 0.1 * float(np.percentile(ref, 99))
    expected_rmse = float(np.sqrt(np.mean((0.1 * ref[mask]) ** 2)))
    assert m["rmse_gy"] == pytest.approx(expected_rmse, rel=1e-5)


def test_relative_dose_error_is_whole_grid() -> None:
    ref = _ref()
    eval_dose = ref.copy()
    eval_dose[4, 5, 6] += 6.0  # one voxel off by 6 Gy
    m = plan_dose_metrics(eval_dose, ref)
    # RDE = (1/N) * L1(diff) / max(ref) * 100 over the WHOLE grid.
    n = ref.size
    expected = 6.0 / float(ref.max()) / n * 100.0
    assert m["relative_dose_error_pct"] == pytest.approx(expected, rel=1e-5)


def test_shape_mismatch_raises() -> None:
    ref = _ref()
    with pytest.raises(ValueError, match="share shape"):
        plan_dose_metrics(ref[..., :-1], ref)
