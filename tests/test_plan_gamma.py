"""Tests for :mod:`src.metrics.plan_gamma`."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.plan_gamma import criterion_label, parse_criteria, plan_gamma
from tests.utils.deps import requires_pymedphys_gamma


def _gaussian_dose(shape=(12, 16, 16)):
    """A smooth 3D Gaussian blob so pymedphys gamma runs quickly and meaningfully."""
    zz, yy, xx = np.indices(shape, dtype=np.float64)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return (10.0 * np.exp(-r2 / (2 * 3.0**2))).astype(np.float32)


# Small/fast gamma params for the synthetic grids.
_FAST = {"interp_fraction": 2, "max_gamma": 2}


def test_parse_criteria_and_label() -> None:
    crit = parse_criteria([[1, 1, 10], [3, 3, 0.1]])
    assert crit == [(1.0, 1.0, 10.0), (3.0, 3.0, 0.1)]
    assert criterion_label((1.0, 1.0, 10.0)) == "1%/1mm/10%"
    assert criterion_label((1.0, 3.0, 0.1)) == "1%/3mm/0.1%"


def test_parse_criteria_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="dose%, distance_mm, cutoff%"):
        parse_criteria([[1, 1]])


@requires_pymedphys_gamma
def test_near_match_passes_fully() -> None:
    # A tiny 0.1% scaling is well inside a 2%/2mm criterion -> all evaluated
    # voxels pass (gamma > 0 but <= 1), so the pass rate is 100%. (Exactly equal
    # doses give gamma == 0 everywhere, which is the degenerate 0/0 case upstream.)
    ref = _gaussian_dose()
    eval_dose = ref * 1.001
    results = plan_gamma(eval_dose, ref, (1.0, 1.0, 1.0), [(2.0, 2.0, 10.0)], _FAST)
    assert len(results) == 1
    assert results[0]["label"] == "2%/2mm/10%"
    assert results[0]["pass_rate_pct"] == pytest.approx(100.0, abs=1e-6)
    assert results[0]["gamma_map"].shape == ref.shape


@requires_pymedphys_gamma
def test_perturbed_dose_lowers_pass_rate() -> None:
    ref = _gaussian_dose()
    # A large, tight perturbation that a 1%/1mm criterion cannot tolerate.
    eval_dose = ref.copy()
    eval_dose[6, 8, 8] += 5.0
    eval_dose[6, 8, 9] -= 5.0
    results = plan_gamma(eval_dose, ref, (1.0, 1.0, 1.0), [(1.0, 1.0, 10.0)], _FAST)
    assert results[0]["pass_rate_pct"] < 100.0


@requires_pymedphys_gamma
def test_inputs_not_mutated() -> None:
    ref = _gaussian_dose()
    eval_dose = ref * 1.02
    ref_before = ref.copy()
    eval_before = eval_dose.copy()
    plan_gamma(eval_dose, ref, (1.0, 1.0, 1.0), [(2.0, 2.0, 10.0)], _FAST)
    # gamma_index zeroes sub-cutoff voxels in place; plan_gamma must pass copies.
    np.testing.assert_array_equal(ref, ref_before)
    np.testing.assert_array_equal(eval_dose, eval_before)


def test_shape_mismatch_raises() -> None:
    ref = _gaussian_dose()
    with pytest.raises(ValueError, match="share shape"):
        plan_gamma(ref[..., :-1], ref, (1.0, 1.0, 1.0), [(2.0, 2.0, 10.0)], _FAST)


@requires_pymedphys_gamma
def test_torch_backend_matches_pymedphys_backend() -> None:
    """The backend switch changes the kernel and nothing else.

    ``plan_gamma`` forwards ``backend`` to ``gamma_index``, which swaps only the
    computation of the gamma array; the pass-rate arithmetic downstream is shared
    verbatim. So both backends must report the same pass rate for the same plan.
    """
    import torch

    dose_ref = _gaussian_dose().astype(np.float64)
    dose_eval = (dose_ref * 1.005).astype(np.float64)
    criteria = [(2.0, 2.0, 10.0)]

    expected = plan_gamma(dose_eval, dose_ref, (1.0, 1.0, 1.0), criteria, _FAST)
    actual = plan_gamma(
        dose_eval, dose_ref, (1.0, 1.0, 1.0), criteria, _FAST,
        backend="torch",
        backend_options={"device": "cpu", "dtype": torch.float64},
    )
    assert actual[0]["pass_rate_pct"] == pytest.approx(
        expected[0]["pass_rate_pct"], abs=1e-9
    )
    np.testing.assert_allclose(
        actual[0]["gamma_map"], expected[0]["gamma_map"], rtol=0, atol=1e-10
    )


def test_unknown_backend_raises() -> None:
    from src.metrics.gamma_pass_rate import gamma_index

    dose = _gaussian_dose((6, 6, 6)).astype(np.float64)
    with pytest.raises(ValueError, match="Unknown gamma backend"):
        gamma_index(
            ground_truth=dose.copy(),
            prediction=dose.copy(),
            scale={"y_max": 10.0, "y_min": 0.0},
            gamma_params={"dose_percent_threshold": 2, "distance_mm_threshold": 2},
            backend="cupy",
        )


@requires_pymedphys_gamma
def test_torch_backend_5d_entry_point_matches_pymedphys() -> None:
    """The ``(N, C, D, H, W)`` entry point agrees across backends.

    The two paths reach the kernel differently: the pymedphys one calls
    ``.detach().cpu().numpy()`` on both volumes and de-normalises in numpy, while
    the torch one de-normalises and thresholds on whatever device the tensors are
    already on and brings back only the gamma array. They must still report the
    same pass rate.
    """
    import torch

    from src.metrics.gamma_pass_rate import gamma_index_torch

    scale = {"y_max": 10.0, "y_min": 0.0}
    normalised = torch.as_tensor(
        _gaussian_dose((10, 12, 12)) / scale["y_max"], dtype=torch.float64
    )[None, None]
    perturbed = normalised * 1.005
    params = {"dose_percent_threshold": 2, "distance_mm_threshold": 2, **_FAST}

    _, expected = gamma_index_torch(normalised, perturbed, scale, params)
    _, actual = gamma_index_torch(
        normalised, perturbed, scale, params,
        backend="torch",
        backend_options={"device": "cpu", "dtype": torch.float64},
    )
    assert actual[0] == pytest.approx(expected[0], abs=1e-9)
    assert actual[1] == pytest.approx(expected[1], abs=1e-9)
