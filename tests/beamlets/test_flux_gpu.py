"""Equivalence tests: ``flux_projection`` (NumPy) vs ``flux_projection_gpu`` (Torch).

The GPU twin must reproduce the NumPy flux *exactly* (the ADoTA model was trained
on the NumPy numerics). We prove this at two levels:

* **Math identity** -- on the Torch CPU device (always available), the float64
  output matches NumPy to round-off (rtol 1e-12), and the float32 cast that the
  pipeline actually stores/consumes is **bit-identical** (``array_equal``).
* **CUDA parity** -- when a GPU is present, the CUDA output matches NumPy to
  float64 round-off and is bit-identical after the float32 cast.

Covering angled/non-angled beams, several sigmas, entrance offsets, shapes and the
``initial_energy`` multiply ensures the whole formula -- not just one path -- agrees.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.beamlets.flux import flux_projection, flux_projection_gpu

torch = pytest.importorskip("torch")

# (entrance[y,z,x], direction[theta_x,theta_y] deg, sigmas_xy, shape, initial_energy)
_CASES = [
    ([30.0, 30.0, 0.0], [0.0, 0.0], (4.0, 3.0), (12, 12, 40), None),
    ([28.5, 31.2, 1.0], [5.0, -3.0], (3.0, 2.0), (12, 14, 40), None),
    ([30.0, 30.0, 0.0], [-8.0, 6.5], (5.0, 5.0), (16, 16, 32), 150.0),
    ([10.0, 50.0, 2.0], [12.0, 0.0], (2.5, 4.5), (20, 18, 24), 0.37),
    ([30.0, 30.0, 0.0], [0.0, 9.0], (3.5, 3.5), (10, 10, 60), None),
]


def _gpu(case, device):
    entrance, direction, sigmas, shape, energy = case
    return flux_projection_gpu(entrance, direction, sigmas, shape, energy, device=device)


def _cpu(case):
    entrance, direction, sigmas, shape, energy = case
    return flux_projection(entrance, direction, sigmas, shape, energy)


@pytest.mark.parametrize("case", _CASES)
def test_torch_cpu_matches_numpy_float64(case) -> None:
    """Math identity: Torch (CPU, float64) reproduces NumPy to round-off."""
    ref = _cpu(case)
    got = _gpu(case, "cpu")
    assert got.shape == ref.shape
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("case", _CASES)
def test_torch_cpu_bit_identical_after_float32_cast(case) -> None:
    """The stored/consumed artifact (float32) is bit-identical to the NumPy path."""
    ref32 = _cpu(case).astype(np.float32)
    got32 = _gpu(case, "cpu").astype(np.float32)
    assert np.array_equal(got32, ref32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("case", _CASES)
def test_cuda_matches_numpy(case) -> None:
    """CUDA path matches NumPy to float64 round-off and is float32-bit-identical."""
    ref = _cpu(case)
    got = _gpu(case, "cuda")
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    assert np.array_equal(got.astype(np.float32), ref.astype(np.float32))


def test_default_energy_and_spacing_paths_agree() -> None:
    """No-energy default and an explicit spacing both agree CPU vs torch."""
    entrance, direction, sigmas, shape = [30.0, 30.0, 0.0], [4.0, -2.0], (3.3, 2.9), (12, 12, 30)
    spacing = np.asarray([2.0, 2.0, 1.0], dtype=np.float32)
    ref = flux_projection(entrance, direction, sigmas, shape, None, spacing)
    got = flux_projection_gpu(entrance, direction, sigmas, shape, None, spacing, device="cpu")
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
    assert np.array_equal(got.astype(np.float32), ref.astype(np.float32))
