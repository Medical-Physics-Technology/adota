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

from src.beamlets.flux import (
    flux_projection,
    flux_projection_gpu,
    flux_projection_gpu_batched,
)

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


# --- Batched twin -------------------------------------------------------------
# The streaming pipeline builds a whole batch's flux in one call. In float64 that
# is the same math as the per-spot path, only reassociated by the batched matmul,
# so it is *not* bit-identical -- it is equal to round-off. The cases below use
# the real gf=2 beamlet geometry (entrance inside the crop, 2mm spacing) rather
# than the tail-only synthetic ``_CASES``, because agreement is only meaningful
# where the projection carries dose: a Gaussian decays to ~1e-36 in its tail,
# where an absolute round-off difference is a huge *relative* one on a value no
# downstream step can represent. Tolerances are therefore stated relative to the
# projection's peak, plus the quantity that actually reaches the model: the crop
# min-max normalized to [0, 1].

_BEAMLET_SHAPE = (30, 30, 160)  # roi_for_factor(2)
_BEAMLET_SPACING = np.asarray([2, 2, 2], dtype=np.float32)
# (entrance[y,z,x], direction[theta_x,theta_y] deg, sigmas_xy)
_BEAMLETS = [
    ([15.0, 15.0, 0.0], [0.0, 0.0], (4.0, 3.0)),
    ([14.2, 16.1, 0.5], [3.0, -2.0], (3.0, 2.0)),
    ([15.8, 14.4, 0.0], [-5.0, 4.0], (5.0, 5.0)),
    ([13.0, 17.0, 1.0], [7.0, 1.5], (2.5, 4.5)),
]


def _beamlet_reference(device: str = None) -> np.ndarray:
    """Per-spot reference: NumPy, or the per-spot GPU twin when ``device`` is set."""
    if device is None:
        return np.stack([
            flux_projection(e, a, s, _BEAMLET_SHAPE, None, _BEAMLET_SPACING)
            for e, a, s in _BEAMLETS
        ])
    return np.stack([
        flux_projection_gpu(e, a, s, _BEAMLET_SHAPE, None, _BEAMLET_SPACING, device=device)
        for e, a, s in _BEAMLETS
    ])


def _beamlet_batched(device: str, dtype) -> np.ndarray:
    return flux_projection_gpu_batched(
        [b[0] for b in _BEAMLETS], [b[1] for b in _BEAMLETS], [b[2] for b in _BEAMLETS],
        _BEAMLET_SHAPE, spacing=_BEAMLET_SPACING, device=device, dtype=dtype,
        return_numpy=True,
    )


def _normalized(batch: np.ndarray) -> np.ndarray:
    """The channel the model is fed: each crop min-max scaled to [0, 1], float32."""
    flat = batch.astype(np.float32).reshape(batch.shape[0], -1)
    lo = flat.min(axis=1, keepdims=True)
    hi = flat.max(axis=1, keepdims=True)
    return (flat - lo) / (hi - lo)


# One float32 ulp on [0, 1] -- the resolution the normalized channel is stored at.
_F32_ULP = float(np.finfo(np.float32).eps)


def test_batched_float64_matches_the_per_spot_numpy_flux() -> None:
    """Batched float64 == the NumPy reference to round-off, relative to the peak."""
    ref = _beamlet_reference()
    got = _beamlet_batched("cpu", torch.float64)
    assert got.shape == ref.shape and got.dtype == np.float64
    # Measured 1.7e-8 of the peak; well under a float32 ulp (1.2e-7).
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-7 * ref.max())


def test_batched_float64_normalized_channel_is_within_one_float32_ulp() -> None:
    """The model's flux channel is the same to the precision it is stored at."""
    ref, got = _normalized(_beamlet_reference()), _normalized(_beamlet_batched("cpu", torch.float64))
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=2 * _F32_ULP)


def test_batched_float32_is_an_order_of_magnitude_less_exact() -> None:
    """float32 is the faster, looser option -- documented, not used by default."""
    ref = _beamlet_reference()
    f64 = _beamlet_batched("cpu", torch.float64)
    f32 = _beamlet_batched("cpu", torch.float32)
    np.testing.assert_allclose(f32, ref, rtol=0.0, atol=1e-5 * ref.max())
    assert np.abs(f32 - ref).max() > 10 * np.abs(f64 - ref).max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_batched_float64_matches_the_per_spot_gpu_flux() -> None:
    """What the streaming pipeline actually swaps: per-spot GPU -> batched GPU."""
    ref = _beamlet_reference(device="cuda")
    got = _beamlet_batched("cuda", torch.float64)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-7 * ref.max())
    np.testing.assert_allclose(
        _normalized(got), _normalized(ref), rtol=0.0, atol=2 * _F32_ULP
    )
