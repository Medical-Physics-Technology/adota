"""Equivalence tests for the batched GPU reinterpretation kernels.

Both run on the CPU device (no CUDA required): the batched kernels must reproduce
the trusted per-item paths they optimise.
  * batched flux (float64)  == the frozen NumPy flux_projection, to round-off;
  * batched rotation (fp32) == the per-item torch grid_sample rotation.
"""
import numpy as np
import pytest
import torch

from src.beamlets.flux import flux_projection, flux_projection_gpu_batched
from src.image_processing.rotation import (
    rotate_beamlet_crop,
    rotate_beamlet_crops_batched,
)


def test_flux_batched_matches_numpy_per_spot():
    shape = (12, 10, 40)  # (z, y, x)
    spacing = np.asarray([2.0, 2.0, 2.0], dtype=np.float32)
    spots = [
        ([5.0, 4.0, 0.0], (0.0, 0.0), (3.0, 3.5)),
        ([6.5, 5.0, 0.0], (1.75, -5.18), (3.2, 3.0)),
        ([4.0, 6.0, 0.0], (-3.56, 4.75), (2.8, 3.1)),
    ]
    ent = [s[0] for s in spots]
    ang = [s[1] for s in spots]
    sig = [s[2] for s in spots]

    batched = flux_projection_gpu_batched(
        ent, ang, sig, shape, spacing=spacing, device="cpu",
        dtype=torch.float64, return_numpy=True,
    )
    assert batched.shape[0] == len(spots)
    for i, (e, a, s) in enumerate(spots):
        ref = flux_projection(e, a, s, shape, spacing=spacing)
        # float64 batched-matmul (bmm) accumulation differs from NumPy at ~1e-8,
        # far below the fp32 the channel is stored/consumed at.
        np.testing.assert_allclose(batched[i], ref, rtol=1e-6, atol=1e-7)


def test_flux_batched_energy_scaling():
    shape = (8, 8, 20)
    ent = [[4.0, 4.0, 0.0], [4.0, 4.0, 0.0]]
    ang = [(0.0, 0.0), (0.0, 0.0)]
    sig = [(3.0, 3.0), (3.0, 3.0)]
    energies = [1.0, 2.5]
    out = flux_projection_gpu_batched(
        ent, ang, sig, shape, initial_energies=energies, device="cpu",
        dtype=torch.float64, return_numpy=True,
    )
    np.testing.assert_allclose(out[1], out[0] * 2.5, rtol=1e-9, atol=1e-10)


def test_rotation_batched_matches_per_item():
    rng = np.random.default_rng(0)
    shape = (30, 30, 160)  # (z, y, x) model grid
    crops = [rng.standard_normal(shape).astype(np.float32) for _ in range(4)]
    angles = [(0.0, 0.0), (1.75, -5.18), (-3.56, 4.75), (2.0, 2.0)]

    for inverse in (False, True):
        batch = rotate_beamlet_crops_batched(
            crops, angles, inverse=inverse, device="cpu", repeats=1, return_numpy=True,
        )
        assert batch.rotated_numpy.shape == (len(crops), *shape)
        for i, (crop, ang) in enumerate(zip(crops, angles)):
            ref, _ = rotate_beamlet_crop(
                crop, ang, inverse=inverse, backend="torch", device="cpu", repeats=1,
            )
            np.testing.assert_allclose(batch.rotated_numpy[i], ref, rtol=1e-5, atol=1e-5)


def test_rotation_batched_validates_inputs():
    crop = np.zeros((4, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        rotate_beamlet_crops_batched([crop], [(0.0, 0.0), (1.0, 1.0)], device="cpu")
    with pytest.raises(ValueError):
        rotate_beamlet_crops_batched([], [], device="cpu")
