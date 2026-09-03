"""Batched model-input preparation == the per-spot loop it replaces.

``prepare_inputs_from_arrays_batched`` builds a whole batch's 2-channel ADoTA
input in one pass -- one contiguous (optionally pinned) host->device copy for the
CT channel and, when the flux was built on the device, no host round trip for the
flux -- instead of calling :func:`prepare_input_from_arrays` per spot and
``torch.stack``-ing. Every step is the same elementwise op or the same per-sample
reduction applied over the batch axis, so the result must be **bit-identical** to
the loop, on both the 1mm (``resize=True``) and 2mm (``resize=False``) paths and
whether or not the flux channel is min-max normalized.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.loaders.dir_based import (
    DEFAULT_SCALE,
    prepare_input_from_arrays,
    prepare_inputs_from_arrays_batched,
)

_ENERGIES = [96.5, 121.0, 150.25, 177.5]


def _crops(shape, n=4, seed=0):
    """``n`` CT (HU) + flux crops of ``shape`` = (z, y, x)."""
    rng = np.random.default_rng(seed)
    ct = [rng.uniform(-1024, 3000, size=shape).astype(np.float32) for _ in range(n)]
    flux = [rng.uniform(0.0, 0.05, size=shape).astype(np.float64) for _ in range(n)]
    return ct, flux


def _loop(ct, flux, energies, **kw):
    """The per-spot reference: prepare each spot, then stack."""
    pairs = [
        prepare_input_from_arrays(c, f, e, scale=DEFAULT_SCALE, **kw)
        for c, f, e in zip(ct, flux, energies)
    ]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


@pytest.mark.parametrize(
    "shape, resize",
    [((30, 30, 160), False), ((60, 60, 320), True)],
    ids=["2mm-no-resize", "1mm-resize"],
)
@pytest.mark.parametrize("normalize_flux", [True, False], ids=["norm", "raw"])
def test_batched_prep_is_bit_identical_to_per_spot_loop(shape, resize, normalize_flux):
    ct, flux = _crops(shape)
    kw = dict(normalize_flux=normalize_flux, downsampling_method="interpolation",
              device=None, resize=resize)
    x_ref, e_ref = _loop(ct, flux, _ENERGIES, **kw)
    x, e = prepare_inputs_from_arrays_batched(
        ct, flux, _ENERGIES, scale=DEFAULT_SCALE, **kw
    )
    assert x.shape == x_ref.shape == (len(ct), 2, 160, 30, 30)
    assert e.shape == e_ref.shape == (len(ct), 1)
    assert torch.equal(x, x_ref), (x - x_ref).abs().max().item()
    assert torch.equal(e, e_ref)


def test_batched_prep_accepts_a_device_tensor_flux_without_host_round_trip():
    """A flux already on the device is consumed in place (same values as numpy)."""
    shape = (30, 30, 160)
    ct, flux = _crops(shape)
    kw = dict(normalize_flux=True, downsampling_method="interpolation",
              device=None, resize=False)
    x_ref, _ = prepare_inputs_from_arrays_batched(
        ct, flux, _ENERGIES, scale=DEFAULT_SCALE, **kw
    )
    flux_tensor = torch.from_numpy(np.stack(flux))  # float64, "resident"
    x, _ = prepare_inputs_from_arrays_batched(
        ct, flux_tensor, _ENERGIES, scale=DEFAULT_SCALE, **kw
    )
    assert torch.equal(x, x_ref)


def test_batched_prep_into_a_reused_staging_buffer_matches():
    """The pinned/pre-allocated CT staging buffer changes nothing numerically."""
    shape = (30, 30, 160)
    ct, flux = _crops(shape)
    kw = dict(normalize_flux=True, downsampling_method="interpolation",
              device=None, resize=False)
    x_ref, e_ref = _loop(ct, flux, _ENERGIES, **kw)

    buffer = torch.empty((8, *shape), dtype=torch.float32)  # larger than the batch
    x, e = prepare_inputs_from_arrays_batched(
        ct, flux, _ENERGIES, scale=DEFAULT_SCALE, ct_buffer=buffer, **kw
    )
    assert torch.equal(x, x_ref) and torch.equal(e, e_ref)

    # Reusing the same buffer for a different batch must not leak the first one.
    ct2, flux2 = _crops(shape, seed=7)
    x2_ref, _ = _loop(ct2, flux2, _ENERGIES, **kw)
    x2, _ = prepare_inputs_from_arrays_batched(
        ct2, flux2, _ENERGIES, scale=DEFAULT_SCALE, ct_buffer=buffer, **kw
    )
    assert torch.equal(x2, x2_ref)
    assert not torch.equal(x2, x_ref)


def test_batched_prep_handles_a_short_final_batch():
    """The last batch is smaller than the buffer; only its rows are used."""
    shape = (30, 30, 160)
    ct, flux = _crops(shape, n=3)
    energies = _ENERGIES[:3]
    kw = dict(normalize_flux=True, downsampling_method="interpolation",
              device=None, resize=False)
    x_ref, e_ref = _loop(ct, flux, energies, **kw)
    buffer = torch.empty((16, *shape), dtype=torch.float32)
    x, e = prepare_inputs_from_arrays_batched(
        ct, flux, energies, scale=DEFAULT_SCALE, ct_buffer=buffer, **kw
    )
    assert x.shape[0] == 3
    assert torch.equal(x, x_ref) and torch.equal(e, e_ref)


def test_batched_prep_avg_pooling_matches_loop():
    """The non-default down-sampling method is batched identically too."""
    shape = (60, 60, 320)
    ct, flux = _crops(shape)
    kw = dict(normalize_flux=True, downsampling_method="avg_pooling",
              device=None, resize=True)
    x_ref, e_ref = _loop(ct, flux, _ENERGIES, **kw)
    x, e = prepare_inputs_from_arrays_batched(
        ct, flux, _ENERGIES, scale=DEFAULT_SCALE, **kw
    )
    assert torch.equal(x, x_ref) and torch.equal(e, e_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_batched_prep_on_cuda_matches_the_per_spot_cuda_loop():
    shape = (30, 30, 160)
    ct, flux = _crops(shape)
    device = torch.device("cuda")
    kw = dict(normalize_flux=True, downsampling_method="interpolation",
              device=device, resize=False)
    x_ref, e_ref = _loop(ct, flux, _ENERGIES, **kw)
    buffer = torch.empty((8, *shape), dtype=torch.float32, pin_memory=True)
    x, e = prepare_inputs_from_arrays_batched(
        ct, flux, _ENERGIES, scale=DEFAULT_SCALE, ct_buffer=buffer, **kw
    )
    assert x.is_cuda and e.is_cuda
    # ``prepare_input_from_arrays`` leaves the scalar energy on the host (the
    # streaming loop moves the stack afterwards); the batched twin returns it on
    # ``device`` directly, which is what the model is fed either way.
    assert torch.equal(x, x_ref)
    assert torch.equal(e.cpu(), e_ref)
