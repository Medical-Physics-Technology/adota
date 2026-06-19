"""Resize-skip in prepare/postprocess for the field-resampling (grid_factor=2) path.

The model always consumes the ``(160, 30, 30)`` grid. The two extraction paths only
differ in whether a resize is needed to reach it:

* ``grid_factor=1`` (default): the 1mm crop ``(60, 60, 320)`` permutes to
  ``(320, 60, 60)`` and is trilinearly down-sampled to ``(160, 30, 30)``.
* ``grid_factor=2``: the crop was already taken on the 2mm grid, so it permutes
  straight to ``(160, 30, 30)`` and the resize is skipped.

These tests prove (a) ``grid_factor=1`` stays byte-identical (F2 guard) and
(b) the skip (``resize=False`` / ``upsample=False``) is **numerically identical**
to running the trilinear resize to-self -- i.e. skipping changes nothing, it only
avoids a wasted no-op resize.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.loaders.dir_based import (
    DEFAULT_SCALE,
    postprocess_prediction,
    prepare_input_from_arrays,
)
from src.utils.scallers import inverse_minmax

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _rng_crops(shape, seed=0):
    rng = np.random.default_rng(seed)
    ct = rng.uniform(-1000.0, 2000.0, size=shape).astype(np.float32)
    flux = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    return ct, flux


def test_grid_factor1_prepare_is_byte_identical_to_explicit_resize() -> None:
    """resize=True on a 1mm crop reproduces the exact (permute -> trilinear) pipeline."""
    ct, flux = _rng_crops((60, 60, 320), seed=1)
    x, energy = prepare_input_from_arrays(ct, flux, 150.0, resize=True)
    assert x.shape == (2, 160, 30, 30)

    # Reproduce the documented pipeline independently.
    s = DEFAULT_SCALE
    ctn = (torch.tensor(ct) - s["min_ct"]) / (s["max_ct"] - s["min_ct"])
    fln = torch.tensor(flux)
    ctn = ctn.permute(2, 0, 1).unsqueeze(0)
    fln = fln.permute(2, 0, 1).unsqueeze(0)
    fln = (fln - fln.min()) / (fln.max() - fln.min())
    ctn = F.interpolate(ctn.unsqueeze(0), size=(160, 30, 30), mode="trilinear",
                        align_corners=False).squeeze(0)
    fln = F.interpolate(fln.unsqueeze(0), size=(160, 30, 30), mode="trilinear",
                        align_corners=False).squeeze(0)
    expected = torch.cat((ctn, fln), dim=0)
    assert torch.equal(x, expected)


def test_resize_skip_equals_resize_to_self_on_2mm_crop() -> None:
    """resize=False on a (30,30,160) 2mm crop == resize=True (a trilinear no-op).

    A trilinear resize to the same size (align_corners=False) samples at the
    integer grid -> identity. So skipping the resize must be bit-identical to
    running it; the skip only avoids the wasted op.
    """
    ct, flux = _rng_crops((30, 30, 160), seed=2)  # 2mm crop -> permutes to (160,30,30)
    x_skip, e_skip = prepare_input_from_arrays(ct, flux, 150.0, resize=False)
    x_self, e_self = prepare_input_from_arrays(ct, flux, 150.0, resize=True)
    assert x_skip.shape == (2, 160, 30, 30)
    assert torch.equal(x_skip, x_self)
    assert torch.equal(e_skip, e_self)


def test_resize_skip_is_pure_permute_no_interpolation() -> None:
    """The CT channel of a skipped 2mm crop is exactly the normalized, permuted crop."""
    ct, flux = _rng_crops((30, 30, 160), seed=3)
    x, _ = prepare_input_from_arrays(ct, flux, 150.0, resize=False)
    s = DEFAULT_SCALE
    ct_expected = ((torch.tensor(ct) - s["min_ct"]) / (s["max_ct"] - s["min_ct"])
                   ).permute(2, 0, 1)
    assert torch.equal(x[0], ct_expected)  # no interpolation touched the CT values


def test_postprocess_upsample_skip_keeps_model_grid_and_denorms() -> None:
    """upsample=False keeps (160,30,30) and de-normalizes identically per voxel."""
    rng = np.random.default_rng(4)
    pred = torch.tensor(rng.uniform(0, 1, size=(1, 1, 160, 30, 30)).astype(np.float32))
    out = postprocess_prediction(pred, upsample=False)
    assert out.shape == (1, 1, 160, 30, 30)
    expected = inverse_minmax(pred.numpy(), DEFAULT_SCALE["min_ds"], DEFAULT_SCALE["max_ds"])
    np.testing.assert_array_equal(out, expected)


def test_postprocess_upsample_true_is_byte_identical_to_explicit() -> None:
    """upsample=True reproduces the exact (320,60,60) trilinear up-sample + de-norm (F2)."""
    rng = np.random.default_rng(5)
    pred = torch.tensor(rng.uniform(0, 1, size=(1, 1, 160, 30, 30)).astype(np.float32))
    out = postprocess_prediction(pred, upsample=True)
    up = F.interpolate(pred, size=(320, 60, 60), mode="trilinear", align_corners=False)
    expected = inverse_minmax(up.numpy(), DEFAULT_SCALE["min_ds"], DEFAULT_SCALE["max_ds"])
    assert out.shape == (1, 1, 320, 60, 60)
    np.testing.assert_array_equal(out, expected)
