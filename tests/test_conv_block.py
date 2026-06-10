"""Unit tests for ConvBlock3D_v2 token_size handling (backlog items #9, #10).

#9 -- token_size is a (height, width) tuple (type hint corrected).
#10 -- token_size is stored as a reusable tuple, not a one-shot generator that
       gets exhausted by the LayerNorm construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.adota.layers import ConvBlock3D_v2

NUM_SLICES = 16


def _layernorm(block):
    for m in block.conv_block:
        if isinstance(m, nn.LayerNorm):
            return m
    return None


def test_downsample_token_size_is_reusable_tuple():
    block = ConvBlock3D_v2(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        token_size=(8, 8),
        num_slices=NUM_SLICES,
        steps=1,
        downsample=True,
        layer_norm=True,
    )
    # #10: still readable after construction (a generator would be exhausted),
    # and reads consistently more than once.
    assert isinstance(block.token_size, tuple)
    assert block.token_size == (4, 4)
    assert block.token_size == (4, 4)  # second read still works

    # Behavior pinned: LayerNorm shape unchanged.
    ln = _layernorm(block)
    assert ln is not None
    assert tuple(ln.normalized_shape) == (8, NUM_SLICES, 4, 4)

    out = block(torch.rand(1, 4, NUM_SLICES, 8, 8))
    assert out.shape == (1, 8, NUM_SLICES, 4, 4)


def test_upsample_token_size_is_reusable_tuple():
    block = ConvBlock3D_v2(
        in_channels=8,
        out_channels=4,
        kernel_size=3,
        token_size=(4, 4),
        num_slices=NUM_SLICES,
        steps=1,
        upsample=True,
        layer_norm=True,
    )
    assert isinstance(block.token_size, tuple)
    assert block.token_size == (8, 8)

    ln = _layernorm(block)
    assert tuple(ln.normalized_shape) == (4, NUM_SLICES, 8, 8)

    out = block(torch.rand(1, 8, NUM_SLICES, 4, 4))
    assert out.shape == (1, 4, NUM_SLICES, 8, 8)
