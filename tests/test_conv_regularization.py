"""Tests for the convolutional regularization options (backlog item #5 + init).

Covers weight standardization, the norm_layer choice (batch/group/none), and the
weight_init option. Defaults must reproduce the original architecture exactly so
existing checkpoints keep loading.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.adota.layers import Conv3D
from src.adota.models import DoTA3D_v3

SHAPE = (2, 16, 24, 24)
KW = dict(
    num_transformers=1,
    num_heads=4,
    num_levels=2,
    enc_features=8,
    kernel_size=3,
    convolutional_steps=1,
    conv_hidden_channels=16,
    dropout_rate=0.0,
    causal=True,
    num_forward=2,
)


def _record():
    return torch.rand(1, *SHAPE), torch.rand(1, 1)


def _conv_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv3d)]


# ── Defaults reproduce the original block ────────────────────────────────────


def test_defaults_use_plain_conv_and_batchnorm():
    model = DoTA3D_v3(input_shape=SHAPE, **KW)
    convs = _conv_modules(model)
    assert convs and all(type(m) is nn.Conv3d for m in convs)  # exactly nn.Conv3d
    assert any(isinstance(m, nn.BatchNorm3d) for m in model.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in model.modules())


def test_defaults_backward_compatible_state_dict():
    a = DoTA3D_v3(input_shape=SHAPE, **KW)
    b = DoTA3D_v3(
        input_shape=SHAPE,
        weight_standardization=False,
        norm_layer="batch",
        weight_init="default",
        **KW,
    )
    assert set(a.state_dict()) == set(b.state_dict())
    b.load_state_dict(a.state_dict())  # must not raise


# ── Weight standardization + norm_layer ──────────────────────────────────────


def test_weight_standardization_uses_conv3d_and_runs():
    model = DoTA3D_v3(
        input_shape=SHAPE, weight_standardization=True, norm_layer="group", **KW
    )
    convs = _conv_modules(model)
    assert convs and all(isinstance(m, Conv3D) for m in convs)
    assert any(isinstance(m, nn.GroupNorm) for m in model.modules())
    assert not any(isinstance(m, nn.BatchNorm3d) for m in model.modules())

    model.train()
    x, e = _record()
    out, _ = model(x, e)  # forward returns (dose, attention)
    assert out.shape == (1, 1, *SHAPE[1:])
    assert torch.isfinite(out).all()


def test_group_norm_groups_divide_channels():
    model = DoTA3D_v3(input_shape=SHAPE, norm_layer="group", **KW)
    groupnorms = [m for m in model.modules() if isinstance(m, nn.GroupNorm)]
    assert groupnorms
    for m in groupnorms:
        assert m.num_channels % m.num_groups == 0


def test_norm_layer_none_removes_conv_norm():
    model = DoTA3D_v3(input_shape=SHAPE, norm_layer="none", **KW)
    assert not any(
        isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)) for m in model.modules()
    )
    model.eval()
    x, e = _record()
    with torch.no_grad():
        out, _ = model(x, e)
    assert out.shape == (1, 1, *SHAPE[1:])


def test_invalid_norm_layer_raises():
    with pytest.raises(ValueError):
        DoTA3D_v3(input_shape=SHAPE, norm_layer="instance", **KW)


# ── Weight initialization ────────────────────────────────────────────────────


def test_weight_init_changes_weights_and_runs():
    torch.manual_seed(0)
    default = DoTA3D_v3(input_shape=SHAPE, **KW)
    torch.manual_seed(0)
    kaiming = DoTA3D_v3(input_shape=SHAPE, weight_init="kaiming", **KW)

    # Same seed but different init scheme => first conv weights must differ.
    assert not torch.allclose(
        _conv_modules(default)[0].weight, _conv_modules(kaiming)[0].weight
    )
    kaiming.train()
    x, e = _record()
    assert torch.isfinite(kaiming(x, e)[0]).all()


def test_weight_init_default_is_noop():
    torch.manual_seed(0)
    a = DoTA3D_v3(input_shape=SHAPE, **KW)
    torch.manual_seed(0)
    b = DoTA3D_v3(input_shape=SHAPE, weight_init="default", **KW)
    assert torch.equal(_conv_modules(a)[0].weight, _conv_modules(b)[0].weight)


def test_invalid_weight_init_raises():
    with pytest.raises(ValueError):
        DoTA3D_v3(input_shape=SHAPE, weight_init="orthogonal", **KW)


# ── Serialization round-trip ─────────────────────────────────────────────────


def test_to_dict_roundtrip_conv_options():
    model = DoTA3D_v3(
        input_shape=SHAPE,
        weight_standardization=True,
        norm_layer="group",
        weight_init="kaiming",
        **KW,
    )
    d = model.to_dict()
    assert d["weight_standardization"] is True
    assert d["norm_layer"] == "group"
    assert d["weight_init"] == "kaiming"

    rebuilt = DoTA3D_v3(**d)
    assert rebuilt.weight_standardization is True
    assert rebuilt.norm_layer == "group"
    assert isinstance(_conv_modules(rebuilt)[0], Conv3D)
