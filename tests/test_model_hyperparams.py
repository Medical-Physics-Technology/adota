"""Tests for model-config fixes (backlog items #1, #2, #3).

* #1 -- ``zero_padding=False`` no longer crashes in forward.
* #2 -- ``dim_feedforward`` is a real, user-settable hyperparameter that
  defaults to ``token_size`` (preserving the original architecture and
  checkpoint compatibility).
* #3 -- the eval-time attention placeholder size is derived from the input
  depth (``x.shape[2] + 1``) instead of the hardcoded ``161``.

All checks emphasize backward compatibility: the defaults must reproduce the
original parameter shapes so existing checkpoints keep loading.
"""

from __future__ import annotations

import torch

from src.adota.models import DoTA3D_v3

# Small model for fast CPU runs. H = W = 24 is divisible by 2**num_levels.
SMALL_SHAPE = (2, 16, 24, 24)
SMALL_KWARGS = dict(
    num_transformers=1,
    num_heads=4,
    num_levels=2,
    enc_features=8,
    kernel_size=3,
    convolutional_steps=1,
    conv_hidden_channels=16,
    dropout_rate=0.1,
    causal=True,
    num_forward=2,
)


def _record(input_shape):
    x = torch.rand(1, *input_shape)
    e = torch.rand(1, 1)
    return x, e


# ── #1: zero_padding=False ───────────────────────────────────────────────────


def test_zero_padding_false_forward_runs():
    """With zero_padding=False the forward pass must run (previously crashed)."""
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, zero_padding=False, **SMALL_KWARGS)
    model.train()
    x, e = _record(SMALL_SHAPE)
    out, attn = model(x, e)  # forward returns (dose, attention)
    assert attn is None  # not computed during training
    # Output keeps the original spatial extent (no padding => no cropping).
    assert out.shape == (1, 1, SMALL_SHAPE[1], SMALL_SHAPE[2], SMALL_SHAPE[3])
    assert torch.isfinite(out).all()


def test_zero_padding_true_still_works():
    """The default zero_padding=True path is unchanged."""
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, zero_padding=True, **SMALL_KWARGS)
    model.train()
    x, e = _record(SMALL_SHAPE)
    out, _ = model(x, e)  # forward returns (dose, attention)
    assert out.shape == (1, 1, SMALL_SHAPE[1], SMALL_SHAPE[2], SMALL_SHAPE[3])
    assert torch.isfinite(out).all()


# ── #2: dim_feedforward ──────────────────────────────────────────────────────


def _ff_linears(model):
    block = model.transformer_layer.transformer_0.feedforward_block
    return block.linear_1, block.linear_2


def test_dim_feedforward_defaults_to_token_size():
    """Unset dim_feedforward => token_size, reproducing the original shapes."""
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, **SMALL_KWARGS)
    assert model.dim_feedforward == model.token_size

    lin1, lin2 = _ff_linears(model)
    # Original code used hidden_dim == embeded_dim == token_size.
    assert lin1.in_features == model.token_size
    assert lin1.out_features == model.token_size
    assert lin2.in_features == model.token_size
    assert lin2.out_features == model.token_size


def test_dim_feedforward_custom_value_changes_only_ff_block():
    """A custom dim_feedforward resizes the feed-forward hidden dim."""
    custom = 64
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, dim_feedforward=custom, **SMALL_KWARGS)
    assert model.dim_feedforward == custom

    lin1, lin2 = _ff_linears(model)
    assert lin1.in_features == model.token_size
    assert lin1.out_features == custom
    assert lin2.in_features == custom
    assert lin2.out_features == model.token_size

    # The model still runs end to end.
    model.eval()
    x, e = _record(SMALL_SHAPE)
    with torch.no_grad():
        out, _ = model(x, e)
    assert out.shape == (1, 1, *SMALL_SHAPE[1:])


def test_dim_feedforward_in_to_dict_and_roundtrip():
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, dim_feedforward=64, **SMALL_KWARGS)
    d = model.to_dict()
    assert d["dim_feedforward"] == 64
    # Rebuild from the serialized hyperparameters.
    rebuilt = DoTA3D_v3(**d)
    assert rebuilt.dim_feedforward == 64


def test_dim_feedforward_backward_compatible_with_legacy_params():
    """Legacy hyperparams (no dim_feedforward) build identical FF shapes."""
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, **SMALL_KWARGS)
    legacy = {k: v for k, v in model.to_dict().items() if k != "dim_feedforward"}
    rebuilt = DoTA3D_v3(**legacy)
    assert rebuilt.dim_feedforward == rebuilt.token_size

    # State-dict keys and shapes must match exactly => old checkpoints load.
    ref_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    new_shapes = {k: tuple(v.shape) for k, v in rebuilt.state_dict().items()}
    assert ref_shapes == new_shapes
    rebuilt.load_state_dict(model.state_dict())  # must not raise


# ── #3: attention placeholder derived from input depth ───────────────────────


def test_attn_placeholder_size_derived_from_depth():
    """With num_transformers=0, the eval attn placeholder is (1, D+1, D+1)."""
    for depth in (16, 8):
        shape = (2, depth, 24, 24)
        kwargs = {**SMALL_KWARGS, "num_transformers": 0}
        model = DoTA3D_v3(input_shape=shape, **kwargs)
        model.eval()
        x, e = _record(shape)
        with torch.no_grad():
            _, attn = model(x, e)
        assert attn.shape == (1, depth + 1, depth + 1)


def test_dead_mask_code_removed():
    """self.mask / generate_subsequent_mask are gone, causal forward still works."""
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, **SMALL_KWARGS)
    assert not hasattr(model, "mask")
    assert not hasattr(model, "generate_subsequent_mask")
    model.train()
    x, e = _record(SMALL_SHAPE)
    assert model(x, e)[0].shape == (1, 1, *SMALL_SHAPE[1:])


def test_attn_placeholder_matches_legacy_161_for_trained_depth():
    """Trained depth 160 reproduces the previous hardcoded 161."""
    shape = (2, 160, 30, 30)
    kwargs = {**SMALL_KWARGS, "num_transformers": 0}
    model = DoTA3D_v3(input_shape=shape, **kwargs)
    model.eval()
    x, e = _record(shape)
    with torch.no_grad():
        _, attn = model(x, e)
    assert attn.shape == (1, 161, 161)


# ── #8: uniform (dose, attention) return contract ────────────────────────────


def test_forward_returns_dose_attention_tuple_in_both_modes():
    """forward always returns (dose, attention); attention is None in training."""
    model = DoTA3D_v3(input_shape=SMALL_SHAPE, **SMALL_KWARGS)
    x, e = _record(SMALL_SHAPE)
    dose_shape = (1, 1, *SMALL_SHAPE[1:])
    seq = SMALL_SHAPE[1] + 1

    # Training: (dose, None)
    model.train()
    out = model(x, e)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == dose_shape  # dose is always first
    assert out[1] is None  # attention not computed during training

    # Eval: (dose, per-head attention tensor)
    model.eval()
    with torch.no_grad():
        out = model(x, e)
    assert isinstance(out, tuple) and len(out) == 2
    dose, attn = out  # unpacking works
    assert dose.shape == dose_shape
    assert out[0] is dose  # dose still first
    assert attn is not None
    assert attn.shape == (1, SMALL_KWARGS["num_heads"], seq, seq)
