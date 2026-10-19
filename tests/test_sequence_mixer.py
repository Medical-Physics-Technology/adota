"""The swappable sequence mixer of DoTA3D_v3: transformer, mLSTM, none.

The mLSTM cell is checked against a step-by-step recurrent reference, the block
for causality with and without look-ahead, and the model for shapes, the
attention return value, checkpoint compatibility of the transformer default
and a training step on every mixer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from src.adota.layers.mixers import SEQUENCE_MIXERS, build_sequence_mixer
from src.adota.layers.mlstm import MLSTMBlockDoTA, look_ahead, mlstm_parallel
from src.adota.models import DoTA3D_v3

SHAPE = (2, 16, 24, 24)
KW = dict(num_transformers=2, num_heads=4, num_levels=2, enc_features=8, kernel_size=3,
          convolutional_steps=1, conv_hidden_channels=16, dropout_rate=0.0, causal=True,
          num_forward=2)
V11_DIR = Path(__file__).resolve().parents[1] / "models" / "DoTA_v3_grid_search_v11"


def _recurrent_reference(q, k, v, i_pre, f_pre, num_heads, eps=1e-6):
    """The mLSTM recurrence of Beck et al. (2024), one step at a time, float64."""
    b, length, width = q.shape
    dh = width // num_heads
    q, k, v = (t.double().reshape(b, length, num_heads, dh) for t in (q, k, v))
    i_pre, f_pre = i_pre.double(), f_pre.double()
    c = torch.zeros(b, num_heads, dh, dh, dtype=torch.float64)
    n = torch.zeros(b, num_heads, dh, dtype=torch.float64)
    m = torch.full((b, num_heads), float("-inf"), dtype=torch.float64)
    out = []
    for t in range(length):
        log_f = torch.nn.functional.logsigmoid(f_pre[:, t])
        m_new = torch.maximum(log_f + m, i_pre[:, t])
        i_gate = torch.exp(i_pre[:, t] - m_new)
        f_gate = torch.exp(log_f + m - m_new)
        kt = k[:, t] / math.sqrt(dh)
        c = f_gate[..., None, None] * c + i_gate[..., None, None] * (v[:, t, :, :, None] * kt[:, :, None, :])
        n = f_gate[..., None] * n + i_gate[..., None] * kt
        num = (c @ q[:, t, :, :, None]).squeeze(-1)
        den = torch.maximum((n * q[:, t]).sum(-1).abs(), torch.exp(-m_new)) + eps
        out.append(num / den[..., None])
        m = m_new
    return torch.stack(out, dim=1).reshape(b, length, width)


@pytest.mark.parametrize("gate_scale", [1.0, 8.0])
def test_parallel_mlstm_equals_the_recurrence(gate_scale):
    torch.manual_seed(0)
    b, length, width, heads = 2, 17, 12, 3
    q, k, v = (torch.randn(b, length, width) for _ in range(3))
    i_pre = gate_scale * torch.randn(b, length, heads)
    f_pre = gate_scale * torch.randn(b, length, heads)
    got = mlstm_parallel(q, k, v, i_pre, f_pre, heads).double()
    want = _recurrent_reference(q, k, v, i_pre, f_pre, heads)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_look_ahead_takes_the_state_num_forward_slices_downstream():
    h = torch.arange(6.0).reshape(1, 6, 1)
    assert look_ahead(h, 0).flatten().tolist() == [0, 1, 2, 3, 4, 5]
    assert look_ahead(h, 2).flatten().tolist() == [2, 3, 4, 5, 5, 5]


@pytest.mark.parametrize("num_forward", [0, 2])
def test_mlstm_block_sees_at_most_num_forward_slices_ahead(num_forward):
    torch.manual_seed(1)
    block = MLSTMBlockDoTA(embeded_dim=16, num_heads=4, dropout=0.0, num_forward=num_forward).eval()
    x = torch.randn(1, 20, 16)
    changed_at = 12
    y = x.clone()
    y[:, changed_at] += torch.randn(16)      # not a constant: LayerNorm removes a constant shift
    with torch.no_grad():
        a, attn = block(x)
        b, _ = block(y)
    assert attn is None
    differs = (a - b).abs().amax(dim=-1).flatten() > 1e-6
    first = changed_at - num_forward
    assert not differs[:first].any()
    assert differs[first:].all()


def test_mixer_names_and_invalid_configurations():
    assert SEQUENCE_MIXERS == ("transformer", "mlstm", "none")
    common = dict(num_heads=4, dim_feedforward=16, dropout=0.0, causal=True, num_forward=0,
                  residual=True)
    assert len(build_sequence_mixer("none", 3, 16, **common)) == 0
    with pytest.raises(ValueError):
        build_sequence_mixer("mamba", 1, 16, **common)
    with pytest.raises(ValueError):
        build_sequence_mixer("mlstm", 1, 16, **{**common, "causal": False})
    with pytest.raises(ValueError):
        MLSTMBlockDoTA(embeded_dim=10, num_heads=3)


@pytest.mark.parametrize("mixer", SEQUENCE_MIXERS)
def test_every_mixer_runs_and_returns_the_dose_shape(mixer):
    torch.manual_seed(0)
    model = DoTA3D_v3(input_shape=SHAPE, sequence_mixer=mixer, **KW)
    x, e = torch.rand(2, *SHAPE), torch.rand(2, 1)
    model.train()
    out, attn = model(x, e)
    assert out.shape == (2, 1, *SHAPE[1:]) and attn is None
    model.eval()
    with torch.no_grad():
        out, attn = model(x, e)
    assert out.shape == (2, 1, *SHAPE[1:])
    assert torch.isfinite(out).all()
    if mixer == "transformer":
        assert attn.shape == (2, KW["num_heads"], SHAPE[1] + 1, SHAPE[1] + 1)
    else:
        # the eval-time placeholder of a mixer without attention
        assert attn.shape == (1, SHAPE[1] + 1, SHAPE[1] + 1) and not attn.any()


def test_none_is_the_model_without_transformers():
    a = DoTA3D_v3(input_shape=SHAPE, sequence_mixer="none", **KW)
    b = DoTA3D_v3(input_shape=SHAPE, **{**KW, "num_transformers": 0})
    assert {k: v.shape for k, v in a.state_dict().items()} == \
        {k: v.shape for k, v in b.state_dict().items()}


def test_transformer_default_keeps_the_checkpoint_layout_and_round_trips():
    legacy = DoTA3D_v3(input_shape=SHAPE, **KW)
    explicit = DoTA3D_v3(input_shape=SHAPE, sequence_mixer="transformer", **KW)
    assert legacy.sequence_mixer == "transformer"
    assert list(legacy.state_dict()) == list(explicit.state_dict())
    assert any(k.startswith("transformer_layer.transformer_0.") for k in legacy.state_dict())
    explicit.load_state_dict(legacy.state_dict())
    for mixer in SEQUENCE_MIXERS:
        model = DoTA3D_v3(input_shape=SHAPE, sequence_mixer=mixer, mlstm_proj_factor=1.5, **KW)
        params = model.to_dict()
        assert params["sequence_mixer"] == mixer and params["mlstm_proj_factor"] == 1.5
        rebuilt = DoTA3D_v3(**params)
        rebuilt.load_state_dict(model.state_dict())


@pytest.mark.parametrize("mixer", SEQUENCE_MIXERS)
def test_one_training_step_reaches_every_parameter(mixer):
    torch.manual_seed(0)
    model = DoTA3D_v3(input_shape=SHAPE, sequence_mixer=mixer, **KW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, e, target = torch.rand(2, *SHAPE), torch.rand(2, 1), torch.rand(2, 1, *SHAPE[1:])
    out, _ = model(x, e)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()


@pytest.mark.integration
def test_v11_checkpoint_loads_into_the_transformer_default():
    hyper, weights = V11_DIR / "hyperparams.json", V11_DIR / "best_model.pth"
    if not (hyper.exists() and weights.exists()):
        pytest.skip(f"v11 weights not present; put hyperparams.json and best_model.pth in {V11_DIR}")
    model = DoTA3D_v3(**json.loads(hyper.read_text()))
    assert model.sequence_mixer == "transformer"
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))


@pytest.mark.gpu
@pytest.mark.parametrize("mixer", SEQUENCE_MIXERS)
def test_a_training_step_on_cuda_at_the_v11_shape(mixer):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    torch.manual_seed(0)
    device = torch.device("cuda")
    hyper = dict(num_levels=4, enc_features=32, kernel_size=3, convolutional_steps=1,
                 conv_hidden_channels=64, num_transformers=1, num_heads=4, dropout_rate=0.1,
                 causal=True, num_forward=2)
    model = DoTA3D_v3(input_shape=(2, 160, 30, 30), sequence_mixer=mixer, **hyper).to(device)
    x = torch.rand(4, 2, 160, 30, 30, device=device)
    e, target = torch.rand(4, 1, device=device), torch.rand(4, 1, 160, 30, 30, device=device)
    out, _ = model(x, e)      # float32, as the training loop runs
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    assert torch.isfinite(loss)
