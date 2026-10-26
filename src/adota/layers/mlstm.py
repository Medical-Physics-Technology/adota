"""The mLSTM block (Beck et al. 2024, arXiv:2405.04517) over the slice sequence.

An alternative sequence mixer to :class:`TransformerEncoderLayerDoTA` with the
same interface: a ``(B, L, d)`` token sequence in, the same shape out, and a
second return value for the attention weights, which a recurrent mixer does not
have (``None``). It reads the sequence in one direction only, from the energy
token through the entrance slice to the distal end, which is the direction the
beam travels.

The block is the pre-up-projection block of the paper: layer norm, an up
projection into an inner width and an output gate, a causal depthwise
convolution, the mLSTM cell (a matrix memory with exponential input and sigmoid
forget gates), a per-head group norm with a learnable skip from the
convolution, the gated down projection, and a residual connection. A layer norm
follows the residual, as in the transformer layer, so both mixers hand the
decoder normalised tokens.

The cell is computed in its parallel form, exactly equivalent to the recurrence
(``tests/test_sequence_mixer.py`` checks it against a step-by-step reference):

    log D[t, s] = sum_{r = s+1..t} log sigmoid(f_r) + i_s      for s <= t
    m_t         = max_s log D[t, s]                              (stabiliser)
    C[t, s]     = (q_t . k_s / sqrt(d_h)) exp(log D[t, s] - m_t)
    h_t         = sum_s C[t, s] v_s / max(|sum_s C[t, s]|, exp(-m_t))

``num_forward`` matches the look-ahead of the transformer's causal mask (a
slice may see ``num_forward`` slices downstream): position ``t`` takes the cell
output of position ``min(t + num_forward, L - 1)``, the state after reading
that far. Pure PyTorch, so it runs on Python 3.10 and exports to ONNX; the
sequence is 161 tokens long, so the quadratic parallel form is cheap.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlstm_parallel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, i_pre: torch.Tensor,
                   f_pre: torch.Tensor, num_heads: int, eps: float = 1e-6) -> torch.Tensor:
    """The mLSTM cell over a whole sequence. ``q, k, v``: ``(B, L, D)``;
    ``i_pre, f_pre``: ``(B, L, num_heads)`` gate pre-activations. Returns
    ``(B, L, D)``. Gates and the stabiliser are computed in float32."""
    b, length, width = q.shape
    dh = width // num_heads

    def heads(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(b, length, num_heads, dh).transpose(1, 2)

    q, k, v = heads(q), heads(k), heads(v)
    log_f = F.logsigmoid(f_pre.float()).transpose(1, 2)            # (B, NH, L)
    cum = torch.cumsum(log_f, dim=-1)
    log_d = cum.unsqueeze(-1) - cum.unsqueeze(-2) + i_pre.float().transpose(1, 2).unsqueeze(-2)
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    log_d = log_d.masked_fill(~causal, float("-inf"))
    m = log_d.amax(dim=-1, keepdim=True)                            # (B, NH, L, 1)
    decay = torch.exp(log_d - m)
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(dh) * decay
    norm = torch.maximum(scores.sum(dim=-1, keepdim=True).abs(), torch.exp(-m)) + eps
    h = (scores / norm) @ v.float()
    return h.to(v.dtype).transpose(1, 2).reshape(b, length, width)


def look_ahead(h: torch.Tensor, num_forward: int) -> torch.Tensor:
    """Position ``t`` takes ``h[:, min(t + num_forward, L - 1)]``."""
    if num_forward <= 0:
        return h
    length = h.shape[1]
    index = torch.clamp(torch.arange(length, device=h.device) + num_forward, max=length - 1)
    return h.index_select(1, index)


class MLSTMBlockDoTA(nn.Module):
    """One mLSTM block; returns ``(x, None)`` like the transformer layer.

    Args:
        embeded_dim: Token width ``d``.
        num_heads: Heads of the matrix memory; must divide the inner width.
        proj_factor: Inner width as a multiple of ``d``. Defaults to 2.
        conv_kernel: Width of the causal depthwise convolution. Defaults to 4.
        dropout: Dropout on the block output. Defaults to 0.1.
        num_forward: Look-ahead in slices, as in the transformer's causal mask.
        residual: If False, the residual connection is dropped (ablation).
    """

    def __init__(self, embeded_dim: int, num_heads: int, proj_factor: float = 2.0,
                 conv_kernel: int = 4, dropout: float = 0.1, num_forward: int = 0,
                 residual: bool = True):
        super().__init__()
        inner = int(round(proj_factor * embeded_dim))
        if inner % num_heads:
            raise ValueError(f"num_heads {num_heads} must divide the mLSTM inner width {inner} "
                             f"(proj_factor {proj_factor} x token size {embeded_dim})")
        self.embeded_dim, self.inner, self.num_heads = embeded_dim, inner, num_heads
        self.num_forward, self.residual = int(num_forward), residual
        self.norm_in = nn.LayerNorm(embeded_dim)
        self.up_projection = nn.Linear(embeded_dim, 2 * inner)
        self.conv = nn.Conv1d(inner, inner, conv_kernel, groups=inner, padding=conv_kernel - 1)
        self.q_projection = nn.Linear(inner, inner)
        self.k_projection = nn.Linear(inner, inner)
        self.v_projection = nn.Linear(inner, inner)
        self.input_gate = nn.Linear(3 * inner, num_heads)
        self.forget_gate = nn.Linear(3 * inner, num_heads)
        self.head_norm = nn.GroupNorm(num_heads, inner)
        self.skip = nn.Parameter(torch.ones(inner))
        self.down_projection = nn.Linear(inner, embeded_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(embeded_dim)
        self._init_gates()

    def _init_gates(self) -> None:
        # As in the paper: the forget gate starts open (bias 3 to 6 over the
        # heads), the input gate small, so early training keeps long memory.
        nn.init.zeros_(self.forget_gate.weight)
        with torch.no_grad():
            self.forget_gate.bias.copy_(torch.linspace(3.0, 6.0, self.num_heads))
        nn.init.zeros_(self.input_gate.weight)
        nn.init.normal_(self.input_gate.bias, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor):
        length = x.shape[1]
        x_m, z = self.up_projection(self.norm_in(x)).chunk(2, dim=-1)
        x_c = F.silu(self.conv(x_m.transpose(1, 2))[..., :length].transpose(1, 2))
        q, k, v = self.q_projection(x_c), self.k_projection(x_c), self.v_projection(x_m)
        gates_in = torch.cat([q, k, v], dim=-1)
        h = mlstm_parallel(q, k, v, self.input_gate(gates_in), self.forget_gate(gates_in),
                           self.num_heads)
        h = look_ahead(h, self.num_forward)
        b = h.shape[0]
        h = self.head_norm(h.reshape(b * length, self.inner)).reshape(b, length, self.inner)
        y = self.down_projection((h + self.skip * x_c) * F.silu(z))
        y = self.dropout(y)
        x = x + y if self.residual else y
        return self.norm_out(x), None
