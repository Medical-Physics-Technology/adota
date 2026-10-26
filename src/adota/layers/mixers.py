"""The sequence mixer of :class:`DoTA3D_v3`, chosen by name.

The mixer runs over the slice sequence between the convolutional encoder and
decoder: the energy token first, then the slices from entrance to distal. Every
mixer takes ``(B, L, d)`` tokens and returns tokens of the same shape plus the
attention weights (``None`` for a mixer without attention), so the U-Net, the
loss and the inputs stay identical whichever mixer a config names.

- ``transformer``: :class:`TransformerEncoderLayerDoTA`, causal attention with
  ``num_forward`` slices of look-ahead (the original ADoTA).
- ``mlstm``: :class:`MLSTMBlockDoTA`, a recurrent matrix memory read in the
  beam direction, with the same look-ahead.
- ``none``: no mixer; the model is a plain 3D U-Net over the token grid.

A new mixer (Mamba is planned) adds a builder here and a name to
:data:`SEQUENCE_MIXERS`; nothing in the model changes.
"""
from __future__ import annotations

import torch.nn as nn

from src.adota.layers.mlstm import MLSTMBlockDoTA
from src.adota.layers.transformer import TransformerEncoderLayerDoTA

SEQUENCE_MIXERS = ("transformer", "mlstm", "none")


def build_sequence_mixer(name: str, num_blocks: int, token_size: int, *, num_heads: int,
                         dim_feedforward: int, dropout: float, causal: bool, num_forward: int,
                         residual: bool, mlstm_proj_factor: float = 2.0,
                         mlstm_conv_kernel: int = 4) -> nn.Sequential:
    """The stack of ``num_blocks`` mixer blocks (empty for ``none``).

    Child names are ``transformer_<i>`` and ``mlstm_<i>``; the transformer's
    names are those of every existing checkpoint, so old state dicts load."""
    if name not in SEQUENCE_MIXERS:
        raise ValueError(f"unknown sequence_mixer {name!r}; expected one of {SEQUENCE_MIXERS}")
    stack = nn.Sequential()
    if name == "none":
        return stack
    if name == "mlstm" and not causal:
        raise ValueError("the mLSTM mixer reads the beam direction only; causal must be True")
    for i in range(num_blocks):
        if name == "transformer":
            block = TransformerEncoderLayerDoTA(
                embeded_dim=token_size, num_heads=num_heads, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True, causal=causal, num_forward=num_forward,
                residual=residual)
        else:
            block = MLSTMBlockDoTA(
                embeded_dim=token_size, num_heads=num_heads, proj_factor=mlstm_proj_factor,
                conv_kernel=mlstm_conv_kernel, dropout=dropout, num_forward=num_forward,
                residual=residual)
        stack.add_module(f"{name}_{i}", block)
    return stack
