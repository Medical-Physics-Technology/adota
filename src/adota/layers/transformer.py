"""The causal-masked transformer stack over the slice sequence.

``TransformerEncoderLayerDoTA`` is a pre-norm encoder layer whose attention is
causally masked along depth, so a slice can only attend to slices upstream of
it -- the physical direction the proton beam travels. The mask is built once
and cached per (length, device).

``PositionalEmbedding`` adds learned per-slice positions; ``LinearProj``
projects the scalar beam energy into token space so it can be summed in.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class TransformerEncoderLayerDoTA(nn.Module):
    # TODO!
    def __init__(
        self, embeded_dim: int, num_heads: int, dropout: float = 0.1, **kwargs
    ):
        super(TransformerEncoderLayerDoTA, self).__init__()
        self.embeded_dim = embeded_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = kwargs.get("batch_first", True)
        # Hidden dimensionality of the feed-forward (Linear) sub-block. Defaults
        # to embeded_dim to preserve the original architecture when unset.
        self.dim_feedforward = kwargs.get("dim_feedforward", self.embeded_dim)
        self.causal = kwargs.get("causal", False)
        self.num_forward = kwargs.get("num_forward", 0)
        # If False, the additive residual connections around the attention and
        # feed-forward sub-blocks are disabled (ablation). Defaults to True to
        # preserve the original behavior. Adds no learnable parameters.
        self.residual = kwargs.get("residual", True)

        # Lazy cache for the causal attention mask, keyed by
        # (sequence_length, device). The mask is constant for a given model, so
        # it is built once and reused instead of rebuilt every forward pass.
        # Kept as a plain dict (not a buffer) so the state_dict is unchanged.
        self._mask_cache: dict = {}

        # num_heads must be a factor of embeded_dim
        assert (
            embeded_dim % num_heads == 0
        ), f"Number of heads must be a factor of embeded_dim: {embeded_dim}"

        self.multihead_attention_block = nn.MultiheadAttention(
            embeded_dim, num_heads, dropout=dropout, batch_first=self.batch_first
        )

        self.feedforward_block = (
            nn.Sequential()
        )  # [Linear -> ReLU -> Linear -> LayerNorm -> Dropout]
        self.feedforward_block.add_module(
            "linear_1", nn.Linear(self.embeded_dim, self.dim_feedforward)
        )
        self.feedforward_block.add_module("relu", nn.ReLU())
        self.feedforward_block.add_module(
            "linear_2", nn.Linear(self.dim_feedforward, self.embeded_dim)
        )

        self.norm_1 = nn.LayerNorm(self.embeded_dim)
        self.norm_2 = nn.LayerNorm(self.embeded_dim)
        # self.dropout_layer = nn.Dropout(dropout) # Adding new, be careful with this.
        # self.last_linear_layer = nn.Linear(self.embeded_dim, self.embeded_dim)

    def forward(self, x):
        # Uniform return: (x, attn_weights). attn_weights is None during training
        # (the per-head weights are not computed, which is faster) and the
        # per-head attention tensor during evaluation.
        attn_mask = self._causal_mask(x.shape[-2], x.device) if self.causal else None

        if self.training:
            mhout, _ = self.multihead_attention_block(x, x, x, attn_mask=attn_mask)
            attn_weights = None
        else:
            # Per-head weights (averaging over heads disabled).
            mhout, attn_weights = self.multihead_attention_block(
                x, x, x, attn_mask=attn_mask, average_attn_weights=False
            )

        if self.residual:
            x = x + mhout
        else:
            x = mhout
        x = self.norm_1(x)

        ffout = self.feedforward_block(x)
        if self.residual:
            x = x + ffout
        else:
            x = ffout
        x = self.norm_2(x)

        return x, attn_weights

    def _causal_mask(self, sequence_length: int, device: torch.device):
        """Return the causal attention mask, building and caching it lazily.

        The mask depends only on ``sequence_length`` and ``self.num_forward``,
        both fixed for a given model, so it is computed once per
        ``(sequence_length, device)`` and reused afterwards. This avoids
        rebuilding the mask and copying it to the device on every forward pass.
        """
        cache_key = (sequence_length, device)
        cached = self._mask_cache.get(cache_key)
        if cached is None:
            cached = self._build_causal_mask(sequence_length, device)
            self._mask_cache[cache_key] = cached
        return cached

    def _build_causal_mask(self, sequence_length: int, device: torch.device):
        """Construct the causal mask (0 where attention is allowed, -inf else).

        Built directly on ``device`` to avoid a host-to-device copy. Produces
        the same float32 values as the previous CPU-built implementation.
        """
        mask = (
            torch.triu(
                torch.ones(sequence_length, sequence_length, device=device),
                diagonal=(-1) * self.num_forward,
            )
            == 1
        ).transpose(
            0, 1
        )  # -1 due to the fact that we are performing transposition.
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens, token_size, **kwargs):
        super(PositionalEmbedding, self).__init__()
        self.num_tokens = num_tokens
        self.token_size = token_size

        self.verbose = kwargs.get("verbose", False)
        logger.debug("Number of tokens: %s", self.num_tokens)

        self.embedding = nn.Embedding(num_tokens, token_size)

        # Position indices are constant; register them as a non-persistent
        # buffer so they move with the module (.to(device)) and are not rebuilt
        # on every forward. persistent=False keeps them out of the state_dict.
        self.register_buffer(
            "positions",
            torch.arange(0, num_tokens, step=1, dtype=torch.int32),
            persistent=False,
        )

    def forward(self, *args):
        return torch.cat(list(args), dim=1) + self.embedding(self.positions)


class LinearProj(nn.Module):
    """Project scalars to token vectors."""

    def __init__(self, token_size):
        super(LinearProj, self).__init__()
        self.token_size = token_size
        self.projection = nn.Linear(1, self.token_size)

    def forward(self, inputs):
        projected = self.projection(inputs)
        return projected.unsqueeze(1)


