"""The ``nn.Module`` building blocks that :class:`DoTA3D_v3` is assembled from.

Split by role so no module exceeds the 500-line limit; every name is re-exported
here, so ``from src.adota.layers import ConvBlock3D_v2`` keeps working.

- ``conv``: ``Conv3D``, ``ConvBlock3D_v2`` -- weight-standardised convolutions.
- ``encoder_decoder``: ``ConvEncoder3D``, ``ConvDecoder3D`` -- the stacks and
  their skip connections.
- ``transformer``: ``TransformerEncoderLayerDoTA``, ``PositionalEmbedding``,
  ``LinearProj`` -- the causal-masked attention over the slice sequence.
- ``tensor_ops``: ``Permute``, ``ReshapeLayer``, ``CroppingLayer`` -- shape
  adapters between the (B, C, D, H, W) and (B, D, H, W, C) conventions.
"""

from src.adota.layers.conv import Conv3D, ConvBlock3D_v2
from src.adota.layers.encoder_decoder import ConvDecoder3D, ConvEncoder3D
from src.adota.layers.tensor_ops import CroppingLayer, Permute, ReshapeLayer
from src.adota.layers.transformer import (
    LinearProj,
    PositionalEmbedding,
    TransformerEncoderLayerDoTA,
)

__all__ = [
    "Conv3D",
    "ConvBlock3D_v2",
    "ConvDecoder3D",
    "ConvEncoder3D",
    "CroppingLayer",
    "LinearProj",
    "Permute",
    "PositionalEmbedding",
    "ReshapeLayer",
    "TransformerEncoderLayerDoTA",
]
