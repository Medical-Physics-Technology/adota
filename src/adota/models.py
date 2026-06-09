import logging

import torch
import numpy as np
import torch.nn as nn

from src.adota.layers import *

logger = logging.getLogger(__name__)

class DoTA3D_v3(nn.Module):
    """Dose Transformer (DoTA) 3D model, version 3.

    Encoder-decoder architecture for proton dose prediction. A convolutional
    encoder maps the input volume into a sequence of per-slice tokens, a stack
    of transformer encoder layers models the energy-conditioned inter-slice
    dependencies, and a convolutional decoder reconstructs the dose volume.

    The model contains two distinct residual / skip pathways, each separately
    controllable for ablation studies:
        1. The convolutional encoder-decoder skip connections (the ``x_hist``
           tensors in ``forward``), which concatenate encoder feature maps into
           the corresponding decoder blocks. Controlled by ``conv_residual``.
        2. The transformer-internal residual connections around the attention
           and feed-forward sub-blocks. Controlled by ``transformer_residual``.

    Both default to True, reproducing the original architecture and keeping
    existing checkpoints and hyperparameter dicts backward compatible.

    Args:
        nn (torch.nn.Module): Base module class from torch.nn.
    """

    def __init__(self, input_shape: tuple, **kwargs):
        """Constructor of the DoTA3D_v3 model.

        Args:
            input_shape (tuple): Shape of the input volume (C, D, H, W).
            output_shape (tuple, optional): Shape of the predicted dose volume.
                Defaults to (1, *input_shape[1:]).
            zero_padding (bool, optional): If True, the spatial dimensions are
                zero-padded to the next power of two. Defaults to True.
            last_activation (bool, optional): If True, a ReLU is applied to the
                final output. Defaults to False.
            num_levels (int, optional): Number of down/upsampling levels in the
                convolutional encoder/decoder. Defaults to 3.
            enc_features (int, optional): Number of channels at the flattened
                encoder output. Defaults to 8.
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.
            convolutional_steps (int, optional): Number of convolutional steps
                per encoder/decoder block. Defaults to 2.
            conv_hidden_channels (int, optional): Number of hidden channels per
                convolutional block. Defaults to 64.
            num_transformers (int, optional): Number of stacked transformer
                encoder layers. Defaults to 1.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            dim_feedforward (int, optional): Hidden dimensionality of the Linear
                feed-forward sub-block inside each transformer layer. If unset
                (None), defaults to ``token_size`` (the original behavior).
            dropout_rate (float, optional): Dropout probability in the attention
                blocks. Defaults to 0.1.
            causal (bool, optional): If True, applies a causal attention mask so
                each slice only attends to previous slices. Defaults to True.
            num_forward (int, optional): Number of look-ahead slices allowed by
                the causal mask. Defaults to 0.
            transformer_residual (bool, optional): If True, the transformer
                encoder layers use additive residual (skip) connections around
                the attention and feed-forward sub-blocks. Set to False to
                ablate them. Adds no learnable parameters, so checkpoints stay
                compatible across both settings. Defaults to True.
            conv_residual (bool, optional): If True, the convolutional decoder
                receives the encoder skip connections (``x_hist``) via channel
                concatenation. Set to False to ablate them; this reduces the
                decoder convolution input channels, so an ablated model has a
                different parameter count from the baseline. Defaults to True.

        Raises:
            ValueError: If the configuration is invalid.
        """
        super(DoTA3D_v3, self).__init__()
        self.input_shape_initialized = input_shape
        self.input_shape = input_shape
        self.output_shape = kwargs.get("output_shape", (1, *input_shape[1:]))
        self.zero_padding = kwargs.get("zero_padding", True)
        self.last_activation = kwargs.get("last_activation", False)

        # Encoder-decoder attributes
        self.num_levels = kwargs.get("num_levels", 3)
        self.enc_features = kwargs.get("enc_features", 8)
        self.kernel_size = kwargs.get("kernel_size", 3)
        self.convolutional_steps = kwargs.get("convolutional_steps", 2)
        self.conv_hidden_channels = kwargs.get("conv_hidden_channels", 64)

        # TransformerEncoderLayer attributes
        self.num_transformers = kwargs.get("num_transformers", 1)
        self.num_heads = kwargs.get("num_heads", 8)
        self.dropout_rate = kwargs.get("dropout_rate", 0.1)
        self.causal = kwargs.get("causal", True)
        self.num_forward = kwargs.get("num_forward", 0)

        # Residual / skip connection ablation flags. Both default to True to
        # preserve the original architecture and backward compatibility.
        self.transformer_residual = kwargs.get("transformer_residual", True)
        self.conv_residual = kwargs.get("conv_residual", True)

        if self.zero_padding:
            self._padded_shape()

        self._calculate_token_size()

        # Feed-forward (Linear) hidden dimensionality of the transformer layers.
        # User-configurable; defaults to token_size, which reproduces the
        # original architecture and keeps existing checkpoints compatible.
        dim_feedforward = kwargs.get("dim_feedforward", None)
        self.dim_feedforward = (
            int(dim_feedforward) if dim_feedforward is not None else self.token_size
        )

        if self.zero_padding:
            # Depending on the input shape, we need to zero pad the input tensor.
            self.zero_padding_layer = nn.ZeroPad3d(self.pad_size)

        if self.causal:
            # Deprecate this attribute
            self.mask = self.generate_subsequent_mask(self.input_shape[1] + 1)

        self.encoder = ConvEncoder3D(
            input_shape=self.input_shape,
            num_levels=self.num_levels,
            enc_features=self.enc_features,
            kernel_size=self.kernel_size,
            conv_steps_per_block=self.convolutional_steps,
            conv_hidden_channels=self.conv_hidden_channels,
        )

        self.linear_projection_energy = LinearProj(self.token_size)
        self.positional_embedding_layer = PositionalEmbedding(
            self.input_shape[1] + 1, self.token_size
        )

        self.transformer_layer = nn.Sequential()

        for i in range(self.num_transformers):
            self.transformer_layer.add_module(
                f"transformer_{i}",
                TransformerEncoderLayerDoTA(
                    embeded_dim=self.token_size,
                    num_heads=self.num_heads,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout_rate,
                    batch_first=True,
                    causal=self.causal,
                    num_forward=self.num_forward,
                    residual=self.transformer_residual,
                ),
            )

        self.reshape_layer = ReshapeLayer(
            (-1, self.input_shape[1] + 1, *self.latent_space_dimension)
        )
        self.cropp_layer = CroppingLayer()

        self.decoder = ConvDecoder3D(
            num_levels=self.num_levels,
            enc_features=self.enc_features,
            kernel_size=self.kernel_size,
            conv_steps_per_block=self.convolutional_steps,
            conv_hidden_channels=self.conv_hidden_channels,
            output_shape=self.output_shape,
            residual=self.conv_residual,
        )

        if self.last_activation:
            self.last_activation_layer = nn.ReLU()

    def forward(self, x: torch.Tensor, energy: torch.Tensor):
        # Token sequence length: depth (input tensor dim 2, i.e. D in
        # (B, C, D, H, W)) plus one energy token. Captured before any padding,
        # which only affects H and W.
        sequence_length = x.shape[2] + 1

        if self.zero_padding:
            x = self.zero_padding_layer(x)

        x, x_hist = self.encoder(x)
        e = self.linear_projection_energy(energy)
        x = self.positional_embedding_layer(e, x)

        for transformer_block in self.transformer_layer:
            if self.training:
                x = transformer_block(x)

            else:
                x, attn_weights = transformer_block(x)

        x = self.reshape_layer(x)
        x = self.cropp_layer(x)
        x = self.decoder(x, x_hist)

        if self.last_activation:
            x = self.last_activation_layer(x)

        # If zero padding was applied, crop the output tensor back to the
        # original H and W. pad_size only exists when zero_padding is enabled.
        if self.zero_padding:
            x = x[
                :,
                :,
                :,
                self.pad_size[0] : -self.pad_size[1],
                self.pad_size[2] : -self.pad_size[3],
            ]
        if self.training:
            return x

        else:
            if len(self.transformer_layer) == 0:
                attn_weights = torch.zeros(
                    [1, sequence_length, sequence_length], dtype=torch.float16
                )
            return x, attn_weights

    def _padded_shape(self):
        # Find the closes power of 2 for the input shape
        original_input_shape = self.input_shape
        closest_power = 2 ** (int(np.log2(self.input_shape[-1])) + 1)
        self.input_shape = (
            self.input_shape[0],
            self.input_shape[1],
            closest_power,
            closest_power,
        )
        self.output_shape = (1, self.input_shape[1], closest_power, closest_power)
        self.pad_size = (
            (closest_power - original_input_shape[-2]) // 2,
            (closest_power - original_input_shape[-2]) // 2,
            (closest_power - original_input_shape[-1]) // 2,
            (closest_power - original_input_shape[-1]) // 2,
            0,
            0,
        )
        logger.debug("New input shape: %s", self.input_shape)
        logger.debug("New output shape: %s", self.output_shape)
        logger.debug("Pad size: %s", self.pad_size)

    def _calculate_token_size(self):
        self.latent_space_dimension = (
            int(self.input_shape[-2] // (2**self.num_levels)),
            int(self.input_shape[-1] // (2**self.num_levels)),
            self.enc_features,
        )  # After permutation layer.
        self.token_size = int(np.prod(self.latent_space_dimension))
        logger.debug("Token size: %s", self.token_size)

    def generate_subsequent_mask(self, sequence_length):
        mask = (
            torch.triu(torch.ones(sequence_length, sequence_length)) == 1
        ).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def to_dict(self):
        model_parameters = {
            "input_shape": self.input_shape_initialized,  # Original input shape
            "output_shape": [
                1,
                *self.input_shape_initialized[1:],
            ],  # Original output shape
            "zero_padding": self.zero_padding,
            "last_activation": self.last_activation,
            "num_levels": self.num_levels,
            "enc_features": self.enc_features,
            "kernel_size": self.kernel_size,
            "convolutional_steps": self.convolutional_steps,
            "conv_hidden_channels": self.conv_hidden_channels,
            "num_transformers": self.num_transformers,
            "num_heads": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout_rate": self.dropout_rate,
            "causal": self.causal,
            "num_forward": self.num_forward,
            "transformer_residual": self.transformer_residual,
            "conv_residual": self.conv_residual,
        }
        return model_parameters
