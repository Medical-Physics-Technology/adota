"""The convolutional encoder and decoder stacks.

``ConvEncoder3D`` maps the (B, C, D, H, W) input volume to a per-slice token
sequence, returning the intermediate feature maps ("history") that
``ConvDecoder3D`` consumes as skip connections on the way back to a dose volume.

Both accept per-block ints or tuples for the step count, hidden channels and
kernel size, so depth and width are configurable from the hyperparameter file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.adota.layers.conv import ConvBlock3D_v2


class ConvEncoder3D(nn.Module):
    """Class ConvEncoder3D.
    General-purpose DoTA Encoder, responsible for convert the input tensor to the
    latent space. This layer do not include zero padding.
    ConvEncoder3D by default perform the permute on the last layer, in order to
    convert the shape from (B, C, D, H, W) to (B, D, H, W, C).

    Attributes:
        - num_levels: int - number of levels in the encoder.
        - enc_features: int - number of channels at the flattened output from the encoder.
        - conv_steps_per_block: int | tuple - number of convolutional steps per
          block. If int is passed, the same number of steps will be used for all
          blocks.
        - conv_hidden_channels: int | tuple - number of hidden channels per block.
          If int is passed, the same number of channels will be used for all blocks.
        - kernel_size: int | tuple - kernel size per block. If int is passed, the
          same kernel size will be used for all blocks.

        - input_shape: tuple - input shape of the tensor. Default: (2, 160, 32, 32)
    """

    def __init__(
        self,
        num_levels: int,
        enc_features: int,
        conv_steps_per_block: int | tuple,
        conv_hidden_channels: int | tuple,
        kernel_size: int | tuple,
        **kwargs,
    ):
        super(ConvEncoder3D, self).__init__()
        self.num_levels = num_levels
        self.enc_features = (
            enc_features  # Number of channels at the flattened output from the encoder
        )
        self.conv_steps_per_block = conv_steps_per_block
        self.conv_hidden_channels = conv_hidden_channels
        self.kernel_size = kernel_size

        self.input_shape = kwargs.get("input_shape", (2, 160, 32, 32))

        # Weight standardization + per-conv norm choice (forwarded to each block).
        self.weight_standardization = kwargs.get("weight_standardization", False)
        self.norm_layer = kwargs.get("norm_layer", "batch")

        # Adjust types depending on passed constructor parameters.
        if isinstance(self.conv_steps_per_block, int):
            self.conv_steps_per_block = [self.conv_steps_per_block] * self.num_levels

        if isinstance(self.conv_hidden_channels, int):
            self.conv_hidden_channels = [self.conv_hidden_channels] * self.num_levels

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * self.num_levels

        self.encoder = nn.Sequential()

        for i in range(self.num_levels):
            self.encoder.add_module(
                f"conv_block_{i}",
                ConvBlock3D_v2(
                    in_channels=(
                        self.input_shape[0]
                        if i == 0
                        else self.conv_hidden_channels[i - 1]
                    ),
                    out_channels=self.conv_hidden_channels[i],
                    kernel_size=self.kernel_size[i],
                    token_size=self._token_size_at_depth(i),
                    num_slices=self.input_shape[1],
                    steps=self.conv_steps_per_block[i],
                    downsample=True,
                    layer_norm=True if i < self.num_levels - 1 else False,
                    weight_standardization=self.weight_standardization,
                    norm_layer=self.norm_layer,
                ),
            )

        self.flattening_conv_block = ConvBlock3D_v2(
            in_channels=self.conv_hidden_channels[-1],
            out_channels=self.enc_features,
            kernel_size=self.kernel_size[-1],
            token_size=self._token_size_at_depth(self.num_levels),
            num_slices=self.input_shape[1],
            steps=self.conv_steps_per_block[-1],
            flatten=True,
            weight_standardization=self.weight_standardization,
            norm_layer=self.norm_layer,
        )

    def forward(self, x):
        x_history = [x]
        for conv_block in self.encoder:
            x = conv_block(x)
            x_history.append(x)

        x = self.flattening_conv_block(x)
        return x, x_history

    def _token_size_at_depth(self, depth: int):
        """Calculates the token size at the given depth."""
        return (
            int(self.input_shape[-2] // (2**depth)),
            int(self.input_shape[-1] // (2**depth)),
        )


class ConvDecoder3D(nn.Module):
    def __init__(
        self,
        num_levels: int,
        enc_features: int,
        conv_steps_per_block: int | tuple,
        conv_hidden_channels: int | tuple,
        kernel_size: int | tuple,
        **kwargs,
    ):
        super(ConvDecoder3D, self).__init__()
        self.num_levels = num_levels
        self.enc_features = enc_features
        self.conv_steps_per_block = conv_steps_per_block
        self.conv_hidden_channels = conv_hidden_channels
        self.kernel_size = kernel_size

        self.output_shape = kwargs.get(
            "output_shape", (1, 160, 32, 32)
        )  # Output is padded.
        self.input_shape = self._calc_input_shape()

        # Adjust types depending on passed constructor parameters.
        if isinstance(self.conv_steps_per_block, int):
            self.conv_steps_per_block = [self.conv_steps_per_block] * self.num_levels

        if isinstance(self.conv_hidden_channels, int):
            self.conv_hidden_channels = [self.conv_hidden_channels] * self.num_levels

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * self.num_levels

        self._assert_in_case_of_passed_list()

        # Assign history filters. By default, we are assuming that Encoder and Decoder are symetrical.
        self.history_filters = kwargs.get(
            "history_filters", [2, *self.conv_hidden_channels][::-1]
        )

        # If False, the encoder-decoder skip (residual) connections are disabled
        # (ablation): the encoder feature maps are no longer concatenated onto
        # the decoder input, so the corresponding input channels are removed.
        # Defaults to True to preserve the original behavior.
        self.residual = kwargs.get("residual", True)

        # Weight standardization + per-conv norm choice (forwarded to each block).
        self.weight_standardization = kwargs.get("weight_standardization", False)
        self.norm_layer = kwargs.get("norm_layer", "batch")

        self.decoder = nn.Sequential()

        for i in range(self.num_levels):
            # When skip connections are enabled, the concatenated encoder
            # feature map adds history_filters[i] channels to the block input.
            # When disabled, no extra channels are added.
            if self.residual:
                skip_channels = self.history_filters[i]
            else:
                skip_channels = 0

            if i == 0:
                in_channels = self.input_shape[0] + skip_channels
            else:
                in_channels = self.conv_hidden_channels[i - 1] + skip_channels

            self.decoder.add_module(
                f"conv_block_{i}",
                ConvBlock3D_v2(
                    in_channels=in_channels,
                    out_channels=(
                        self.conv_hidden_channels[i]
                        if i < self.num_levels - 1
                        else self.output_shape[0]
                    ),
                    kernel_size=self.kernel_size[i],
                    token_size=self._token_size_at_depth(self.num_levels - i),
                    num_slices=self.input_shape[1],
                    steps=self.conv_steps_per_block[i],
                    upsample=True,
                    layer_norm=True if i < self.num_levels - 1 else False,
                    weight_standardization=self.weight_standardization,
                    norm_layer=self.norm_layer,
                ),
            )

    def forward(self, x, x_history):
        for i, conv_block in enumerate(self.decoder):
            if self.residual:
                x = torch.cat([x, x_history[-(i + 1)]], dim=1)
            x = conv_block(x)
        return x

    def _calc_input_shape(self):
        return (
            self.enc_features,
            self.output_shape[1],
            int(self.output_shape[-2] // (2**self.num_levels)),
            int(self.output_shape[-1] // (2**self.num_levels)),
        )

    def _assert_in_case_of_passed_list(self):
        assert (
            len(self.conv_steps_per_block) == self.num_levels
        ), f"Length of conv_steps_per_block must be equal to num_levels: {self.num_levels}"
        assert (
            len(self.conv_hidden_channels) == self.num_levels
        ), f"Length of conv_hidden_channels must be equal to num_levels: {self.num_levels}"
        assert (
            len(self.kernel_size) == self.num_levels
        ), f"Length of kernel_size must be equal to num_levels: {self.num_levels}"

    def _token_size_at_depth(self, depth: int):
        """Calculates the token size at the given depth."""
        return (
            int(self.output_shape[-2] // (2**depth)),
            int(self.output_shape[-1] // (2**depth)),
        )


