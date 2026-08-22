"""Weight-standardised 3D convolution and the ADoTA conv block.

``Conv3D`` standardises its weights before every ``conv3d`` call, which
stabilises training at the small batch sizes the 3D volumes force.
``ConvBlock3D_v2`` stacks those convolutions with a normalisation and an
activation into the unit the encoder and decoder are built from, and handles
the down/up-sampling stride.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.adota.layers.tensor_ops import Permute


class Conv3D(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        eps=1e-8,
    ):

        super(Conv3D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.eps = eps

    def forward(self, x):
        return F.conv3d(
            x,
            self.weights_standardization(self.weight, self.eps),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @staticmethod
    def weights_standardization(weight, eps=1e-8):
        c_out, c_in, *kernel_size = weight.shape
        weight = weight.view(c_out, -1)

        var, mean = torch.var_mean(weight, dim=1, keepdim=True)
        weight = (weight - mean) / torch.sqrt(var + eps)

        return weight.view(c_out, c_in, *kernel_size)


class ConvBlock3D_v2(nn.Module):
    """Class repreenting a ConvBlock3D layer. ConvBlock is responsible for
    processing an input signal and perform downsampling / upsampling.
    In the paper nomenclature, this class represents both Convolutional Encoder Layer and Convolutional Decoder Layer.

    Args:
        nn (torch.nn.Module): Base module class from torch.nn.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        token_size: tuple[int, int],
        num_slices: int,
        **kwargs,
    ):
        """Constructor of ConvBlock3D class.

        Args:
            in_channels (int): Number of input channels (C_in in paper).
            out_channels (int): Number of output channels (C_out in paper).
            kernel_size (int | tuple): Kernel size (k). If int is passed, the
                isotropic kernel is constructed with the same size in all dimensions.
            token_size (tuple[int, int]): The (height, width) of the feature map at
                this depth, used to build the LayerNorm shape.
            num_slices (int): Number of slices (D in paper).
            **kwargs: Additional arguments:
                - steps (int): Number of convolutional steps in the block.
                - downsample (bool): If True, the block performs downsampling.
                - upsample (bool): If True, the block performs upsampling.
                - flatten (bool): If True, the block flattens the output tensor.
                - layer_norm (bool): If True, the block applies layer normalization.

        Raises:
            ValueError: If both downsample and upsample are True at the same time.
        """
        super(ConvBlock3D_v2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_slices = num_slices
        self.token_size = token_size  # (height, width) of the feature map at this depth.

        self.steps = kwargs.get("steps", 2)

        self.downsample = kwargs.get("downsample", False)
        self.upsample = kwargs.get("upsample", False)
        self.flatten = kwargs.get("flatten", False)

        self.layer_norm = kwargs.get("layer_norm", True)

        # Weight standardization + per-conv normalization choice. Defaults
        # (False / "batch") reproduce the original nn.Conv3d + BatchNorm3d block,
        # keeping existing checkpoints compatible. norm_layer in
        # {"batch", "group", "none"}; weight standardization pairs naturally with
        # "group" (GroupNorm).
        self.weight_standardization = kwargs.get("weight_standardization", False)
        self.norm_layer = kwargs.get("norm_layer", "batch")
        self.num_groups = kwargs.get("num_groups", 32)

        self.conv_block = nn.Sequential()

        for i in range(self.steps):
            self.conv_block.append(
                self._construct_convblock(
                    self.in_channels if i == 0 else self.out_channels,
                    self.out_channels,
                    self.kernel_size,
                )
            )

        if self.downsample and self.upsample:
            raise ValueError(
                "Both downsample and upsample cannot be True at the same time."
            )

        if self.downsample:
            self.conv_block.append(nn.MaxPool3d(kernel_size=(1, 2, 2)))
            self.token_size = tuple(ts // 2 for ts in self.token_size)

        if self.upsample:
            self.conv_block.append(
                nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
            )
            self.token_size = tuple(ts * 2 for ts in self.token_size)

        if self.layer_norm:
            self.conv_block.append(
                nn.LayerNorm([self.out_channels, self.num_slices, *self.token_size])
                # Test with BatchNorm3d instead of LayerNorm:
                # nn.BatchNorm3d([self.out_channels, self.num_slices, *self.token_size])
            )

        if self.flatten:
            self.conv_block.append(Permute(dims=(0, 2, 3, 4, 1)))
            self.conv_block.append(nn.Flatten(start_dim=2))

    def forward(self, x):
        # [ConvBlock, ] * steps -> Downsample / Upsample -> LayerNorm -> Flatten
        x = self.conv_block(x)
        return x

    def _construct_convblock(
        self, in_channels: int, out_channels: int, kernel_size: int | tuple
    ):
        """Constructs the convblock as described in the https://arxiv.org/abs/1505.04597 paper.

        With the defaults (weight_standardization=False, norm_layer="batch") this
        is the original nn.Conv3d -> BatchNorm3d -> ReLU block.
        """
        conv_cls = Conv3D if self.weight_standardization else nn.Conv3d
        layers = [conv_cls(in_channels, out_channels, kernel_size, padding="same")]

        norm = self._make_norm(out_channels)
        if norm is not None:
            layers.append(norm)

        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _make_norm(self, num_channels: int):
        """Builds the per-conv normalization layer per ``norm_layer``."""
        if self.norm_layer == "batch":
            return nn.BatchNorm3d(num_channels)
        if self.norm_layer == "group":
            # gcd guarantees num_groups divides num_channels (GroupNorm requires it).
            num_groups = math.gcd(self.num_groups, num_channels) or 1
            return nn.GroupNorm(num_groups, num_channels)
        if self.norm_layer == "none":
            return None
        raise ValueError(
            f"Unknown norm_layer: {self.norm_layer!r} (expected 'batch', 'group', or 'none')"
        )
