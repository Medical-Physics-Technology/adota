from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


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


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class ConvEncoder3D(nn.Module):
    """Class ConvEncoder3D.
    General-purpose DoTA Encoder, responsible for convert the input tensor to the latent space. This layer do not include zero padding.
    ConvEncoder3D by default perform the permute on the last layer, in order to convert the shape from (B, C, D, H, W) to (B, D, H, W, C).

    Attributes:
        - num_levels: int - number of levels in the encoder.
        - enc_features: int - number of channels at the flattened output from the encoder.
        - conv_steps_per_block: int | tuple - number of convolutional steps per block. If int is passed, the same number of steps will be used for all blocks.
        - conv_hidden_channels: int | tuple - number of hidden channels per block. If int is passed, the same number of channels will be used for all blocks.
        - kernel_size: int | tuple - kernel size per block. If int is passed, the same kernel size will be used for all blocks.

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

        self.decoder = nn.Sequential()

        for i in range(self.num_levels):
            self.decoder.add_module(
                f"conv_block_{i}",
                ConvBlock3D_v2(
                    in_channels=(
                        self.input_shape[0] + self.history_filters[i]
                        if i == 0
                        else self.conv_hidden_channels[i - 1] + self.history_filters[i]
                    ),
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
                ),
            )

    def forward(self, x, x_history):
        for i, conv_block in enumerate(self.decoder):
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
        self.hidden_dim = kwargs.get("hidden_dim", self.embeded_dim)
        self.causal = kwargs.get("causal", False)
        self.num_forward = kwargs.get("num_forward", 0)

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
            "linear_1", nn.Linear(self.embeded_dim, self.hidden_dim)
        )
        self.feedforward_block.add_module("relu", nn.ReLU())
        self.feedforward_block.add_module(
            "linear_2", nn.Linear(self.hidden_dim, self.embeded_dim)
        )

        self.norm_1 = nn.LayerNorm(self.embeded_dim)
        self.norm_2 = nn.LayerNorm(self.embeded_dim)
        # self.dropout_layer = nn.Dropout(dropout) # Adding new, be careful with this.
        # self.last_linear_layer = nn.Linear(self.embeded_dim, self.embeded_dim)

    def forward(self, x):
        if self.training:
            if self.causal:
                attn_mask = self._causal_mask(x.shape[-2]).to(x.device)
                mhout, _ = self.multihead_attention_block(x, x, x, attn_mask=attn_mask)

            else:
                mhout, _ = self.multihead_attention_block(x, x, x)

        else:
            if self.causal:
                attn_mask = self._causal_mask(x.shape[-2]).to(x.device)
                # Weights are average over all heads.
                mhout, atten_weights = self.multihead_attention_block(
                    x, x, x, attn_mask=attn_mask, average_attn_weights=False
                )

            else:
                # Weights are average over all heads.
                mhout, atten_weights = self.multihead_attention_block(
                    x, x, x, average_attn_weights=False
                )

        x = x + mhout
        x = self.norm_1(x)

        ffout = self.feedforward_block(x)
        x = x + ffout
        x = self.norm_2(x)

        if self.training:
            return x

        else:
            return x, atten_weights

    def _causal_mask(self, sequence_length: int):
        """Creates a causal mask for the input tensor."""
        mask = (
            torch.triu(
                torch.ones(sequence_length, sequence_length),
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

        # """Creates a causal mask for the input tensor."""
        # mask = (torch.tril(torch.ones(sequence_length, sequence_length), diagonal=num_forward))
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # return mask


class ReshapeLayer(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    def __init__(self, shape: tuple, permute: bool = True):
        super(ReshapeLayer, self).__init__()
        self.shape = shape
        self.permute = permute

    def forward(self, x):
        if self.permute:
            x = x.view(*self.shape)
            x = torch.permute(x, (0, 4, 1, 2, 3))
            return x

        else:
            return x.view(*self.shape)


class CroppingLayer(nn.Module):
    def __init__(self, start_idx: int = 1):
        # Require already permuted data: (B, C, D, H, W)
        super(CroppingLayer, self).__init__()
        self.start_idx = start_idx

    def forward(self, x):
        return x[:, :, self.start_idx :, ...]


class PositionalEmbedding(nn.Module):
    def __init__(self, num_tokens, token_size, **kwargs):
        super(PositionalEmbedding, self).__init__()
        self.num_tokens = num_tokens
        self.token_size = token_size

        self.verbose = kwargs.get("verbose", False)
        print(f"Number of tokens: {self.num_tokens}") if self.verbose else None

        self.embedding = nn.Embedding(num_tokens, token_size)

    def forward(self, *args):
        positions = torch.arange(
            0,
            self.num_tokens,
            step=1,
            dtype=torch.int32,
            device=self.embedding.weight.device,
        )
        return torch.cat(list(args), dim=1) + self.embedding(positions)


class LinearProj(nn.Module):
    """Project scalars to token vectors."""

    def __init__(self, token_size):
        super(LinearProj, self).__init__()
        self.token_size = token_size
        self.projection = nn.Linear(1, self.token_size)

    def forward(self, inputs):
        projected = self.projection(inputs)
        return projected.unsqueeze(1)


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        token_size: tuple,
        num_slices: int,
        conv_steps: int = 1,
        downsample: bool = False,
        upsample: bool = False,
        flatten: bool = False,
        batch_normalization: bool = False,
        layer_norm: bool = True,
        **kwargs,
    ):

        super(ConvBlock3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Size-related attributes
        self.token_size = token_size
        self.num_slices = num_slices

        # Default attributes
        self.conv_steps = conv_steps
        self.downsample = downsample
        self.upsample = upsample
        self.flatten = flatten
        self.batch_normalization = batch_normalization
        self.layer_norm = layer_norm

        self.blocks = nn.ModuleList()

        for i in range(self.conv_steps):
            self.blocks.append(
                nn.Conv3d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    padding="same",
                )
                # Currently not used - this implementation use a trick with wieghts standardization.
                # Conv3D(
                #    in_channels if i == 0 else out_channels,
                #    out_channels,
                #    kernel_size,
                #    padding='same',
                # )
            )
            if self.batch_normalization:
                self.blocks.append(nn.BatchNorm3d(out_channels))

            # self.blocks.append(nn.LeakyReLU(negative_slope=0.05))
            self.blocks.append(nn.ReLU())

        if self.downsample:
            self.blocks.append(nn.MaxPool3d(kernel_size=(1, 2, 2)))
            self.token_size = (ts // 2 for ts in self.token_size)

        if self.upsample:
            self.blocks.append(nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear"))
            self.token_size = (ts * 2 for ts in self.token_size)

        if self.layer_norm:
            self.blocks.append(
                nn.LayerNorm([out_channels, num_slices, *self.token_size])
            )

        if self.flatten:
            # If flatten is True, provided tensor must be permuted, to have a following data format: (B, D, H, W, C).
            self.blocks.append(nn.Flatten(start_dim=2))

    def forward(self, x):
        for block in self.blocks:
            if isinstance(block, nn.Flatten):
                x = torch.permute(
                    x, (0, 2, 3, 4, 1)
                )  # (B, C, D, H, W) -> (B, D, H, W, C)
            x = block(x)
        return x


class ConvBlock3D_v2(nn.Module):
    """Class repreenting a ConvBlock3D layer. ConvBlock is responsible for processing an input signal and perform downsampling / upsampling.
    In the paper nomenclature, this class represents both Convolutional Encoder Layer and Convolutional Decoder Layer.

    Args:
        nn (torch.nn.Module): Base module class from torch.nn.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        token_size: int,
        num_slices: int,
        **kwargs,
    ):
        """Constructor of ConvBlock3D class.

        Args:
            in_channels (int): Number of input channels (C_in in paper).
            out_channels (int): Number of output channels (C_out in paper).
            kernel_size (int | tuple): Kernel size (k). If int is passed, the isotropic kernel is constructed with the same size in all dimensions.
            token_size (int):
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
        self.token_size = token_size  # Token represents the physical slice of the CT. It is a tuple of 3 elements (channels, height, width)

        self.steps = kwargs.get("steps", 2)

        self.downsample = kwargs.get("downsample", False)
        self.upsample = kwargs.get("upsample", False)
        self.flatten = kwargs.get("flatten", False)

        self.layer_norm = kwargs.get("layer_norm", True)

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
            self.token_size = (ts // 2 for ts in self.token_size)

        if self.upsample:
            self.conv_block.append(
                nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
            )
            self.token_size = (ts * 2 for ts in self.token_size)

        if self.layer_norm:
            self.conv_block.append(
                nn.LayerNorm([self.out_channels, self.num_slices, *self.token_size])
                # nn.BatchNorm3d([self.out_channels, self.num_slices, *self.token_size]) # Test with BatchNorm3d instead of LayerNorm
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
        """Constructs the convblock as described in the https://arxiv.org/abs/1505.04597 paper."""
        conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        return conv_block
