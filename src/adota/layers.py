from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


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
        token_size: tuple[int, int],
        num_slices: int,
        **kwargs,
    ):
        """Constructor of ConvBlock3D class.

        Args:
            in_channels (int): Number of input channels (C_in in paper).
            out_channels (int): Number of output channels (C_out in paper).
            kernel_size (int | tuple): Kernel size (k). If int is passed, the isotropic kernel is constructed with the same size in all dimensions.
            token_size (tuple[int, int]): The (height, width) of the feature map at this depth, used to build the LayerNorm shape.
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
