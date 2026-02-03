import torch
import numpy as np
import torch.nn as nn

from src.adota.layers import *

class DoTA3D_v3(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_shape: tuple, **kwargs):
        """_summary_

        Args:
            input_shape (tuple): _description_
            ourput_shape (tuple, optional): _description_. Defaults to (1, *input_shape[1:]).
            zero_padding (bool, optional): _description_. Defaults to True.
            last_activation (bool, optional): _description_. Defaults to False.
            num_levels (int, optional): _description_. Defaults to 3.
            enc_features (int, optional): _description_. Defaults to 8.
            kernel_size (int, optional): _description_. Defaults to 3.
            convolutional_steps (int, optional): _description_. Defaults to 2.
            conv_hidden_channels (int, optional): _description_. Defaults to 64.
            num_transformers (int, optional): _description_. Defaults to 1.
            num_heads (int, optional): _description_. Defaults to 8.
            dropout_rate (float, optional): _description_. Defaults to 0.1.
            causal (bool, optional): _description_. Defaults to True.
            num_forward (int, optional): _description_. Defaults to 0.
        Raises:
            ValueError: _description_
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

        if self.zero_padding:
            self._padded_shape()

        self._calculate_token_size()

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
                    dim_feedforward=self.token_size,
                    dropout=self.dropout_rate,
                    batch_first=True,
                    causal=self.causal,
                    num_forward=self.num_forward,
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
        )

        if self.last_activation:
            self.last_activation_layer = nn.ReLU()

    def forward(self, x: torch.Tensor, energy: torch.Tensor):
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

        # If zero padding, crop the output tensor
        # Based on the padding size, we need to crop the output tensor.
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
                attn_weights = torch.zeros([1, 161, 161], dtype=torch.float16)
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
        print("New input shape: ", self.input_shape)
        print("New output shape: ", self.output_shape)
        print("Pad size: ", self.pad_size)

    def _calculate_token_size(self):
        self.latent_space_dimension = (
            int(self.input_shape[-2] // (2**self.num_levels)),
            int(self.input_shape[-1] // (2**self.num_levels)),
            self.enc_features,
        )  # After permutation layer.
        self.token_size = np.prod(self.latent_space_dimension)
        print("Token size: ", self.token_size)

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
            "dropout_rate": self.dropout_rate,
            "causal": self.causal,
            "num_forward": self.num_forward,
        }
        return model_parameters
