"""Shape adapters between the convolutional and token tensor conventions.

The convolutional parts work in (B, C, D, H, W); the transformer works in
(B, D, H, W, C). ``Permute`` and ``ReshapeLayer`` move between them, and
``CroppingLayer`` drops leading slices so the decoder output lines up with the
target dose grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


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


