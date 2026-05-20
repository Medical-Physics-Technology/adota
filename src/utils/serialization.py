"""Serialization helpers for ADoTA scripts.

Exposes :class:`NumpyEncoder`, a :class:`json.JSONEncoder` subclass that
understands the NumPy and PyTorch scalar / array types commonly produced
by training and analysis scripts so they can be written to JSON without
manual ``.item()`` / ``.tolist()`` calls at the call site.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars / arrays and torch tensors.

    Conversions performed in :meth:`default`:

    - ``np.ndarray``         -> ``list`` via :meth:`numpy.ndarray.tolist`.
    - ``np.integer``         -> Python ``int``.
    - ``np.floating``        -> Python ``float``.
    - ``np.bool_``           -> Python ``bool``.
    - ``np.complex_``        -> ``{"real": float, "imag": float}``.
    - ``torch.Tensor``       -> ``list`` (detached, moved to CPU).

    Anything else is delegated to the base encoder, which raises
    ``TypeError`` for unknown types.

    Example:
        >>> json.dumps({"loss": np.float32(0.5)}, cls=NumpyEncoder)
        '{"loss": 0.5}'
    """

    def default(self, obj: Any) -> Any:  # noqa: D401 - JSONEncoder hook
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complexfloating):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)
