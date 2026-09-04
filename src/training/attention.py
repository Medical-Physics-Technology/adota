"""Attention-map snapshots for transformer interpretability.

Once every N epochs the training loop runs a fixed "canary" validation sample
through the model with attention capture enabled and writes the resulting maps
under ``run_dir/attention/``, so attention behaviour can be compared across
epochs on identical input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Attention canary snapshot ───────────────────────────────────────────────


def save_attention_snapshot(
    *,
    model: torch.nn.Module,
    canary_x: torch.Tensor,
    canary_energy: torch.Tensor,
    run_dir: Path,
    epoch: int,
) -> Optional[Path]:
    """Save attention maps for a fixed validation sample.

    Skipped silently when the model has no transformer blocks (the
    forward pass returns a zero placeholder in that case).

    Args:
        model: Model in eval mode (we don't toggle modes here).
        canary_x: Single-sample input ``(1, C, D, H, W)`` already on
            the right device.
        canary_energy: Single-sample energy ``(1, 1)`` already on the
            right device.
        run_dir: Run directory; the file is written to
            ``run_dir/attention/epoch_NNNN.npy``.
        epoch: Current epoch.

    Returns:
        Path to the saved file, or ``None`` if attention is not
        meaningful for this model.
    """
    num_transformers = getattr(model, "num_transformers", 0)
    if num_transformers == 0:
        return None

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(canary_x, canary_energy)
    finally:
        if was_training:
            model.train()

    if not isinstance(outputs, tuple) or len(outputs) < 2:
        return None
    attn = outputs[1]
    out_path = run_dir / "attention" / f"epoch_{epoch:04d}.npy"
    np.save(out_path, attn.detach().cpu().numpy())
    return out_path
