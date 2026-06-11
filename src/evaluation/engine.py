"""The shared per-sample evaluation engine.

:func:`evaluate` owns the parts every Tier-1 script duplicates: pulling a
:class:`~src.evaluation.sources.Sample` from a source, moving its tensors to the
device, running ``torch.no_grad()`` inference, timing, the skip protocol, and the
progress bar. It delegates the divergent part (metric computation + result
dataclass construction) to a per-script ``per_sample_fn`` callback that receives
an :class:`InferenceContext`.

Part 1 keeps the current numerics exactly: ``torch.no_grad`` (not
``inference_mode``), batch size 1, and NumPy/CPU metrics inside the callbacks.
The resolved ``device`` is threaded through so on-device metrics (GPR today, the
batched on-GPU path in Part 2) can use it.

``calc_time`` is measured around ``next(source)`` + the device move + the forward
pass, i.e. it includes the per-sample load, matching the scripts' current
``start_time = perf_counter()`` placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Optional

import torch
from tqdm import tqdm

from src.evaluation.sources import Sample, SampleSource
from src.utils.scallers import inverse_minmax


def denorm_pair(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    scale: dict,
) -> tuple:
    """De-normalize ground truth and prediction to physical units (NumPy).

    Mirrors the exact expression the scripts use today: the ground truth is
    unsqueezed to match the prediction's ``(1, 1, D, H, W)`` shape, then both go
    through :func:`inverse_minmax` on CPU.

    Args:
        y: Normalized ground-truth dose, shape ``(1, D, H, W)`` (on any device).
        y_pred: Normalized prediction, shape ``(1, 1, D, H, W)``.
        scale: Scale dict with ``min_ds`` / ``max_ds``.

    Returns:
        ``(y_np, y_pred_np)`` NumPy arrays in physical units.
    """
    y_np = inverse_minmax(
        y.unsqueeze(0).detach().cpu().numpy(),
        scale["min_ds"],
        scale["max_ds"],
    )
    y_pred_np = inverse_minmax(
        y_pred.detach().cpu().numpy(),
        scale["min_ds"],
        scale["max_ds"],
    )
    return y_np, y_pred_np


@dataclass
class InferenceContext:
    """Everything a ``per_sample_fn`` needs for one sample.

    Attributes:
        sample: The source :class:`Sample` (id + ``extra`` such as beamlet
            angles).
        x: Model input moved to ``device``, shape ``(C, D, H, W)``.
        energy: Energy tensor moved to ``device``.
        y: Ground-truth dose moved to ``device``, shape ``(1, D, H, W)``.
        y_pred: Model output dose, shape ``(1, 1, D, H, W)``, on ``device``.
        device: The resolved device the inference ran on.
        calc_time: Seconds for load + device move + forward (informational).
    """

    sample: Sample
    x: torch.Tensor
    energy: torch.Tensor
    y: torch.Tensor
    y_pred: torch.Tensor
    device: torch.device
    calc_time: float

    @property
    def sample_id(self) -> str:
        return self.sample.sample_id

    @property
    def extra(self) -> dict:
        return self.sample.extra

    def denorm(self, scale: dict) -> tuple:
        """Convenience wrapper for :func:`denorm_pair` on this context."""
        return denorm_pair(self.y, self.y_pred, scale)


def evaluate(
    model: torch.nn.Module,
    source: SampleSource,
    *,
    device: torch.device,
    per_sample_fn: Callable[[InferenceContext], Optional[Any]],
    show_progress: bool = True,
    desc: str = "Evaluating",
    postfix_fn: Optional[Callable[[Any], dict]] = None,
) -> list:
    """Run per-sample evaluation over ``source``.

    Args:
        model: The loaded model (already on ``device``).
        source: Any :class:`SampleSource` (DirSource, H5Source, ...).
        device: Target device for inference (from ``resolve_device``).
        per_sample_fn: Callback mapping an :class:`InferenceContext` to a result
            object, or ``None`` to skip the sample (zero-flux / energy guards).
        show_progress: Whether to show a tqdm bar.
        desc: Progress-bar description.
        postfix_fn: Optional callback mapping a (non-None) result to a dict of
            tqdm postfix fields.

    Returns:
        List of non-``None`` results, in source order.
    """
    results: list = []
    total = len(source)
    progress = tqdm(total=total, desc=desc, disable=not show_progress)
    iterator = iter(source)

    try:
        while True:
            t0 = perf_counter()
            try:
                sample = next(iterator)
            except StopIteration:
                break

            x = sample.x.to(device)
            energy = sample.energy.to(device)
            y = sample.y.to(device)

            with torch.no_grad():
                y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]

            calc_time = perf_counter() - t0

            ctx = InferenceContext(
                sample=sample,
                x=x,
                energy=energy,
                y=y,
                y_pred=y_pred,
                device=device,
                calc_time=calc_time,
            )
            result = per_sample_fn(ctx)
            progress.update(1)

            if result is None:
                continue
            results.append(result)
            if postfix_fn is not None:
                progress.set_postfix(**postfix_fn(result))
    finally:
        progress.close()

    return results
