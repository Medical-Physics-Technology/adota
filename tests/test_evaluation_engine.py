"""Unit tests for src/evaluation/engine.py.

Uses a fake model and a fake source on CPU, so no dataset/checkpoint is needed.
Exercises the skeleton: device move, no_grad inference, the skip protocol,
ordering, calc_time presence, and the denorm helper.
"""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.engine import InferenceContext, denorm_pair, evaluate
from src.evaluation.sources import Sample


class _FakeModel(torch.nn.Module):
    """Returns (dose, attention); dose = mean of input broadcast to dose shape."""

    def forward(self, x, energy):
        # x: (1, C, D, H, W) -> dose (1, 1, D, H, W)
        b, c, d, h, w = x.shape
        dose = x.mean(dim=1, keepdim=True)
        return dose, None


class _FakeSource:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)


def _make_sample(sid: str, fill: float) -> Sample:
    x = torch.full((2, 4, 3, 3), fill)
    energy = torch.tensor([0.5])
    y = torch.full((1, 4, 3, 3), fill)
    return Sample(sample_id=sid, x=x, energy=energy, y=y, extra={"k": sid})


def test_evaluate_runs_each_sample_and_preserves_order():
    model = _FakeModel()
    samples = [_make_sample(f"s{i}", float(i)) for i in range(3)]
    source = _FakeSource(samples)

    seen_ids = []

    def per_sample_fn(ctx: InferenceContext):
        seen_ids.append(ctx.sample_id)
        # x/energy/y moved to (cpu) device; y_pred is the model output
        assert ctx.x.shape == (2, 4, 3, 3)
        assert ctx.y_pred.shape == (1, 1, 4, 3, 3)
        assert ctx.device == torch.device("cpu")
        assert ctx.calc_time >= 0.0
        assert ctx.extra["k"] == ctx.sample_id
        return {"id": ctx.sample_id}

    results = evaluate(
        model, source, device=torch.device("cpu"),
        per_sample_fn=per_sample_fn, show_progress=False,
    )

    assert seen_ids == ["s0", "s1", "s2"]
    assert [r["id"] for r in results] == ["s0", "s1", "s2"]


def test_evaluate_skips_on_none():
    model = _FakeModel()
    samples = [_make_sample(f"s{i}", float(i)) for i in range(4)]
    source = _FakeSource(samples)

    def per_sample_fn(ctx: InferenceContext):
        # skip the odd-indexed ids
        if ctx.sample_id in {"s1", "s3"}:
            return None
        return ctx.sample_id

    results = evaluate(
        model, source, device=torch.device("cpu"),
        per_sample_fn=per_sample_fn, show_progress=False,
    )
    assert results == ["s0", "s2"]


def test_evaluate_inference_runs_under_no_grad():
    model = _FakeModel()
    source = _FakeSource([_make_sample("s0", 1.0)])

    def per_sample_fn(ctx: InferenceContext):
        assert not ctx.y_pred.requires_grad
        return 1

    evaluate(
        model, source, device=torch.device("cpu"),
        per_sample_fn=per_sample_fn, show_progress=False,
    )


def test_denorm_pair_matches_inverse_minmax():
    scale = {"min_ds": 0.0, "max_ds": 10.0}
    y = torch.rand(1, 4, 3, 3)
    y_pred = torch.rand(1, 1, 4, 3, 3)

    y_np, y_pred_np = denorm_pair(y, y_pred, scale)

    # Shapes: y is unsqueezed to match the prediction's leading dims.
    assert y_np.shape == (1, 1, 4, 3, 3)
    assert y_pred_np.shape == (1, 1, 4, 3, 3)

    expected_y = y.unsqueeze(0).numpy() * (10.0 - 0.0) + 0.0
    expected_pred = y_pred.numpy() * (10.0 - 0.0) + 0.0
    assert np.allclose(y_np, expected_y)
    assert np.allclose(y_pred_np, expected_pred)


def test_context_denorm_method_matches_function():
    scale = {"min_ds": -5.0, "max_ds": 25277028.0}
    sample = _make_sample("s0", 0.3)
    y = sample.y
    y_pred = torch.rand(1, 1, 4, 3, 3)
    ctx = InferenceContext(
        sample=sample, x=sample.x, energy=sample.energy, y=y,
        y_pred=y_pred, device=torch.device("cpu"), calc_time=0.0,
    )
    a = ctx.denorm(scale)
    b = denorm_pair(y, y_pred, scale)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])
