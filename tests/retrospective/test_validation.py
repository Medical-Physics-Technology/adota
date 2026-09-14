"""The evaluation subsample, the per-epoch loss and the per-sample metric set."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.active_learning.retrospective.validation import (
    MetricSettings,
    draw_eval_subsample,
    evaluate_loss,
    evaluate_metrics,
    summarise_metrics,
)
from src.adota.config import DEFAULT_GAMMA_PARAMS

D, H, W = 40, 8, 8
SCALE = {"min_ds": 0.0, "max_ds": 1.0e6, "min_ct": -1024, "max_ct": 3071,
         "min_energy": 70.0, "max_energy": 270.0}


class FluxEcho(torch.nn.Module):
    """Predicts the flux channel, so the truth can be planted in the input."""

    def forward(self, x, energy):
        return x[:, 1:2], None


def bragg(peak: float, plateau: bool = False) -> torch.Tensor:
    depth = torch.arange(D, dtype=torch.float32)
    profile = 0.3 + 0.7 * torch.exp(-((depth - peak) ** 2) / (2 * 2.0 ** 2))
    if not plateau:
        profile[depth > peak + 5] = 0.0
    else:
        profile = torch.clamp(profile, min=0.85)      # never falls below 80 percent
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    lateral = torch.exp(-((yy - 3.5) ** 2 + (xx - 3.5) ** 2) / (2 * 1.5 ** 2)).float()
    return (profile[:, None, None] * lateral[None]).unsqueeze(0)


def make_loader(truths, predictions):
    ct = torch.zeros(len(truths), 1, D, H, W)
    x = torch.cat([ct, torch.stack(predictions)], dim=1)
    y = torch.stack(truths)
    e = torch.full((len(truths), 1), 0.25)
    return DataLoader(TensorDataset(x, e, y), batch_size=2, shuffle=False)


def settings() -> MetricSettings:
    return MetricSettings(scale=SCALE, gamma_params=dict(DEFAULT_GAMMA_PARAMS),
                          resolution_mm=(2.0, 2.0, 2.0), gamma_backend="torch")


def test_draw_eval_subsample_is_fixed_and_inside_v():
    val = [f"v{i}" for i in range(100)]
    a = draw_eval_subsample(val, 10, seed=5)
    assert a == draw_eval_subsample(val, 10, seed=5) and len(set(a)) == 10
    assert set(a) <= set(val) and a != draw_eval_subsample(val, 10, seed=6)
    assert draw_eval_subsample(val, 500, seed=1) == val


def test_near_perfect_prediction_scores_near_perfectly():
    # Not bit-identical: the pass-rate denominator counts gamma > 0, and an exact
    # copy has gamma == 0 everywhere (a quirk shared with every other caller).
    truth = bragg(20.0)
    near = truth * 1.002
    loader = make_loader([truth, truth], [near, near])
    frame = evaluate_metrics(FluxEcho(), loader, ["p", "q"], device=torch.device("cpu"),
                             settings=settings())
    assert frame["sample_id"].tolist() == ["p", "q"]
    assert np.allclose(frame["gpr"], 1.0) and (frame["mape_pct"] < 0.5).all()
    assert np.allclose(frame["dr80_mm"], 0.0, atol=0.05)
    summary = summarise_metrics(frame)
    assert summary["gpr_frac_below_95"] == 0.0 and summary["dr80_defined_fraction"] == 1.0


def test_a_range_shift_shows_up_in_dr80_and_a_plateau_is_undefined():
    truth, shifted, plateau = bragg(20.0), bragg(22.0), bragg(20.0, plateau=True)
    loader = make_loader([truth, plateau], [shifted, plateau])
    frame = evaluate_metrics(FluxEcho(), loader, ["shift", "plateau"],
                             device=torch.device("cpu"), settings=settings())
    assert frame.loc[0, "dr80_mm"] == pytest.approx(4.0, abs=0.5)   # 2 voxels of 2 mm
    assert frame.loc[0, "gpr"] < 1.0
    assert np.isnan(frame.loc[1, "dr80_mm"])                        # the plateau guard
    summary = summarise_metrics(frame)
    assert summary["dr80_defined_fraction"] == 0.5 and summary["n_dr80"] == 1.0


def test_evaluate_loss_weights_the_two_components():
    truth = bragg(20.0)
    loader = make_loader([truth, truth, truth], [truth, bragg(21.0), truth])
    out = evaluate_loss(FluxEcho(), loader, device=torch.device("cpu"), weight_mse=0.7,
                        weight_ps=0.3, settings=settings())
    assert out["n"] == 3 and out["loss_mse_mean"] > 0.0
    assert out["loss_combined_mean"] == pytest.approx(
        0.7 * out["loss_mse_mean"] + 0.3 * out["loss_ps_mean"])
