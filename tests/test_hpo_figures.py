"""Tests of the HPO figures: :mod:`src.figures.hpo` and the pure helpers of
``scripts/active_learning/hpo_figures.py``. Synthetic data only, CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.active_learning.hpo_figures import finalist_order, pick_beamlets, selected_epoch
from src.figures.hpo import (
    beamlet_planes,
    depth_dose,
    finalist_beamlet_figure,
    hpo_learning_curves_figure,
    lateral_profile,
    model_styles,
    search_tier,
    shown_depth,
)


def _beamlet(depth=40, side=10, peak=25):
    """A Gaussian pencil with a Bragg-like peak at ``peak`` along depth."""
    z = np.arange(depth)[:, None, None]
    y = np.arange(side)[None, :, None] - side / 2 + 0.5
    x = np.arange(side)[None, None, :] - side / 2 + 0.5
    idd = np.where(z <= peak, 0.3 + 0.7 * np.exp(-((z - peak) / 3.0) ** 2), np.exp(-((z - peak) / 1.5) ** 2))
    return (idd * np.exp(-(x ** 2 + y ** 2) / 4.0)).astype(np.float32)


def test_pick_beamlets_rules():
    gpr = pd.DataFrame({"a": [0.99, 0.90, 0.50, 0.95], "b": [0.97, 0.40, 0.55, 0.93]},
                       index=["s1", "s2", "s3", "s4"])
    picks = pick_beamlets(gpr, ["typical", "disagreement", "hardest"])
    assert picks["disagreement"] == "s2"          # std of (0.90, 0.40) is the largest
    assert picks["hardest"] == "s3"               # mean 0.525
    means = gpr.mean(axis=1)
    assert picks["typical"] == (means - means.median()).abs().idxmin()
    with pytest.raises(ValueError, match="unknown picks"):
        pick_beamlets(gpr, ["best"])


def test_finalist_order_groups_by_mixer_roles_first():
    selection = {"C0": {"trial": "transformer_021"}, "C1": {"trial": "transformer_016"},
                 "X0": {"trial": "mlstm_004"}, "U0": {"trial": "none_029"}}
    order = finalist_order(selection, ["none_017", "transformer_012", "none_029", "mlstm_004",
                                       "transformer_016", "transformer_021"])
    assert order == ["transformer_021", "transformer_016", "transformer_012", "mlstm_004",
                     "none_029", "none_017"]


def test_selected_epoch_takes_the_highest_finished_rung(tmp_path: Path):
    for budget, seg in ((3, {"objective_epoch": 2}), (9, {"objective_epoch": 7}),
                        (27, {"objective_epoch": None, "failed": "worker exit 1"})):
        (tmp_path / f"segment_{budget:03d}.json").write_text(json.dumps(seg))
    assert selected_epoch(tmp_path) == 8          # epoch index 7, the 8th epoch
    assert selected_epoch(tmp_path / "missing") is None


def test_search_tier():
    assert [search_tier(e, [3, 9, 27]) for e in (1, 3, 4, 9, 16, 27)] == [0, 0, 1, 1, 2, 2]


def test_planes_profiles_and_depth():
    dose = _beamlet()
    peak = (25, 5, 5)
    axial, coronal = beamlet_planes(dose, peak)
    assert axial.shape == coronal.shape == (40, 10)
    np.testing.assert_allclose(depth_dose(dose, 2.0), dose.sum(axis=(1, 2)) * 4.0)
    np.testing.assert_array_equal(lateral_profile(dose, peak), dose[25, 5, :])
    assert 25 < shown_depth(dose, margin_mm=4.0, voxel_mm=2.0) <= 40


def test_finalist_beamlet_figure_writes_and_returns_curves(tmp_path: Path):
    gt = _beamlet()
    ct = np.random.default_rng(0).uniform(-1000, 1000, size=gt.shape).astype(np.float32)
    preds = {"model A": gt * 0.9, "model B": np.roll(gt, 2, axis=0)}
    paths, curves = finalist_beamlet_figure(ct, gt, preds, figure_path=str(tmp_path / "beamlet"), title="t",
                                            notes={"model A": "pass rate 90%"}, caption="c")
    assert {p.suffix for p in paths} == {".svg", ".pdf", ".png"} and all(p.is_file() for p in paths)
    assert list(curves["idd"].columns) == ["depth_mm", "Ground truth", "model A", "model B"]
    assert curves["idd"]["Ground truth"].max() == pytest.approx(100.0)
    assert curves["profile"]["model A"].max() == pytest.approx(0.9 * curves["profile"]["Ground truth"].max())


def test_learning_curves_figure_writes(tmp_path: Path):
    rows, info = [], []
    for mixer in ("transformer", "none"):
        for k, last in enumerate((3, 9, 27, 16)):
            trial = f"{mixer}_{k:03d}"
            rows += [{"trial": trial, "mixer": mixer, "epoch": e, "train_loss": 1.0 / e, "val_loss": 1.2 / e}
                     for e in range(1, last + 1)]
            info.append({"trial": trial, "mixer": mixer, "failed": last == 16,
                         "label": "C0" if last == 27 else "", "selected_epoch": 20 if last == 27 else np.nan})
    paths = hpo_learning_curves_figure(pd.DataFrame(rows), pd.DataFrame(info), [3, 9, 27],
                                       str(tmp_path / "curves"), title="t", caption="c")
    assert {p.suffix for p in paths} == {".svg", ".pdf", ".png"} and all(p.is_file() for p in paths)


def test_model_styles_one_hue_per_mixer():
    styles = model_styles({"C0": "transformer", "t2": "transformer", "X0": "mlstm"})
    assert styles["C0"]["color"] != styles["t2"]["color"]
    assert styles["C0"]["linestyle"] != styles["t2"]["linestyle"]
    assert styles["X0"]["linestyle"] == styles["C0"]["linestyle"]  # first of its mixer
