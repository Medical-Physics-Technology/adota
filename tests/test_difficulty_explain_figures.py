"""Smoke tests of the difficulty-explanation figures on synthetic data (CPU, no dataset)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.acquisition.explain.profiles import BeamAxisProfile, BeamletInputs
from src.figures.difficulty_explain import (
    GROUP_PALETTE,
    DepthMark,
    beam_path_figure,
    contrast_figure,
    contribution_figure,
    decile_profile_figure,
    group_colors,
)

GROUPS = ["depth and energy [max_grad_depth_mm]", "edges around the peak [sobel_th_mean]",
          "tissue changes in the peak zone [total_hu_change]", "bone along the path [bone_frac]",
          "lateral heterogeneity [lat_std]", "range uncertainty [wepl_gradient]"]
MEMBERS = ["C0_transformer_021", "C1_transformer_016", "X0_mlstm_004", "U0_none_029", "T012_transformer_012"]
SCORERS = ["ensemble"] + MEMBERS
EXAMPLES = ["easy1", "med1", "hard1"]
EXT = ("png", "pdf", "svg")


def _synthetic_beamlet(d=160, n=30, dz=2.0):
    ct = np.zeros((d, n, n))
    ct[50:56] = 1000.0
    lat = (np.arange(n) - 15.0) * dz
    gauss = np.exp(-0.5 * (lat[:, None] ** 2 + lat[None, :] ** 2) / 8.0 ** 2)
    depth = np.arange(d) * dz
    bragg = np.exp(-0.5 * ((depth - 200.0) / 8.0) ** 2) + 0.3 * (depth < 200.0)
    bragg /= bragg.max()
    dose = bragg[:, None, None] * gauss[None]
    idd = dose.sum(axis=(1, 2))
    idd /= idd.max()
    peak = tuple(int(i) for i in np.unravel_index(np.argmax(dose), dose.shape))
    inputs = BeamletInputs("syn", "grid", ct, np.broadcast_to(gauss, (d, n, n)).copy(), 120.0, mc_dose=dose * 0.98)
    profile = BeamAxisProfile(
        depth_mm=depth, hu_axis=ct[:, peak[1], peak[2]], hu_flux_weighted=ct.mean(axis=(1, 2)) * 0.8,
        rsp_axis=1.0 + 0.0016 * ct[:, peak[1], peak[2]], wepl_axis_mm=np.cumsum(1.0 + 0.0016 * ct[:, 15, 15]) * dz,
        idd_analytic=idd, idd_mc=idd * 0.98, analytic_dose=dose, peak_index=peak, bp_slice=int(np.argmax(idd)),
        bp_zone_mm=(184.0, 216.0), mc_bp_slice=int(np.argmax(idd)), sphere_radius_mm=20.0, dz_mm=dz)
    return inputs, profile


def _contrib_tables():
    rng = np.random.default_rng(0)
    rows, tots = [], []
    for s in SCORERS:
        for b in EXAMPLES:
            vals = rng.normal(0, 0.4, len(GROUPS))
            rows += [{"beamlet_id": b, "scorer": s, "group": g, "contribution": float(v)} for g, v in zip(GROUPS, vals)]
            tots.append({"scorer": s, "beamlet_id": b, "score": float(vals.sum()) + 1.0, "baseline": 1.0})
    return pd.DataFrame(rows), pd.DataFrame(tots)


def _contrast_table(pairs):
    rng = np.random.default_rng(1)
    rows = []
    for h, e in pairs:
        for s in SCORERS:
            for rank, g in enumerate(GROUPS):
                rows.append({"scorer": s, "item": g, "harder": h, "easier": e, "difference": float(rng.normal(0, 0.3)),
                             "rank": rank})
    return pd.DataFrame(rows)


def _assert_written(tmp_path, stem="x"):
    for ext in EXT:
        f = tmp_path / f"{stem}.{ext}"
        assert f.exists() and f.stat().st_size > 1000


def test_group_colors_stable_under_order():
    a = group_colors(["b", "a", "c", "a"])
    assert a == group_colors(["c", "b", "a"])
    assert len(set(a.values())) == 3 and a["a"] == GROUP_PALETTE[0]
    assert len(set(GROUP_PALETTE)) >= 10


def test_contribution_figure_writes_and_orders(tmp_path, monkeypatch):
    contrib, totals = _contrib_tables()
    seen = []
    import matplotlib.axes as maxes
    orig = maxes.Axes.barh

    def spy(self, y, width, *a, **k):
        seen.append(tuple(k["color"]))
        return orig(self, y, width, *a, **k)

    monkeypatch.setattr(maxes.Axes, "barh", spy)
    paths = contribution_figure(contrib, totals, examples=EXAMPLES, example_labels={"easy1": "Easy", "med1": "Medium",
                                "hard1": "Hard"}, scorers=SCORERS, colors=group_colors(GROUPS),
                                figure_path=str(tmp_path / "x"), title="t")
    assert {p.suffix for p in paths} == {".png", ".pdf", ".svg"}
    _assert_written(tmp_path)
    assert len(seen) == len(SCORERS) * len(EXAMPLES) and len(set(seen)) == 1


def test_contrast_figure_writes(tmp_path):
    pairs = [("med1", "easy1"), ("hard1", "med1")]
    labels = {pairs[0]: "Medium vs easy", pairs[1]: "Hard vs medium"}
    contrast_figure(_contrast_table(pairs), pairs=pairs, pair_labels=labels, colors=group_colors(GROUPS),
                    figure_path=str(tmp_path / "x"), title="t")
    _assert_written(tmp_path)


@pytest.mark.parametrize("with_mc", [True, False])
def test_beam_path_figure_writes(tmp_path, with_mc):
    inputs, profile = _synthetic_beamlet()
    if not with_mc:
        from dataclasses import replace
        profile = replace(profile, idd_mc=None, mc_bp_slice=None)
    marks = [DepthMark("Bone slab", 100.0, 112.0, "#d62728"), DepthMark("Peak", 200.0, 200.0, "#2ca02c")]
    beam_path_figure(inputs, profile, title="Hard beamlet", marks=marks, figure_path=str(tmp_path / "x"))
    _assert_written(tmp_path)


def _decile_profile():
    rng = np.random.default_rng(2)
    rows = []
    for panel in ("all", "lung"):
        for d in range(1, 11):
            vals = rng.normal(0, 0.4, 4)
            n = 5 if (panel == "lung" and d == 3) else 100 + d
            rows += [{"panel": panel, "decile": d, "group": f"g{i}", "mean_contribution": float(v), "n_beamlets": n,
                      "n_patients": 7, "mean_total": float(vals.sum())} for i, v in enumerate(vals)]
    return pd.DataFrame(rows)


def test_decile_profile_figure_writes_png_only(tmp_path):
    colors = group_colors([f"g{i}" for i in range(4)])
    paths = decile_profile_figure(_decile_profile(), panels=["all", "lung"], panel_titles={"all": "All"},
                                  colors=colors, figure_path=str(tmp_path / "x"), title="t", formats=("png",))
    assert [p.name for p in paths] == ["x.png"]
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["x.png"] and (tmp_path / "x.png").stat().st_size > 1000
    assert not (tmp_path / "x.svg").exists() and not (tmp_path / "x.pdf").exists()
