"""Unit tests for per-beamlet analysis helpers (data-free)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.beamlets.beamlet_analysis import (
    BeamletAnalysisConfig,
    _aggregate,
    _stats,
    assert_join,
    mc_support_bbox_gy,
    per_spot_metrics,
)
from src.beamlets.plan_spots import spot_id


def _si(rows):
    return pd.DataFrame(rows, columns=["beam_idx", "layer_idx", "spot_idx"])


def test_assert_join_ok_and_mismatch():
    si = _si([(0, 0, 0), (0, 1, 0), (0, 1, 1)])
    recs = [{"id": spot_id(b, l, s)} for b, l, s in [(0, 0, 0), (0, 1, 0), (0, 1, 1)]]
    assert_join(si, recs)  # no raise
    bad = [{"id": spot_id(0, 0, 0)}, {"id": spot_id(9, 9, 9)}, {"id": spot_id(0, 1, 1)}]
    with pytest.raises(ValueError):
        assert_join(si, bad)
    with pytest.raises(ValueError):  # count mismatch
        assert_join(si, recs[:2])


def test_mc_support_bbox_reshape_and_placement():
    # 3x3x3 grid, one column, two known voxels; verify F-order unravel + Gy scale
    nx, ny, nz = 3, 3, 3
    grid = [nx, ny, nz]
    # place dose at (x,y,z)=(1,2,0) and (2,2,1); F-order flat = x + nx*(y + ny*z)
    def flat(x, y, z):
        return x + nx * (y + ny * z)
    data = np.zeros(nx * ny * nz, dtype=np.float32)
    data[flat(1, 2, 0)] = 2.0
    data[flat(2, 2, 1)] = 4.0
    M = sp.csc_matrix(data.reshape(-1, 1))
    si = pd.DataFrame({"mu": [10.0], "opentps_rescaling": [0.5]})  # w = 5.0
    box, (z0, y0, x0), n = mc_support_bbox_gy(M, si, 0, grid, margin=0)
    assert n == 2
    # values scaled by w=5.0; placed at their (z-z0, y-y0, x-x0)
    assert box[0 - z0, 2 - y0, 1 - x0] == pytest.approx(2.0 * 5.0)
    assert box[1 - z0, 2 - y0, 2 - x0] == pytest.approx(4.0 * 5.0)
    assert box.sum() == pytest.approx((2.0 + 4.0) * 5.0)


def test_per_spot_metrics_identical_crops():
    # identical MC and ADoTA crops -> perfect agreement
    rng = np.random.default_rng(0)
    D, H, W = 40, 12, 12
    depth = np.exp(-((np.arange(D) - 25) ** 2) / (2 * 5.0**2))  # Bragg-like along depth
    lat = np.exp(-((np.arange(H)[:, None] - 6) ** 2 + (np.arange(W)[None, :] - 6) ** 2) / (2 * 2.0**2))
    crop = (lat[:, :, None] * depth[None, None, :]).astype(np.float64)  # (H,W,D)
    m = per_spot_metrics(crop.copy(), crop.copy(), BeamletAnalysisConfig())
    assert m["integral_ratio"] == pytest.approx(1.0)
    assert m["mape_pct"] == pytest.approx(0.0, abs=1e-6)
    assert m["corr_high_dose"] == pytest.approx(1.0, abs=1e-6)
    assert m["r80_diff_mm"] == pytest.approx(0.0, abs=1e-6)
    # (GPR is not asserted here: identical distributions give all-zero gamma, a
    # 0/0 edge case in gamma_index that real, noisy beamlets never hit.)


def test_per_spot_metrics_scaled_prediction():
    # ADoTA = 1.10 * MC: integral ratio 1.10, MAPE 10%, correlation still 1
    D, H, W = 40, 12, 12
    depth = np.exp(-((np.arange(D) - 25) ** 2) / (2 * 5.0**2))
    lat = np.exp(-((np.arange(H)[:, None] - 6) ** 2 + (np.arange(W)[None, :] - 6) ** 2) / (2 * 2.0**2))
    mc = (lat[:, :, None] * depth[None, None, :]).astype(np.float64)
    m = per_spot_metrics(mc.copy(), (1.10 * mc).copy(), BeamletAnalysisConfig())
    assert m["integral_ratio"] == pytest.approx(1.10, abs=1e-4)
    assert m["mape_pct"] == pytest.approx(10.0, abs=1e-3)
    assert m["corr_high_dose"] == pytest.approx(1.0, abs=1e-6)


def test_aggregate_stratifies_and_mu_weights():
    df = pd.DataFrame({
        "energy_mev": [120.0, 160.0, 200.0],
        "mu_fraction": [0.2, 0.3, 0.5],
        "gpr_2pct_2mm": [98.0, 90.0, 80.0],
        "gpr_3pct_3mm": [99.0, 95.0, 92.0],
        "mape_pct": [1.0, 2.0, 3.0], "rmse_gy": [0.1, 0.2, 0.3],
        "corr_high_dose": [0.99, 0.9, 0.8], "r80_diff_mm": [0.0, 1.0, -1.0],
        "integral_ratio": [1.0, 1.0, 1.0],
    })
    cfg = BeamletAnalysisConfig(energy_split_mev=150.0)
    agg = _aggregate(df, cfg, {"primaries_per_beamlet": 1e5}, "somedir")
    assert agg["below_150MeV"]["n_spots"] == 1
    assert agg["at_or_above_150MeV"]["n_spots"] == 2
    # MU fraction carried by the >=150 spots
    assert agg["mu_fraction_above_split"] == pytest.approx(0.8)
    # unweighted vs MU-weighted overall GPR2/2 differ (weights skewed to low-GPR spots)
    ov = agg["overall"]["gpr_2pct_2mm"]
    assert ov["mean"] == pytest.approx((98 + 90 + 80) / 3, abs=1e-3)
    assert ov["mu_weighted_mean"] == pytest.approx(0.2 * 98 + 0.3 * 90 + 0.5 * 80, abs=1e-3)


def test_stats_empty():
    assert _stats(pd.DataFrame(columns=["mu_fraction"]))["n_spots"] == 0
