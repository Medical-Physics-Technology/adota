"""Beamlet-versus-HU decomposition of voxel errors (synthetic, CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine_drivers.voxel_decomposition import additive_fit, decompose, decomposition_table


def _cells(hu_effect: float, n_beam: int = 60, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    bins = np.array([-1000.0, -400.0, 0.0, 400.0, 900.0])
    rows, meta = [], []
    for k in range(n_beam):
        level = rng.gamma(2.0, 2.0)
        meta.append({"sample_id": f"s{k}", "patient_id": f"p{k // 3}", "site": "thoracic"})
        for lo in bins:
            n = int(rng.integers(20, 200))
            mean = level + hu_effect * (lo >= 300) + rng.normal(0, 0.05)
            rows.append({"sample_id": f"s{k}", "engine": "E", "band": "50-100%", "bin_low_hu": lo, "n": n,
                         "sum_abs": n * mean, "sum_signed": n * mean})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def test_additive_fit_recovers_two_exact_effects():
    a = np.repeat(np.arange(4), 3)
    b = np.tile(np.arange(3), 4)
    y = np.array([1.0, 2.0, 3.0, 4.0])[a] + np.array([0.0, 0.5, -0.5])[b]
    w = np.arange(1, 13, dtype=float)
    fit, _, _ = additive_fit(y, w, a, b, 4, 3)
    assert np.allclose(fit, y)


def test_no_material_effect_gives_zero_hu_increment_and_contrast():
    cells, meta = _cells(0.0)
    arr_y = cells["sum_abs"] / cells["n"]
    out = decompose(arr_y.to_numpy(), cells["n"].to_numpy(float), pd.factorize(cells["sample_id"])[0],
                    pd.factorize(cells["bin_low_hu"], sort=True)[0], np.sort(cells["bin_low_hu"].unique()))
    assert out["r2_beamlet"] > 0.95 and out["hu_increment"] < 0.01 and abs(out["bone_minus_soft"]) < 0.05


def test_a_bone_effect_is_found_within_beamlets():
    cells, meta = _cells(3.0)
    table = decomposition_table(cells, meta, ["50-100%"], values=("abs",), n_boot=30, seed=1, scopes=["all"])
    row = table.iloc[0]
    assert abs(row["bone_minus_soft"] - 3.0) < 0.1
    assert row["hu_increment"] > 0.1 and row["hu_increment_ci_low"] > 0.05
