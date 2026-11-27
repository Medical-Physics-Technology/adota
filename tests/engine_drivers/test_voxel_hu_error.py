"""Voxel-level dose error against HU (synthetic, CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine_drivers.voxel_hu_error import (
    BANDS,
    HU_EDGES,
    SUMS,
    _beamlet_sums,
    beamlet_medians,
    pooled,
    tissue_pooled,
)


def test_hu_edges_put_the_class_boundaries_on_bin_edges():
    for edge in (-1000.0, -500.0, 0.0, 300.0):
        assert edge in HU_EDGES
    assert HU_EDGES[0] <= -1024 and HU_EDGES[-1] >= 3071


def _cell(band: int, hu: float) -> int:
    return band * (HU_EDGES.size - 1) + int(np.searchsorted(HU_EDGES, hu, side="right") - 1)


def test_beamlet_sums_bin_by_band_and_hu_with_global_and_local_errors():
    mc = np.array([100.0, 40.0, 5.0, 0.05])          # max 100: bands 50-100, 10-50, 0.1-10, below cutoff
    dose = np.array([110.0, 30.0, 6.0, 1.0])
    hu = np.array([0.0, -800.0, 1200.0, 0.0])
    sums = _beamlet_sums(hu, mc, dose, cutoff=0.001, local_min=0.10)
    assert sums[:, 0].sum() == 3                       # the 0.05 voxel is below 0.1% of the maximum
    peak = sums[_cell(2, 0.0)]
    assert np.allclose(peak, [1, 10.0, 10.0, 1, 10.0])  # e = +10% of max, local = 10%
    mid = sums[_cell(1, -800.0)]
    assert np.allclose(mid, [1, 10.0, -10.0, 1, 25.0])  # local = 10/40
    low = sums[_cell(0, 1200.0)]
    assert np.allclose(low, [1, 1.0, 1.0, 0, 0.0])      # below 10%: no local error


def _cells_and_meta(n_pat: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for p in range(n_pat):
        for b in range(3):
            sid = f"p{p}b{b}"
            meta.append({"sample_id": sid, "patient_id": f"P{p}", "site": "thoracic"})
            for engine, bias in (("ADoTA", 1.0), ("Pencil beam", 5.0)):
                for lo in (-1000.0, 0.0, 400.0):
                    n = int(rng.integers(5, 30))
                    rows.append({"sample_id": sid, "engine": engine, "band": BANDS[2][0], "bin_low_hu": lo,
                                 "n": n, "sum_abs": n * bias, "sum_signed": n * bias, "n_local": n,
                                 "sum_local": n * 2 * bias})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def test_pooled_means_are_voxel_weighted_and_intervals_contain_them():
    cells, meta = _cells_and_meta()
    out = pooled(cells, meta, n_boot=100, seed=1)
    pba = out[out["engine"] == "Pencil beam"]
    assert np.allclose(pba["abs_mean"], 5.0) and np.allclose(pba["local_mean"], 10.0)
    assert (out["abs_ci_low"] <= out["abs_mean"] + 1e-9).all() and (out["abs_ci_high"] >= out["abs_mean"] - 1e-9).all()
    assert (out["patients"] == 12).all()
    assert out["voxels"].sum() == cells["n"].sum()


def test_tissue_classes_split_at_minus_500_and_300():
    cells, meta = _cells_and_meta()
    out = tissue_pooled(cells, meta, n_boot=50, seed=1)
    assert set(out["tissue"]) == {"below -500 HU", "-500 to 300 HU", "above 300 HU"}
    assert out.groupby("engine")["voxels"].sum().eq(cells.groupby("engine")["n"].sum()).all()


def test_beamlet_medians_drop_sparse_cells():
    cells, meta = _cells_and_meta()
    cells.loc[0, "n"] = 3
    out = beamlet_medians(cells, meta, min_voxels=10)
    assert out["beamlets"].sum() == (cells["n"] >= 10).sum()
    assert set(SUMS) <= set(cells.columns)
