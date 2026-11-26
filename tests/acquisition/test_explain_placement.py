"""Placement of beamlets against a reference population (CPU, synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.acquisition.explain.placement import percentile_of, placement_table, robust_scale

COLUMNS = [
    "beamlet_id", "feature", "value", "percentile_ref", "robust_z", "ref_median",
    "ref_q25", "ref_q75", "ref_scale", "scale_method", "n_reference",
]  # fmt: skip


def test_percentile_of_midrank():
    out = percentile_of(np.array([1, 2, 3, 4]), np.array([2.5, 0, 5, 2, np.nan]))
    assert out[:4] == pytest.approx([50.0, 0.0, 100.0, 37.5])
    assert np.isnan(out[4])


def test_percentile_of_drops_reference_nan():
    assert percentile_of(np.array([1, np.nan, 2, 3, 4]), np.array([2.5]))[0] == pytest.approx(50.0)


def test_robust_scale_normal():
    ref = np.random.default_rng(0).standard_normal(200_000)
    med, scale, method = robust_scale(ref)
    assert method == "iqr"
    assert abs(med) < 0.02
    assert scale == pytest.approx(1.0, rel=0.02)


def test_robust_scale_zero_iqr_has_no_usable_scale():
    # IQR = 0 forces at least half the sample onto the median, hence MAD = 0 as well.
    ref = np.concatenate([np.zeros(800), np.random.default_rng(1).standard_normal(200)])
    assert np.quantile(ref, 0.75) - np.quantile(ref, 0.25) == 0
    assert np.median(np.abs(ref - np.median(ref))) == 0
    assert robust_scale(ref)[2] == "degenerate"


def test_robust_scale_constant_degenerate_and_nan_z():
    ref = pd.DataFrame({"a": np.full(50, 3.0)})
    beam = pd.DataFrame({"a": [3.0, 4.0]}, index=["x", "y"])
    med, scale, method = robust_scale(ref["a"].to_numpy())
    assert (med, method) == (3.0, "degenerate") and np.isnan(scale)
    table = placement_table(ref, beam, ["a"])
    assert table["robust_z"].isna().all()
    assert (table["scale_method"] == "degenerate").all()


def test_placement_table_lognormal():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"a": rng.lognormal(size=5000), "b": rng.lognormal(size=5000)})
    med = ref["a"].median()
    beam = pd.DataFrame({"a": [med, 10 * med], "b": [1.0, 2.0]}, index=["p", "q"])
    table = placement_table(ref, beam, ["a", "b"])
    assert list(table.columns) == COLUMNS
    assert len(table) == 4
    assert list(table["beamlet_id"]) == ["p", "p", "q", "q"]
    assert list(table["feature"]) == ["a", "b", "a", "b"]
    row = table.iloc[0]
    assert row["robust_z"] == pytest.approx(0.0, abs=1e-12)
    assert row["percentile_ref"] == pytest.approx(50.0, abs=0.1)
    assert row["n_reference"] == 5000
    assert table.iloc[2]["robust_z"] > 0
