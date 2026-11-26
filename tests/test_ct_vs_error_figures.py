"""Synthetic, CPU-only tests for :mod:`src.figures.ct_vs_error`."""
from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from src.figures.ct_vs_error import (
    _is_reversed,
    _title_lines,
    error_vs_feature_grid,
    feature_vs_error_grid,
)

SITES = ["thoracic", "abdominal", "pelvic"]
FEATURES = ["depth_mm", "mean_hu"]
ERRORS = ["gpr_2pct_2mm_10pct", "mape_pct"]
FEATURE_LABELS = {"depth_mm": "Water-equivalent depth [mm]", "mean_hu": "Mean HU on path [HU]"}
ERROR_LABELS = {"gpr_2pct_2mm_10pct": "Gamma pass rate 2%/2mm [%]", "mape_pct": "MAPE [%]"}


def synthetic_tables():
    rng = np.random.default_rng(0)
    n = 600
    site = rng.choice(SITES, n)
    depth = rng.normal(150, 40, n) + 20 * (site == "thoracic")
    hu = rng.normal(0, 150, n)
    gpr = np.clip(100 - 0.05 * depth + rng.normal(0, 3, n), 60, 100)
    mape = np.abs(rng.normal(2 + 0.02 * depth, 1.5))
    points = pd.DataFrame({"sample_id": np.arange(n), "site": site, "depth_mm": depth, "mean_hu": hu,
                           "gpr_2pct_2mm_10pct": gpr, "mape_pct": mape})
    ebins: List[Dict] = []
    fbins: List[Dict] = []
    for s in SITES:
        d = points[points["site"] == s]
        for e in ERRORS:
            for f in FEATURES:
                for b, g in d.groupby(pd.qcut(d[e], 5, labels=False)):
                    ebins.append({"error": e, "feature": f, "site": s, "bin": int(b), "error_median": g[e].median(),
                                  "n": len(g), "feature_median": g[f].median(), "feature_q25": g[f].quantile(0.25),
                                  "feature_q75": g[f].quantile(0.75)})
                for b, g in d.groupby(pd.qcut(d[f], 5, labels=False)):
                    fbins.append({"error": e, "feature": f, "site": s, "bin": int(b), "feature_median": g[f].median(),
                                  "n": len(g), "error_median": g[e].median(), "error_q25": g[e].quantile(0.25),
                                  "error_q75": g[e].quantile(0.75)})
    stats = pd.DataFrame([{"error": e, "feature": f, "scope": sc, "spearman": 0.3, "ci_low": 0.2, "ci_high": 0.4}
                          for e in ERRORS for f in FEATURES for sc in ["pooled", "energy_partial"] + SITES])
    return points, pd.DataFrame(ebins), pd.DataFrame(fbins), stats


def _kwargs(path):
    return dict(features=FEATURES, errors=ERRORS, feature_labels=FEATURE_LABELS, error_labels=ERROR_LABELS,
                reversed_errors=["gpr_2pct_2mm_10pct"], sites=SITES, title="Test", figure_path=str(path),
                caption="Beamlets outside the 0.5-99.5 percentile range are not drawn.")


@pytest.mark.parametrize("formats", [("png",), ("png", "pdf")])
def test_feature_vs_error_grid_writes_requested_formats(tmp_path, formats):
    points, ebins, _, stats = synthetic_tables()
    paths = feature_vs_error_grid(points, ebins, stats, formats=formats, **_kwargs(tmp_path / "a"))
    assert [p.suffix for p in paths] == [f".{f}" for f in formats]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"a.{f}" for f in formats)


@pytest.mark.parametrize("formats", [("png",), ("png", "pdf")])
def test_error_vs_feature_grid_writes_requested_formats(tmp_path, formats):
    points, _, fbins, stats = synthetic_tables()
    paths = error_vs_feature_grid(points, fbins, stats, formats=formats, **_kwargs(tmp_path / "b"))
    assert [p.suffix for p in paths] == [f".{f}" for f in formats]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"b.{f}" for f in formats)


def test_title_lines_format_numbers_exactly():
    stats = pd.DataFrame([
        {"error": "e", "feature": "f", "scope": "pooled", "spearman": 0.35, "ci_low": 0.31, "ci_high": 0.40},
        {"error": "e", "feature": "f", "scope": "energy_partial", "spearman": 0.2, "ci_low": 0.15, "ci_high": 0.25},
        {"error": "e", "feature": "f", "scope": "thoracic", "spearman": -0.3, "ci_low": -0.35, "ci_high": -0.25},
    ])
    lines = _title_lines(stats, "e", "f", ["thoracic", "pelvic"])
    assert lines[0] == "pooled rho +0.35 [+0.31, +0.40]; energy-controlled +0.20 [+0.15, +0.25]"
    assert lines[1] == "thor -0.30 [-0.35, -0.25]"


def test_title_lines_skip_missing_scopes():
    stats = pd.DataFrame([
        {"error": "e", "feature": "f", "scope": "pooled", "spearman": 0.1, "ci_low": 0.0, "ci_high": 0.2},
    ])
    assert _title_lines(stats, "e", "f", ["thoracic"]) == ["pooled rho +0.10 [+0.00, +0.20]"]
    assert _title_lines(stats, "other", "f", ["thoracic"]) == []


def test_is_reversed():
    assert _is_reversed("gpr_2pct_2mm_10pct", ["gpr_2pct_2mm_10pct"])
    assert not _is_reversed("mape_pct", ["gpr_2pct_2mm_10pct"])
