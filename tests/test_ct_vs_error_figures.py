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


def test_title_lines_skip_non_finite_scopes():
    from src.figures.ct_vs_error import _title_lines
    stats = pd.DataFrame([{"error": "e", "feature": "energy_mev", "scope": "pooled", "spearman": 0.3,
                           "ci_low": 0.2, "ci_high": 0.4},
                          {"error": "e", "feature": "energy_mev", "scope": "energy_partial", "spearman": float("nan"),
                           "ci_low": float("nan"), "ci_high": float("nan")}])
    lines = _title_lines(stats, "e", "energy_mev", ["thoracic"])
    assert lines == ["pooled rho +0.30 [+0.20, +0.40]"]


def test_axis_label_states_direction_only_when_reversed():
    from src.figures.ct_vs_error import _axis_label
    assert _axis_label("GPR", False, True) == "GPR"
    assert "worse to the right" in _axis_label("GPR", True, True)
    assert "worse upward" in _axis_label("GPR", True, False)


def _summary_inputs():
    rng = np.random.default_rng(3)
    stats = []
    for f in ("a", "b", "depth"):
        for scope in ("pooled", "thoracic", "abdominal", "pelvic"):
            r = float(rng.uniform(-0.3, 0.6))
            stats.append({"error": "mape_pct", "feature": f, "scope": scope, "spearman": r, "ci_low": r - 0.05,
                          "ci_high": r + 0.05})
    stats = pd.DataFrame(stats)
    curves = pd.DataFrame([{"feature": f, "site": s, "bin": k, "feature_median": float(k), "value": float(k + i)}
                           for f in ("a", "b") for i, s in enumerate(("thoracic", "abdominal", "pelvic"))
                           for k in range(1, 6)])
    return stats, curves


def test_correlation_overview_and_shape_grid_write_the_formats(tmp_path):
    from src.figures.ct_vs_error import correlation_overview, feature_shape_grid
    stats, curves = _summary_inputs()
    sites = ["thoracic", "abdominal", "pelvic"]
    paths = correlation_overview({"E1": stats, "E2": stats}, features=["a", "b"], reference_features=["depth"],
                                 feature_names={"a": "Feature A"}, error="mape_pct", sites=sites, xlabel="rho",
                                 title="t", figure_path=str(tmp_path / "ov"), formats=("png",))
    assert [p.suffix for p in paths] == [".png"] and paths[0].stat().st_size > 1000
    paths = feature_shape_grid({"E1": curves, "E2": curves}, {"E1": {"a": 0.3}}, features=["a", "b"],
                               feature_names={"a": ("Feature A", "mm")}, sites=sites, ylabel="MAPE", title="t",
                               figure_path=str(tmp_path / "sh"), formats=("png", "pdf"))
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
