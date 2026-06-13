"""Smoke tests for :mod:`src.figures.plan_comparison`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.figures.plan_comparison import dose_comparison_metrics, plan_dose_comparison


def _volumes(shape=(12, 16, 20)):
    rng = np.random.default_rng(0)
    ct = rng.uniform(-1000, 1000, size=shape).astype(np.float32)
    dose_b = np.zeros(shape, dtype=np.float32)
    dose_b[6, 8, 5:15] = 100.0  # a beam-like deposit
    dose_a = dose_b * 1.05  # 5% high, like the real result
    return ct, dose_a, dose_b


def test_metrics() -> None:
    _, dose_a, dose_b = _volumes()
    m = dose_comparison_metrics(dose_a, dose_b, threshold=0.1)
    assert m["peak_ratio"] == pytest.approx(1.05)
    assert m["integral_ratio"] == pytest.approx(1.05)
    assert m["mape_high_dose_pct"] == pytest.approx(5.0, abs=1e-3)


def test_writes_all_formats_and_caption(tmp_path: Path) -> None:
    ct, dose_a, dose_b = _volumes()
    paths = plan_dose_comparison(ct, dose_a, dose_b, str(tmp_path / "cmp"), dpi=80)
    suffixes = {p.suffix for p in paths}
    assert {".svg", ".pdf", ".png"} <= suffixes
    assert all(p.is_file() for p in paths)
    # No metric text on the figure; the caption sidecar holds it instead.
    caption = tmp_path / "cmp_caption.txt"
    assert caption.is_file()
    assert "peak ratio" in caption.read_text()


def test_grid_disabled(tmp_path: Path) -> None:
    ct, dose_a, dose_b = _volumes()
    paths = plan_dose_comparison(
        ct, dose_a, dose_b, str(tmp_path / "cmp2"), grid=False, write_caption=False, dpi=80
    )
    assert all(p.is_file() for p in paths)
    assert not (tmp_path / "cmp2_caption.txt").exists()


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    ct, dose_a, dose_b = _volumes()
    with pytest.raises(ValueError, match="share shape"):
        plan_dose_comparison(ct, dose_a[..., :-1], dose_b, str(tmp_path / "x"))
