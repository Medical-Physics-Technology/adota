"""Smoke tests for :mod:`src.figures.gamma_comparison`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.figures.gamma_comparison import plan_gamma_figure


def _ct_and_results(shape=(10, 14, 18), n_crit=2):
    rng = np.random.default_rng(0)
    ct = rng.uniform(-1000, 1000, size=shape).astype(np.float32)
    results = []
    for i in range(n_crit):
        gm = np.zeros(shape, dtype=np.float32)
        gm[3:7, 5:9, 4:14] = 0.5 + 0.1 * i  # evaluated, passing
        gm[5, 7, 9] = 1.5  # one failing voxel -> exercises the contour branch
        results.append(
            {
                "criterion": (1.0 + i, 1.0 + i, 10.0),
                "label": f"{1 + i}%/{1 + i}mm/10%",
                "pass_rate_pct": 95.0 - i,
                "gamma_map": gm,
            }
        )
    return ct, results


def test_writes_all_formats_and_caption(tmp_path: Path) -> None:
    ct, results = _ct_and_results(n_crit=2)
    paths = plan_gamma_figure(
        ct, results, (5, 7, 9), str(tmp_path / "gamma"), dpi=80
    )
    suffixes = {p.suffix for p in paths}
    assert {".svg", ".pdf", ".png"} <= suffixes
    assert all(p.is_file() for p in paths)
    caption = tmp_path / "gamma_caption.txt"
    assert caption.is_file()
    assert "Gamma pass rates" in caption.read_text()


def test_handles_five_criteria(tmp_path: Path) -> None:
    ct, results = _ct_and_results(n_crit=5)
    paths = plan_gamma_figure(
        ct, results, (5, 7, 9), str(tmp_path / "g5"), write_caption=False, dpi=80
    )
    assert all(p.is_file() for p in paths)
    assert not (tmp_path / "g5_caption.txt").exists()


def test_empty_results_raises(tmp_path: Path) -> None:
    ct, _ = _ct_and_results()
    with pytest.raises(ValueError, match="empty"):
        plan_gamma_figure(ct, [], (5, 7, 9), str(tmp_path / "x"))


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    ct, results = _ct_and_results()
    results[0]["gamma_map"] = results[0]["gamma_map"][..., :-1]
    with pytest.raises(ValueError, match="!= ct shape"):
        plan_gamma_figure(ct, results, (5, 7, 9), str(tmp_path / "x"))
