"""Smoke test for :func:`src.figures.single_beam_ground_truth.ground_truth_figure`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.figures.single_beam_ground_truth import ground_truth_figure


def test_writes_svg_pdf_png(tmp_path: Path) -> None:
    shape = (160, 30, 30)                                   # (D, H, W) on the 2 mm model grid
    ct = np.random.default_rng(0).uniform(-1000, 1000, size=shape).astype(np.float32)
    flux = np.zeros(shape, dtype=np.float32)
    flux[:, 15, 15] = 1.0
    depth = np.arange(shape[0], dtype=np.float32)
    dose = np.zeros(shape, dtype=np.float32)
    dose[:, 14:17, 14:17] = np.exp(-((depth - 100.0) / 8.0) ** 2)[:, None, None] + 0.2 * (depth < 100)[:, None, None]
    paths = ground_truth_figure(ct, flux, dose, str(tmp_path / "gt"), title="t")
    assert {p.suffix for p in paths} == {".svg", ".pdf", ".png"}
    assert all(p.is_file() for p in paths)
