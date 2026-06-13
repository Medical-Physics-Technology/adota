"""Smoke tests for :func:`src.figures.single_beam.beamlet_input_figure`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.figures.single_beam import beamlet_input_figure


def _ct_and_flux(shape=(8, 8, 20)):
    ct = np.random.default_rng(0).uniform(-1000, 1000, size=shape).astype(np.float32)
    flux = np.zeros(shape, dtype=np.float32)
    flux[shape[0] // 2, shape[1] // 2, :] = 1.0  # a tube along depth
    return ct, flux


def test_writes_svg_pdf_png(tmp_path: Path) -> None:
    ct, flux = _ct_and_flux()
    paths = beamlet_input_figure(
        ct, flux, str(tmp_path / "spot_input"),
        initial_energy=150.0, beamlet_angles=(0.1, 0.2), spot_id="b00_l000_s0000",
    )
    suffixes = {p.suffix for p in paths}
    assert suffixes == {".svg", ".pdf", ".png"}
    assert all(p.is_file() for p in paths)


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    ct, flux = _ct_and_flux()
    with pytest.raises(ValueError, match="same shape"):
        beamlet_input_figure(ct, flux[..., :-1], str(tmp_path / "x"))


def test_non_3d_raises(tmp_path: Path) -> None:
    arr = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="z, y, x"):
        beamlet_input_figure(arr, arr, str(tmp_path / "x"))
