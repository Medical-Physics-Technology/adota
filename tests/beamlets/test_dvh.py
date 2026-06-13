"""Tests for :mod:`src.beamlets.dvh` (OpenTPS-adapted DVH)."""

from __future__ import annotations

import numpy as np
import pytest

from src.beamlets.dvh import DVH


def test_uniform_dose() -> None:
    mask = np.ones((4, 4, 4), bool)
    dose = np.full((4, 4, 4), 50.0)
    dvh = DVH(mask, dose, (1.0, 1.0, 1.0), name="t")
    assert dvh.Dmin == 50.0
    assert dvh.Dmax == 50.0
    assert dvh.Dmean == pytest.approx(50.0)
    assert dvh.D95 == pytest.approx(50.0, abs=0.5)
    assert dvh.histogram[1].max() == pytest.approx(100.0)


def test_two_value_dose() -> None:
    dose = np.zeros((2, 4, 4))
    dose[0] = 20.0
    dose[1] = 60.0
    mask = np.ones((2, 4, 4), bool)
    dvh = DVH(mask, dose, (1.0, 1.0, 1.0))
    assert dvh.Dmin == 20.0
    assert dvh.Dmax == 60.0
    assert dvh.Dmean == pytest.approx(40.0)
    # 98% of the volume gets >= ~20 Gy; 2% gets >= ~60 Gy.
    assert dvh.D98 == pytest.approx(20.0, abs=2.0)
    assert dvh.D2 == pytest.approx(60.0, abs=2.0)


def test_compute_vg() -> None:
    dose = np.zeros((2, 4, 4))
    dose[0] = 20.0
    dose[1] = 60.0
    dvh = DVH(np.ones((2, 4, 4), bool), dose, (1.0, 1.0, 1.0))
    assert dvh.compute_vg(50.0) == pytest.approx(50.0, abs=2.0)  # half gets >= 50 Gy
    assert dvh.compute_vg(10.0) == pytest.approx(100.0, abs=2.0)
    # absolute volume: half of 32 voxels * 1 mm^3 / 1000 = 0.016 cm^3
    assert dvh.compute_vg(50.0, return_percentage=False) == pytest.approx(0.016, abs=1e-3)


def test_empty_mask_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        DVH(np.zeros((4, 4, 4), bool), np.ones((4, 4, 4)), (1.0, 1.0, 1.0))


def test_metrics_dict() -> None:
    dvh = DVH(np.ones((3, 3, 3), bool), np.full((3, 3, 3), 30.0), (1.0, 1.0, 1.0))
    m = dvh.metrics()
    assert set(m) >= {"Dmin", "Dmean", "Dmax", "D95", "D2", "n_voxels"}
    assert m["n_voxels"] == 27
