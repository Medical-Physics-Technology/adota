"""Tests for beamlet range-fidelity metrics (src.metrics.range_metrics)."""

import numpy as np
import pytest

from src.metrics.range_metrics import (
    compute_range_metrics,
    integrated_depth_dose,
    range_metric_deltas,
)

DZ = 2.0  # mm, beam-axis voxel spacing


def _synthetic_idd(n=160, peak_mm=120.0, dz=DZ, sigma_mm=8.0, falloff_mm=6.0):
    """Bragg-like IDD: gaussian rise to the peak, exponential distal fall-off."""
    z = np.arange(n) * dz
    idd = np.where(
        z <= peak_mm,
        np.exp(-0.5 * ((z - peak_mm) / sigma_mm) ** 2),
        np.exp(-(z - peak_mm) / falloff_mm),
    )
    return idd


def test_peak_and_distal_ordering():
    idd = _synthetic_idd(peak_mm=120.0)
    m = compute_range_metrics(idd, DZ)
    # Peak near 120 mm (within one fine-grid step).
    assert m.r100_mm == pytest.approx(120.0, abs=DZ)
    # Distal depths strictly increase as the level decreases, all beyond the peak.
    assert m.r100_mm <= m.r90_mm <= m.r80_mm <= m.r50_mm <= m.r20_mm
    assert m.dfw_mm == pytest.approx(m.r20_mm - m.r80_mm, abs=1e-6)
    assert m.dfw_mm > 0


def test_subvoxel_resolution_beats_grid():
    """R80 should land off the 2 mm grid thanks to interpolation."""
    idd = _synthetic_idd(peak_mm=121.0, falloff_mm=5.0)
    m = compute_range_metrics(idd, DZ, oversample=20)
    # Not snapped to an integer multiple of the voxel size.
    assert abs((m.r80_mm / DZ) - round(m.r80_mm / DZ)) > 1e-3


def test_empty_curve_returns_nan():
    m = compute_range_metrics(np.zeros(160), DZ)
    assert np.isnan(m.r80_mm)
    assert np.isnan(m.r100_mm)
    assert m.peak_dose == 0.0


def test_deltas_sign_convention():
    mc = _synthetic_idd(peak_mm=120.0, falloff_mm=5.0)
    # Prediction overshoots range and has a smoother (wider) distal edge.
    pred = _synthetic_idd(peak_mm=123.0, falloff_mm=9.0)
    mc_m = compute_range_metrics(mc, DZ)
    pred_m = compute_range_metrics(pred, DZ)
    d = range_metric_deltas(pred_m, mc_m)
    assert d["r80_delta_mm"] > 0  # prediction ranges deeper
    assert d["dfw_delta_mm"] > 0  # prediction fall-off is wider (smoother)


def test_integrated_depth_dose_shape():
    dose = np.ones((160, 30, 30))
    idd = integrated_depth_dose(dose)
    assert idd.shape == (160,)
    assert idd[0] == pytest.approx(900.0)


def test_idd_requires_3d():
    with pytest.raises(ValueError):
        integrated_depth_dose(np.ones((160, 30)))
