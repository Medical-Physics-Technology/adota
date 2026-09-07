"""The Bortfeld analytic Bragg curve behaves like a Bragg curve."""
from __future__ import annotations

import numpy as np
import pytest

from src.acquisition.bragg_curve import bragg_curve, peak_depth_mm, straggling_sigma_cm
from src.processing.range_energy import energy_to_range_mm


@pytest.mark.parametrize("energy_mev", [70.0, 100.0, 150.0, 200.0, 220.0])
def test_distal_80_percent_depth_is_the_input_range(energy_mev):
    """Bortfeld's R0 is the distal 80 percent depth, so feeding it the Grevillot
    R80 must give a curve whose distal 80 percent point lands back on R80."""
    r80 = energy_to_range_mm(energy_mev)
    z = np.arange(0.0, r80 + 40.0, 0.05)
    d = bragg_curve(z, r80)
    peak = z[np.argmax(d)]
    distal_80 = z[(z > peak) & (d < 0.8 * d.max())][0]
    assert abs(distal_80 - r80) < 0.3, (energy_mev, distal_80, r80)
    assert peak < r80                                # the maximum sits proximal of R80
    assert np.isfinite(d).all() and (d >= 0).all()


def test_zero_beyond_the_straggling_tail():
    r80 = 150.0
    tail_end = r80 + 5.0 * 10.0 * straggling_sigma_cm(r80 / 10.0)
    assert bragg_curve(np.array([tail_end + 1.0, tail_end + 50.0]), r80).max() == 0.0


def test_entrance_to_peak_ratio_is_physical():
    """A clinical pristine peak is 1.5 to 4 times the entrance dose."""
    r80 = energy_to_range_mm(150.0)
    z = np.arange(0.0, r80 + 20.0, 0.1)
    d = bragg_curve(z, r80)
    assert 1.5 < d.max() / d[0] < 4.5


def test_straggling_grows_with_range():
    assert straggling_sigma_cm(5.0) < straggling_sigma_cm(15.0) < straggling_sigma_cm(25.0)


def test_peak_depth_helper_matches_dense_scan():
    r80 = energy_to_range_mm(120.0)
    z = np.arange(0.0, r80 + 20.0, 0.1)
    assert abs(peak_depth_mm(r80) - z[np.argmax(bragg_curve(z, r80))]) < 0.15


def test_zero_range_gives_zero_dose():
    assert bragg_curve(np.linspace(0, 10, 5), 0.0).max() == 0.0
