"""Path features of the engine-error drivers: air before the skin and low-density path before the peak."""
from __future__ import annotations

import numpy as np

from src.engine_drivers.geometry import MIN_RUN, SKIN_HU, path_lengths

AIR = -1000.0
TISSUE = 0.0
LUNG = -800.0


def _profile(*runs):
    """Concatenate (value, count) runs into a beam-axis HU profile."""
    return np.concatenate([np.full(n, v, dtype=float) for v, n in runs])


def test_defaults_are_documented_values():
    assert SKIN_HU == -500.0
    assert MIN_RUN == 5


def test_thin_shell_in_air_is_not_the_skin():
    hu = _profile((AIR, 10), (TISSUE, 2), (AIR, 3), (TISSUE, 20))
    air, low = path_lengths(hu, peak_slice=len(hu))
    assert air == 15 * 2.0
    # the 3 air slices between the shell and the skin lie before the skin, so they are not lung
    assert low == 0.0


def test_single_slice_shell_is_not_the_skin():
    hu = _profile((AIR, 4), (TISSUE, 1), (AIR, 4), (TISSUE, 10))
    air, _ = path_lengths(hu, peak_slice=len(hu))
    assert air == 9 * 2.0


def test_air_gap_scales_with_dz():
    hu = _profile((AIR, 7), (TISSUE, 10))
    air, _ = path_lengths(hu, peak_slice=len(hu), dz_mm=3.0)
    assert air == 21.0


def test_run_of_exactly_min_run_is_the_skin():
    hu = _profile((AIR, 6), (TISSUE, MIN_RUN), (AIR, 4))
    air, _ = path_lengths(hu, peak_slice=len(hu))
    assert air == 12.0


def test_lowdensity_between_skin_and_peak_counted_and_after_peak_ignored():
    hu = _profile((AIR, 4), (TISSUE, 5), (LUNG, 3), (TISSUE, 5), (LUNG, 4), (TISSUE, 5))
    peak = 17   # first slice after the second tissue block
    air, low = path_lengths(hu, peak_slice=peak)
    assert air == 8.0
    assert low == 3 * 2.0
    # moving the peak past the second lung block picks it up
    _, low_all = path_lengths(hu, peak_slice=len(hu))
    assert low_all == 7 * 2.0


def test_threshold_slice_counts_as_low_density():
    hu = _profile((AIR, 2), (TISSUE, 5), (SKIN_HU, 2), (TISSUE, 5))
    _, low = path_lengths(hu, peak_slice=len(hu))
    assert low == 2 * 2.0


def test_peak_before_skin_gives_no_lowdensity():
    hu = _profile((AIR, 10), (TISSUE, 5), (LUNG, 3), (TISSUE, 5))
    air, low = path_lengths(hu, peak_slice=3)
    assert air == 20.0
    assert low == 0.0


def test_peak_beyond_profile_is_clipped():
    hu = _profile((AIR, 2), (TISSUE, 5), (LUNG, 2))
    _, low = path_lengths(hu, peak_slice=10_000)
    assert low == 4.0


def test_no_dense_run_means_all_air_and_no_lowdensity():
    hu = _profile((AIR, 10), (TISSUE, 3), (AIR, 7))
    air, low = path_lengths(hu, peak_slice=15)
    assert air == len(hu) * 2.0
    assert low == 0.0


def test_all_air_profile():
    hu = np.full(12, AIR)
    assert path_lengths(hu, peak_slice=6, dz_mm=2.5) == (30.0, 0.0)


def test_min_run_one_is_first_slice_above_threshold():
    hu = _profile((AIR, 10), (TISSUE, 2), (AIR, 3), (TISSUE, 20))
    air, low = path_lengths(hu, peak_slice=len(hu), min_run=1)
    assert air == 10 * 2.0
    assert low == 3 * 2.0


def test_custom_skin_threshold():
    hu = _profile((AIR, 3), (-300.0, 6), (TISSUE, 5))
    air_default, _ = path_lengths(hu, peak_slice=len(hu))
    air_strict, _ = path_lengths(hu, peak_slice=len(hu), skin_hu=-100.0)
    assert air_default == 6.0
    assert air_strict == 18.0
