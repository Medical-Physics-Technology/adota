"""Unit + A/B parity tests for :mod:`src.beamlets.flux`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.flux import flux_projection, flux_spatial_spread
from tests.utils.bdl import build_bdl_text

_SYNTH_BDL = build_bdl_text()


@pytest.fixture()
def synth_bdl(tmp_path: Path) -> BeamDataLibrary:
    path = tmp_path / "bdl.txt"
    path.write_text(_SYNTH_BDL)
    return BeamDataLibrary.from_file(path)


def test_flux_spatial_spread_nearest_energy(synth_bdl: BeamDataLibrary) -> None:
    # 120 is closest to the 100 MeV row -> its spot sizes.
    assert flux_spatial_spread(synth_bdl, 120.0) == (4.0, 3.0)
    # 190 is closest to the 200 MeV row.
    assert flux_spatial_spread(synth_bdl, 190.0) == (3.0, 2.0)


def test_flux_projection_shape_and_positivity() -> None:
    shape = (10, 10, 20)
    flux = flux_projection(
        beamlet_entrence=(5.0, 5.0, 0.0),
        beamlet_direction=(0.0, 0.0),
        sigmas_xy=(3.0, 2.0),
        shape=shape,
    )
    assert flux.shape == shape
    assert np.all(flux >= 0.0)


def test_flux_projection_peak_on_entrance_zero_angle() -> None:
    """With zero angles the Gaussian centre sits at the entrance, constant in depth.

    Pins the flux placement (suspect S6): the lateral maximum must land on the
    entrance coordinate and not drift along the depth axis.
    """
    shape = (12, 14, 30)
    y0, x0 = 4.0, 9.0  # beamlet_entrence = (x_0, y_0, z_0) in the source's order
    flux = flux_projection(
        beamlet_entrence=(x0, y0, 0.0),
        beamlet_direction=(0.0, 0.0),
        sigmas_xy=(2.0, 2.0),
        shape=shape,
    )
    # flux peaks at (axis0 = y_0, axis1 = x_0); constant along depth (axis2).
    flat_idx = np.argmax(flux[:, :, 0])
    peak_axis0, peak_axis1 = np.unravel_index(flat_idx, flux[:, :, 0].shape)
    assert (peak_axis0, peak_axis1) == (round(y0), round(x0))
    # Same lateral pattern at every depth slice (zero angle => depth-invariant).
    np.testing.assert_allclose(flux[:, :, 0], flux[:, :, -1])


# --- A/B parity vs datagenerator -------------------------------------------


# --- Spot-size memoization ----------------------------------------------------
# ``flux_spatial_spread`` used to run a pandas argsort over the whole energy table
# once per spot. It is now memoized on the BDL instance. The memo must be
# invisible: same answers, no leakage between libraries, and no effect on the
# table it reads.

def test_spot_size_cache_starts_empty_and_fills_on_use(synth_bdl: BeamDataLibrary) -> None:
    assert synth_bdl.spot_size_cache == {}
    first = flux_spatial_spread(synth_bdl, 120.0)
    assert synth_bdl.spot_size_cache == {120.0: first}
    # A second distinct energy adds an entry rather than replacing one.
    flux_spatial_spread(synth_bdl, 190.0)
    assert set(synth_bdl.spot_size_cache) == {120.0, 190.0}


def test_cached_result_equals_the_uncached_computation(synth_bdl: BeamDataLibrary) -> None:
    """A hit returns exactly what a miss computes, across the whole table."""
    energies = [60.0, 99.9, 120.0, 120.5, 155.0, 190.0, 250.0]
    uncached = []
    for e in energies:
        synth_bdl.spot_size_cache.clear()  # force a miss every time
        uncached.append(flux_spatial_spread(synth_bdl, e))
    synth_bdl.spot_size_cache.clear()
    warm = [flux_spatial_spread(synth_bdl, e) for e in energies]   # populate
    hits = [flux_spatial_spread(synth_bdl, e) for e in energies]   # all hits
    assert warm == uncached
    assert hits == uncached


def test_cache_is_per_instance_and_does_not_leak(tmp_path, synth_bdl: BeamDataLibrary) -> None:
    """Two libraries with different tables must not share memoized sigmas."""
    other = BeamDataLibrary(
        nozzle_isocenter=synth_bdl.nozzle_isocenter,
        smx=synth_bdl.smx,
        smy=synth_bdl.smy,
        energy_table=synth_bdl.energy_table.assign(
            SpotSize1x=synth_bdl.energy_table["SpotSize1x"] * 2.0,
            SpotSize1y=synth_bdl.energy_table["SpotSize1y"] * 3.0,
        ),
        source_path=tmp_path / "other_bdl.txt",
    )
    mine = flux_spatial_spread(synth_bdl, 120.0)
    theirs = flux_spatial_spread(other, 120.0)
    assert theirs == (mine[0] * 2.0, mine[1] * 3.0)
    assert synth_bdl.spot_size_cache != other.spot_size_cache


def test_cache_does_not_participate_in_equality_or_repr(synth_bdl: BeamDataLibrary) -> None:
    """The memo is an implementation detail, not part of the library's value."""
    before = repr(synth_bdl)
    flux_spatial_spread(synth_bdl, 120.0)
    assert synth_bdl.spot_size_cache  # populated
    assert repr(synth_bdl) == before  # but invisible


def test_flux_projection_is_unchanged_by_a_warm_cache(synth_bdl: BeamDataLibrary) -> None:
    """End result: the flux array is byte-identical cold vs warm."""
    synth_bdl.spot_size_cache.clear()
    cold = flux_projection([6.0, 6.0, 0.0], [2.0, -1.0],
                           flux_spatial_spread(synth_bdl, 120.0), (12, 12, 24))
    warm = flux_projection([6.0, 6.0, 0.0], [2.0, -1.0],
                           flux_spatial_spread(synth_bdl, 120.0), (12, 12, 24))
    assert np.array_equal(cold, warm)


def test_parity_flux_spatial_spread(synth_bdl: BeamDataLibrary, datagenerator_utils) -> None:
    theirs = datagenerator_utils.flux_spatial_spread(synth_bdl.energy_table, 120.0)
    ours = flux_spatial_spread(synth_bdl, 120.0)
    assert ours == pytest.approx(theirs)


def test_parity_flux_projection(datagenerator_utils) -> None:
    rng = np.random.default_rng(11)
    shape = (8, 9, 15)
    for _ in range(5):
        entrance = rng.uniform(0, 8, size=3)
        direction = rng.uniform(-5, 5, size=2)
        sigmas = rng.uniform(1.5, 4.0, size=2)
        ours = flux_projection(entrance, direction, sigmas, shape)
        theirs = datagenerator_utils.flux_projection(entrance, direction, sigmas, shape)
        np.testing.assert_allclose(ours, theirs, rtol=1e-10, atol=1e-12)
