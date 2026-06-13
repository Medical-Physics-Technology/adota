"""Unit + A/B parity tests for :mod:`src.beamlets.flux`."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.flux import flux_projection, flux_spatial_spread

_SYNTH_BDL = """\
    Nozzle exit to Isocenter distance
    400.0
    SMX to Isocenter distance
    2000.0
    SMY to Isocenter distance
    2500.0
    NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x SpotSize1y Divergence1y Correlation1y
    100.0 100.0 1.0 1000.0 1.0 4.0 0.003 0.5 3.0 0.004 0.6
    200.0 200.0 0.5 2000.0 1.0 3.0 0.002 0.3 2.0 0.003 0.4
"""


@pytest.fixture()
def synth_bdl(tmp_path: Path) -> BeamDataLibrary:
    path = tmp_path / "bdl.txt"
    path.write_text(dedent(_SYNTH_BDL))
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
