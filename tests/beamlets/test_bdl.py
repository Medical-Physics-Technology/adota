"""Unit + A/B parity tests for :mod:`src.beamlets.bdl`."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from src.beamlets.bdl import (
    BeamDataLibrary,
    angles_to_spot_position,
    spot_position_to_angles,
)

PLAN_BDL = Path(
    "/scratch/mstryja/opentps_plans/"
    "Prostate-AEC-001_Publication_Prostate-AEC-001_100MperBeam/bdl.txt"
)

# A tiny two-energy synthetic BDL with the OpenTPS header layout.
_SYNTH_BDL = """\
    --synthetic beam model--

    Nozzle exit to Isocenter distance
    400.0

    SMX to Isocenter distance
    2000.0

    SMY to Isocenter distance
    2500.0

    Beam parameters
    2 energies

    NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x SpotSize1y Divergence1y Correlation1y
    100.0 100.0 1.0 1000.0 1.0 4.0 0.003 0.5 3.0 0.004 0.6
    200.0 200.0 0.5 2000.0 1.0 3.0 0.002 0.3 2.0 0.003 0.4
"""


@pytest.fixture()
def synth_bdl(tmp_path: Path) -> BeamDataLibrary:
    path = tmp_path / "bdl.txt"
    path.write_text(dedent(_SYNTH_BDL))
    return BeamDataLibrary.from_file(path)


def test_from_file_parses_geometry_and_table(synth_bdl: BeamDataLibrary) -> None:
    assert synth_bdl.nozzle_isocenter == 400.0
    assert synth_bdl.smx == 2000.0
    assert synth_bdl.smy == 2500.0
    assert synth_bdl.distances == (400.0, 2000.0, 2500.0)
    assert list(synth_bdl.nominal_energy) == [100.0, 200.0]
    assert "SpotSize1x" in synth_bdl.energy_table.columns


def test_compute_mu_to_protons_interpolates(synth_bdl: BeamDataLibrary) -> None:
    # Exact at a tabulated energy, linear in between.
    assert synth_bdl.compute_mu_to_protons(100.0) == 1000.0
    assert synth_bdl.compute_mu_to_protons(200.0) == 2000.0
    assert synth_bdl.compute_mu_to_protons(150.0) == pytest.approx(1500.0)


def test_spot_sizes_interpolate_and_extrapolate(synth_bdl: BeamDataLibrary) -> None:
    assert synth_bdl.spot_sizes(100.0) == pytest.approx((4.0, 3.0))
    assert synth_bdl.spot_sizes(150.0) == pytest.approx((3.5, 2.5))
    # interp1d with fill_value='extrapolate' continues the trend past the table.
    sx, sy = synth_bdl.spot_sizes(250.0)
    assert sx == pytest.approx(2.5)
    assert sy == pytest.approx(1.5)


def test_divergences_and_correlations(synth_bdl: BeamDataLibrary) -> None:
    assert synth_bdl.divergences(100.0) == pytest.approx((0.003, 0.004))
    assert synth_bdl.correlations(200.0) == pytest.approx((0.3, 0.4))


def test_angle_round_trip() -> None:
    d_smx, d_smy = 2014.9, 2584.1
    y, z = 12.3, -7.8
    theta_y, theta_z = spot_position_to_angles(y, z, d_smx, d_smy)
    y2, z2 = angles_to_spot_position(theta_y, theta_z, d_smx, d_smy)
    assert (y2, z2) == pytest.approx((y, z))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Beam data library not found"):
        BeamDataLibrary.from_file(tmp_path / "nope.txt")


# --- A/B parity vs datagenerator -------------------------------------------


@pytest.mark.skipif(not PLAN_BDL.is_file(), reason="real plan bdl.txt not available")
def test_parity_geometry_with_datagenerator(datagenerator_utils) -> None:
    """Distances match datagenerator's bdl_extract_geo on the real bdl.txt."""
    bdl = BeamDataLibrary.from_file(PLAN_BDL)
    _, (d_nozzle, d_smx, d_smy) = datagenerator_utils.bdl_extract_geo(str(PLAN_BDL))
    assert bdl.distances == pytest.approx((d_nozzle, d_smx, d_smy))


@pytest.mark.skipif(not PLAN_BDL.is_file(), reason="real plan bdl.txt not available")
def test_parity_angles_with_datagenerator(datagenerator_utils) -> None:
    """Angle conversion matches datagenerator fed the same BDL geometry."""
    bdl = BeamDataLibrary.from_file(PLAN_BDL)
    rng = np.random.default_rng(3)
    for _ in range(10):
        y, z = rng.uniform(-50, 50, size=2)
        ours = spot_position_to_angles(y, z, bdl.d_smx, bdl.d_smy)
        theirs = datagenerator_utils.spot_position_to_angles(y, z, str(PLAN_BDL))
        assert ours == pytest.approx(theirs)
