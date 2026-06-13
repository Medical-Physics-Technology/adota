"""Tests for :mod:`src.beamlets.dose_scaling` (OpenTPS eV/g/proton -> Gy)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import SimpleITK as sitk

from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.dose_scaling import (
    delivered_protons,
    dose_to_gy_factor,
    load_dose_gy,
)
from src.loaders.plan_parser import ControlPoint, Field, Fraction, Plan, Spot

_EVG_TO_GY = 1.602176e-19 * 1000.0

_BDL = """\
    Nozzle exit to Isocenter distance
    400.0
    SMX to Isocenter distance
    2000.0
    SMY to Isocenter distance
    2500.0
    NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x SpotSize1y Divergence1y Correlation1y
    100.0 100.0 1.0 1000.0 1.0 4.0 0.003 0.5 3.0 0.004 0.6
    200.0 200.0 0.8 3000.0 1.0 3.5 0.003 0.4 2.8 0.004 0.5
"""


@pytest.fixture()
def bdl(tmp_path: Path) -> BeamDataLibrary:
    p = tmp_path / "bdl.txt"
    p.write_text(dedent(_BDL))
    return BeamDataLibrary.from_file(p)


def _cp(weights, energy: float) -> ControlPoint:
    return ControlPoint(
        index=0, spot_tuned_id=0, cumulative_msw=0.0, energy_mev=energy,
        range_shifter_setting="", iso_to_rs_distance=0.0, rs_wet=0.0,
        spots=[Spot(x=0.0, y=0.0, weight=float(w)) for w in weights],
    )


def _plan(n_fractions: int = 1) -> Plan:
    # Field with two control points: meterset 4 @100 MeV, meterset 6 @200 MeV.
    fld = Field(1, 0.0, 90.0, 0.0, (0.0, 0.0, 0.0), "", "", [_cp([1, 3], 100.0), _cp([2, 4], 200.0)])
    plan = Plan(name="t", total_msw=10.0, fractions=[Fraction(1, [1], [fld])])
    plan.n_treatment_fractions = n_fractions
    return plan


def test_delivered_protons(bdl: BeamDataLibrary) -> None:
    # meterset 4 @ ProtonsMU(100)=1000, meterset 6 @ ProtonsMU(200)=3000.
    assert delivered_protons(_plan(), bdl) == pytest.approx(4 * 1000.0 + 6 * 3000.0)


def test_gy_factor_uses_n_fractions(bdl: BeamDataLibrary) -> None:
    dp = 4 * 1000.0 + 6 * 3000.0
    assert dose_to_gy_factor(_plan(n_fractions=1), bdl) == pytest.approx(dp * _EVG_TO_GY)
    # n_treatment_fractions multiplies the factor.
    assert dose_to_gy_factor(_plan(n_fractions=5), bdl) == pytest.approx(dp * _EVG_TO_GY * 5)
    # explicit override wins.
    assert dose_to_gy_factor(_plan(n_fractions=5), bdl, n_fractions=2) == pytest.approx(
        dp * _EVG_TO_GY * 2
    )


def test_load_dose_gy_scales_and_preserves_geometry(bdl: BeamDataLibrary, tmp_path: Path) -> None:
    arr = np.full((3, 4, 5), 2.0, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin((-1.0, -2.0, 3.0))
    img.SetSpacing((1.0, 1.0, 2.0))
    dose_path = tmp_path / "Dose.mhd"
    sitk.WriteImage(img, str(dose_path))

    factor = dose_to_gy_factor(_plan(), bdl)
    gy = load_dose_gy(dose_path, _plan(), bdl)
    out = sitk.GetArrayFromImage(gy)
    np.testing.assert_allclose(out, 2.0 * factor, rtol=1e-6)
    assert gy.GetOrigin() == (-1.0, -2.0, 3.0)
    assert gy.GetSpacing() == (1.0, 1.0, 2.0)
