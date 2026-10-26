"""MCsquare beamlet dose units: Gy per MU, MCsquare's protons per MU, and the legacy factor."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.utils.unit_conversions import (
    EV_PER_G_TO_GY,
    legacy_to_gy_per_1e6_protons,
    mcsquare_protons_per_mu,
    mcsquare_to_gy_per_mu,
    mcsquare_to_gy_per_proton,
    to_gy,
)
from tests.utils.bdl import build_bdl_text


def write_bdl(path, header="--UPenn beam model (double gaussian)--"):
    path.write_text(build_bdl_text(preamble=header))
    return path


NOMINAL = [70.0, 100.0, 150.0]
PPM = [2.0e6, 3.0e6, 4.0e6]


def test_protons_per_mu_interpolates_and_extrapolates_like_mcsquare():
    assert mcsquare_protons_per_mu(100.0, NOMINAL, PPM) == pytest.approx(3.0e6)
    assert mcsquare_protons_per_mu(125.0, NOMINAL, PPM) == pytest.approx(3.5e6)
    assert mcsquare_protons_per_mu(160.0, NOMINAL, PPM) == pytest.approx(4.2e6)   # extrapolated, not clamped
    assert mcsquare_protons_per_mu(60.0, NOMINAL, PPM) == pytest.approx(2.0e6 - 10 * 1.0e6 / 30)


def test_gy_per_mu_and_per_proton(tmp_path):
    from src.beamlets.bdl import BeamDataLibrary

    bdl = BeamDataLibrary.from_file(write_bdl(tmp_path / "bdl.txt"))
    e = float(bdl.nominal_energy[1])
    dose = np.array([1.0e7, 2.0e7])
    ppm = mcsquare_protons_per_mu(e, bdl.nominal_energy, bdl.protons_mu)
    assert mcsquare_to_gy_per_mu(dose, e, bdl) == pytest.approx(dose * 1.602176e-16 * ppm)
    assert mcsquare_to_gy_per_proton(1.0) == pytest.approx(EV_PER_G_TO_GY) == pytest.approx(1.602176e-16)


def test_non_upenn_bdl_is_refused(tmp_path):
    path = write_bdl(tmp_path / "bdl.txt", header="--Grevillot beam model--")
    with pytest.raises(ValueError, match="not a UPenn BDL"):
        mcsquare_to_gy_per_mu(1.0, 100.0, path)


def test_legacy_factor_is_unchanged_and_to_gy_warns():
    assert legacy_to_gy_per_1e6_protons(1.0) == 1.60218e-10      # exactly the old to_gy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert to_gy(2.0) == legacy_to_gy_per_1e6_protons(2.0)
    assert caught and caught[0].category is DeprecationWarning
