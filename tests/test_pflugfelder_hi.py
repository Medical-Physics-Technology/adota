"""
Tests for Pflugfelder heterogeneity index.

Phantoms use the project's data shape: (160, 30, 30), 2 mm isotropic.
"""

import numpy as np
import pytest

from src.processing.pflugfelder_hi import (
    compute_pflugfelder_hi,
    compute_wepl_map,
    pflugfelder_hi,
)
from src.processing.rsp import hu_to_rsp

RESOLUTION = (2.0, 2.0, 2.0)
SHAPE = (160, 30, 30)


def _make_uniform_phantom(hu_value: float = 0.0):
    """Pure water (HU=0) phantom with uniform flux and dose peak at slice 80."""
    ct = np.full(SHAPE, hu_value, dtype=np.float64)
    flux = np.ones(SHAPE, dtype=np.float64)
    dose = np.zeros(SHAPE, dtype=np.float64)
    # Gaussian-ish IDD peak at slice 80
    for z in range(SHAPE[0]):
        dose[z] = np.exp(-0.5 * ((z - 80) / 5.0) ** 2)
    return ct, flux, dose


class TestWEPLMap:
    def test_water_wepl_equals_geometric_depth(self):
        """For water (RSP=1), WEPL should equal physical depth."""
        ct = np.zeros(SHAPE, dtype=np.float64)  # HU=0 → RSP≈1
        bp_depth_mm = 100.0  # 50 slices × 2 mm
        wepl = compute_wepl_map(ct, RESOLUTION, bp_depth_mm)

        assert wepl.shape == (30, 30)
        # RSP for HU=0 is 1.0, so WEPL = 50 slices × 2mm × 1.0 = 100 mm
        np.testing.assert_allclose(wepl, bp_depth_mm, atol=0.5)

    def test_dense_material_wepl_greater_than_water(self):
        """Bone (HU=1000) should give WEPL > physical depth."""
        ct = np.full(SHAPE, 1000.0, dtype=np.float64)
        bp_depth_mm = 100.0
        wepl = compute_wepl_map(ct, RESOLUTION, bp_depth_mm)

        assert np.all(wepl > bp_depth_mm)


class TestPflugfelderHI:
    def test_homogeneous_water_hi_zero(self):
        """Uniform water phantom: all rays identical → HI = 0."""
        ct, flux, dose = _make_uniform_phantom(hu_value=0.0)
        result = pflugfelder_hi(ct, flux, dose, RESOLUTION)

        assert result["hi"] == pytest.approx(0.0, abs=1e-10)
        assert result["wepl_mean"] > 0
        assert result["wepl_std"] == pytest.approx(0.0, abs=1e-10)

    def test_lateral_slab_hi_positive(self):
        """Half bone / half water laterally → HI > 0."""
        ct, flux, dose = _make_uniform_phantom(hu_value=0.0)
        # Left half = bone (HU=1000), right half = water (HU=0)
        ct[:, :, :15] = 1000.0

        result = pflugfelder_hi(ct, flux, dose, RESOLUTION)
        assert result["hi"] > 0.01, f"Expected HI > 0.01, got {result['hi']}"

    def test_axial_slab_hi_zero(self):
        """Bone slab spanning full lateral extent → still homogeneous laterally."""
        ct, flux, dose = _make_uniform_phantom(hu_value=0.0)
        # Bone slab at slices 30-40 across entire lateral extent
        ct[30:40, :, :] = 1000.0

        result = pflugfelder_hi(ct, flux, dose, RESOLUTION)
        # All rays pass through same slab → same WEPL → HI ≈ 0
        assert result["hi"] == pytest.approx(0.0, abs=1e-10)
        # But WEPL_mean should be elevated vs pure water
        water_ct, water_flux, water_dose = _make_uniform_phantom(0.0)
        water_result = pflugfelder_hi(water_ct, water_flux, water_dose, RESOLUTION)
        assert result["wepl_mean"] > water_result["wepl_mean"]

    def test_compute_pflugfelder_hi_empty_mask(self):
        """All-zero flux → returns zeros gracefully."""
        wepl = np.ones((30, 30))
        flux_2d = np.zeros((30, 30))
        result = compute_pflugfelder_hi(wepl, flux_2d)
        assert result["hi"] == 0.0


class TestRSPConsistency:
    def test_water_rsp_is_one(self):
        """HU=0 should give RSP ≈ 1.0."""
        rsp = hu_to_rsp(np.array([0.0]))
        np.testing.assert_allclose(rsp, 1.0, atol=0.02)

    def test_air_rsp_near_zero(self):
        """HU=-1000 should give RSP close to 0."""
        rsp = hu_to_rsp(np.array([-1000.0]))
        assert rsp[0] < 0.05
