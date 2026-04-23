"""
Tests for Interface Severity Index (ISI).

Synthetic phantoms use the project's standard grid (160, 30, 30) at
2 mm isotropic spacing.  Tissue classes are induced by injecting
representative HU values:

    Air = -1000, Water/Soft = 0, Bone ≈ 1200 (cortical).
"""

import numpy as np
import pytest

from src.processing.interface_severity import (
    _build_sphere_mask,
    build_severity_matrix,
    compute_interface_severity_map,
    compute_isi_metrics,
    interface_severity,
)

from src.processing.tissue_decomposition import N_TISSUE_CLASSES, segment_tissue

RESOLUTION = (2.0, 2.0, 2.0)
SHAPE = (160, 30, 30)

HU_AIR = -1000.0
HU_WATER = 0.0
HU_BONE = 1200.0


BP_K, BP_Y, BP_X = 80, 15, 15


def _uniform_phantom(hu_value: float = HU_WATER):
    """Uniform HU phantom + uniform flux + 3-D Gaussian dose peak at (80, 15, 15).

    The dose peak is both axially and laterally localised so that
    ``interface_severity``'s ``argmax`` picks a well-defined BP voxel
    at the grid centre — inside any bone cube placed at [70:90, 10:20, 10:20].
    """
    ct = np.full(SHAPE, hu_value, dtype=np.float64)
    flux = np.ones(SHAPE, dtype=np.float64)
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    dose = np.exp(
        -0.5
        * (
            ((zz - BP_K) / 5.0) ** 2
            + ((yy - BP_Y) / 3.0) ** 2
            + ((xx - BP_X) / 3.0) ** 2
        )
    )
    return ct, flux, dose


# ─────────────────────────────────────────────────────────────────────────
# Severity matrix properties
# ─────────────────────────────────────────────────────────────────────────


class TestSeverityMatrix:
    def test_shape_and_symmetry(self):
        for mode in ["rsp_sq", "rsp_abs", "density_sq"]:
            W = build_severity_matrix(mode=mode)
            assert W.shape == (N_TISSUE_CLASSES, N_TISSUE_CLASSES)
            np.testing.assert_allclose(W, W.T, atol=1e-12)

    def test_zero_diagonal(self):
        for mode in ["rsp_sq", "rsp_abs", "density_sq"]:
            W = build_severity_matrix(mode=mode)
            np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)

    def test_nonneg(self):
        for mode in ["rsp_sq", "rsp_abs", "density_sq"]:
            W = build_severity_matrix(mode=mode)
            assert (W >= 0).all()

    def test_bone_air_dominates_soft_air(self):
        """Bone↔Air severity should exceed Soft-tissue↔Air severity."""
        W = build_severity_matrix(mode="rsp_sq")
        c_air = segment_tissue(np.array([HU_AIR]))[0]
        c_soft = segment_tissue(np.array([HU_WATER]))[0]
        c_bone = segment_tissue(np.array([HU_BONE]))[0]

        w_bone_air = W[c_bone, c_air]
        w_soft_air = W[c_soft, c_air]
        w_bone_soft = W[c_bone, c_soft]

        assert w_bone_air > w_soft_air, (
            f"expected bone-air ({w_bone_air:.3f}) > soft-air ({w_soft_air:.3f})"
        )
        assert w_bone_air > w_bone_soft, (
            f"expected bone-air ({w_bone_air:.3f}) > bone-soft ({w_bone_soft:.3f})"
        )

    def test_custom_mode_validates_shape(self):
        with pytest.raises(ValueError):
            build_severity_matrix(mode="custom", custom=np.zeros((3, 3)))

    def test_custom_mode_returns_supplied_matrix(self):
        custom = np.random.rand(N_TISSUE_CLASSES, N_TISSUE_CLASSES)
        W = build_severity_matrix(mode="custom", custom=custom)
        np.testing.assert_allclose(W, custom)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            build_severity_matrix(mode="bogus")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────
# Severity map properties (on raw class volumes)
# ─────────────────────────────────────────────────────────────────────────


class TestSeverityMap:
    def test_homogeneous_volume_zero_severity(self):
        """All-same-class volume → severity map is zero everywhere."""
        class_vol = np.full(SHAPE, 5, dtype=np.int64)  # any single class
        W = build_severity_matrix(mode="rsp_sq")
        sev = compute_interface_severity_map(class_vol, W)
        assert sev.sum() == 0.0

    def test_single_axial_interface_symmetric_assignment(self):
        """
        Half-bone / half-water along z: the interface at z_mid should
        show up on both neighbouring slices with half the weight each.
        """
        ct = np.full(SHAPE, HU_WATER)
        ct[80:, :, :] = HU_BONE
        class_vol = segment_tissue(ct)
        W = build_severity_matrix(mode="rsp_sq")

        sev = compute_interface_severity_map(class_vol, W, axial_only=True)

        c_water = class_vol[0, 0, 0]
        c_bone = class_vol[80, 0, 0]
        expected_total = W[c_water, c_bone] * SHAPE[1] * SHAPE[2]
        np.testing.assert_allclose(sev.sum(), expected_total, rtol=1e-10)

        # Both sides of the interface receive 0.5 * w
        assert np.all(sev[79] > 0)
        assert np.all(sev[80] > 0)
        # Everything else should be zero (pure homogeneous slabs)
        assert sev[:79].sum() == 0.0
        assert sev[81:].sum() == 0.0

    def test_axial_only_smaller_than_isotropic(self):
        """
        Bone cube inside water: isotropic counts 6 faces, axial-only
        counts only the 2 axial faces → isotropic > axial-only.
        """
        ct = np.full(SHAPE, HU_WATER)
        ct[70:90, 10:20, 10:20] = HU_BONE  # bone cube
        class_vol = segment_tissue(ct)
        W = build_severity_matrix(mode="rsp_sq")

        sev_iso = compute_interface_severity_map(class_vol, W, axial_only=False)
        sev_axial = compute_interface_severity_map(class_vol, W, axial_only=True)

        assert sev_iso.sum() > sev_axial.sum()
        # Axial-only is exactly the axial contribution of isotropic
        assert sev_axial.sum() < sev_iso.sum() * 0.5 + 1e-9

    def test_bone_in_air_exceeds_bone_in_water(self):
        """Bone-in-air interfaces should be far more severe than bone-in-water."""
        W = build_severity_matrix(mode="rsp_sq")

        ct_air = np.full(SHAPE, HU_AIR)
        ct_air[70:90, 10:20, 10:20] = HU_BONE
        sev_air = compute_interface_severity_map(segment_tissue(ct_air), W).sum()

        ct_water = np.full(SHAPE, HU_WATER)
        ct_water[70:90, 10:20, 10:20] = HU_BONE
        sev_water = compute_interface_severity_map(segment_tissue(ct_water), W).sum()

        assert sev_air > sev_water, (
            f"bone-in-air sev={sev_air:.2f} should exceed "
            f"bone-in-water sev={sev_water:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────
# ISI aggregation & sphere masking
# ─────────────────────────────────────────────────────────────────────────


class TestISIAggregation:
    def test_empty_mask_returns_zeros(self):
        sev = np.ones(SHAPE)
        mask = np.zeros(SHAPE, dtype=bool)
        out = compute_isi_metrics(sev, mask)
        assert out == {"isi_sum": 0.0, "isi_max": 0.0, "isi_mean": 0.0, "isi_axial_sum": 0.0}

    def test_flux_mask_excludes_low_flux_voxels(self):
        sev = np.ones(SHAPE)
        region = np.ones(SHAPE, dtype=bool)
        flux = np.zeros(SHAPE)
        flux[80, 15, 15] = 1.0  # single high-flux voxel
        out = compute_isi_metrics(sev, region, flux=flux, flux_threshold_frac=0.5)
        assert out["isi_sum"] == pytest.approx(1.0)
        assert out["isi_max"] == pytest.approx(1.0)
        assert out["isi_mean"] == pytest.approx(1.0)

    def test_axial_sum_passthrough(self):
        sev = np.ones(SHAPE)
        axial = np.full(SHAPE, 0.5)
        region = np.ones(SHAPE, dtype=bool)
        out = compute_isi_metrics(sev, region, axial_severity_map=axial)
        assert out["isi_axial_sum"] == pytest.approx(0.5 * np.prod(SHAPE))

    def test_sphere_mask_size(self):
        """Sphere at centre with radius 4 mm on 2 mm grid → ~33 voxels (4/3·π·2³)."""
        mask = _build_sphere_mask(SHAPE, (80, 15, 15), 4.0, RESOLUTION)
        n_expected = 4.0 / 3.0 * np.pi * (4.0 / 2.0) ** 3  # r/dz = 2 voxels
        assert 10 < mask.sum() < 60
        assert mask[80, 15, 15]
        # Outside-sphere voxels are False
        assert not mask[0, 0, 0]


# ─────────────────────────────────────────────────────────────────────────
# End-to-end wrapper
# ─────────────────────────────────────────────────────────────────────────


class TestInterfaceSeverityE2E:
    def test_uniform_water_all_zero(self):
        ct, flux, dose = _uniform_phantom(HU_WATER)
        out = interface_severity(ct, flux, dose, RESOLUTION)
        assert out["isi_sum"] == pytest.approx(0.0)
        assert out["isi_max"] == pytest.approx(0.0)
        assert out["isi_mean"] == pytest.approx(0.0)
        assert out["isi_axial_sum"] == pytest.approx(0.0)

    def test_bone_cube_in_water_positive(self):
        """
        Synthetic phantom: bone cube centred on the BP voxel,
        surrounded by water.  The cube is sized so all 6 faces fit
        inside the sphere at r=15 mm (~7.5 voxel radius).
        """
        ct, flux, dose = _uniform_phantom(HU_WATER)
        ct[77:83, 12:18, 12:18] = HU_BONE  # 6×6×6 cube at BP (all faces in-sphere)
        out = interface_severity(ct, flux, dose, RESOLUTION, sphere_radius_mm=15.0)
        assert out["isi_sum"] > 0
        assert out["isi_max"] > 0
        assert out["isi_axial_sum"] > 0
        # axial_sum counts only 2 of 6 faces, so must be < full sum
        assert out["isi_axial_sum"] < out["isi_sum"]

    def test_bone_in_air_vs_bone_in_water(self):
        """
        Classic "bone-air dominates": replace water background with
        air; expect a much higher ISI because Δρ(bone, air) ≫ Δρ(bone, water).
        """
        ct_w, flux, dose = _uniform_phantom(HU_WATER)
        ct_w[70:90, 10:20, 10:20] = HU_BONE
        out_w = interface_severity(ct_w, flux, dose, RESOLUTION, sphere_radius_mm=15.0)

        ct_a, _, _ = _uniform_phantom(HU_AIR)
        ct_a[70:90, 10:20, 10:20] = HU_BONE
        out_a = interface_severity(ct_a, flux, dose, RESOLUTION, sphere_radius_mm=15.0)

        assert out_a["isi_sum"] > out_w["isi_sum"], (
            f"bone-in-air ({out_a['isi_sum']:.2f}) should exceed "
            f"bone-in-water ({out_w['isi_sum']:.2f})"
        )
        # The Δρ ratio is ~2.3× (bone-air vs bone-water) → severity ratio
        # for rsp_sq is ~5×; require at least 2× to be conservative.
        assert out_a["isi_sum"] >= 2.0 * out_w["isi_sum"]

    def test_sphere_radius_sensitivity(self):
        """
        Larger sphere should capture at least as much severity as a
        smaller sphere centred at the same BP voxel (monotonicity).
        """
        ct, flux, dose = _uniform_phantom(HU_WATER)
        ct[70:90, 10:20, 10:20] = HU_BONE
        small = interface_severity(ct, flux, dose, RESOLUTION, sphere_radius_mm=6.0)
        large = interface_severity(ct, flux, dose, RESOLUTION, sphere_radius_mm=20.0)
        assert large["isi_sum"] >= small["isi_sum"]

    def test_severity_modes_agree_in_sign(self):
        """All three physics-based modes should rank bone-in-air > bone-in-water."""
        ct_w, flux, dose = _uniform_phantom(HU_WATER)
        ct_w[70:90, 10:20, 10:20] = HU_BONE
        ct_a, _, _ = _uniform_phantom(HU_AIR)
        ct_a[70:90, 10:20, 10:20] = HU_BONE
        for mode in ["rsp_sq", "rsp_abs", "density_sq"]:
            out_w = interface_severity(
                ct_w, flux, dose, RESOLUTION, severity_mode=mode, sphere_radius_mm=15.0
            )
            out_a = interface_severity(
                ct_a, flux, dose, RESOLUTION, severity_mode=mode, sphere_radius_mm=15.0
            )
            assert out_a["isi_sum"] > out_w["isi_sum"], f"mode={mode} fails"
