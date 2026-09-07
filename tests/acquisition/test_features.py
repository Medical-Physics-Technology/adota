"""The metric orchestrator returns the full feature vector on either dose."""
from __future__ import annotations

import numpy as np

from src.acquisition.features import FEATURE_NAMES, FeatureConfig, compute_features
from src.acquisition.surrogate import analytic_dose


def _phantom():
    D, H, W = 160, 30, 30
    ct = np.zeros((D, H, W))
    # Heterogeneity placed where a 150 MeV beam peaks (about 150 mm, z ~ 75), so
    # the 5 mm spheres and the Bragg-peak zone actually contain it.
    ct[60:100, :, :15] = 900.0                      # a bone wedge under half the beam
    ct[60:100, :, 15:] = -700.0                     # lung under the other half
    yy, xx = np.mgrid[0:H, 0:W]
    flux = np.repeat(np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / 18.0)[None], D, axis=0)
    return ct, flux


def test_all_thirty_features_are_finite_on_the_analytic_dose():
    ct, flux = _phantom()
    dose = analytic_dose(ct, flux, 150.0, 2.0)
    out = compute_features(ct, flux, 150.0, dose)
    missing = [n for n in FEATURE_NAMES if n not in out]
    assert not missing, missing
    assert all(np.isfinite(out[n]) for n in FEATURE_NAMES)
    assert out["energy_mev"] == 150.0 and out["ct_max_hu"] == 900.0
    assert 0 < out["bp_range_min_mm"] < out["bp_range_max_mm"] <= 320.0
    assert out["n_density_regions"] >= 1


def test_lateral_heterogeneity_raises_the_edge_and_wepl_spread_metrics():
    ct, flux = _phantom()
    water = np.zeros_like(ct)
    hetero = compute_features(ct, flux, 150.0, analytic_dose(ct, flux, 150.0, 2.0))
    homo = compute_features(water, flux, 150.0, analytic_dose(water, flux, 150.0, 2.0))
    assert hetero["sum_sobel_bp"] > homo["sum_sobel_bp"] == 0.0
    assert hetero["wepl_std"] > homo["wepl_std"]
    assert hetero["isi_sum"] > homo["isi_sum"]
    # Bone under one half and lung under the other average to soft tissue in the
    # flux-weighted 1D profile, so the along-beam density metrics rightly see
    # nothing here; that is what the axial phantom below is for.
    assert hetero["total_hu_change"] == 0.0


def test_axial_heterogeneity_raises_the_along_beam_density_metrics():
    D, H, W = 160, 30, 30
    ct = np.zeros((D, H, W))
    ct[55:65] = 900.0                               # full-width bone slab
    ct[65:80] = -700.0                              # then lung, both proximal of the peak
    yy, xx = np.mgrid[0:H, 0:W]
    flux = np.repeat(np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / 18.0)[None], D, axis=0)
    water = np.zeros_like(ct)
    hetero = compute_features(ct, flux, 150.0, analytic_dose(ct, flux, 150.0, 2.0))
    homo = compute_features(water, flux, 150.0, analytic_dose(water, flux, 150.0, 2.0))
    assert hetero["n_density_regions"] > homo["n_density_regions"]
    assert hetero["total_hu_change"] > homo["total_hu_change"] == 0.0
    assert hetero["max_hu_jump"] > 500.0             # the lung-to-water step inside the zone


def test_default_config_is_the_reference_run():
    cfg = FeatureConfig()
    assert cfg.region_method == "sphere" and cfg.sphere_radius_mm == 5.0 and cfg.sobel_use_raw
    assert cfg.resolution_mm == (2.0, 2.0, 2.0)
