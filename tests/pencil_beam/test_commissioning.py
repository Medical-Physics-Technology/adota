"""Commissioning fits on synthetic beamlets with known parameters."""
from __future__ import annotations

import numpy as np
import pytest

from src.pencil_beam.commissioning import (
    CommissioningConfig,
    DepthProfile,
    commissioning_phantom,
    depth_profile,
    distal_r80,
    fit_energy_node,
    fit_primary_sigma,
    kernel_halo,
    physical_to_wet,
)


def _plane(sigma, n=61, halo_sigma=None, halo_weight=0.0):
    z, y = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
    r2 = (z - n / 2) ** 2 + (y - n / 2) ** 2
    out = (1 - halo_weight) / sigma ** 2 * np.exp(-r2 / (2 * sigma ** 2))
    if halo_weight:
        st2 = sigma ** 2 + halo_sigma ** 2
        out += halo_weight / st2 * np.exp(-r2 / (2 * st2))
    return out


def test_depth_profile_recovers_sigma_and_idd():
    dose = np.stack([_plane(s) for s in (3.0, 4.0, 5.0)], axis=-1)       # (z, y, x)
    prof = depth_profile(dose, weight=2.0)
    assert prof.sigma == pytest.approx([3.0, 4.0, 5.0], abs=0.02)
    assert prof.idd == pytest.approx(dose.sum(axis=(0, 1)) / 2.0)


def test_primary_sigma_fit_inverts_the_kernel_halo():
    halo_sigma, halo_weight = 6.0, 0.1
    plane = _plane(4.0, n=101, halo_sigma=halo_sigma, halo_weight=halo_weight)
    assert fit_primary_sigma(plane, halo_sigma, halo_weight) == pytest.approx(4.0, abs=0.01)
    single = depth_profile(plane[..., None]).sigma[0]
    assert single > 4.05                         # the single-Gaussian width the kernel would widen again


def test_kernel_halo_is_bounded():
    s, w = kernel_halo(np.linspace(0, 150, 50), 106.0)
    assert np.all(s >= 0) and np.all((w >= 0) & (w <= 0.9))


def test_distal_r80_and_wet():
    wet = np.arange(200, dtype=float)
    idd = np.where(wet <= 100, 1.0, np.clip(1 - (wet - 100) / 10, 0, 1))
    assert distal_r80(wet, idd) == pytest.approx(102.0, abs=0.01)
    assert physical_to_wet(np.array([10.0, 30.0]), 20, 1.02, 0.001) == pytest.approx([0.01, 0.02 + 10.2])


def test_fit_energy_node_reproduces_kernel_width():
    """sigma_air(d) + sigma_mcs(z) at the kernel's d reproduces the synthetic width at any SSD."""
    cfg = CommissioningConfig(energies=[100.0], surface_to_iso_mm=[100.0, 175.0, 250.0, 325.0])
    coef = np.array([1e-6, -2e-3, 6.0])
    x = np.arange(300) + 0.5
    wet = physical_to_wet(x, cfg.air_front_mm, 1.0, 0.0)
    idd = np.where(wet < 80, 1.0, np.clip(1 - (wet - 80) / 2, 0, 1))
    src = 2300.0

    def truth(s):
        # in water the kernel's d = dist - z_w + 0.7 R80 is constant along the axis
        return np.polyval(coef, src - s + 0.7 * 80.0) + 0.01 * wet

    profiles = [DepthProfile(x, idd, truth(s), truth(s), x * 0, x * 0) for s in cfg.surface_to_iso_mm]
    node = fit_energy_node(100.0, profiles, [wet] * 4, cfg.surface_to_iso_mm, cfg.air_front_mm, src, cfg)
    assert node["r80"] == pytest.approx(80.0, abs=0.6)
    for s in (100.0, 210.0, 325.0):                        # 210 is held out of the fit
        d = src - s + 0.7 * node["r80"]
        z = np.array([10.0, 40.0, 70.0])
        model = np.polyval(node["sigma_air_coef"], d) + np.interp(z, node["depths"], node["sigmas"])
        expect = np.polyval(coef, src - s + 0.7 * 80.0) + 0.01 * z
        assert model == pytest.approx(expect, abs=0.01)


def test_phantom_guards_mcsquare_nozzle_transport():
    cfg = CommissioningConfig()
    ct, iso = commissioning_phantom(cfg, 100.0)
    assert ct.GetSize() == (cfg.air_front_mm + cfg.water_length_mm, cfg.lateral_size_mm, cfg.lateral_size_mm)
    assert iso[0] == cfg.air_front_mm + 100.0
    for bad in (0.0, 50.0, 450.0):            # nozzle > 300 mm upstream, or inside the grid
        with pytest.raises(ValueError):
            commissioning_phantom(cfg, bad)
