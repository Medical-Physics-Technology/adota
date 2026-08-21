"""Fast (no-MC) tests for the robustness orchestration logic."""
import numpy as np
import pytest

from src.mc_generation.robustness import (
    RobustnessConfig,
    build_angle_grid,
    resolve_gantry,
)


def test_angle_grid_shape_indices_and_values():
    grid = build_angle_grid((-2.0, 2.0), (-2.0, 2.0), 18)
    assert len(grid) == 18 * 18
    ixs = {g[0] for g in grid}
    iys = {g[1] for g in grid}
    assert ixs == set(range(18)) and iys == set(range(18))
    # endpoints of the sweep are exactly the requested range
    txs = sorted({g[2] for g in grid})
    assert txs[0] == pytest.approx(-2.0) and txs[-1] == pytest.approx(2.0)
    # row-major over theta_x: first 18 entries share ix=0 and span all theta_y
    assert all(g[0] == 0 for g in grid[:18])
    assert sorted(g[3] for g in grid[:18]) == pytest.approx(list(np.linspace(-2, 2, 18)))


def test_gantry_fixed():
    cfg = RobustnessConfig(gantry_mode="fixed", gantry_value=90.0)
    assert resolve_gantry(cfg, "any/uid") == 90.0


def test_gantry_bimodal_is_seeded_and_in_range():
    cfg = RobustnessConfig(gantry_mode="bimodal_random",
                           gantry_ranges=((30.0, 120.0), (240.0, 330.0)), gantry_seed=1234)
    # reproducible per patient uid
    a = resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")
    b = resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")
    assert a == b
    # different patients generally differ
    c = resolve_gantry(cfg, "NSCLC/LUNG1-006/9.9.9")
    assert a != c
    # every draw falls in one of the two bimodal lobes
    for uid in [f"p/{i}" for i in range(50)]:
        g = resolve_gantry(cfg, uid)
        assert (30.0 <= g <= 120.0) or (240.0 <= g <= 330.0)


def test_gantry_uniform_random_is_seeded_and_in_range():
    cfg = RobustnessConfig(gantry_mode="uniform_random", gantry_min=0.0, gantry_max=360.0,
                           gantry_seed=1234)
    a = resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")
    assert a == resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")   # reproducible
    assert a != resolve_gantry(cfg, "NSCLC/LUNG1-006/9.9.9")   # patient-specific
    draws = [resolve_gantry(cfg, f"p/{i}") for i in range(200)]
    assert all(0.0 <= g < 360.0 for g in draws)
    assert max(draws) - min(draws) > 180.0                     # actually spread out


def test_gantry_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_gantry(RobustnessConfig(gantry_mode="spiral"), "x")


def test_robustness_config_from_dict_defaults_and_overrides():
    from src.mc_generation.robustness import robustness_config_from_dict
    r = {"energies": [80.0, 120.0, 140.0], "gantry_mode": "uniform_random",
         "gantry_max": 360.0, "grid_n": 18, "num_primaries": 1e6,
         "experiment_prefix": "patient_set"}
    cfg = robustness_config_from_dict(r, grid_n=3, num_primaries=5e5)
    assert cfg.energies == [80.0, 120.0, 140.0]
    assert cfg.gantry_mode == "uniform_random"
    assert cfg.grid_n == 3                       # CLI override wins over yaml
    assert cfg.num_primaries == 5e5              # CLI override wins
    assert cfg.experiment_prefix == "patient_set"
    assert cfg.rotate_to_canonical is True       # default


def test_ct_rotation_preserves_isocenter_at_grid_center():
    """Rotate+expand about the grid-centre isocenter must keep it the new centre."""
    import SimpleITK as sitk
    from src.beamlets.rotation import rotate_ct_around_isocenter
    from src.mc_generation.geometry import extraction_isocenter_physical

    arr = np.zeros((40, 60, 80), dtype=np.float32)  # (z, y, x); non-cubic on purpose
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    iso0 = extraction_isocenter_physical(img)
    rot = rotate_ct_around_isocenter(img, -(210.0 - 90.0), iso0, expand=True)
    iso1 = extraction_isocenter_physical(rot)
    # grid grows (no clipping) but the isocenter stays put to sub-voxel accuracy
    assert rot.GetSize() != img.GetSize()
    assert np.allclose(np.asarray(iso0), np.asarray(iso1), atol=1.0)
