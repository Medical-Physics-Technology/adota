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
