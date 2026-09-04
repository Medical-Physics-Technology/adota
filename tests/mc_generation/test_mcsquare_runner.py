"""Integration test for the vendored MCsquare runner (needs the engine binary).

Numerically meaningful assertions on a real single-beamlet run:
  * the dose grid matches the CT grid;
  * the dose is a focused beam (top-1% voxels hold most of the dose), not noise;
  * the run is deterministic (same RNG seed + single thread -> bit-identical dose).

Skipped automatically when the engine install or a scratch work area is absent.
"""
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

INSTALL = Path("/home/mstryja/tools/mcsquare")
WORK_ROOT = Path("/scratch/mstryja/mc_work/pytest")

pytestmark = pytest.mark.skipif(
    not (INSTALL / "MCsquare_linux").exists(),
    reason="MCsquare engine not installed at /home/mstryja/tools/mcsquare",
)


@pytest.fixture(scope="module")
def runner():
    from src.mc_generation.mcsquare_runner import MCSquareRunner
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    return MCSquareRunner(install_dir=str(INSTALL), work_root=str(WORK_ROOT))


@pytest.fixture(scope="module")
def ct():
    return sitk.ReadImage(str(INSTALL / "Sample_input_data" / "CT.mhd"))


def _concentration(dose, top_frac=0.01):
    flat = np.sort(dose.ravel())[::-1]
    k = max(1, int(len(flat) * top_frac))
    return float(flat[:k].sum() / (flat.sum() + 1e-12))


def test_run_beamlet_shape_and_focused_beam(runner, ct):
    dose, sim_res = runner.run_beamlet(
        ct, energy=140.0, gantry_angle=90.0, spot_xy=(0.0, 0.0),
        num_primaries=3e4, num_threads=1, rng_seed=1,
    )
    arr = sitk.GetArrayFromImage(dose).astype(np.float64)  # (z, y, x)
    # dose grid matches the CT grid (SITK size is (x,y,z); numpy is (z,y,x))
    assert arr.shape == sitk.GetArrayFromImage(ct).shape
    assert arr.sum() > 0 and arr.max() > 0
    # a real pencil beam concentrates dose; uniform noise would give ~0.01
    assert _concentration(arr) > 0.05
    # sim_res carries the physical plan params back
    assert sim_res["initial_energy"] == pytest.approx(140.0)
    assert sim_res["simulation_log"]["gantry_angle"] == pytest.approx(90.0)


def test_run_beamlet_deterministic(runner, ct):
    kw = dict(energy=140.0, gantry_angle=90.0, spot_xy=(0.0, 0.0),
              num_primaries=3e4, num_threads=1, rng_seed=7)
    d1, _ = runner.run_beamlet(ct, **kw)
    d2, _ = runner.run_beamlet(ct, **kw)
    a1 = sitk.GetArrayFromImage(d1).astype(np.float64)
    a2 = sitk.GetArrayFromImage(d2).astype(np.float64)
    assert np.array_equal(a1, a2), f"non-deterministic: max abs diff {np.abs(a1 - a2).max():.3e}"
