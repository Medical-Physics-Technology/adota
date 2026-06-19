"""Flux equivalence on the 2x2x2 grid (field-resampling P2).

The flux at 2mm must be the **same physical beamlet** as at 1mm: identical optics
(sigma in mm), identical entrance/peak physical position, and -- because the flux
is the analytic Gaussian sampled on a coarser grid with the entrance + sigma scaled
by the same spacing -- the normalized 2mm flux is *exactly* the normalized 1mm flux
sampled at the coinciding voxels. (The difference vs the trilinear ``downsample(1mm)``
that the model trained on is a separate, model-level question -- not asserted here.)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.beamlets.flux import flux_projection, flux_projection_gpu

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _norm(a: np.ndarray) -> np.ndarray:
    return (a - a.min()) / (a.max() - a.min())


def _flux_pair(sigmas, direction, re1, shape1=(60, 60, 320)):
    """Flux for the same physical beamlet at 1mm and 2mm (entrance scaled by spacing)."""
    shape2 = tuple(s // 2 for s in shape1)
    re2 = [e / 2.0 for e in re1]
    f1 = flux_projection(re1, direction, sigmas, shape1, spacing=np.array([1.0, 1.0, 1.0]))
    f2 = flux_projection(re2, direction, sigmas, shape2, spacing=np.array([2.0, 2.0, 2.0]))
    return f1, f2


# (sigmas_mm, direction_deg, entrance_1mm) -- even entrance so the 2mm/1mm grids coincide.
_CASES = [
    ((3.5, 3.0), [0.0, 0.0], [30.0, 30.0, 0.0]),
    ((4.0, 4.0), [5.0, -3.0], [30.0, 30.0, 0.0]),
    ((2.8, 3.3), [-6.0, 4.0], [28.0, 32.0, 0.0]),
]


def _sigma_mm(flux: np.ndarray, spacing: float, axis: int, depth: int) -> float:
    """Lateral 1-sigma in mm from the 2nd moment of a fixed-depth slice along ``axis``."""
    sl = flux[:, :, depth]  # (z, y) lateral plane
    prof = sl.sum(axis=1 - axis)
    idx = np.arange(prof.size)
    centre = (prof * idx).sum() / prof.sum()
    var = (prof * (idx - centre) ** 2).sum() / prof.sum()
    return float(np.sqrt(var) * spacing)


@pytest.mark.parametrize("sigmas,direction,re1", _CASES)
def test_2mm_flux_is_subsample_of_1mm(sigmas, direction, re1) -> None:
    """Normalized 2mm flux == normalized 1mm flux at coinciding voxels (same beamlet)."""
    f1, f2 = _flux_pair(sigmas, direction, re1)
    assert f2.shape == tuple(s // 2 for s in f1.shape)
    sub = _norm(f1)[::2, ::2, ::2]
    np.testing.assert_allclose(_norm(f2), sub, atol=1e-9)


def test_2mm_optics_sigma_mm_invariant() -> None:
    """The lateral sigma in mm is identical on the 1mm and 2mm grids (optics preserved)."""
    sigmas = (3.5, 3.0)
    f1, f2 = _flux_pair(sigmas, [0.0, 0.0], [30.0, 30.0, 0.0])
    for axis in (0, 1):  # the two lateral axes (axis 2 is depth)
        s1 = _sigma_mm(f1, 1.0, axis, depth=10)
        s2 = _sigma_mm(f2, 2.0, axis, depth=5)
        assert abs(s1 - s2) < 0.02, (axis, s1, s2)


def test_2mm_peak_physical_position_invariant() -> None:
    """The flux peak maps to the same physical (z,y,x) mm on both grids."""
    f1, f2 = _flux_pair((3.5, 3.0), [4.0, -2.0], [30.0, 30.0, 0.0])
    p1 = np.array(np.unravel_index(int(f1.argmax()), f1.shape)) * 1.0
    p2 = np.array(np.unravel_index(int(f2.argmax()), f2.shape)) * 2.0
    np.testing.assert_allclose(p2, p1, atol=2.0)  # within one 2mm voxel


def test_2mm_normalized_integral_conserved() -> None:
    """Physical integral of the normalized flux (unit-peak Gaussian) is grid-invariant."""
    f1, f2 = _flux_pair((3.5, 3.0), [0.0, 0.0], [30.0, 30.0, 0.0])
    i1 = _norm(f1).sum() * 1.0**3
    i2 = _norm(f2).sum() * 2.0**3
    assert i2 == pytest.approx(i1, rel=2e-3)


def test_2mm_depth_extent_matches_1mm() -> None:
    f1, f2 = _flux_pair((3.5, 3.0), [0.0, 0.0], [30.0, 30.0, 0.0])
    assert f2.shape[2] * 2 == f1.shape[2] * 1  # 160*2 == 320*1 mm


def _lateral_centroid(flux: np.ndarray, depth: int) -> float:
    prof = flux[:, :, depth].sum(axis=1)
    idx = np.arange(prof.size)
    return float((prof * idx).sum() / prof.sum())


def test_cell_center_alignment_removes_shift_vs_downsample() -> None:
    """The cell-center 2mm flux is centroid-aligned with downsample(1mm); grid-point isn't.

    Guards the bug: ``align_corners=False`` (training/per-beamlet) samples the 2mm
    voxel at the 1mm cell-centre (``2j+0.5``), so the field-level 2mm grid must be
    offset by the same half voxel (done in ``expanded_reference_grid`` via the
    ``+(f-1)/2*spacing`` origin shift -> here the entrance shifts by ``-(f-1)/(2f)``).
    A naive grid-point entrance is shifted by +0.25 of a 2mm voxel (= +0.5mm).
    """
    sigmas, direction = (3.5, 3.0), [4.0, -2.0]
    f1 = flux_projection([30.0, 30.0, 0.0], direction, sigmas, (60, 60, 320),
                         spacing=np.array([1.0, 1.0, 1.0]))
    # Down-sample exactly as the loader does: permute (z,y,x)->(D,H,W) then resize.
    dm = F.interpolate(
        torch.tensor(f1).permute(2, 0, 1)[None, None], size=(160, 30, 30),
        mode="trilinear", align_corners=False,
    )[0, 0].numpy()
    down = np.transpose(dm, (1, 2, 0))  # (160,30,30) -> (30,30,160) = (z,y,x)

    grid_point = flux_projection([15.0, 15.0, 0.0], direction, sigmas, (30, 30, 160),
                                 spacing=np.array([2.0, 2.0, 2.0]))
    cell_center = flux_projection([14.75, 14.75, 0.0], direction, sigmas, (30, 30, 160),
                                  spacing=np.array([2.0, 2.0, 2.0]))

    c_down = _lateral_centroid(_norm(down), 5)
    assert abs(_lateral_centroid(_norm(grid_point), 5) - c_down) > 0.2   # bug present
    assert abs(_lateral_centroid(_norm(cell_center), 5) - c_down) < 0.05  # fixed


@pytest.mark.parametrize("sigmas,direction,re1", _CASES)
def test_gpu_2mm_matches_numpy_2mm(sigmas, direction, re1) -> None:
    """The GPU flux at 2mm is float32-identical to the NumPy flux at 2mm."""
    re2 = [e / 2.0 for e in re1]
    ref = flux_projection(re2, direction, sigmas, (30, 30, 160), spacing=np.array([2.0, 2.0, 2.0]))
    for dev in (["cuda"] if torch.cuda.is_available() else ["cpu"]):
        got = flux_projection_gpu(
            re2, direction, sigmas, (30, 30, 160),
            spacing=np.array([2.0, 2.0, 2.0]), device=dev,
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)
        assert np.array_equal(got.astype(np.float32), ref.astype(np.float32))
