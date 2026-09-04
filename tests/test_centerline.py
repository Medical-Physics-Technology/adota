"""Unit tests for the beam-centerline geometry (src.beamlets.centerline)."""
import numpy as np
import pytest

from src.beamlets.centerline import (
    BeamLine,
    beam_line_from_metadata,
    render_centerline,
)


def test_beam_line_from_metadata_conventions():
    # entrance [depth, e1, e2] in ROI voxels; downsample 2 -> record frame.
    # axis0 <- e2/ds + origin, axis1 <- e1/ds + origin, origin = -(ds-1)/(2 ds).
    line = beam_line_from_metadata([0.0, 30.0, 40.0], [1.0, -2.0], downsample=2.0)
    assert line.a0 == pytest.approx(40.0 / 2 - 0.25)
    assert line.a1 == pytest.approx(30.0 / 2 - 0.25)
    assert line.b0 == pytest.approx(np.tan(np.deg2rad(1.0)))
    assert line.b1 == pytest.approx(-np.tan(np.deg2rad(-2.0)))


def test_lateral_center_is_linear():
    line = BeamLine(a0=10.0, a1=12.0, b0=0.02, b1=-0.03)
    c0, c1 = line.lateral_center(np.array([0, 100]))
    assert c0 == pytest.approx([10.0, 12.0])
    assert c1 == pytest.approx([12.0, 9.0])


def test_render_binary_one_voxel_per_slice():
    line = BeamLine(a0=15.0, a1=15.0, b0=0.0, b1=0.0)
    vol = render_centerline(line, (30, 30, 20), mode="binary")
    assert vol.shape == (30, 30, 20)
    assert vol.dtype == np.float32
    # exactly one lit voxel per depth slice, at (15, 15)
    for d in range(20):
        assert vol[:, :, d].sum() == 1.0
        assert vol[15, 15, d] == 1.0


def test_render_soft_peak_on_axis_and_range():
    line = BeamLine(a0=15.0, a1=15.0, b0=0.0, b1=0.0)
    vol = render_centerline(line, (30, 30, 10), mode="soft", sigma=1.5)
    assert vol.shape == (30, 30, 10)
    # peak ~1 on the axis, decays away, stays in [0, 1]
    assert vol[15, 15, 0] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= vol.min() and vol.max() <= 1.0
    assert vol[15, 15, 0] > vol[17, 15, 0] > vol[19, 15, 0]


def test_render_soft_follows_drift():
    # a beam drifting +0.5 vox/slice in axis0 should peak at a0 + 0.5 d
    line = BeamLine(a0=5.0, a1=15.0, b0=0.5, b1=0.0)
    vol = render_centerline(line, (30, 30, 8), mode="soft", sigma=1.0)
    for d in range(8):
        peak_row = np.argmax(vol[:, 15, d])
        assert peak_row == pytest.approx(round(5.0 + 0.5 * d), abs=1)


def test_degenerate_straight_beam_is_vertical():
    line = beam_line_from_metadata([0.0, 30.0, 30.0], [0.0, 0.0], downsample=2.0)
    assert line.b0 == 0.0 and line.b1 == 0.0
    vol = render_centerline(line, (30, 30, 50), mode="binary")
    centers = [np.unravel_index(np.argmax(vol[:, :, d]), (30, 30)) for d in range(50)]
    assert len(set(centers)) == 1  # same lateral voxel at every depth


def test_render_invalid_mode_raises():
    with pytest.raises(ValueError):
        render_centerline(BeamLine(1, 1, 0, 0), (5, 5, 5), mode="nope")


def test_render_out_of_bounds_line_is_empty_binary():
    # a line outside the lateral grid should light no voxels (no crash)
    line = BeamLine(a0=100.0, a1=100.0, b0=0.0, b1=0.0)
    vol = render_centerline(line, (30, 30, 5), mode="binary")
    assert vol.sum() == 0.0
