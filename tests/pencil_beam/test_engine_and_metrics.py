"""MC-frame -> DoseCUDA geometry, agreement metrics, and (on a GPU) the real kernels."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.pencil_beam.comparison import _region
from src.pencil_beam.engine import mc_frame_to_dosecuda
from src.pencil_beam.metrics import criteria_from_config, dose_agreement, dose_bbox

KERNELS = Path(os.environ.get("ADOTA_DOSECUDA_KERNELS", "/home/mstryja/tools/dosecuda/current/dose_kernels.so"))


def _ct(size=(200, 80, 60), origin=(-100.0, 20.0, 5.0)):
    img = sitk.GetImageFromArray(np.zeros(size[::-1], dtype=np.float32))
    img.SetOrigin(origin)
    return img


def test_mc_isocenter_maps_to_world_and_gantry_270():
    iso_world, gantry = mc_frame_to_dosecuda(_ct(), [100.0, 40.0, 30.0])
    assert iso_world == pytest.approx([-100.0 - 0.5 + 100.0, 20.0 - 0.5 + 40.0, 5.0 - 0.5 + 30.0])
    assert gantry == 270.0
    with pytest.raises(ValueError):
        mc_frame_to_dosecuda(_ct(), [0, 0, 0], mc_gantry=0.0)


def test_criteria_and_bbox_and_agreement():
    assert [c.key for c in criteria_from_config(None)][0] == "g3_3_10"
    assert criteria_from_config([{"dose": 2, "dist": 2, "cutoff": 5}])[0].key == "g2_2_5"
    vol = np.zeros((20, 20, 50))
    vol[8:12, 9:11, 5:30] = 1.0
    box = dose_bbox([vol], margin=2)
    assert box == (slice(6, 14), slice(7, 13), slice(3, 32))
    agree = dose_agreement(vol, vol)
    assert agree["r80_delta_mm"] == 0.0 and agree["mean_abs_diff_pct"] == 0.0


def test_comparison_window_is_roi_plus_margin():
    region = _region((300, 400, 500), (150, 200, 0), (60, 60, 320), 30)
    assert region == (slice(90, 210), slice(140, 260), slice(0, 500))
    clipped = _region((100, 100, 500), (10, 95, 0), (60, 60, 320), 30)
    assert clipped == (slice(0, 70), slice(35, 100), slice(0, 500))


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.skipif(not KERNELS.is_file(), reason=(
    f"DoseCUDA kernels not built at {KERNELS}; run scripts/pencil_beam/provision_dosecuda.sh "
    "or set ADOTA_DOSECUDA_KERNELS"))
def test_off_axis_spot_lands_where_mcsquare_puts_it(tmp_path):
    """MC spot (X, Y) -> image (y, z) offset (-X, +Y) at the isocenter plane (beamlet_ray, g=-90)."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from src.pencil_beam.engine import PencilBeamEngine
    from src.pencil_beam.rlsp import write_rlsp_csv
    from tests.pencil_beam.test_rlsp_and_model import _model

    model = _model()
    model.write(tmp_path)
    write_rlsp_csv(tmp_path / "HU_RLSP.csv", np.array([-1000.0, 0.0]), np.array([0.001, 1.0]))
    ct = _ct(size=(200, 100, 100), origin=(0.0, 0.0, 0.0))
    iso = [100.0, 50.0, 50.0]
    eng = PencilBeamEngine(KERNELS, tmp_path)
    upstream = iso[0] - 40.5                                  # plane x = 40 (inside the 77 mm range)
    for spot, axis, sign, vsad in (((20.0, 0.0), 1, -1, model.vsadx), ((0.0, 20.0), 0, +1, model.vsady)):
        dose = eng.dose(ct, 100.0, [spot], iso).dose
        plane = dose[:, :, 40]
        centre = (np.indices(plane.shape)[axis] * plane).sum() / plane.sum() + 0.5
        assert centre - 50.0 == pytest.approx(sign * 20.0 * (1 - upstream / vsad), abs=0.1)
