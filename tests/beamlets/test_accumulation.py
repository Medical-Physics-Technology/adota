"""Tests for :mod:`src.beamlets.accumulation`.

Covers the deposit placement/clipping, the decisive crop -> deposit inverse
round trip (deposit lands exactly where the crop came from), and an end-to-end
extract -> accumulate on a synthetic plan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.beamlets.accumulation import (
    AccumulationConfig,
    accumulate_dose,
    deposit_crop,
    run_accumulation,
)
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.extraction import ExtractionConfig, run_extraction
from src.loaders.plan_directory import PlanDirectory
from src.loaders.plan_parser import ControlPoint, Field, Fraction, Plan, Spot

ROI = (4, 4, 8)
DISTANCES = (420.0, 2014.9, 2584.1)

_BDL = """\
Nozzle exit to Isocenter distance
420.0
SMX to Isocenter distance
2014.9
SMY to Isocenter distance
2584.1
NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x SpotSize1y Divergence1y Correlation1y
100.0 100.0 1.0 1000.0 1.0 4.0 0.003 0.5 3.0 0.004 0.6
150.0 150.0 0.8 1500.0 1.0 3.5 0.003 0.4 2.8 0.004 0.5
"""


def test_deposit_crop_placement() -> None:
    grid = np.zeros((10, 12, 20), dtype=np.float32)
    crop = np.ones(ROI, dtype=np.float32)
    deposit_crop(grid, crop, crp=(5, 6, 9), weight=2.0, roi_size=ROI)
    # Window z 3:7, y 4:8, x 0:8 holds 2.0; everything else is 0.
    assert np.all(grid[3:7, 4:8, 0:8] == 2.0)
    assert grid.sum() == pytest.approx(2.0 * 4 * 4 * 8)


def test_deposit_crop_edge_clipping() -> None:
    grid = np.zeros((10, 12, 20), dtype=np.float32)
    crop = np.arange(np.prod(ROI), dtype=np.float32).reshape(ROI)
    # crp near the z=0 / y=0 corner: window starts at -1 in z and y.
    deposit_crop(grid, crop, crp=(1, 1, 4), weight=1.0, roi_size=ROI)
    # Only the in-bounds part (crop[1:4, 1:4, :]) lands at grid[0:3, 0:3, :].
    np.testing.assert_array_equal(grid[0:3, 0:3, 0:8], crop[1:4, 1:4, 0:8])
    assert grid[3:, :, :].sum() == 0.0


def test_deposit_is_exact_inverse_of_crop() -> None:
    """Depositing an extracted crop reproduces the source window exactly (no rotation)."""
    rng = np.random.default_rng(0)
    arr = rng.uniform(-500, 500, size=(12, 16, 24)).astype(np.float32)
    image = sitk.GetImageFromArray(arr)
    image.SetOrigin((-4.0, -8.0, 2.0))
    image.SetSpacing((1.0, 1.0, 1.0))
    iso = image.TransformContinuousIndexToPhysicalPoint([8.0, 8.0, 6.0])

    crop, _entrance, crp, oob = extract_beamlet_roi(image, *DISTANCES, (0.0, 0.0), iso, ROI)
    assert not oob

    grid = np.zeros_like(arr)
    deposit_crop(grid, crop, crp, weight=1.0, roi_size=ROI)

    iz, iy, _ix = crp
    np.testing.assert_array_equal(
        grid[iz - 2 : iz + 2, iy - 2 : iy + 2, 0:8], arr[iz - 2 : iz + 2, iy - 2 : iy + 2, 0:8]
    )


def _cp(n_spots: int, energy: float) -> ControlPoint:
    return ControlPoint(
        index=0, spot_tuned_id=0, cumulative_msw=0.0, energy_mev=energy,
        range_shifter_setting="", iso_to_rs_distance=0.0, rs_wet=0.0,
        spots=[Spot(x=float(i), y=float(-i), weight=float(i + 1)) for i in range(n_spots)],
    )


def _plan() -> Plan:
    iso = (10.0, 10.0, 5.0)
    f1 = Field(1, 0.0, 90.0, 0.0, iso, "", "", [_cp(2, 100.0)])
    f2 = Field(2, 0.0, -90.0, 0.0, iso, "", "", [_cp(2, 150.0)])
    return Plan(name="synth", total_msw=10.0, fractions=[Fraction(1, [1, 2], [f1, f2])])


@pytest.fixture()
def plan_directory(tmp_path: Path) -> PlanDirectory:
    arr = np.random.default_rng(1).uniform(-200, 200, size=(10, 20, 20)).astype(np.float32)
    ct = sitk.GetImageFromArray(arr)
    ct.SetOrigin((-5.0, -5.0, 2.0))
    ct.SetSpacing((1.0, 1.0, 1.0))
    bdl_path = tmp_path / "bdl.txt"
    bdl_path.write_text(_BDL)
    return PlanDirectory(
        plan_dir=tmp_path, plan=_plan(), ct=ct, contours={}, config={},
        bdl_path=bdl_path, bdl_text=_BDL, mc_dose_path=None,
    )


def test_accumulate_end_to_end(plan_directory, tmp_path: Path) -> None:
    """Extract flux then accumulate; the dose image matches the CT grid, is non-negative."""
    beamlets = tmp_path / "adota_beamlets"
    run_extraction(plan_directory, beamlets, ExtractionConfig(roi_size=ROI, save_overlays=False))

    dose_image, summary = accumulate_dose(
        plan_directory, beamlets, AccumulationConfig(dose_source="flux")
    )
    assert dose_image.GetSize() == plan_directory.ct.GetSize()
    assert dose_image.GetSpacing() == plan_directory.ct.GetSpacing()
    assert dose_image.GetOrigin() == plan_directory.ct.GetOrigin()
    assert summary["n_spots"] == 4
    assert summary["n_fields"] == 2

    arr = sitk.GetArrayFromImage(dose_image)
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) > 0.0  # flux was deposited somewhere


def test_run_accumulation_writes_dose_adota(plan_directory, tmp_path: Path) -> None:
    beamlets = tmp_path / "adota_beamlets"
    run_extraction(plan_directory, beamlets, ExtractionConfig(roi_size=ROI, save_overlays=False))

    out = tmp_path / "Dose_ADoTA.mhd"
    summary = run_accumulation(plan_directory, beamlets, out)
    assert out.is_file()
    assert summary["output_path"] == str(out)
    # The written image carries the CT geometry.
    written = sitk.ReadImage(str(out))
    assert written.GetSize() == plan_directory.ct.GetSize()


def test_refuses_to_overwrite_mc_dose(plan_directory, tmp_path: Path) -> None:
    beamlets = tmp_path / "adota_beamlets"
    run_extraction(plan_directory, beamlets, ExtractionConfig(roi_size=ROI, save_overlays=False))
    with pytest.raises(ValueError, match="read-only"):
        run_accumulation(plan_directory, beamlets, tmp_path / "Dose.mhd")


def test_prediction_source_moveaxis_and_deposit(plan_directory, tmp_path: Path) -> None:
    """The prediction path squeezes + moveaxis (1,1,320,60,60) -> (z,y,x) and deposits."""
    beamlets = tmp_path / "adota_beamlets"
    run_extraction(plan_directory, beamlets, ExtractionConfig(roi_size=ROI, save_overlays=False))

    # Write a model-shaped prediction per spot: (1,1,D,H,W) = (1,1,8,4,4) for ROI.
    h, w, d = ROI
    for meta in beamlets.glob("*_sim_res.json"):
        sid = meta.name[: -len("_sim_res.json")]
        pred = np.ones((1, 1, d, h, w), dtype=np.float32)  # depth-first
        np.save(beamlets / f"{sid}_ds_pred.npy", pred)

    dose_image, summary = accumulate_dose(
        plan_directory, beamlets, AccumulationConfig(dose_source="prediction")
    )
    assert summary["dose_source"] == "prediction"
    arr = sitk.GetArrayFromImage(dose_image)
    assert float(arr.max()) > 0.0  # the (moveaxis'd) prediction was deposited
    assert dose_image.GetSize() == plan_directory.ct.GetSize()


def test_accumulate_oblique_field_outputs_original_size(tmp_path: Path) -> None:
    """An oblique field grows the expanded grid; accumulation returns to CT size.

    Phase 3: extraction rotates into an expanded grid (here genuinely larger),
    and accumulation rebuilds that grid, deposits, and de-rotates back into the
    original CT grid so the dose is exactly the CT size.
    """
    import json

    arr = np.random.default_rng(2).uniform(-200, 200, size=(10, 20, 20)).astype(np.float32)
    ct = sitk.GetImageFromArray(arr)
    ct.SetOrigin((-5.0, -5.0, 2.0))
    ct.SetSpacing((1.0, 1.0, 1.0))
    bdl_path = tmp_path / "bdl.txt"
    bdl_path.write_text(_BDL)

    # Single oblique field: gantry 60 -> adjusted angle 30 deg, off-centre iso.
    iso = (15.0, 10.0, 5.0)
    fld = Field(1, 0.0, 60.0, 0.0, iso, "", "", [_cp(2, 100.0)])
    plan = Plan("obl", 10.0, [Fraction(1, [1], [fld])])
    pd = PlanDirectory(
        plan_dir=tmp_path, plan=plan, ct=ct, contours={}, config={},
        bdl_path=bdl_path, bdl_text=_BDL, mc_dose_path=None,
    )

    beamlets = tmp_path / "adota_beamlets"
    run_extraction(pd, beamlets, ExtractionConfig(roi_size=ROI, save_overlays=False))

    # The expanded rotated grid stored per spot is larger than the CT.
    meta = json.loads(next(beamlets.glob("*_sim_res.json")).read_text())
    assert (
        meta["image_size"][0] > ct.GetSize()[0]
        or meta["image_size"][1] > ct.GetSize()[1]
    )

    dose_image, _ = accumulate_dose(pd, beamlets, AccumulationConfig(dose_source="flux"))
    assert dose_image.GetSize() == ct.GetSize()  # back to the original CT size
    assert float(sitk.GetArrayFromImage(dose_image).max()) > 0.0
