"""Orchestration tests for :mod:`src.beamlets.extraction`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.beamlets.extraction import ExtractionConfig, run_extraction
from src.loaders.plan_directory import PlanDirectory
from src.loaders.plan_parser import ControlPoint, Field, Fraction, Plan, Spot

ROI = (4, 4, 8)

_BDL = """\
Nozzle exit to Isocenter distance
400.0
SMX to Isocenter distance
2000.0
SMY to Isocenter distance
2500.0
NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x SpotSize1y Divergence1y Correlation1y
100.0 100.0 1.0 1000.0 1.0 4.0 0.003 0.5 3.0 0.004 0.6
150.0 150.0 0.8 1500.0 1.0 3.5 0.003 0.4 2.8 0.004 0.5
"""


def _cp(n_spots: int, energy: float) -> ControlPoint:
    return ControlPoint(
        index=0,
        spot_tuned_id=0,
        cumulative_msw=0.0,
        energy_mev=energy,
        range_shifter_setting="",
        iso_to_rs_distance=0.0,
        rs_wet=0.0,
        spots=[Spot(x=float(i), y=float(-i), weight=float(i + 1)) for i in range(n_spots)],
    )


def _plan() -> Plan:
    iso = (10.0, 10.0, 5.0)
    f1 = Field(1, 0.0, 90.0, 0.0, iso, "", "", [_cp(2, 100.0)])
    f2 = Field(2, 0.0, -90.0, 0.0, iso, "", "", [_cp(2, 150.0)])
    return Plan(name="synth", total_msw=10.0, fractions=[Fraction(1, [1, 2], [f1, f2])])


@pytest.fixture()
def plan_directory(tmp_path: Path) -> PlanDirectory:
    arr = np.random.default_rng(0).uniform(-200, 200, size=(10, 20, 20)).astype(np.float32)
    ct = sitk.GetImageFromArray(arr)
    ct.SetOrigin((-5.0, -5.0, 2.0))
    ct.SetSpacing((1.0, 1.0, 1.0))
    bdl_path = tmp_path / "bdl.txt"
    bdl_path.write_text(_BDL)
    return PlanDirectory(
        plan_dir=tmp_path,
        plan=_plan(),
        ct=ct,
        contours={},
        config={},
        bdl_path=bdl_path,
        bdl_text=_BDL,
        mc_dose_path=None,
    )


def test_extraction_writes_files_and_manifest(plan_directory, tmp_path: Path) -> None:
    out = tmp_path / "adota_beamlets"
    manifest = run_extraction(
        plan_directory, out, ExtractionConfig(roi_size=ROI, save_overlays=False)
    )

    assert manifest["n_spots"] == 4
    assert manifest["n_fields"] == 2
    assert (out / "manifest.json").is_file()

    for spot_id in manifest["spot_ids"]:
        ct_arr = np.load(out / f"{spot_id}_ct.npy")
        flux = np.load(out / f"{spot_id}_flux.npy")
        assert ct_arr.shape == ROI
        assert flux.shape == ROI
        meta = json.loads((out / f"{spot_id}_sim_res.json").read_text())
        for key in (
            "crp_numpy_ct",
            "relative_weight",
            "oob",
            "rays_entrence_point",
            "rays_entrence_point_proj",
            "image_origin",
        ):
            assert key in meta


def test_deterministic_ids_stable_across_reruns(plan_directory, tmp_path: Path) -> None:
    out = tmp_path / "adota_beamlets"
    m1 = run_extraction(plan_directory, out, ExtractionConfig(roi_size=ROI, save_overlays=False))
    ct_first = np.load(out / f"{m1['spot_ids'][0]}_ct.npy")
    m2 = run_extraction(
        plan_directory, out, ExtractionConfig(roi_size=ROI, overwrite=True, save_overlays=False)
    )
    assert m1["spot_ids"] == m2["spot_ids"]
    np.testing.assert_array_equal(ct_first, np.load(out / f"{m2['spot_ids'][0]}_ct.npy"))


def test_overwrite_guard(plan_directory, tmp_path: Path) -> None:
    out = tmp_path / "adota_beamlets"
    run_extraction(plan_directory, out, ExtractionConfig(roi_size=ROI, save_overlays=False))
    with pytest.raises(FileExistsError, match="not empty"):
        run_extraction(plan_directory, out, ExtractionConfig(roi_size=ROI, save_overlays=False))


def test_overwrite_clears_stale_files(plan_directory, tmp_path: Path) -> None:
    """overwrite=True wipes the dir so a stale spot from a prior run cannot survive."""
    out = tmp_path / "adota_beamlets"
    run_extraction(plan_directory, out, ExtractionConfig(roi_size=ROI, save_overlays=False))
    stale = out / "bZZ_l999_s9999_sim_res.json"
    stale.write_text("{}")
    assert stale.exists()

    run_extraction(
        plan_directory, out, ExtractionConfig(roi_size=ROI, overwrite=True, save_overlays=False)
    )
    assert not stale.exists()  # the stale file was removed
    assert (out / "manifest.json").is_file()  # the fresh run still wrote its outputs


def test_subset_n_spots_and_beams(plan_directory, tmp_path: Path) -> None:
    out = tmp_path / "adota_beamlets"
    manifest = run_extraction(
        plan_directory,
        out,
        ExtractionConfig(roi_size=ROI, n_spots=1, beams=[0], save_overlays=False),
    )
    assert manifest["n_spots"] == 1
    assert manifest["n_fields"] == 1


def test_overlay_png_written(plan_directory, tmp_path: Path) -> None:
    out = tmp_path / "adota_beamlets"
    run_extraction(plan_directory, out, ExtractionConfig(roi_size=ROI, save_overlays=True))
    pngs = list((out / "overlays").glob("*.png"))
    assert len(pngs) == 2  # one per field
