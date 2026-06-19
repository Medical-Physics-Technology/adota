"""Streaming pipeline equivalence: fused == staged dose, no beamlets written.

`run_streaming_pipeline` fuses extract+infer+accumulate in memory; it must produce
the **same** ``Dose_ADoTA`` as the staged `run_extraction -> run_inference ->
run_accumulation` path (it reuses the same crop/flux, the shared
`prepare_input_from_arrays` / `postprocess_prediction`, and the same `deposit_crop`
+ de-rotation), and must write **no** per-beamlet files. (Real-plan equivalence
with the production model is checked by `scripts/verify_streaming_equivalence.py`.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch
import torch.nn as nn

from src.beamlets.accumulation import AccumulationConfig, run_accumulation
from src.beamlets.extraction import ExtractionConfig, run_extraction
from src.beamlets.inference import InferenceConfig, run_inference
from src.beamlets.streaming import StreamingConfig, run_streaming_pipeline
from src.loaders.plan_directory import PlanDirectory
from src.loaders.plan_parser import ControlPoint, Field, Fraction, Plan, Spot

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


class _TinyModel(nn.Module):
    """Deterministic stand-in: maps the 2-ch input to a (B,1,160,30,30) dose."""

    def forward(self, x, e):  # noqa: D401 - matches DoTA (returns (dose, attention))
        dose = (x[:, 0:1] * 0.3 + x[:, 1:2] * 0.2).clamp(0.0, 1.0)
        return dose, None


def _cp(n_spots: int, energy: float) -> ControlPoint:
    return ControlPoint(
        index=0, spot_tuned_id=0, cumulative_msw=0.0, energy_mev=energy,
        range_shifter_setting="", iso_to_rs_distance=0.0, rs_wet=0.0,
        spots=[Spot(x=float(i), y=float(-i), weight=float(i + 1)) for i in range(n_spots)],
    )


def _plan() -> Plan:
    iso = (16.0, 28.0, 20.0)
    f1 = Field(1, 0.0, 90.0, 0.0, iso, "", "", [_cp(5, 100.0)])
    f2 = Field(2, 0.0, -90.0, 0.0, iso, "", "", [_cp(5, 150.0)])
    return Plan(name="synth", total_msw=10.0, fractions=[Fraction(1, [1, 2], [f1, f2])])


@pytest.fixture()
def plan_directory(tmp_path: Path) -> PlanDirectory:
    arr = np.random.default_rng(0).uniform(-200, 200, size=(40, 56, 56)).astype(np.float32)
    ct = sitk.GetImageFromArray(arr)
    ct.SetOrigin((-5.0, -5.0, 2.0))
    ct.SetSpacing((1.0, 1.0, 1.0))
    bdl_path = tmp_path / "bdl.txt"
    bdl_path.write_text(_BDL)
    return PlanDirectory(
        plan_dir=tmp_path, plan=_plan(), ct=ct, contours={}, config={},
        bdl_path=bdl_path, bdl_text=_BDL, mc_dose_path=None,
    )


def test_streaming_matches_staged_dose(plan_directory, tmp_path: Path) -> None:
    model = _TinyModel().eval()
    device = torch.device("cpu")

    # Staged: extract (CPU flux) -> infer -> accumulate.
    beamlets = tmp_path / "adota_beamlets"
    run_extraction(
        plan_directory, beamlets,
        ExtractionConfig(save_overlays=False, flux_on_gpu=False),
    )
    run_inference(beamlets, model, device, InferenceConfig(batch_size=4))
    staged_dose = tmp_path / "Dose_staged.mhd"
    run_accumulation(
        plan_directory, beamlets, staged_dose,
        AccumulationConfig(dose_source="prediction"),
    )

    # Streaming: fused, no disk.
    streamed_dose = tmp_path / "Dose_streamed.mhd"
    run_streaming_pipeline(
        plan_directory, model, device, streamed_dose,
        StreamingConfig(batch_size=4, flux_on_gpu=False),
    )

    a = sitk.GetArrayFromImage(sitk.ReadImage(str(staged_dose)))
    b = sitk.GetArrayFromImage(sitk.ReadImage(str(streamed_dose)))
    assert a.shape == b.shape == sitk.GetArrayFromImage(plan_directory.ct).shape
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6)
    assert float(b.max()) > 0.0  # dose was actually deposited


def test_streaming_writes_no_beamlets(plan_directory, tmp_path: Path) -> None:
    model = _TinyModel().eval()
    out = tmp_path / "Dose_ADoTA.mhd"
    summary = run_streaming_pipeline(
        plan_directory, model, torch.device("cpu"), out,
        StreamingConfig(batch_size=4, flux_on_gpu=False),
    )
    assert out.is_file()
    assert summary["n_spots"] == 10 and summary["n_fields"] == 2
    assert not (tmp_path / "adota_beamlets").exists()  # nothing persisted


def test_streaming_calibration_scales(plan_directory, tmp_path: Path) -> None:
    model = _TinyModel().eval()
    base = run_streaming_pipeline(
        plan_directory, model, torch.device("cpu"), tmp_path / "d0.mhd",
        StreamingConfig(batch_size=8, flux_on_gpu=False),
    )
    cal = run_streaming_pipeline(
        plan_directory, model, torch.device("cpu"), tmp_path / "d1.mhd",
        StreamingConfig(batch_size=8, flux_on_gpu=False, calibration_factor=1.05),
    )
    assert cal["dose_sum"] == pytest.approx(base["dose_sum"] * 1.05, rel=1e-4)


# --- Field-level 2mm resampling (grid_factor=2) -------------------------------

def _dose_centroid(D: np.ndarray) -> np.ndarray:
    idx = np.indices(D.shape)
    w = D / D.sum()
    return np.array([(idx[k] * w).sum() for k in range(3)])


def test_streaming_grid_factor2_runs_and_labels(plan_directory, tmp_path: Path) -> None:
    """grid_factor=2 streams end-to-end: same grid, dose deposited, labelled, no disk."""
    model = _TinyModel().eval()
    out = tmp_path / "Dose_ADoTA_2mm.mhd"
    summary = run_streaming_pipeline(
        plan_directory, model, torch.device("cpu"), out,
        StreamingConfig(batch_size=4, flux_on_gpu=False, grid_factor=2),
    )
    D = sitk.GetArrayFromImage(sitk.ReadImage(str(out)))
    assert D.shape == sitk.GetArrayFromImage(plan_directory.ct).shape  # on the CT grid
    assert float(D.max()) > 0.0
    assert summary["grid_factor"] == 2 and summary["grid_mode"] == "2mm_field"
    assert summary["n_spots"] == 10
    assert not (tmp_path / "adota_beamlets").exists()  # still disk-free


def test_grid_factor2_preserves_plan_dose_physically(plan_directory, tmp_path: Path) -> None:
    """The 2mm-field plan dose matches the 1mm per-beamlet dose on the CT grid.

    This is the physically meaningful check (test (5)/(4)): both doses are de-rotated
    onto the same CT grid, so a genuine spatial shift would show as a centroid shift
    -- not the crop-window index offset that confounds an array-index comparison.
    Thresholds are loose because the synthetic CT is pure noise; the real-model gate
    is P6. (1mm path is the reference; gf=1 stays byte-identical, guarded above.)
    """
    model = _TinyModel().eval()
    cfg = dict(batch_size=8, flux_on_gpu=False)
    run_streaming_pipeline(plan_directory, model, torch.device("cpu"),
                           tmp_path / "d1.mhd", StreamingConfig(grid_factor=1, **cfg))
    run_streaming_pipeline(plan_directory, model, torch.device("cpu"),
                           tmp_path / "d2.mhd", StreamingConfig(grid_factor=2, **cfg))
    D1 = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "d1.mhd")))
    D2 = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "d2.mhd")))

    m = D1 > 0.1 * D1.max()
    assert float(np.corrcoef(D1[m], D2[m])[0, 1]) > 0.90       # shape preserved
    shift = np.abs(_dose_centroid(D2) - _dose_centroid(D1))    # voxels (= mm here)
    assert shift.max() < 0.5, shift                            # no physical shift
    assert D2.sum() / D1.sum() == pytest.approx(1.0, abs=0.05)  # dose conserved


def test_grid_factor2_single_spot_centroid(plan_directory, tmp_path: Path) -> None:
    """A single beamlet lands at the same physical centroid at 2mm and 1mm."""
    model = _TinyModel().eval()
    cfg = dict(batch_size=1, flux_on_gpu=False, beams=[1], n_spots=1)
    run_streaming_pipeline(plan_directory, model, torch.device("cpu"),
                           tmp_path / "s1.mhd", StreamingConfig(grid_factor=1, **cfg))
    run_streaming_pipeline(plan_directory, model, torch.device("cpu"),
                           tmp_path / "s2.mhd", StreamingConfig(grid_factor=2, **cfg))
    D1 = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "s1.mhd")))
    D2 = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "s2.mhd")))
    assert float(D1.max()) > 0.0 and float(D2.max()) > 0.0
    shift = np.abs(_dose_centroid(D2) - _dose_centroid(D1))
    assert shift.max() < 0.6, shift  # sub-voxel; single spot is noisier than the plan
