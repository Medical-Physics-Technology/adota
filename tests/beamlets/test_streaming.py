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
