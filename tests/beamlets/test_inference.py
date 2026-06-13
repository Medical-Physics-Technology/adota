"""Tests for :mod:`src.beamlets.inference` (plumbing, batch-invariance, discovery)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from src.beamlets.extraction import ExtractionConfig, run_extraction
from src.beamlets.inference import InferenceConfig, discover_spot_ids, run_inference
from src.loaders.plan_directory import PlanDirectory
from src.loaders.plan_parser import ControlPoint, Field, Fraction, Plan, Spot

ROI = (4, 4, 8)

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


class _StubModel(torch.nn.Module):
    """Deterministic, parameter-free stand-in: (x, e) -> ((B,1,160,30,30), None)."""

    def forward(self, x: torch.Tensor, e: torch.Tensor):
        out = x.mean(dim=1, keepdim=True) + e.view(-1, 1, 1, 1, 1)
        return out, None


def _cp(n_spots: int, energy: float) -> ControlPoint:
    return ControlPoint(
        index=0, spot_tuned_id=0, cumulative_msw=0.0, energy_mev=energy,
        range_shifter_setting="", iso_to_rs_distance=0.0, rs_wet=0.0,
        spots=[Spot(x=float(i), y=float(-i), weight=float(i + 1)) for i in range(n_spots)],
    )


def _plan() -> Plan:
    iso = (10.0, 10.0, 5.0)
    f1 = Field(1, 0.0, 90.0, 0.0, iso, "", "", [_cp(3, 100.0)])
    f2 = Field(2, 0.0, -90.0, 0.0, iso, "", "", [_cp(2, 150.0)])
    return Plan(name="synth", total_msw=10.0, fractions=[Fraction(1, [1, 2], [f1, f2])])


@pytest.fixture()
def beamlets_dir(tmp_path: Path) -> Path:
    arr = np.random.default_rng(3).uniform(-200, 200, size=(10, 20, 20)).astype(np.float32)
    ct = sitk.GetImageFromArray(arr)
    ct.SetOrigin((-5.0, -5.0, 2.0))
    ct.SetSpacing((1.0, 1.0, 1.0))
    bdl_path = tmp_path / "bdl.txt"
    bdl_path.write_text(_BDL)
    pd = PlanDirectory(
        plan_dir=tmp_path, plan=_plan(), ct=ct, contours={}, config={},
        bdl_path=bdl_path, bdl_text=_BDL, mc_dose_path=None,
    )
    out = tmp_path / "adota_beamlets"
    run_extraction(pd, out, ExtractionConfig(roi_size=ROI, save_overlays=False))
    return out


def test_discover_requires_ct_flux_and_meta(beamlets_dir: Path) -> None:
    ids = discover_spot_ids(beamlets_dir)
    assert len(ids) == 5
    # Removing a flux file drops that spot from discovery.
    (beamlets_dir / f"{ids[0]}_flux.npy").unlink()
    assert ids[0] not in discover_spot_ids(beamlets_dir)
    assert len(discover_spot_ids(beamlets_dir)) == 4


def test_inference_writes_predictions(beamlets_dir: Path) -> None:
    summary = run_inference(
        beamlets_dir, _StubModel(), torch.device("cpu"), InferenceConfig(batch_size=2)
    )
    assert summary["n_spots"] == 5
    assert summary["n_batches"] == 3  # ceil(5/2)
    for spot_id in discover_spot_ids(beamlets_dir):
        pred = np.load(beamlets_dir / f"{spot_id}_ds_pred.npy")
        assert pred.shape == (1, 1, 320, 60, 60)  # upsampled + de-normalized


def test_prediction_is_batch_invariant(beamlets_dir: Path) -> None:
    """A spot's prediction is independent of the batch it was grouped into."""
    model = _StubModel()
    run_inference(beamlets_dir, model, torch.device("cpu"), InferenceConfig(batch_size=1))
    single = {
        s: np.load(beamlets_dir / f"{s}_ds_pred.npy") for s in discover_spot_ids(beamlets_dir)
    }
    run_inference(beamlets_dir, model, torch.device("cpu"), InferenceConfig(batch_size=5))
    for spot_id, pred_single in single.items():
        np.testing.assert_array_equal(
            pred_single, np.load(beamlets_dir / f"{spot_id}_ds_pred.npy")
        )


def test_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No complete beamlet"):
        run_inference(tmp_path / "empty", _StubModel(), torch.device("cpu"))
