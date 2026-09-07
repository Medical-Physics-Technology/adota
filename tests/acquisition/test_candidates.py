"""The single call: candidates on a full CT come back scored, or flagged."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from src.acquisition.candidates import BeamletCandidate, prepare_ct, score_candidates
from src.acquisition.features import FEATURE_NAMES
from src.acquisition.scorer import SPARSE, DifficultyScorer
from src.beamlets.bdl import BeamDataLibrary
from src.mc_generation.sweep import RobustnessConfig


def _bdl() -> BeamDataLibrary:
    """A synthetic beam model with the HPTC distances and 3 mm spots."""
    energies = np.arange(70.0, 231.0, 10.0)
    table = pd.DataFrame({"NominalEnergy": energies, "MeanEnergy": energies, "ProtonsMU": 1e8,
                          "SpotSize1x": 3.0, "SpotSize1y": 3.0})
    return BeamDataLibrary(nozzle_isocenter=500.0, smx=2014.9, smy=2584.1, energy_table=table,
                           source_path=Path("synthetic"))


def _torso() -> sitk.Image:
    """A 1 mm CT with an elliptical soft-tissue torso and a bone rod along z (array is z, y, x).

    140 slices thick: a 0.5 degree steering shifts the 60 mm ROI by 22 mm along z,
    which an 80-slice phantom cannot hold.
    """
    arr = np.full((140, 260, 420), -1024.0, dtype=np.float32)
    zz, yy, xx = np.mgrid[0:140, 0:260, 0:420]
    torso = ((yy - 130) ** 2 / 100 ** 2 + (xx - 210) ** 2 / 150 ** 2) < 1.0
    arr[torso] = 30.0
    arr[((yy - 130) ** 2 + (xx - 250) ** 2) < 12 ** 2] = 900.0         # a rib-like rod in the beam
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    return image


def _scorer(tmp_path: Path) -> DifficultyScorer:
    grids = {name: np.linspace(0, 1, 101).tolist() for name in FEATURE_NAMES}
    grids["wepl_mean"] = np.linspace(0, 300, 101).tolist()
    data = {"percentile_grids_dev": grids,
            "weights": {SPARSE: {"intercept": 0.0, "coefficients": {"wepl_mean": 1.0, "total_hu_change": 0.5}}}}
    path = tmp_path / "scorer.json"
    path.write_text(json.dumps(data))
    return DifficultyScorer.load(path, SPARSE)


@pytest.fixture(scope="module")
def prepared_ct():
    return prepare_ct(_torso())


def test_candidates_are_scored_in_order_with_geometry_grouped_by_gantry(prepared_ct, tmp_path):
    cands = [BeamletCandidate(90.0, 100.0, 0.0, 0.0, "a"),
             BeamletCandidate(90.0, 120.0, 0.5, -0.5, "b"),
             BeamletCandidate(90.0, 100.0, 0.0, 0.0, "c")]
    frame = score_candidates(prepared_ct, cands, _bdl(), {"sparse": _scorer(tmp_path)},
                             config=RobustnessConfig(rotate_to_canonical=True), prepared=True)
    assert list(frame.candidate_id) == ["a", "b", "c"]
    assert set(FEATURE_NAMES) <= set(frame.columns) and "score_sparse" in frame.columns
    assert frame.valid.all(), frame[["candidate_id", "valid", "reason"]]
    valid = frame
    assert np.isfinite(valid[list(FEATURE_NAMES)].to_numpy(float)).all()
    # identical candidates score identically; the 120 MeV beam reaches deeper water-equivalent depth
    a, c = frame.set_index("candidate_id").loc[["a", "c"]].score_sparse
    assert a == c
    assert frame.set_index("candidate_id").loc["b", "wepl_mean"] > frame.set_index("candidate_id").loc["a", "wepl_mean"]


def test_peak_leaving_the_crop_is_invalid_not_an_error(prepared_ct, tmp_path):
    """230 MeV has a 300 mm water range; through a 300 mm torso it does not stop inside the crop."""
    frame = score_candidates(prepared_ct, [BeamletCandidate(90.0, 230.0, 0.0, 0.0, "hot")],
                             _bdl(), {"sparse": _scorer(tmp_path)}, prepared=True)
    row = frame.iloc[0]
    assert not row.valid and row.reason == "peak_outside_crop" and np.isnan(row.score_sparse)


def test_ray_missing_the_ct_is_invalid_not_an_error(prepared_ct, tmp_path):
    frame = score_candidates(prepared_ct, [BeamletCandidate(90.0, 100.0, 12.0, 12.0, "wild")],
                             _bdl(), {}, prepared=True)
    assert not frame.iloc[0].valid and frame.iloc[0].reason == "roi_out_of_bounds"


def test_random_gantry_rotates_to_canonical(prepared_ct, tmp_path):
    frame = score_candidates(prepared_ct, [BeamletCandidate(135.0, 100.0, 0.0, 0.0, "oblique")],
                             _bdl(), {"sparse": _scorer(tmp_path)}, prepared=True)
    row = frame.iloc[0]
    assert row.mc_gantry_deg == 90.0 and row.ct_rotation_deg == pytest.approx(-45.0)
    assert row.valid, row.reason
