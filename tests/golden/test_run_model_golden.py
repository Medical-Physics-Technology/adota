"""Characterization (golden) test for ``scripts/run_model.py``.

Pins the per-sample metric CSV produced by the current ``evaluate_single_sample``
+ ``save_results_csv`` on a fixed slice of the lung test set, so the upcoming
``src/evaluation/`` refactor can be proven behavior-preserving.

The harness calls ``evaluate_single_sample`` directly (not ``evaluate_samples``)
so the golden is independent of the ``i == 169`` skip removal and does not write
``_ds_pred.npy`` files into the dataset directory.

Skipped unless the checkpoint and dataset are present (mirrors the smoke test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the sibling _goldenlib importable regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _goldenlib as gl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LUNG = Path("/scratch/mstryja/DoTA_dataset_v2/lung_testset_paper")
MODEL_DIR = PROJECT_ROOT / "models" / "DoTA_v3_grid_search_v11"
MODEL_PATH = MODEL_DIR / "best_model.pth"
HP_PATH = MODEL_DIR / "hyperparams.json"

N_SLICE = 5

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and HP_PATH.exists() and LUNG.is_dir()),
    reason="golden test requires the checkpoint + lung dataset on /scratch",
)


def test_run_model_golden(tmp_path):
    gl.set_determinism()

    from scripts.run_model import (
        discover_sample_ids,
        evaluate_samples,
        save_results_csv,
    )
    from src.adota.config import get_device
    from src.adota.utils import load_model
    from src.schemas.configs import EvaluationConfig

    device = get_device(0)
    model = load_model(MODEL_PATH, HP_PATH, device)
    config = EvaluationConfig()

    sample_ids = sorted(discover_sample_ids(LUNG))[:N_SLICE]
    assert sample_ids, "no sample ids discovered in the lung test set"

    # Drive the refactored engine path; save_predictions=False to avoid mutating
    # the dataset directory.
    results = evaluate_samples(
        model=model,
        sample_ids=sample_ids,
        test_data_path=LUNG,
        config=config,
        device=device,
        downsampling_method="interpolation",
        show_progress=False,
        save_predictions=False,
    )

    out = tmp_path / "results.csv"
    save_results_csv(results, out)

    status, msgs = gl.check_against_golden("run_model", out.read_text())
    if status == "captured":
        pytest.skip("baseline captured: " + "; ".join(msgs))
    assert status == "match", "golden mismatch:\n" + "\n".join(msgs)
