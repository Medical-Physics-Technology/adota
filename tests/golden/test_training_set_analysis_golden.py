"""Characterization (golden) test for ``scripts/training_set_analysis.py``.

Pins the per-beamlet metric CSV produced by the current ``evaluate_single_sample``
+ ``save_results_csv`` on a fixed slice of the HDF5 dataset.

This baseline is captured against the CURRENT code, which shares the same MAPE
argument-order bug as run_model_h5py. The agreed fix changes the ``mape_pct``
column only; the golden is re-captured at that point and the diff reviewed.

Per-sample skip guards (zero-flux, energy threshold) are part of the script's
behavior and are exercised here exactly as in production.

Skipped unless the checkpoint and HDF5 dataset are present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _goldenlib as gl  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
H5_PATH = Path(
    "/scratch/mstryja/DoTA_dataset_v2/"
    "trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
)
EXCLUDE_FILE = Path(
    "/home/mstryja/projects/dota_pytorch/auxilary_files/"
    "IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt"
)
MODEL_DIR = PROJECT_ROOT / "models" / "DoTA_v3_grid_search_v11"
MODEL_PATH = MODEL_DIR / "best_model.pth"
HP_PATH = MODEL_DIR / "hyperparams.json"

N_SLICE = 5

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and HP_PATH.exists() and H5_PATH.exists()),
    reason="golden test requires the checkpoint + HDF5 dataset on /scratch",
)


def _fixed_record_ids() -> list[str]:
    """Deterministic slice: sorted H5 keys minus excluded ids, first N."""
    import h5py

    with h5py.File(str(H5_PATH), "r") as ds:
        all_ids = sorted(ds.keys())

    excluded: set[str] = set()
    if EXCLUDE_FILE.exists():
        excluded = {line.strip() for line in EXCLUDE_FILE.read_text().splitlines() if line.strip()}

    kept = [rid for rid in all_ids if rid not in excluded]
    return kept[:N_SLICE]


def test_training_set_analysis_golden(tmp_path):
    gl.set_determinism()

    from scripts.training_set_analysis import evaluate_samples, save_results_csv
    from src.adota.config import get_device
    from src.adota.utils import load_model
    from src.loaders.generator import H5PYGenerator
    from src.schemas.configs import AnalysisConfig

    record_ids = _fixed_record_ids()
    assert record_ids, "no record ids discovered in the HDF5 dataset"

    device = get_device(0)
    model = load_model(MODEL_PATH, HP_PATH, device)
    config = AnalysisConfig()

    dataset = H5PYGenerator(
        file_path=str(H5_PATH),
        indexes=record_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )

    # Drive the refactored engine path (per-sample skip guards preserved).
    results = evaluate_samples(
        model=model,
        record_ids=record_ids,
        dataset=dataset,
        config=config,
        device=device,
        show_progress=False,
        compute_gpr=True,
    )

    assert results, "all sampled beamlets were skipped; widen the slice"

    out = tmp_path / "results.csv"
    save_results_csv(results, out)

    status, msgs = gl.check_against_golden("training_set_analysis", out.read_text())
    if status == "captured":
        pytest.skip("baseline captured: " + "; ".join(msgs))
    assert status == "match", "golden mismatch:\n" + "\n".join(msgs)
