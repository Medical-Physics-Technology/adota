"""Unit tests for src/evaluation/sources.py.

Asserts each source yields tensors equal to a direct call of the underlying
loader (DirSource vs get_single_record; H5Source vs dataset[i]) on a fixed slice
of the real data. Skipped when that data is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LUNG = Path("/scratch/mstryja/DoTA_dataset_v2/lung_testset_paper")
H5_PATH = Path(
    "/scratch/mstryja/DoTA_dataset_v2/"
    "trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
)
EXCLUDE_FILE = Path(
    "/home/mstryja/projects/dota_pytorch/auxilary_files/"
    "IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt"
)


# ── DirSource ────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not LUNG.is_dir(), reason="lung dataset not available")
def test_dirsource_matches_loader():
    from scripts.run_model import discover_sample_ids
    from src.evaluation.sources import DirSource, Sample
    from src.loaders.dir_based import get_single_record
    from src.schemas.configs import EvaluationConfig

    config = EvaluationConfig()
    ids = sorted(discover_sample_ids(LUNG))[:3]
    source = DirSource(
        LUNG,
        ids,
        scale=config.scale,
        normalize_flux=config.normalize_flux,
        downsampling_method="interpolation",
        beamlet_angle=True,
    )

    assert len(source) == len(ids)

    samples = list(source)
    assert [s.sample_id for s in samples] == ids

    for sample in samples:
        assert isinstance(sample, Sample)
        x, energy, y, ba = get_single_record(
            sample.sample_id,
            str(LUNG),
            scale=config.scale,
            normalize_flux=config.normalize_flux,
            downsampling_method="interpolation",
            beamlet_angle=True,
        )
        assert torch.equal(sample.x, x)
        assert torch.equal(sample.energy, energy)
        assert torch.equal(sample.y, y)
        assert sample.extra["beamlet_angles"] == ba


# ── H5Source ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not H5_PATH.exists(), reason="HDF5 dataset not available")
def test_h5source_matches_dataset():
    import h5py

    from src.evaluation.sources import H5Source, Sample
    from src.loaders.generator import H5PYGenerator

    with h5py.File(str(H5_PATH), "r") as ds:
        all_ids = sorted(ds.keys())
    excluded = set()
    if EXCLUDE_FILE.exists():
        excluded = {ln.strip() for ln in EXCLUDE_FILE.read_text().splitlines() if ln.strip()}
    record_ids = [r for r in all_ids if r not in excluded][:3]

    dataset = H5PYGenerator(
        file_path=str(H5_PATH),
        indexes=record_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )

    source = H5Source(dataset, record_ids)
    assert len(source) == len(dataset)

    samples = list(source)
    assert [s.sample_id for s in samples] == record_ids

    for idx, sample in enumerate(samples):
        assert isinstance(sample, Sample)
        x, energy, y = dataset[idx]
        assert torch.equal(sample.x, x)
        assert torch.equal(sample.energy, energy)
        assert torch.equal(sample.y, y)
        assert sample.extra == {}


@pytest.mark.skipif(not H5_PATH.exists(), reason="HDF5 dataset not available")
def test_h5source_defaults_record_ids_from_dataset():
    import h5py

    from src.evaluation.sources import H5Source
    from src.loaders.generator import H5PYGenerator

    with h5py.File(str(H5_PATH), "r") as ds:
        all_ids = sorted(ds.keys())
    excluded = set()
    if EXCLUDE_FILE.exists():
        excluded = {ln.strip() for ln in EXCLUDE_FILE.read_text().splitlines() if ln.strip()}
    record_ids = [r for r in all_ids if r not in excluded][:3]

    dataset = H5PYGenerator(
        file_path=str(H5_PATH),
        indexes=record_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=True,
    )

    source = H5Source(dataset)  # record_ids omitted -> taken from dataset
    assert [s.sample_id for s in source] == list(dataset.record_ids)


def test_h5source_length_mismatch_raises():
    # Pure-Python guard; no data needed. Use a tiny stand-in dataset.
    from src.evaluation.sources import H5Source

    class _FakeDS:
        record_ids = ["a", "b"]

        def __len__(self):
            return 2

    with pytest.raises(ValueError):
        H5Source(_FakeDS(), ["only-one"])
