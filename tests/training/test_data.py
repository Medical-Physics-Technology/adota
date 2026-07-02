"""Unit tests for :mod:`src.training.data`.

These assert exact, numerically-checkable behavior of the data-pipeline
helpers that used to be inline in ``scripts/train_adota.py``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.data import (
    LimitedLoader,
    collate_h5,
    limited_loader,
    load_record_ids,
    train_val_split,
)


# ── train_val_split ─────────────────────────────────────────────────────────


def test_train_val_split_sizes_and_partition() -> None:
    ids = [f"r{i:02d}" for i in range(20)]
    train, val = train_val_split(ids, test_size=0.2, random_state=42)

    # Exact split sizes (round(20 * 0.2) == 4).
    assert len(val) == 4
    assert len(train) == 16
    # Disjoint and exhaustive.
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(ids)
    # Each list is sorted by original index.
    assert val == sorted(val)
    assert train == sorted(train)


def test_train_val_split_is_deterministic() -> None:
    ids = [f"r{i}" for i in range(50)]
    a = train_val_split(ids, test_size=0.3, random_state=7)
    b = train_val_split(ids, test_size=0.3, random_state=7)
    assert a == b
    # A different seed gives a different partition.
    c = train_val_split(ids, test_size=0.3, random_state=8)
    assert a != c


# ── load_record_ids ─────────────────────────────────────────────────────────


def test_load_record_ids_applies_exclusion(tmp_path: Path) -> None:
    h5_path = tmp_path / "ds.h5"
    keys = ["a", "b", "c", "d"]
    with h5py.File(h5_path, "w") as f:
        for k in keys:
            f.create_dataset(k, data=np.zeros(2))

    excl_path = tmp_path / "excl.txt"
    excl_path.write_text("b\n\nd\n")  # blank line must be ignored

    out = load_record_ids(h5_path, excl_path)
    assert out == ["a", "c"]


def test_load_record_ids_without_exclusion(tmp_path: Path) -> None:
    h5_path = tmp_path / "ds.h5"
    with h5py.File(h5_path, "w") as f:
        for k in ["x", "y"]:
            f.create_dataset(k, data=np.zeros(1))
    assert sorted(load_record_ids(h5_path, None)) == ["x", "y"]


# ── collate_h5 ──────────────────────────────────────────────────────────────


def test_collate_h5_shapes_and_values() -> None:
    samples = []
    for i in range(3):
        x = torch.full((2, 4, 4, 4), float(i))
        e = torch.tensor(float(i) * 0.1)
        y = torch.full((1, 4, 4, 4), float(i) + 0.5)
        samples.append((x, e, y))

    X, E, Y = collate_h5(samples)

    assert X.shape == (3, 2, 4, 4, 4)
    assert E.shape == (3, 1)
    assert Y.shape == (3, 1, 4, 4, 4)
    assert X.is_contiguous() and Y.is_contiguous()
    # Values are preserved per sample.
    assert torch.equal(E.squeeze(1), torch.tensor([0.0, 0.1, 0.2]))
    assert torch.equal(X[1], torch.full((2, 4, 4, 4), 1.0))
    assert torch.equal(Y[2], torch.full((1, 4, 4, 4), 2.5))


# ── LimitedLoader / limited_loader ──────────────────────────────────────────


def _toy_loader(n: int) -> DataLoader:
    ds = TensorDataset(torch.arange(n).unsqueeze(1).float())
    return DataLoader(ds, batch_size=1, shuffle=False)


def test_limited_loader_caps_iteration_and_len() -> None:
    loader = _toy_loader(10)
    limited = LimitedLoader(loader, max_batches=3)
    assert len(limited) == 3
    assert sum(1 for _ in limited) == 3
    # Dataset is still exposed for downstream code.
    assert limited.dataset is loader.dataset


def test_limited_loader_none_is_passthrough() -> None:
    loader = _toy_loader(5)
    assert limited_loader(loader, None) is loader
    wrapped = limited_loader(loader, 2)
    assert isinstance(wrapped, LimitedLoader)
    assert len(wrapped) == 2
