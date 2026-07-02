"""End-to-end training smoke test on a tiny slice of the real dataset.

Trains DoTA3D_v3 with a real optimizer and the real ``LMSE`` loss for a few
epochs on a handful of records pulled through the actual ``H5PYGenerator`` data
pipeline (same dataset / exclusion file / settings as the training script). The
point is not accuracy but to confirm the whole data -> forward -> loss ->
backward -> step -> eval pipeline runs and keeps working after the refactors.

Skips cleanly when the dataset is not available on the machine.
"""

from __future__ import annotations

import os
import subprocess

import pytest
import torch
from torch.utils.data import DataLoader

from src.adota.models import DoTA3D_v3
from src.loaders.generator import H5PYGenerator
from src.training.data import collate_h5
from src.training.losses import LMSE

DATASET_PATH = (
    "/scratch/mstryja/DoTA_dataset_v2/"
    "trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
)
EXCLUDED_PATH = (
    "/home/mstryja/projects/dota_pytorch/auxilary_files/"
    "IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt"
)

# Real model config (scripts/config_train_adota.yaml).
INPUT_SHAPE = (2, 160, 30, 30)
MODEL_KWARGS = dict(
    num_transformers=1,
    num_heads=4,
    num_levels=4,
    enc_features=32,
    kernel_size=3,
    convolutional_steps=2,
    conv_hidden_channels=64,
    dropout_rate=0.0,  # deterministic so the overfit signal is clean
    causal=True,
    zero_padding=True,
    last_activation=False,
    num_forward=2,
)
N_RECORDS = 8
BATCH = 2
EPOCHS = 3
SEED = 1234

pytestmark = pytest.mark.skipif(
    not (os.path.exists(DATASET_PATH) and os.path.exists(EXCLUDED_PATH)),
    reason="real dataset / exclusion file not available on this machine",
)


def _pick_device() -> torch.device:
    """Freest visible CUDA device (avoids GPUs busy with training), else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except Exception:
        return torch.device("cuda:0")
    best_idx, best_free = None, -1
    for line in out.strip().splitlines():
        idx, free = (int(p.strip()) for p in line.split(",")[:2])
        if idx < torch.cuda.device_count() and free > best_free:
            best_idx, best_free = idx, free
    return torch.device(f"cuda:{best_idx}" if best_idx is not None else "cuda:0")


# Stack H5PYGenerator samples via the shared production collate (single source
# of truth in src.training.data).
_collate = collate_h5


def test_training_smoke_on_real_data_slice():
    device = _pick_device()
    torch.manual_seed(SEED)

    # Tiny slice of the real dataset through the real loader (first N usable ids).
    dataset = H5PYGenerator(
        file_path=DATASET_PATH,
        indexes=list(range(N_RECORDS)),
        augmentation=False,  # deterministic for the overfit check
        cropp=True,  # deterministic crop around the Bragg peak (val-loader path)
        normalize=False,
        normalize_flux_only=True,
        flux_mode="analytical",
        indexes_to_exclude_list=EXCLUDED_PATH,
    )
    assert len(dataset) == N_RECORDS

    loader = DataLoader(
        dataset, batch_size=BATCH, shuffle=False, num_workers=0, collate_fn=_collate
    )

    model = DoTA3D_v3(input_shape=INPUT_SHAPE, **MODEL_KWARGS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = LMSE()

    # ── Training loop (mirrors scripts/train_adota.py inner step) ──
    model.train()
    epoch_losses = []
    for _ in range(EPOCHS):
        total, n = 0.0, 0
        for x, e, y in loader:
            x, e, y = x.to(device), e.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x, e)[0]  # forward returns (dose, attention)
            assert out.shape == (x.shape[0], 1, *INPUT_SHAPE[1:])
            loss = loss_fn(out, y)
            assert torch.isfinite(loss), "non-finite training loss"
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n += 1
        epoch_losses.append(total / n)

    # Overfitting a handful of real records must reduce the loss.
    assert epoch_losses[-1] < epoch_losses[0], f"loss did not decrease: {epoch_losses}"

    # ── Inference path (eval returns dose + attention) ──
    model.eval()
    x, e, _ = next(iter(loader))
    with torch.no_grad():
        dose, attn = model(x[:1].to(device), e[:1].to(device))
    assert dose.shape == (1, 1, *INPUT_SHAPE[1:])
    assert torch.isfinite(dose).all()
    seq = INPUT_SHAPE[1] + 1  # depth + energy token
    assert attn.shape[-2:] == (seq, seq)
