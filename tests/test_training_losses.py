"""End-to-end and property tests for :mod:`src.training.losses`.

The integration tests run the full inference pipeline once per session:

1. Load the released DoTA checkpoint from ``models/DoTA_v3_grid_search_v11/``.
2. Preprocess each sample under ``data/example_inputs/`` via
   :func:`src.loaders.dir_based.get_single_record` (normalize CT / dose /
   energy, normalize flux per-sample, trilinear-interpolate to the model
   input shape, stack CT and flux into a 2-channel volume).
3. Run inference with ``torch.no_grad()``.
4. Compute each loss against the interpolated ground-truth dose.

The cached inference results feed parametrized per-sample tests (one
test case per sample id) and a final summary test that prints a
per-sample table of all three loss values.

Property tests on top of the pipeline tests cover:

- Identity: ``loss(y, y) == 0`` for each loss.
- Gradient flow: backward through ``LMSE`` produces finite gradients.
- Shape-mismatch raises :class:`ValueError`.
- :class:`TwoObjectiveBalancer` returns weights that sum to one, gives
  the heavier weight to the larger loss, updates its running state, and
  clears it on :meth:`~TwoObjectiveBalancer.reset`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch

from src.adota.config import DEFAULT_SCALE, get_device
from src.adota.models import DoTA3D_v3
from src.adota.utils import load_model
from src.loaders.dir_based import get_single_record
from src.training.losses import LMSE, LPS, LossLPD, TwoObjectiveBalancer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "example_inputs"
MODEL_DIR = PROJECT_ROOT / "models" / "DoTA_v3_grid_search_v11"
MODEL_PATH = MODEL_DIR / "best_model.pth"
HYPERPARAMS_PATH = MODEL_DIR / "hyperparams.json"


def _discover_sample_ids() -> List[str]:
    return sorted(p.name.replace("_ct.npy", "") for p in DATA_DIR.glob("*_ct.npy"))


SAMPLE_IDS: List[str] = _discover_sample_ids()

if not SAMPLE_IDS:
    pytest.skip(
        f"No example samples found in {DATA_DIR}",
        allow_module_level=True,
    )


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def device() -> torch.device:
    return get_device(0)


@pytest.fixture(scope="session")
def model(device: torch.device) -> DoTA3D_v3:
    if not MODEL_PATH.exists() or not HYPERPARAMS_PATH.exists():
        pytest.skip(f"Model checkpoint or hyperparams missing under {MODEL_DIR}")
    return load_model(MODEL_PATH, HYPERPARAMS_PATH, device)


@pytest.fixture(scope="session")
def inference_results(
    model: DoTA3D_v3,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Run inference on every example sample once and cache the tensors.

    Returns:
        Mapping ``sample_id -> {"y_pred": tensor, "y_gt": tensor,
        "energy": tensor}``, with tensors on ``device`` and shape
        ``(1, 1, 160, 30, 30)`` for dose volumes.
    """
    results: Dict[str, Dict[str, torch.Tensor]] = {}
    for sample_id in SAMPLE_IDS:
        x, energy, y = get_single_record(
            sample_id,
            str(DATA_DIR),
            scale=DEFAULT_SCALE,
            normalize_flux=True,
            downsampling_method="interpolation",
            beamlet_angle=False,
        )
        x = x.unsqueeze(0).to(device)
        energy = energy.unsqueeze(0).to(device)
        y_gt = y.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x, energy)
        y_pred = output[0] if isinstance(output, tuple) else output

        results[sample_id] = {
            "y_pred": y_pred,
            "y_gt": y_gt,
            "energy": energy,
        }
    return results


# ── Integration tests: full pipeline per sample ─────────────────────────────


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_lmse_on_inference(
    inference_results: Dict[str, Dict[str, torch.Tensor]],
    sample_id: str,
) -> None:
    """LMSE on the model output: scalar, finite, non-negative."""
    sample = inference_results[sample_id]
    loss = LMSE()(sample["y_pred"], sample["y_gt"])

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0
    logger.info("LMSE | %s | %.6e", sample_id, loss.item())


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_lps_on_inference(
    inference_results: Dict[str, Dict[str, torch.Tensor]],
    sample_id: str,
) -> None:
    """LPS on the model output: scalar, finite, non-negative."""
    sample = inference_results[sample_id]
    loss = LPS()(sample["y_pred"], sample["y_gt"])

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0.0
    logger.info("LPS  | %s | %.6e", sample_id, loss.item())


@pytest.mark.parametrize("sample_id", SAMPLE_IDS)
def test_losslpd_on_inference(
    inference_results: Dict[str, Dict[str, torch.Tensor]],
    sample_id: str,
) -> None:
    """LossLPD on the model output: scalar, finite, in ``[0, clamp_max]``."""
    loss_fn = LossLPD()
    sample = inference_results[sample_id]
    loss = loss_fn(sample["y_pred"], sample["y_gt"])

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert 0.0 <= loss.item() <= loss_fn.clamp_max
    logger.info("LPD  | %s | %.6e", sample_id, loss.item())


def test_pipeline_loss_summary(
    inference_results: Dict[str, Dict[str, torch.Tensor]],
    capsys: pytest.CaptureFixture,
) -> None:
    """Aggregate report: per-sample LMSE / LPS / LossLPD plus mean and std.

    The table is printed to stdout (use ``pytest -s`` to view).
    """
    lmse_fn, lps_fn, lpd_fn = LMSE(), LPS(), LossLPD()

    rows: List[tuple] = []
    for sample_id, sample in inference_results.items():
        rows.append(
            (
                sample_id,
                lmse_fn(sample["y_pred"], sample["y_gt"]).item(),
                lps_fn(sample["y_pred"], sample["y_gt"]).item(),
                lpd_fn(sample["y_pred"], sample["y_gt"]).item(),
            )
        )

    values = np.array([(r[1], r[2], r[3]) for r in rows])
    header = f"\n{'sample_id':<40} {'LMSE':>12} {'LPS':>12} {'LPD':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for sample_id, lmse, lps, lpd in rows:
        print(f"{sample_id:<40} {lmse:>12.4e} {lps:>12.4e} {lpd:>12.4e}")
    print(sep)
    print(
        f"{'mean':<40} {values[:, 0].mean():>12.4e} "
        f"{values[:, 1].mean():>12.4e} {values[:, 2].mean():>12.4e}"
    )
    print(
        f"{'std':<40} {values[:, 0].std():>12.4e} "
        f"{values[:, 1].std():>12.4e} {values[:, 2].std():>12.4e}"
    )

    assert np.all(np.isfinite(values))


# ── Property tests: identity (loss(y, y) == 0) ──────────────────────────────


def test_lmse_identity() -> None:
    torch.manual_seed(0)
    y = torch.rand(2, 1, 160, 30, 30) + 0.1
    assert LMSE()(y, y).item() == pytest.approx(0.0, abs=1e-7)


def test_lps_identity() -> None:
    torch.manual_seed(0)
    y = torch.rand(2, 1, 160, 30, 30) + 0.1
    assert LPS()(y, y).item() == pytest.approx(0.0, abs=1e-7)


def test_losslpd_identity() -> None:
    torch.manual_seed(0)
    y = torch.rand(2, 1, 160, 30, 30) + 0.1
    assert LossLPD()(y, y).item() == pytest.approx(0.0, abs=1e-5)


# ── Property tests: gradient flow & shape validation ────────────────────────


def test_lmse_gradient_flow() -> None:
    torch.manual_seed(0)
    y_gt = torch.rand(2, 1, 160, 30, 30) + 0.1
    y_pred = torch.rand_like(y_gt).requires_grad_(True)
    LMSE()(y_pred, y_gt).backward()
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all().item()


def test_lps_gradient_flow() -> None:
    torch.manual_seed(0)
    y_gt = torch.rand(2, 1, 160, 30, 30) + 0.1
    y_pred = torch.rand_like(y_gt).requires_grad_(True)
    LPS()(y_pred, y_gt).backward()
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all().item()


@pytest.mark.parametrize("loss_cls", [LMSE, LPS, LossLPD])
def test_loss_raises_on_shape_mismatch(loss_cls: type) -> None:
    a = torch.zeros(1, 1, 16, 8, 8)
    b = torch.zeros(1, 1, 16, 8, 9)
    with pytest.raises(ValueError):
        loss_cls()(a, b)


# ── Property tests: TwoObjectiveBalancer ────────────────────────────────────


def test_balancer_weights_sum_to_one() -> None:
    w1, w2 = TwoObjectiveBalancer().get_weights(
        torch.tensor(0.7), torch.tensor(0.3)
    )
    assert (w1 + w2).item() == pytest.approx(1.0, abs=1e-6)
    assert w1.item() > 0 and w2.item() > 0


def test_balancer_larger_loss_gets_larger_weight() -> None:
    """Softmax-over-log gives the larger weight to the larger magnitude."""
    w1, w2 = TwoObjectiveBalancer().get_weights(
        torch.tensor(0.9), torch.tensor(0.1)
    )
    assert w1.item() > w2.item()


def test_balancer_running_state_updates_and_resets() -> None:
    balancer = TwoObjectiveBalancer(smoothing=0.5)
    assert balancer.running_weights is None

    balancer.get_weights(torch.tensor(0.4), torch.tensor(0.6))
    first = balancer.running_weights
    assert first is not None

    balancer.get_weights(torch.tensor(0.9), torch.tensor(0.1))
    second = balancer.running_weights
    assert second is not None
    assert not torch.allclose(first, second)

    balancer.reset()
    assert balancer.running_weights is None


def test_balancer_rejects_invalid_smoothing() -> None:
    with pytest.raises(ValueError):
        TwoObjectiveBalancer(smoothing=1.5)
