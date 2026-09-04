"""Plan-scale checks for the torch gamma backend against the OpenTPS corpus.

Marked ``integration``, ``slow`` and ``gpu``: this reads real plan directories,
computes a full-volume gamma on both backends, and needs a CUDA device. When the
corpus is not on the machine the tests skip with a reason naming
``$ADOTA_GAMMA_CORPUS`` and what it should contain, per the repository rule that
a missing resource is a skip and never a failure.

The synthetic parity tests in ``test_gamma_torch.py`` are the ones that prove the
algorithm; these prove it still holds at 10^8 voxels, where the search converges
over a real dose distribution rather than over noise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.metrics.gamma_benchmark import (
    VOXEL_PARITY_PLANS,
    corpus_dir,
    corpus_skip_reason,
    load_plan_case,
    voxel_parity,
)
from src.metrics.gamma_rungs import run_cpu_case, run_torch_case

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
]

# The fastest plan in the corpus (270 x 500 x 500). One criterion of it is about
# 11 s on the CPU backend and 2 s on a GPU, which is affordable for a test.
PLAN = VOXEL_PARITY_PLANS[0]
CRITERION = (3.0, 3.0, 10.0)

# The brief's correctness gate: the torch backend must land within 0.01
# percentage points of the pymedphys pass rate for the same criterion.
PASS_RATE_TOLERANCE_PP = 0.01


def _require_corpus():
    reason = corpus_skip_reason([PLAN])
    if reason is not None:
        pytest.skip(reason)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip(
            "No CUDA device available; the torch gamma backend is exercised on "
            "the CPU device by tests/test_gamma_torch.py."
        )


@pytest.fixture(scope="module")
def plan_case():
    _require_corpus()
    return load_plan_case(corpus_dir() / PLAN)


@pytest.fixture(scope="module")
def cpu_result(plan_case):
    return run_cpu_case(plan_case, criteria=[CRITERION], tag="test-cpu")


def test_gpu_pass_rate_matches_pymedphys(plan_case, cpu_result):
    """Rung 3 against rung 1 on a full clinical grid."""
    _require_cuda()
    gpu_result = run_torch_case(
        plan_case, device="cuda:0", dtype="float64",
        criteria=[CRITERION], tag="test-gpu",
    )
    expected = cpu_result["criteria"][0]["pass_rate_pct"]
    actual = gpu_result["criteria"][0]["pass_rate_pct"]
    assert abs(actual - expected) <= PASS_RATE_TOLERANCE_PP, (
        f"{PLAN} {CRITERION}: torch backend gave {actual:.6f}% against "
        f"pymedphys {expected:.6f}%"
    )


def test_gpu_float32_stays_within_the_precision_budget(plan_case, cpu_result):
    """float32 is the kernel's default dtype, so it gets its own budget: 0.1 pp."""
    _require_cuda()
    gpu_result = run_torch_case(
        plan_case, device="cuda:0", dtype="float32",
        criteria=[CRITERION], tag="test-gpu32",
    )
    expected = cpu_result["criteria"][0]["pass_rate_pct"]
    actual = gpu_result["criteria"][0]["pass_rate_pct"]
    assert abs(actual - expected) <= 0.1, (
        f"{PLAN} {CRITERION}: float32 gave {actual:.6f}% against pymedphys "
        f"{expected:.6f}%"
    )


def test_gpu_gamma_map_agrees_voxel_by_voxel(plan_case, cpu_result, tmp_path):
    """Pass rates can agree while the maps differ, so compare the maps too.

    The number that matters is how many evaluated voxels cross the gamma = 1
    decision boundary: a voxel moving 1.5 -> 1.6 changes nothing, one moving
    0.999 -> 1.001 changes the result.
    """
    _require_cuda()
    cpu_maps = run_cpu_case(
        plan_case, criteria=[CRITERION], maps_dir=tmp_path, tag="cpu"
    )
    gpu_maps = run_torch_case(
        plan_case, device="cuda:0", dtype="float64",
        criteria=[CRITERION], maps_dir=tmp_path, tag="gpu",
    )
    reference = np.load(cpu_maps["criteria"][0]["gamma_map_path"])
    other = np.load(gpu_maps["criteria"][0]["gamma_map_path"])

    stats = voxel_parity(reference, other)
    assert stats["n_evaluated"] > 0
    assert stats["n_evaluated_disagree"] == 0
    # Well under one in a million evaluated voxels may change side.
    assert stats["boundary_cross_frac"] < 1e-6, stats
