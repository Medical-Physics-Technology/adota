"""Tests for scripts/validation_adota.py and its helpers.

Covers the new logic (per-sample metric assembly, nan-aware aggregation, the
split-consistency guard, and the comparison-table renderer) on synthetic data,
so they run without the H5 dataset or a GPU.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts.validation_adota import (
    SampleResult,
    _check_split_consistency,
    _log_cell,
    aggregate_run,
    compute_sample_metrics,
    select_epoch_from_logs,
)
from src.schemas.configs import RunRef, ValidationExperimentConfig
from src.tables.results import render_comparison_table
from src.training.losses import LMSE

# Identity de-norm (min_ds=0, max_ds=1) keeps synthetic values interpretable.
SCALE = {
    "min_ds": 0.0,
    "max_ds": 1.0,
    "min_energy": 70.0,
    "max_energy": 270.0,
}
D, H, W = 80, 8, 8
DZ_MM = 2.0


def _volume_from_profile(profile: np.ndarray) -> torch.Tensor:
    """Make a (1, D, H, W) dose whose lateral sum equals ``profile`` (D,)."""
    vol = np.broadcast_to(
        profile[:, None, None] / (H * W), (D, H, W)
    ).astype(np.float32)
    return torch.from_numpy(vol.copy()).unsqueeze(0)


def _gaussian_profile(center: float, sigma: float = 6.0) -> np.ndarray:
    d = np.arange(D, dtype=float)
    return np.exp(-((d - center) ** 2) / (2 * sigma**2))


def _ctx(y: torch.Tensor, y_pred_vol: torch.Tensor, energy_norm: float = 0.5):
    """Build a minimal InferenceContext-like object for the metric fn."""
    return SimpleNamespace(
        sample_id="sample_0",
        y=y,  # (1, D, H, W)
        y_pred=y_pred_vol.unsqueeze(0),  # (1, 1, D, H, W)
        energy=torch.tensor(energy_norm),
    )


def _metrics(ctx) -> SampleResult:
    return compute_sample_metrics(
        ctx,
        scale=SCALE,
        loss_mse_fn=LMSE(),
        mape_mask_frac=0.1,
        range_dz_mm=DZ_MM,
        range_oversample=20,
        range_min_peak_dose=0.0,
    )


# ── Per-sample metrics ──────────────────────────────────────────────────────


def test_identical_pred_gt_gives_zero_errors() -> None:
    prof = _gaussian_profile(center=40)
    y = _volume_from_profile(prof)
    res = _metrics(_ctx(y, y))  # y_pred volume == gt volume

    assert res.lmse == pytest.approx(0.0, abs=1e-6)
    assert res.mape_pct == pytest.approx(0.0, abs=1e-4)
    assert res.rmse_gy == pytest.approx(0.0, abs=1e-9)
    assert res.rde_pct == pytest.approx(0.0, abs=1e-4)
    assert res.r80_err_mm == pytest.approx(0.0, abs=1e-6)
    assert res.r20_err_mm == pytest.approx(0.0, abs=1e-6)
    assert res.dfw_err_mm == pytest.approx(0.0, abs=1e-6)
    # Energy de-normalizes: 0.5 -> 70 + 0.5*(270-70) = 170 MeV.
    assert res.energy_mev == pytest.approx(170.0)


def test_range_error_equals_injected_depth_shift() -> None:
    k = 3  # voxels deeper => +k*dz mm range error
    gt = _volume_from_profile(_gaussian_profile(center=40))
    pred = _volume_from_profile(_gaussian_profile(center=40 + k))
    res = _metrics(_ctx(gt, pred))

    expected_mm = k * DZ_MM
    # Sub-voxel extraction (oversample 20 => 0.1 voxel grid); allow ~0.5 mm.
    assert res.r80_err_mm == pytest.approx(expected_mm, abs=0.5)
    assert res.r20_err_mm == pytest.approx(expected_mm, abs=0.5)
    # Same shape, just shifted => distal fall-off width unchanged.
    assert res.dfw_err_mm == pytest.approx(0.0, abs=0.5)
    assert res.r80_err_mm > 0  # signed: prediction is deeper


# ── Aggregation ─────────────────────────────────────────────────────────────


def test_aggregate_run_is_nan_aware() -> None:
    def mk(r80: float) -> SampleResult:
        return SampleResult(
            sample_id="s", energy_mev=170.0, mape_pct=5.0, lmse=1e-3,
            rmse_gy=1e-5, rde_pct=2.0, r80_err_mm=r80, r20_err_mm=0.0,
            dfw_err_mm=0.0,
        )

    agg = aggregate_run([mk(1.0), mk(3.0), mk(float("nan"))])
    # The NaN sample is excluded from the range stat.
    assert agg["r80_err_mm"]["n"] == 2
    assert agg["r80_err_mm"]["mean"] == pytest.approx(2.0)
    # Non-NaN metrics use all three.
    assert agg["mape_pct"]["n"] == 3
    assert agg["mape_pct"]["mean"] == pytest.approx(5.0)


# ── Split-consistency guard ─────────────────────────────────────────────────


def _write_run_config(run_dir: Path, split_random_state: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset_path": "/data/ds.h5",
                "train_test_split": 0.2,
                "split_random_state": split_random_state,
            }
        )
    )


def test_split_guard_raises_on_mismatch_when_strict(tmp_path: Path) -> None:
    cfg = ValidationExperimentConfig(
        dataset_path="/data/ds.h5", train_test_split=0.2,
        split_random_state=42, strict_split_check=True,
    )
    rd = tmp_path / "run_bad"
    _write_run_config(rd, split_random_state=999)
    with pytest.raises(ValueError, match="split/scale mismatch"):
        _check_split_consistency(cfg, RunRef(name="bad", run_dir=str(rd)), rd)


def test_split_guard_warns_when_not_strict(tmp_path: Path) -> None:
    cfg = ValidationExperimentConfig(
        dataset_path="/data/ds.h5", train_test_split=0.2,
        split_random_state=42, strict_split_check=False,
    )
    rd = tmp_path / "run_bad"
    _write_run_config(rd, split_random_state=999)
    # Should not raise.
    _check_split_consistency(cfg, RunRef(name="bad", run_dir=str(rd)), rd)


def test_split_guard_passes_on_match(tmp_path: Path) -> None:
    cfg = ValidationExperimentConfig(
        dataset_path="/data/ds.h5", train_test_split=0.2, split_random_state=42,
    )
    rd = tmp_path / "run_ok"
    _write_run_config(rd, split_random_state=42)
    _check_split_consistency(cfg, RunRef(name="ok", run_dir=str(rd)), rd)


# ── Table renderer ──────────────────────────────────────────────────────────


def test_render_comparison_table_shape_and_format() -> None:
    table = render_comparison_table(
        ["baseline", "ablation_x"],
        ["MAPE [%]", "LMSE"],
        [["6.190 +/- 1.2", "8.4e-03 +/- 1e-3"], ["7.0 +/- 1.5", "9e-03 +/- 1e-3"]],
        row_header="Run",
    )
    lines = table.splitlines()
    assert lines[0].startswith("| Run")
    assert "MAPE [%]" in lines[0] and "LMSE" in lines[0]
    assert set(lines[1].replace("|", "").strip()) <= {"-"}  # separator row
    assert "baseline" in lines[2]
    assert "ablation_x" in lines[3]
    # Every data row exposes both metric cells.
    assert "6.190 +/- 1.2" in table and "9e-03 +/- 1e-3" in table


def test_render_comparison_table_rejects_ragged_cells() -> None:
    with pytest.raises(ValueError):
        render_comparison_table(["a", "b"], ["m1", "m2"], [["1", "2"]])


# ── Log-based selection (paper reproduction) ────────────────────────────────


def _write_metrics_jsonl(path: Path, rows: list) -> None:
    import json

    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_select_epoch_from_logs_picks_min_val_mape(tmp_path: Path) -> None:
    mpath = tmp_path / "metrics.jsonl"
    _write_metrics_jsonl(
        mpath,
        [
            {"epoch": 0, "val": {"mape_pct_mean": 9.0, "loss_combined_mean": 0.05}},
            {"epoch": 1, "val": {"mape_pct_mean": 6.0, "loss_combined_mean": 0.02}},
            # Lowest MAPE here, but NOT the lowest loss.
            {"epoch": 2, "val": {"mape_pct_mean": 5.5, "loss_combined_mean": 0.03}},
        ],
    )
    epoch, val = select_epoch_from_logs(mpath, "val_mape")
    assert epoch == 2
    assert val["mape_pct_mean"] == 5.5

    # select_by val_loss picks the min-loss epoch instead.
    epoch_loss, _ = select_epoch_from_logs(mpath, "val_loss")
    assert epoch_loss == 1


def test_select_epoch_rejects_unknown_metric(tmp_path: Path) -> None:
    mpath = tmp_path / "metrics.jsonl"
    _write_metrics_jsonl(mpath, [{"epoch": 0, "val": {"mape_pct_mean": 1.0}}])
    with pytest.raises(ValueError):
        select_epoch_from_logs(mpath, "nonsense")


def test_log_cell_formats_mean_std_and_handles_missing() -> None:
    val = {"mape_pct_mean": 6.017, "mape_pct_std": 4.674, "loss_mse_mean": 6.7e-3}
    assert _log_cell(val, "mape_pct_mean", "mape_pct_std", "{:.2f}") == "6.02 +/- 4.67"
    # No std key -> mean only.
    assert _log_cell(val, "loss_mse_mean", None, "{:.2e}") == "6.70e-03"
    # Missing metric -> n/a.
    assert _log_cell(val, "absent", "x", "{:.2f}") == "n/a"
