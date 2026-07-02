"""Cross-run validation experiment for ADoTA (inference only).

Evaluates several trained ADoTA runs on one shared validation set and reports a
``Run x Metric`` table with ``mean +/- std`` cells, to enrich the ablation
study. For every run it loads the model from its run directory, runs inference
over the identical validation set, and computes per-sample:

* MAPE [%]            (de-normalized dose, 10%-of-peak mask; matches training val)
* LMSE                (normalized MSE = the training objective)
* RMSE [Gy] / RDE [%] (de-normalized dose)
* R80 / R20 / DFW range errors [mm]  (signed prediction - ground-truth)

No training is performed.

Usage:
    uv run python scripts/validation_adota.py --config scripts/config_validation_adota.yaml
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import (  # noqa: E402
    denormalize_energy,
    load_yaml_config,
    setup_logging,
    setup_run_directory,
)
from src.adota.models import DoTA3D_v3  # noqa: E402
from src.evaluation.cli import resolve_device  # noqa: E402
from src.evaluation.engine import denorm_pair  # noqa: E402
from src.evaluation.outputs import CsvColumn, save_results_csv  # noqa: E402
from src.loaders.generator import H5PYGenerator  # noqa: E402
from src.metrics.classic import (  # noqa: E402
    calculate_pure_mape,
    calculate_relative_dose_error,
    calculate_rmse,
)
from src.metrics.range_metrics import (  # noqa: E402
    compute_range_metrics,
    integrated_depth_dose,
    range_metric_deltas,
)
from src.schemas.configs import RunRef, ValidationExperimentConfig  # noqa: E402
from src.tables.results import render_comparison_table  # noqa: E402
from src.training.data import collate_h5, load_record_ids, train_val_split  # noqa: E402
from src.training.losses import LMSE  # noqa: E402
from src.utils.unit_conversions import to_gy  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(help="ADoTA cross-run validation experiment (inference only).")

# Split / data fields that must match between the shared config and each run,
# so all models are scored on the identical validation set in the same units.
_SPLIT_GUARD_KEYS = ("dataset_path", "train_test_split", "split_random_state")


@dataclass
class SampleResult:
    """Per-sample metrics for one validation record under one run's model."""

    sample_id: str
    energy_mev: float
    mape_pct: float
    lmse: float
    rmse_gy: float
    rde_pct: float
    r80_err_mm: float
    r20_err_mm: float
    dfw_err_mm: float


# Metric columns: (attribute, header, value format). Order drives the table.
_METRIC_COLUMNS = [
    ("mape_pct", "MAPE [%]", "{:.3f}"),
    ("lmse", "LMSE", "{:.4e}"),
    ("rmse_gy", "RMSE [Gy]", "{:.4e}"),
    ("rde_pct", "RDE [%]", "{:.3f}"),
    ("r80_err_mm", "R80 err [mm]", "{:.3f}"),
    ("r20_err_mm", "R20 err [mm]", "{:.3f}"),
    ("dfw_err_mm", "DFW err [mm]", "{:.3f}"),
]


# ── Config plumbing ─────────────────────────────────────────────────────────


def _build_config(yaml_path: Path) -> ValidationExperimentConfig:
    """Load the YAML into a :class:`ValidationExperimentConfig`."""
    raw = load_yaml_config(yaml_path)
    valid_keys = {f.name for f in fields(ValidationExperimentConfig)}
    merged = {k: v for k, v in raw.items() if k in valid_keys}
    merged["runs"] = [
        RunRef(**r) if isinstance(r, dict) else r for r in raw.get("runs", [])
    ]
    return ValidationExperimentConfig(**merged)


def _check_split_consistency(
    cfg: ValidationExperimentConfig, run: RunRef, run_dir: Path
) -> None:
    """Verify a run's saved config shares the experiment's split + scale.

    Guards against silently comparing models that were validated on different
    splits. Raises (strict) or logs a warning depending on ``strict_split_check``.
    """
    run_cfg_path = run_dir / "config.yaml"
    if not run_cfg_path.exists():
        msg = f"[{run.name}] no config.yaml in {run_dir}; cannot verify split."
        if cfg.strict_split_check:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return

    run_cfg = load_yaml_config(run_cfg_path)
    mismatches = [
        f"{key}: experiment={getattr(cfg, key)!r} run={run_cfg.get(key)!r}"
        for key in _SPLIT_GUARD_KEYS
        if run_cfg.get(key) != getattr(cfg, key)
    ]
    if run_cfg.get("scale") not in (None, cfg.scale):
        mismatches.append("scale differs from experiment scale")

    if mismatches:
        msg = f"[{run.name}] split/scale mismatch: " + "; ".join(mismatches)
        if cfg.strict_split_check:
            raise ValueError(msg)
        logger.warning("%s (continuing; strict_split_check is off)", msg)


# ── Shared validation set ───────────────────────────────────────────────────


def _build_val_loader(cfg: ValidationExperimentConfig):
    """Build the shared, batched validation DataLoader (identical for all runs).

    Returns ``(loader, val_ids)``. ``shuffle=False`` keeps the iteration order
    aligned with ``val_ids`` so per-sample results map back to record ids.
    """
    excluded = (
        Path(cfg.excluded_indexes_file) if cfg.excluded_indexes_file else None
    )
    record_ids = load_record_ids(Path(cfg.dataset_path), excluded)
    _, val_ids = train_val_split(
        record_ids,
        test_size=cfg.train_test_split,
        random_state=cfg.split_random_state,
    )
    if cfg.n_samples is not None and cfg.n_samples < len(val_ids):
        val_ids = val_ids[: cfg.n_samples]
        logger.warning("Capped val set to first %d samples (n_samples).", len(val_ids))

    dataset = H5PYGenerator(
        file_path=cfg.dataset_path,
        indexes=val_ids,
        augmentation=False,
        cropp=True,
        normalize=False,
        normalize_flux_only=cfg.normalize_flux_only,
        flux_mode=cfg.flux_mode,
        indexes_to_exclude_list=cfg.excluded_indexes_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate_h5,
    )
    logger.info(
        "Shared validation set: %d samples | batch_size=%d num_workers=%d",
        len(dataset), cfg.batch_size, cfg.num_workers,
    )
    return loader, val_ids


def load_run_model(
    run_dir: Path, checkpoint_fname: str, device: torch.device
) -> DoTA3D_v3:
    """Build a model from ``hyperparams.json`` and load a run's checkpoint.

    Handles both checkpoint layouts: a structured ``CheckpointManager`` snapshot
    (a dict with a ``model`` key, plus RNG/optimizer state) and a bare
    ``state_dict``. ``weights_only=False`` is required because the structured
    snapshot pickles NumPy / Python RNG state alongside the weights.
    """
    hyperparams_path = run_dir / "hyperparams.json"
    ckpt_path = run_dir / "checkpoints" / checkpoint_fname
    if not hyperparams_path.exists():
        raise FileNotFoundError(f"missing hyperparams.json in {run_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint {ckpt_path}")

    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)
    model = DoTA3D_v3(**hyperparams).to(device).eval()

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    return model


# ── Per-sample metrics ──────────────────────────────────────────────────────


def compute_sample_metrics(
    ctx,
    *,
    scale: dict,
    loss_mse_fn: LMSE,
    mape_mask_frac: float,
    range_dz_mm: float,
    range_oversample: int,
    range_min_peak_dose: float,
) -> SampleResult:
    """Compute all per-sample metrics for one :class:`InferenceContext`.

    Mirrors the de-normalization and MAPE masking used in training validation
    (:mod:`src.training.validation`). Range deltas are NaN when the GT curve is
    empty; the caller aggregates with nan-aware reducers.
    """
    # Normalized objective (shapes: y_pred (1,1,D,H,W), y (1,D,H,W) -> (1,1,..)).
    lmse = float(loss_mse_fn(ctx.y_pred, ctx.y.unsqueeze(0)).item())

    # De-normalize to physical units (MeV/g), then to Gy for the dose metrics.
    y_np, y_pred_np = denorm_pair(ctx.y, ctx.y_pred, scale)
    mask = y_pred_np > mape_mask_frac * float(np.max(y_pred_np))
    mape = (
        float(calculate_pure_mape(y_np[mask], y_pred_np[mask])) if mask.any() else 0.0
    )
    y_gt_gy = to_gy(y_np)
    y_pred_gy = to_gy(y_pred_np)
    rmse = float(calculate_rmse(y_pred_gy, y_gt_gy))
    rde = float(calculate_relative_dose_error(y_pred_gy, y_gt_gy))

    # Range fidelity on the laterally integrated depth-dose (beam axis).
    rm_pred = compute_range_metrics(
        integrated_depth_dose(np.squeeze(y_pred_gy)),
        range_dz_mm,
        oversample=range_oversample,
        min_peak_dose=range_min_peak_dose,
    )
    rm_gt = compute_range_metrics(
        integrated_depth_dose(np.squeeze(y_gt_gy)),
        range_dz_mm,
        oversample=range_oversample,
        min_peak_dose=range_min_peak_dose,
    )
    deltas = range_metric_deltas(rm_pred, rm_gt)

    return SampleResult(
        sample_id=ctx.sample_id,
        energy_mev=denormalize_energy(float(ctx.energy.item()), scale),
        mape_pct=mape,
        lmse=lmse,
        rmse_gy=rmse,
        rde_pct=rde,
        r80_err_mm=deltas["r80_delta_mm"],
        r20_err_mm=deltas["r20_delta_mm"],
        dfw_err_mm=deltas["dfw_delta_mm"],
    )


@torch.no_grad()
def evaluate_run(
    model: torch.nn.Module,
    loader: DataLoader,
    val_ids: List[str],
    device: torch.device,
    cfg: ValidationExperimentConfig,
    loss_mse_fn: LMSE,
    desc: str = "",
) -> List[SampleResult]:
    """Run batched inference and per-sample metrics for one model.

    The forward pass and H5 reads are batched (DataLoader workers); per-sample
    metrics are then computed by slicing the batch. The model is in eval mode,
    so outputs are batch-size-invariant (BatchNorm uses running stats).
    """
    model.eval()
    results: List[SampleResult] = []
    idx = 0
    total = len(loader)
    heartbeat = max(1, total // 10)
    for bi, (X, E, Y) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        E = E.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        out = model(X, E)[0]  # (B, 1, D, H, W); forward returns (dose, attention)

        for b in range(out.shape[0]):
            ctx = SimpleNamespace(
                sample_id=val_ids[idx],
                y=Y[b],  # (1, D, H, W)
                y_pred=out[b : b + 1],  # (1, 1, D, H, W)
                energy=E[b],  # (1,)
            )
            results.append(
                compute_sample_metrics(
                    ctx,
                    scale=cfg.scale,
                    loss_mse_fn=loss_mse_fn,
                    mape_mask_frac=cfg.mape_mask_frac,
                    range_dz_mm=cfg.range_dz_mm,
                    range_oversample=cfg.range_oversample,
                    range_min_peak_dose=cfg.range_min_peak_dose,
                )
            )
            idx += 1

        if (bi + 1) % heartbeat == 0 or bi == total - 1:
            logger.info("%s batch %d/%d (%d samples)", desc, bi + 1, total, idx)
    return results


def aggregate_run(results: List[SampleResult]) -> Dict[str, Dict[str, float]]:
    """Nan-aware mean/std per metric over a run's per-sample results."""
    agg: Dict[str, Dict[str, float]] = {}
    for attr, _header, _fmt in _METRIC_COLUMNS:
        vals = np.array([getattr(r, attr) for r in results], dtype=float)
        agg[attr] = {
            "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
            "std": float(np.nanstd(vals)) if vals.size else float("nan"),
            "n": int(np.count_nonzero(~np.isnan(vals))),
        }
    return agg


# ── Output ──────────────────────────────────────────────────────────────────


def _cell(stats: Dict[str, float], fmt: str) -> str:
    return f"{fmt.format(stats['mean'])} +/- {fmt.format(stats['std'])}"


def _save_per_sample_csv(results: List[SampleResult], path: Path) -> None:
    columns = [
        CsvColumn(name="sample_id", row=lambda r: r.sample_id),
        CsvColumn(
            name="energy_mev", row=lambda r: f"{r.energy_mev:.2f}",
            summary_extract=lambda r: r.energy_mev, summary_fmt=".2f",
        ),
    ]
    for attr, header, fmt in _METRIC_COLUMNS:
        columns.append(
            CsvColumn(
                name=header,
                row=(lambda a, f: (lambda r: f.format(getattr(r, a))))(attr, fmt),
                summary_extract=(lambda a: (lambda r: getattr(r, a)))(attr),
                summary_fmt=".6g",
            )
        )
    save_results_csv(results, path, columns, logger=logger)


# ── Log-based reporting (paper-faithful: min-val-MAPE epoch from metrics.jsonl) ─

# Selection metric -> the metrics.jsonl val key minimised to pick the epoch.
_SELECT_KEYS = {"val_mape": "mape_pct_mean", "val_loss": "loss_combined_mean"}

# Logged columns: (header, mean_key, std_key | None, value format).
_LOG_COLUMNS = [
    ("MAPE [%]", "mape_pct_mean", "mape_pct_std", "{:.3f}"),
    ("RDE [%]", "rde_pct_mean", "rde_pct_std", "{:.3f}"),
    ("RMSE [Gy]", "rmse_gy_mean", "rmse_gy_std", "{:.4e}"),
    ("LMSE", "loss_mse_mean", None, "{:.4e}"),  # no per-sample std logged
]


def select_epoch_from_logs(metrics_path: Path, select_by: str):
    """Pick the epoch minimising ``select_by`` from a run's metrics.jsonl.

    Returns ``(epoch, val_dict)`` where ``val_dict`` is that epoch's logged
    ``val`` block (metrics computed over the full validation set during
    training). Reproduces the published "checkpoint minimising validation MAPE"
    selection.
    """
    if select_by not in _SELECT_KEYS:
        raise ValueError(
            f"select_by must be one of {sorted(_SELECT_KEYS)}, got {select_by!r}"
        )
    key = _SELECT_KEYS[select_by]
    rows = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no metric rows in {metrics_path}")

    def _val(r):
        return r.get("val", {}).get(key, float("inf"))

    best = min(rows, key=_val)
    return int(best["epoch"]), best.get("val", {})


def _log_cell(val: Dict[str, Any], mean_key: str, std_key, fmt: str) -> str:
    """Format ``mean +/- std`` (or ``mean``) for a logged metric, or ``n/a``."""
    mean = val.get(mean_key)
    if mean is None:
        return "n/a"
    if std_key is None:
        return fmt.format(mean)
    std = val.get(std_key)
    return f"{fmt.format(mean)} +/- {fmt.format(std)}" if std is not None else fmt.format(mean)


# ── Output helpers ──────────────────────────────────────────────────────────


def _save_comparison(
    run_dir: Path,
    row_labels: List[str],
    headers: List[str],
    cells: List[List[str]],
    *,
    caption: str,
) -> None:
    """Render the table to the log and save comparison.md + comparison.csv."""
    table = render_comparison_table(row_labels, headers, cells, row_header="Run")
    logger.info("\n%s", table)
    (run_dir / "comparison.md").write_text(f"# {caption}\n\n" + table + "\n")
    with open(run_dir / "comparison.csv", "w") as f:
        f.write(",".join(["run", *headers]) + "\n")
        for label, row in zip(row_labels, cells):
            f.write(",".join([label, *[c.replace(",", ";") for c in row]]) + "\n")
    logger.info("Comparison table : %s", run_dir / "comparison.md")
    logger.info("Comparison CSV   : %s", run_dir / "comparison.csv")


# ── CLI ─────────────────────────────────────────────────────────────────────


@app.command()
def main(
    config: Annotated[
        Optional[Path], typer.Option(help="Path to the validation YAML config.")
    ] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index. -1 for CPU.")
    ] = None,
    n_samples: Annotated[
        Optional[int], typer.Option(help="Cap the val set to the first N samples.")
    ] = None,
    verbose: Annotated[bool, typer.Option(help="DEBUG-level logging.")] = False,
) -> None:
    if config is None:
        raise typer.BadParameter("--config is required.")
    cfg = _build_config(config)
    if device_index is not None:
        cfg.device_index = device_index
    if n_samples is not None:
        cfg.n_samples = n_samples
    if not cfg.runs:
        raise typer.BadParameter("at least one run must be listed under 'runs'.")
    if cfg.report_mode not in ("logs", "inference"):
        raise typer.BadParameter("report_mode must be 'logs' or 'inference'.")

    run_dir = setup_run_directory(
        Path(cfg.output_dir), prefix=f"{cfg.experiment_name}_", subdirs=()
    )
    setup_logging(run_dir, verbose=verbose, log_filename="validation.log")
    if config.exists():
        shutil.copy2(config, run_dir / "config_input.yaml")

    logger.info("=" * 70)
    logger.info("ADoTA cross-run validation experiment (report_mode=%s)", cfg.report_mode)
    logger.info("Output dir : %s", run_dir)
    logger.info("Runs       : %d", len(cfg.runs))
    logger.info("=" * 70)

    if cfg.report_mode == "logs":
        _run_logs_mode(cfg, run_dir)
    else:
        _run_inference_mode(cfg, run_dir, verbose=verbose)

    logger.info("Run directory    : %s", run_dir)


def _run_logs_mode(cfg: ValidationExperimentConfig, run_dir: Path) -> None:
    """Paper-faithful reporting: logged metrics at the min-val-MAPE epoch."""
    headers = ["Epoch", "N", *[h for h, *_ in _LOG_COLUMNS]]
    row_labels: List[str] = []
    cells: List[List[str]] = []

    for run in cfg.runs:
        rd = Path(run.run_dir)
        _check_split_consistency(cfg, run, rd)
        epoch, val = select_epoch_from_logs(rd / "metrics.jsonl", cfg.select_by)
        logger.info(
            "[%s] selected epoch %d (min %s): MAPE=%s RDE=%s",
            run.name, epoch, cfg.select_by,
            val.get("mape_pct_mean"), val.get("rde_pct_mean"),
        )
        row_labels.append(run.name)
        cells.append(
            [str(epoch), str(val.get("n_samples", "?"))]
            + [_log_cell(val, mk, sk, fmt) for _h, mk, sk, fmt in _LOG_COLUMNS]
        )

    _save_comparison(
        run_dir, row_labels, headers, cells,
        caption=(
            f"ADoTA ablation (logged metrics at the epoch minimising {cfg.select_by}; "
            "cells = mean +/- std over the full validation set)"
        ),
    )


def _run_inference_mode(
    cfg: ValidationExperimentConfig, run_dir: Path, *, verbose: bool
) -> None:
    """Fresh batched inference on each run's checkpoint (adds range errors)."""
    if not cfg.dataset_path:
        raise typer.BadParameter("dataset_path is required for inference mode.")
    device = resolve_device(cfg.device_index)
    logger.info("Device     : %s", device)

    val_loader, val_ids = _build_val_loader(cfg)
    loss_mse_fn = LMSE()
    headers = [h for _a, h, _f in _METRIC_COLUMNS]
    row_labels: List[str] = []
    cells: List[List[str]] = []

    for run in cfg.runs:
        rd = Path(run.run_dir)
        _check_split_consistency(cfg, run, rd)
        ckpt_name = run.checkpoint_fname or cfg.checkpoint_fname
        logger.info("[%s] loading %s", run.name, rd / "checkpoints" / ckpt_name)
        model = load_run_model(rd, ckpt_name, device)

        results = evaluate_run(
            model, val_loader, val_ids, device, cfg, loss_mse_fn, desc=f"[{run.name}]"
        )
        agg = aggregate_run(results)
        row_labels.append(run.name)
        cells.append([_cell(agg[attr], fmt) for attr, _h, fmt in _METRIC_COLUMNS])
        _save_per_sample_csv(results, run_dir / f"per_sample_{run.name}.csv")
        logger.info("[%s] done: %d samples", run.name, len(results))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _save_comparison(
        run_dir, row_labels, headers, cells,
        caption=(
            f"ADoTA validation comparison ({len(val_ids)} samples, checkpoint="
            f"{cfg.checkpoint_fname}; cells = mean +/- std)"
        ),
    )


if __name__ == "__main__":
    app()
