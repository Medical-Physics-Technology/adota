"""Regenerate the established CT+Sobel and 3D-dose figures for the worst
range-error beamlets.

Given a completed advanced-metrics run (its ``results.csv``), this selects the
top-N beamlets by |ΔR80| and re-renders, for each, the two publication figures
the analysis already knows how to make:

  * ``<id>_ct_seg.png`` -- CT (+ smoothed CT, tissue segmentation) with the
    Sobel edge-magnitude map and the Bragg-peak range band, so you can see
    whether a strong interface sits on the beam path;
  * ``<id>_E<energy>MeV.svg`` -- the MC-vs-ADoTA 3D dose distribution
    (axial + sagittal + BEV + IDD with Bragg-peak markers).

It reuses :func:`src.figures.advanced_metrics.generate_figures_for_records`
verbatim, so these are identical in style to the paper figures; only the
selection (worst range error, rather than best/worst GPR) is new.

Usage:
    uv run python scripts/range_failure_diagnostics.py \\
        --run-dir /scratch/mstryja/adota_runs/20260702_175724 \\
        --config scripts/config_analysis_advanced_metrics.yaml
"""

import logging
from dataclasses import fields
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer

from src.adota.config import DEFAULT_GAMMA_PARAMS, load_yaml_config, setup_logging
from src.adota.utils import load_model
from src.evaluation.cli import resolve_device
from src.figures.advanced_metrics import generate_figures_for_records
from src.loaders.generator import H5PYGenerator
from src.schemas.configs import AdvancedAnalysisConfig as AnalysisConfig
from src.schemas.results import SampleRecord

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)
app = typer.Typer(help="Regenerate CT+Sobel and 3D-dose figures for worst-range beamlets")


def _records_from_csv(df: pd.DataFrame) -> list[SampleRecord]:
    """Rebuild SampleRecord objects from a results.csv slice (typed by field)."""
    field_types = {f.name: f.type for f in fields(SampleRecord)}
    records: list[SampleRecord] = []
    for _, row in df.iterrows():
        kwargs = {}
        for name in field_types:
            if name not in row:
                continue
            val = row[name]
            kwargs[name] = int(val) if name == "n_density_regions" else val
        kwargs["sample_id"] = str(row["sample_id"])
        records.append(SampleRecord(**kwargs))
    return records


def _build_config(yaml_config: dict) -> AnalysisConfig:
    """AnalysisConfig with the fields the figure generator reads (YAML-overridable)."""
    gamma_params = {**DEFAULT_GAMMA_PARAMS, **(yaml_config.get("gamma_params") or {})}
    return AnalysisConfig(
        resolution=tuple(yaml_config.get("resolution", [2.0, 2.0, 2.0])),
        smoothing_method=yaml_config.get("smoothing_method", "gaussian"),
        smoothing_sigma=yaml_config.get("smoothing_sigma", 1.0),
        proximal_fraction=yaml_config.get("proximal_fraction", 0.50),
        fall_fraction=yaml_config.get("fall_fraction", 0.10),
        sobel_use_raw=yaml_config.get("sobel_use_raw", False),
        gamma_params=gamma_params,
    )


@app.command()
def main(
    run_dir: Annotated[
        Path, typer.Option(help="Completed advanced-metrics run dir (has results.csv)")
    ],
    config: Annotated[
        Optional[Path], typer.Option(help="Advanced-metrics YAML (model / h5 / params)")
    ] = None,
    h5_path: Annotated[Optional[Path], typer.Option(help="Override HDF5 path")] = None,
    model_name: Annotated[Optional[str], typer.Option(help="Override model dir name")] = None,
    model_fname: Annotated[Optional[str], typer.Option(help="Model weights filename")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA device (-1 CPU)")] = None,
    n_worst: Annotated[int, typer.Option(help="Number of worst-|ΔR80| beamlets")] = 8,
    sample_ids: Annotated[
        Optional[str], typer.Option(help="Comma-separated ids (overrides n_worst)")
    ] = None,
) -> None:
    """Render CT+Sobel and 3D-dose diagnostics for the worst-range beamlets."""
    yaml_config = load_yaml_config(config) if config is not None else {}
    h5_path = Path(h5_path or yaml_config["h5_path"])
    model_name = model_name or yaml_config.get("model_name")
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )

    out_dir = run_dir / "figures" / "range_diagnostics" / "composite"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir, verbose=False, log_filename="range_diagnostics.log")

    # ── Select the worst-range beamlets ─────────────────────────────────
    df = pd.read_csv(run_dir / "results.csv")
    if sample_ids:
        ids = [s.strip() for s in sample_ids.split(",") if s.strip()]
        sel = df[df["sample_id"].isin(ids)].copy()
    else:
        order = df["r80_delta_mm"].abs().sort_values(ascending=False).index
        sel = df.loc[order].head(n_worst).copy()
    ids = [str(s) for s in sel["sample_id"].tolist()]
    logger.info("Worst-range beamlets (|ΔR80|): %s",
                list(zip(ids, sel["r80_delta_mm"].round(1).tolist())))

    # ── Model + dataset (match the analysis run's loader settings) ──────
    device = resolve_device(device_index)
    model = load_model(
        PROJECT_ROOT / "models" / model_name / model_fname,
        PROJECT_ROOT / "models" / model_name / "hyperparams.json",
        device,
    )
    dataset = H5PYGenerator(
        file_path=str(h5_path), indexes=ids, augmentation=False,
        cropp=True, normalize=False, normalize_flux_only=True,
    )
    # Rebuild records in dataset order so id -> index lookup is consistent.
    order_map = {rid: i for i, rid in enumerate(dataset.record_ids)}
    sel = sel.assign(_ord=sel["sample_id"].map(order_map)).sort_values("_ord")
    records = _records_from_csv(sel)

    config_obj = _build_config(yaml_config)
    generate_figures_for_records(
        model=model,
        records=records,
        record_ids=list(dataset.record_ids),
        dataset=dataset,
        config=config_obj,
        device=device,
        output_dir=out_dir,
        category_label="worst_range",
    )
    logger.info("Done. Figures in %s", out_dir)


if __name__ == "__main__":
    app()
