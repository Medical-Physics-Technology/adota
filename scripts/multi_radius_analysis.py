"""
Multi-Radius Heterogeneity Analysis

For every beamlet in an existing results.csv, loads the CT grid from
HDF5, reuses the Bragg peak location recorded in the CSV, and computes
σ_HU and TV at multiple sphere radii.

The output is a new results CSV that contains all original columns plus
additional columns ``sigma_hu_{r}mm`` and ``tv_{r}mm`` for each radius.

No model inference is performed; this is a pure CT texture analysis.

Usage:
    uv run python scripts/multi_radius_analysis.py \\
        --config scripts/config_multi_radius_analysis.yaml
"""

import csv
import logging
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import h5py
import numpy as np
import typer
from tqdm import tqdm

# ── Project root ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training_set_analysis import (
    _extract_sphere_voxels,
    compute_bp_sigma_hu,
    compute_bp_tv,
)
from src.adota.config import (
    DEFAULT_SCALE,
    load_yaml_config,
    setup_logging,
    setup_run_directory,
)
from src.augmentation.geo_augmenations import cropp_around_index
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)

app = typer.Typer(help="Multi-radius σ_HU / TV analysis")


# ── CSV I/O ─────────────────────────────────────────────────────────────────


def load_input_csv(csv_path: Path) -> list[dict]:
    """Load the source results.csv, skipping summary rows at the bottom."""
    rows: list[dict] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Summary rows have empty sample_id or are statistics labels
            sid = row.get("sample_id", "").strip()
            if not sid or sid in ("mean", "std", "min", "max", ""):
                continue
            rows.append(row)
    return rows


def save_output_csv(
    rows: list[dict],
    radii: list[float],
    output_path: Path,
) -> None:
    """Write the enriched CSV with multi-radius columns appended."""
    if not rows:
        logger.warning("No rows to save")
        return

    # Build fieldnames: original columns + new radius columns
    original_fields = [
        "sample_id",
        "energy_mev",
        "bp_depth",
        "bp_y",
        "bp_x",
        "sigma_hu",
        "tv",
        "cv",
        "label",
        "gpr_pct",
        "rmse_gy",
        "mape_pct",
        "rde_pct",
        "calc_time_s",
    ]
    radius_fields = []
    for r in radii:
        tag = f"{r:.0f}mm" if r == int(r) else f"{r}mm"
        radius_fields.append(f"sigma_hu_{tag}")
        radius_fields.append(f"tv_{tag}")
    fieldnames = original_fields + radius_fields

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logger.info(f"Results CSV saved to {output_path} ({len(rows)} rows)")


# ── Core computation ────────────────────────────────────────────────────────


def compute_multi_radius_metrics(
    ct_hu: np.ndarray,
    bp_idx: tuple[int, int, int],
    radii: list[float],
    resolution: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> dict[str, float]:
    """Compute σ_HU and TV for multiple sphere radii at one Bragg peak.

    Returns a dict with keys like ``sigma_hu_5mm``, ``tv_5mm``, etc.
    """
    result: dict[str, float] = {}
    for r in radii:
        tag = f"{r:.0f}mm" if r == int(r) else f"{r}mm"
        result[f"sigma_hu_{tag}"] = compute_bp_sigma_hu(ct_hu, bp_idx, r, resolution)
        result[f"tv_{tag}"] = compute_bp_tv(ct_hu, bp_idx, r, resolution)
    return result


def process_beamlets(
    rows: list[dict],
    h5_path: Path,
    radii: list[float],
    resolution: tuple[float, float, float],
    scale: dict,
    show_progress: bool = True,
) -> list[dict]:
    """Enrich each CSV row with multi-radius σ_HU and TV.

    Opens the HDF5 file once and iterates over all rows, loading CT
    data per sample, denormalising to HU, and computing sphere metrics.
    """
    enriched: list[dict] = []
    n_skipped = 0

    iterator = (
        tqdm(rows, desc="Computing multi-radius metrics") if show_progress else rows
    )

    with h5py.File(h5_path, "r") as ds:
        available_ids = set(ds.keys())

        for row in iterator:
            sample_id = row["sample_id"]

            if sample_id not in available_ids:
                logger.warning(f"Sample {sample_id} not found in HDF5, skipping")
                n_skipped += 1
                continue

            # Load CT and dose grids from HDF5 (raw shape: H, W, D)
            record = ds[sample_id]
            ct_raw = record["ct"][:]
            dose_raw = record["dose"][:]
            flux_raw = record["flux"][:]
            initial_energy = float(record.attrs["initial_energy"])

            # Crop to (30, 30, 160) around the Bragg peak, matching
            # the H5PYGenerator with cropp=True / augmentation=False.
            ct_cropped, _, _, _ = cropp_around_index(
                ct_raw, flux_raw, dose_raw, initial_energy
            )

            # Permute to (D, H, W) = (160, 30, 30), same as generator
            ct_grid = np.transpose(ct_cropped, (2, 0, 1))

            # Denormalise to Hounsfield Units.
            # The HDF5 stores normalised CT in [0, 1]; the generator
            # with normalize=False does NOT re-scale, but
            # training_set_analysis.py calls inverse_minmax on x[0]
            # (the CT channel) to recover HU.
            ct_hu = inverse_minmax(
                ct_grid.astype(np.float32),
                scale["min_ct"],
                scale["max_ct"],
            )

            # Bragg peak location from CSV
            bp_idx = (
                int(row["bp_depth"]),
                int(row["bp_y"]),
                int(row["bp_x"]),
            )

            # Compute σ_HU and TV at all radii
            metrics = compute_multi_radius_metrics(ct_hu, bp_idx, radii, resolution)

            # Format new columns
            for key, val in metrics.items():
                row[key] = f"{val:.4f}"

            enriched.append(row)

            if show_progress:
                r_str = ", ".join(
                    f"{metrics[f'sigma_hu_{r:.0f}mm' if r == int(r) else f'sigma_hu_{r}mm']:.0f}"
                    for r in radii
                )
                iterator.set_postfix_str(f"σ=[{r_str}]")

    if n_skipped:
        logger.info(f"Skipped {n_skipped}/{len(rows)} samples (not found in HDF5)")

    return enriched


# ── CLI ─────────────────────────────────────────────────────────────────────


@app.command()
def main(
    results_csv: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the source results.csv from a previous run"),
    ] = None,
    h5_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the HDF5 dataset file"),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML configuration file"),
    ] = None,
    radii: Annotated[
        Optional[str],
        typer.Option(
            help="Comma-separated list of sphere radii in mm " "(e.g. '5,10,15,20,30')"
        ),
    ] = None,
    no_progress: Annotated[
        Optional[bool], typer.Option(help="Disable progress bar")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose output")
    ] = None,
) -> None:
    """Compute σ_HU and TV at multiple radii for every beamlet.

    Reads an existing results.csv (from training_set_analysis.py),
    loads CT data from the HDF5 dataset, and appends per-radius
    heterogeneity columns. No model inference is performed.

    Can be configured via CLI arguments, a YAML config file (--config),
    or both. CLI arguments take precedence.
    """
    # ── Load & merge config ─────────────────────────────────────────────
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    results_csv = results_csv or (
        Path(yaml_config["results_csv"]) if "results_csv" in yaml_config else None
    )
    h5_path = h5_path or (
        Path(yaml_config["h5_path"]) if "h5_path" in yaml_config else None
    )

    # Parse radii
    radii_list: list[float]
    if radii is not None:
        radii_list = [float(r.strip()) for r in radii.split(",")]
    elif "radii" in yaml_config:
        radii_list = [float(r) for r in yaml_config["radii"]]
    else:
        radii_list = [5.0, 10.0, 15.0, 20.0, 30.0]

    resolution = tuple(yaml_config.get("resolution", [2.0, 2.0, 2.0]))
    scale = yaml_config.get("scale", DEFAULT_SCALE.copy())

    no_progress = (
        no_progress
        if no_progress is not None
        else yaml_config.get("no_progress", False)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    # ── Validate ────────────────────────────────────────────────────────
    if results_csv is None:
        raise typer.BadParameter(
            "RESULTS_CSV is required (via CLI argument or YAML config)"
        )
    if h5_path is None:
        raise typer.BadParameter(
            "H5_PATH is required (via CLI argument or YAML config)"
        )
    if not results_csv.exists():
        raise typer.BadParameter(f"Results CSV not found: {results_csv}")
    if not h5_path.exists():
        raise typer.BadParameter(f"HDF5 file not found: {h5_path}")

    # ── Setup run directory & logging ───────────────────────────────────
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir, subdirs=())
    log_file = setup_logging(
        run_dir, verbose=verbose, log_filename="multi_radius_analysis.log"
    )

    if config_path is not None:
        shutil.copy2(config_path, run_dir / config_path.name)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Log file:      {log_file}")

    # ── Log configuration ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RUN CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Source CSV:  {results_csv}")
    logger.info(f"HDF5 file:   {h5_path}")
    logger.info(f"Radii [mm]:  {radii_list}")
    logger.info(f"Resolution:  {resolution}")
    logger.info(f"CT scale:    min={scale['min_ct']}, max={scale['max_ct']}")
    logger.info("=" * 60)

    # ── Load source CSV ─────────────────────────────────────────────────
    rows = load_input_csv(results_csv)
    logger.info(f"Loaded {len(rows)} beamlets from {results_csv}")

    if not rows:
        logger.error("No data rows found in the source CSV")
        raise typer.Exit(code=1)

    # ── Compute ─────────────────────────────────────────────────────────
    start_time = perf_counter()

    enriched = process_beamlets(
        rows=rows,
        h5_path=h5_path,
        radii=radii_list,
        resolution=resolution,
        scale=scale,
        show_progress=not no_progress,
    )

    elapsed = perf_counter() - start_time
    logger.info(f"Computation finished in {elapsed:.1f}s")

    # ── Save output ─────────────────────────────────────────────────────
    output_csv = run_dir / "results.csv"
    save_output_csv(enriched, radii_list, output_csv)

    # ── Summary statistics ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Beamlets processed: {len(enriched)}")
    logger.info(f"Total time:         {elapsed:.1f}s")
    logger.info(f"Per beamlet:        {1000.0 * elapsed / max(len(enriched), 1):.1f}ms")
    logger.info("")

    for r in radii_list:
        tag = f"{r:.0f}mm" if r == int(r) else f"{r}mm"
        sigmas = [float(row[f"sigma_hu_{tag}"]) for row in enriched]
        tvs = [float(row[f"tv_{tag}"]) for row in enriched]
        logger.info(
            f"  r = {r:5.1f} mm  │  "
            f"σ_HU = {np.mean(sigmas):8.2f} ± {np.std(sigmas):8.2f}  │  "
            f"TV = {np.mean(tvs):10.2f} ± {np.std(tvs):10.2f}"
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Done! Results saved to: {run_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
