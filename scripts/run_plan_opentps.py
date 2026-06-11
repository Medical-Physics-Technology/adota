"""End-to-end ADoTA plan pipeline CLI (skeleton).

Given an OpenTPS plan directory this script will eventually extract per-spot
ADoTA inputs, run inference, accumulate the predicted dose, and compare it
against the MCsquare reference (see ``docs/beamlet_extraction_integration_plan.md``).

This first iteration wires up the foundation only:

1. load the ADoTA model into memory,
2. load the plan directory (CT grid + contours via SimpleITK, ``PlanPencil.txt``,
   beam data library, ``config.txt``, MC reference dose path),
3. parse the plan, and
4. print a preview of everything that was loaded.

The downstream stages (``--stages extract,infer,accumulate,compare``) are added
in later tasks; for now the script stops after the preview.

Usage:
    uv run python scripts/run_plan_opentps.py \\
        --config scripts/config_run_plan_opentps.yaml
"""

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import pydicom
import typer

# Add the project root to the path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import load_yaml_config, setup_logging, setup_run_directory
from src.adota.utils import load_model
from src.dcm.load_data import list_all_files
from src.evaluation.cli import resolve_device
from src.loaders.plan_directory import load_plan_directory

logger = logging.getLogger(__name__)

app = typer.Typer(help="End-to-end ADoTA plan pipeline (OpenTPS plan directory).")

# Decision: all heavy outputs live on /scratch, never under /home (see the
# integration plan, section 2).
DEFAULT_RUNS_DIR = Path("/scratch/mstryja/tmp_adota/runs")


def load_ct_from_dicom_dir(dicom_dir: Path, logger: logging.Logger) -> list:
    """Load CT images from a directory of DICOM files (DORMANT).

    Kept from the original HPTC script for the DICOM-input path. The plan
    pipeline reads ``CT.mhd`` (the grid MCsquare simulated on) instead, so this
    is currently unused; retained for plans delivered as DICOM CT.

    Args:
        dicom_dir: Directory containing DICOM CT files.
        logger: Logger instance.

    Returns:
        CT ``pydicom`` datasets sorted by instance number.
    """
    file_lists = list_all_files(str(dicom_dir), maxDepth=0)
    dicom_files = file_lists.get("Dicom", [])

    if not dicom_files:
        logger.error(f"No DICOM files found in {dicom_dir}")
        return []

    ct_slices = []
    for file_path in dicom_files:
        try:
            dcm = pydicom.dcmread(file_path)
            if getattr(dcm, "Modality", "Unknown") == "CT":
                ct_slices.append(dcm)
        except Exception as exc:
            logger.warning(f"Could not read {file_path}: {exc}")

    ct_slices.sort(key=lambda x: getattr(x, "InstanceNumber", 0))
    logger.info(f"Loaded {len(ct_slices)} CT slices from {dicom_dir}")
    return ct_slices


@app.command()
def main(
    plan_dir: Annotated[
        Optional[Path],
        typer.Option(help="OpenTPS plan directory (CT.mhd, PlanPencil.txt, ...)."),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML configuration file."),
    ] = None,
    model_name: Annotated[
        Optional[str], typer.Option(help="Model directory name under models/.")
    ] = None,
    model_fname: Annotated[
        Optional[str], typer.Option(help="Model weights filename.")
    ] = None,
    bdl_path: Annotated[
        Optional[Path],
        typer.Option(help="Beam data library path (default: plan-local bdl.txt)."),
    ] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index (-1 for CPU).")
    ] = None,
    max_fields: Annotated[
        Optional[int], typer.Option(help="Fields to expand per fraction in preview.")
    ] = None,
    max_control_points: Annotated[
        Optional[int],
        typer.Option(help="Control points to expand per field in preview."),
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose/debug logging.")
    ] = None,
) -> None:
    """Load the model + plan directory, parse the plan, and print a preview.

    Configurable via CLI flags, a YAML config (``--config``), or both. CLI
    flags take precedence over YAML values.
    """
    # Load YAML config if provided.
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    # Merge: CLI overrides YAML, YAML overrides defaults.
    plan_dir = plan_dir if plan_dir is not None else yaml_config.get("plan_dir")
    model_name = model_name or yaml_config.get("model_name")
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    bdl_path = bdl_path if bdl_path is not None else yaml_config.get("bdl_path")
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )
    max_fields = (
        max_fields if max_fields is not None else yaml_config.get("max_fields", 2)
    )
    max_control_points = (
        max_control_points
        if max_control_points is not None
        else yaml_config.get("max_control_points", 3)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    # Validate required arguments.
    if plan_dir is None:
        raise typer.BadParameter("PLAN_DIR is required (via --plan-dir or YAML config)")
    if model_name is None:
        raise typer.BadParameter(
            "MODEL_NAME is required (via --model-name or YAML config)"
        )

    plan_dir = Path(plan_dir)
    bdl_path = Path(bdl_path) if bdl_path is not None else None

    # Setup run directory and logging on /scratch.
    runs_dir = Path(yaml_config.get("runs_dir", DEFAULT_RUNS_DIR))
    run_dir = setup_run_directory(runs_dir, prefix="plan_", subdirs=())
    log_file = setup_logging(run_dir, verbose=verbose, log_filename="run.log")
    if config_path is not None:
        import shutil

        shutil.copy2(config_path, run_dir / config_path.name)

    logger.info("=" * 70)
    logger.info("ADoTA plan pipeline")
    logger.info("=" * 70)
    logger.info("Run directory : %s", run_dir)
    logger.info("Log file      : %s", log_file)
    logger.info("Plan directory: %s", plan_dir)

    # --- Stage 0a: load the model into memory --------------------------------
    model_hub = PROJECT_ROOT / "models"
    model_path = model_hub / model_name / model_fname
    hyperparams_path = model_hub / model_name / "hyperparams.json"

    device = resolve_device(device_index)
    logger.info("Device        : %s", device)
    logger.info("Loading model : %s", model_path)
    model = load_model(model_path, hyperparams_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded  : %s (%s parameters)", type(model).__name__, f"{n_params:,}"
    )

    # --- Stage 0b: load + parse the plan directory ---------------------------
    logger.info("Loading plan directory ...")
    plan_directory = load_plan_directory(plan_dir, bdl_path=bdl_path)

    # --- Stage 0c: preview ---------------------------------------------------
    preview = plan_directory.summary(
        max_fields=max_fields, max_control_points=max_control_points
    )
    logger.info("\n%s\n%s\n%s", "-" * 70, preview, "-" * 70)

    logger.info("=" * 70)
    logger.info("Plan loaded and parsed. Downstream stages not yet implemented.")
    logger.info("=" * 70)


if __name__ == "__main__":
    app()
