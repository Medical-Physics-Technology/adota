"""Script to run HPTC model with CT images, treatment plan, and configuration."""

import logging
import sys
from datetime import datetime
from pathlib import Path

from pprint import pprint

import pydicom
import typer

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcm.load_data import list_all_files
from loaders.plan_parser import parse_plan

app = typer.Typer(
    help="Run HPTC model with CT images, treatment plan, and configuration."
)

# Setup logging
RUNS_DIR = Path(__file__).parent.parent / "runs"


def setup_logging() -> Path:
    """Setup logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "logs.out"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def parse_config_file(config_path: Path) -> dict:
    """Parse configuration file and return a dictionary of parameters.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary with configuration parameters.
    """
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Split on first whitespace, handle inline comments
            parts = line.split("#")[0].strip()  # Remove inline comments
            if not parts:
                continue

            # Split key and value
            tokens = parts.split(None, 1)  # Split on whitespace, max 2 parts
            if len(tokens) >= 2:
                key = tokens[0]
                value = tokens[1].strip()
                config[key] = value
            elif len(tokens) == 1:
                config[tokens[0]] = None

    return config


def load_ct_from_dicom_dir(dicom_dir: Path, logger: logging.Logger) -> list:
    """Load CT images from a directory containing DICOM files.

    Args:
        dicom_dir: Path to directory containing DICOM CT files.
        logger: Logger instance.

    Returns:
        List of pydicom Dataset objects sorted by instance number.
    """
    file_lists = list_all_files(str(dicom_dir), maxDepth=0)
    dicom_files = file_lists.get("Dicom", [])

    if not dicom_files:
        logger.error(f"No DICOM files found in {dicom_dir}")
        return []

    ct_slices = []
    corrupted_files = []

    for file_path in dicom_files:
        try:
            dcm = pydicom.dcmread(file_path)
            modality = getattr(dcm, "Modality", "Unknown")
            if modality == "CT":
                ct_slices.append(dcm)
            else:
                logger.debug(f"Skipping non-CT file {file_path} (Modality: {modality})")
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            corrupted_files.append(file_path)

            # Try to read with force=True
            try:
                dcm_partial = pydicom.dcmread(file_path, force=True)
                logger.info("=" * 60)
                logger.info(f"PARTIAL DICOM HEADER for {file_path}:")
                logger.info("=" * 60)
                logger.info(f"\n{dcm_partial}")
                logger.info("=" * 60)
            except Exception as e2:
                logger.error(f"Could not read even partial header: {e2}")

    # Sort by InstanceNumber if available
    ct_slices.sort(key=lambda x: getattr(x, "InstanceNumber", 0))

    logger.info(f"Loaded {len(ct_slices)} CT slices from {dicom_dir}")
    if corrupted_files:
        logger.warning(f"Corrupted/problematic files: {len(corrupted_files)}")

    return ct_slices


@app.command()
def run(
    ct_dir: Path = typer.Argument(
        ...,
        help="Path to the directory containing CT DICOM files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    plan_file: Path = typer.Argument(
        ...,
        help="Path to the PlanPencil treatment plan file (.txt).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    config_file: Path = typer.Argument(
        ...,
        help="Path to the configuration file (.txt).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Run HPTC model with CT images, treatment plan, and configuration."""
    log_file = setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)
    logger.info("HPTC Model Run")
    logger.info("=" * 60)

    # Log input files
    logger.info(f"\nInput CT directory: {ct_dir}")
    logger.info(f"Input plan file: {plan_file}")
    logger.info(f"Input config file: {config_file}")
    logger.info("-" * 50)

    # Parse configuration file
    logger.info("\nParsing configuration file...")
    config = parse_config_file(config_file)
    logger.info(f"Configuration parameters loaded: {len(config)} parameters")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Parse treatment plan using the new parser
    logger.info("\nParsing treatment plan file...")
    plan = parse_plan(str(plan_file))
    logger.info(f"Plan name: {plan.name}")
    logger.info(f"Number of treatment fractions: {plan.n_treatment_fractions}")
    logger.info(f"Number of fraction definitions: {len(plan.fractions)}")
    logger.info(f"Total meterset weight: {plan.total_msw}")

    # Count total fields across all fractions
    total_fields = sum(len(fr.fields) for fr in plan.fractions)
    logger.info(f"Number of fields: {total_fields}")

    for fraction in plan.fractions:
        logger.info(f"\n  Fraction {fraction.id}:")
        for fld in fraction.fields:
            logger.info(f"\n    Field {fld.id}:")
            logger.info(f"      Gantry angle: {fld.gantry_angle}°")
            logger.info(f"      Couch angle: {fld.couch_angle}°")
            logger.info(f"      Isocenter position: {fld.isocenter}")
            logger.info(f"      Number of control points: {len(fld.control_points)}")
            logger.info(f"      Final cumulative MSW: {fld.final_cumulative_msw}")

            if fld.rs_id:
                logger.info(f"      Range shifter ID: {fld.rs_id}")
                logger.info(f"      Range shifter type: {fld.rs_type}")

            # Summary of control points
            if fld.control_points:
                energies = [cp.energy_mev for cp in fld.control_points]
                total_spots = sum(len(cp.spots) for cp in fld.control_points)
                logger.info(
                    f"      Energy range: {min(energies):.1f} - {max(energies):.1f} MeV"
                )
                logger.info(f"      Total spots: {total_spots}")

    # Only single fraction plans are supported
    sim_res_dicts = []
    for fr in plan.fractions:
        for field in fr.fields:
            for control_point in field.control_points:
                for spot in control_point.spots:
                    _field = {}
                    _field["simulation_log"] = {}
                    _field["simulation_log"]["gantry_angle"] = (-1) * (
                        field.gantry_angle - 90
                    )  # Adjusting to the coordinate system
                    _field["simulation_log"]["isocenter"] = field.isocenter
                    _field["simulation_log"]["energy"] = [control_point.energy_mev]
                    _field["simulation_log"]["bixelgrid_shifts_xy"] = [[spot.x, spot.y]]
                    _field["simulation_log"]["relative_weight"] = (
                        spot.weight / plan.total_msw
                    )

                    sim_res_dicts.append(_field)

    

    pprint(sim_res_dicts[0])
    pprint(sim_res_dicts[-1])
    print(f"\nTotal spots processed: {len(sim_res_dicts)}")

    # Load CT images
    logger.info("\nLoading CT images...")
    ct_slices = load_ct_from_dicom_dir(ct_dir, logger)

    if ct_slices:
        # Display CT info
        first_slice = ct_slices[0]
        logger.info(f"\nCT Image Information:")
        logger.info(f"  Patient ID: {getattr(first_slice, 'PatientID', 'Unknown')}")
        logger.info(f"  Patient Name: {getattr(first_slice, 'PatientName', 'Unknown')}")
        logger.info(f"  Study Date: {getattr(first_slice, 'StudyDate', 'Unknown')}")
        logger.info(
            f"  Rows x Columns: {getattr(first_slice, 'Rows', 'Unknown')} x {getattr(first_slice, 'Columns', 'Unknown')}"
        )
        logger.info(
            f"  Pixel Spacing: {getattr(first_slice, 'PixelSpacing', 'Unknown')}"
        )
        logger.info(
            f"  Slice Thickness: {getattr(first_slice, 'SliceThickness', 'Unknown')}"
        )
        logger.info(f"  Number of slices: {len(ct_slices)}")

    logger.info("\n" + "=" * 60)
    logger.info("Input parsing complete. Ready for model execution.")
    logger.info("=" * 60)
    logger.info(f"\nLog saved to: {log_file}")


if __name__ == "__main__":
    app()
