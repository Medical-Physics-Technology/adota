"""
Texture analysis utilities.

Shared helpers for the CT texture analysis pipeline:
- Run directory creation
- Logging setup
- Image format enum & file discovery
- Configuration loading
"""

import logging
import shutil
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
import typer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image format
# ---------------------------------------------------------------------------


class ImageFormat(str, Enum):
    """Supported CT image file formats."""

    MHD = "mhd"
    DCM = "dcm"
    NPY = "npy"


EXTENSION_MAP: dict[ImageFormat, list[str]] = {
    ImageFormat.MHD: [".mhd"],
    ImageFormat.DCM: [".dcm", ".dicom"],
    ImageFormat.NPY: [".npy"],
}


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------


def setup_run_directory(runs_dir: Path) -> Path:
    """Create a timestamped run directory for texture analysis results.

    Args:
        runs_dir: Base directory for all runs.

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"texture_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, verbose: bool = False) -> Path:
    """Configure logging to both console and file.

    Args:
        run_dir: Directory where log file will be stored.
        verbose: Whether to enable debug-level logging.

    Returns:
        Path to the log file.
    """
    log_file = run_dir / "texture_analysis.log"
    log_level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_file


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------


def discover_images(input_path: Path, fmt: ImageFormat) -> list[Path]:
    """Discover CT image files under *input_path*.

    If *input_path* is a file it is returned directly (after extension check).
    If it is a directory, all matching files are collected recursively.

    Args:
        input_path: A single file or a directory to scan.
        fmt: Expected image format.

    Returns:
        Sorted list of discovered file paths.
    """
    extensions = EXTENSION_MAP[fmt]

    if input_path.is_file():
        if input_path.suffix.lower() in extensions:
            return [input_path]
        raise typer.BadParameter(
            f"File {input_path} does not match expected format '{fmt.value}' "
            f"(expected extensions: {extensions})"
        )

    if not input_path.is_dir():
        raise typer.BadParameter(f"Path {input_path} is not a file or directory.")

    files = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in extensions)
    if not files:
        raise typer.BadParameter(f"No {fmt.value} files found under {input_path}")
    return files


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and return the YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    return cfg


def save_config_copy(config_path: Path, run_dir: Path) -> Path:
    """Copy the configuration file into the run directory for reproducibility.

    Args:
        config_path: Original config file path.
        run_dir: Destination run directory.

    Returns:
        Path to the copied file.
    """
    dest = run_dir / config_path.name
    shutil.copy2(config_path, dest)
    return dest
