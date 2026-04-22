"""
Shared configuration constants and boilerplate for ADoTA scripts.

This module is the **single source of truth** for:
- Default scaling constants (``DEFAULT_SCALE``)
- Default gamma-index parameters (``DEFAULT_GAMMA_PARAMS``)
- Energy denormalization (``denormalize_energy``)
- Run-directory creation (``setup_run_directory``)
- Logging setup (``setup_logging``)
- Device selection (``get_device``)
- YAML config loading (``load_yaml_config``)

All analysis/inference scripts should import from here instead of
defining their own copies.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import torch
import yaml

logger = logging.getLogger(__name__)

# ── Project root (repo top-level) ──────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# ── Default scaling constants ───────────────────────────────────────────────

DEFAULT_SCALE: dict[str, float] = {
    "min_ds": 0.0,
    "max_ds": 25277028.0,
    "min_ct": -1024,
    "max_ct": 3071,
    "min_energy": 70.0,
    "max_energy": 270.0,
}

# ── Default gamma-index parameters ─────────────────────────────────────────

DEFAULT_GAMMA_PARAMS: dict[str, object] = {
    "dose_percent_threshold": 2,
    "distance_mm_threshold": 2,
    "interp_fraction": 10,
    "max_gamma": 2,
    "lower_percent_dose_cutoff": 10,
    "random_subset": None,
    "local_gamma": False,
    "quiet": True,
}


# ── Energy helpers ──────────────────────────────────────────────────────────


def denormalize_energy(energy_normalized: float, scale: dict) -> float:
    """Convert normalised energy back to MeV.

    Args:
        energy_normalized: Energy value in [0, 1] range.
        scale: Scaling dict containing ``min_energy`` and ``max_energy``.

    Returns:
        Energy in MeV.
    """
    return (
        energy_normalized * (scale["max_energy"] - scale["min_energy"])
        + scale["min_energy"]
    )


# ── Run directory & logging ─────────────────────────────────────────────────


def setup_run_directory(
    runs_dir: Path,
    prefix: str = "",
    subdirs: Sequence[str] = ("figures",),
) -> Path:
    """Create a timestamped run directory.

    Args:
        runs_dir: Base directory for all runs.
        prefix: Optional prefix before the timestamp
            (e.g. ``"texture_"`` → ``texture_20260420_120000``).
        subdirs: Sub-directories to create inside the run directory.
            Defaults to ``("figures",)``.  Pass an empty sequence to
            skip.

    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{prefix}{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sd in subdirs:
        (run_dir / sd).mkdir(exist_ok=True)
    return run_dir


def setup_logging(
    run_dir: Path,
    verbose: bool = False,
    log_filename: str = "evaluation.log",
) -> Path:
    """Configure logging to both console and file.

    Args:
        run_dir: Directory where the log file will be stored.
        verbose: Whether to enable debug-level logging.
        log_filename: Name of the log file inside *run_dir*.

    Returns:
        Path to the log file.
    """
    log_file = run_dir / log_filename
    log_level = logging.DEBUG if verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return log_file


def copy_config(config_path: Path, run_dir: Path) -> None:
    """Copy a configuration file into a run directory for reproducibility.

    Args:
        config_path: Source config file.
        run_dir: Destination directory.
    """
    shutil.copy2(config_path, run_dir / config_path.name)
    logger.info(f"Config file copied to {run_dir / config_path.name}")


# ── Device helpers ──────────────────────────────────────────────────────────


def get_device(device_index: int = 0) -> torch.device:
    """Get the appropriate torch device.

    Args:
        device_index: CUDA device ordinal.  Use ``-1`` to force CPU.

    Returns:
        A :class:`torch.device` instance.
    """
    if torch.cuda.is_available() and device_index >= 0:
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


# ── YAML config loading ────────────────────────────────────────────────────


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary with configuration values.

    Raises:
        typer.BadParameter: If the file does not exist or cannot be
        parsed (when *typer* is available), otherwise a plain
        ``FileNotFoundError`` / ``yaml.YAMLError``.
    """
    if not config_path.exists():
        # Try typer-friendly error; fall back to plain exception
        try:
            import typer

            raise typer.BadParameter(f"Config file not found: {config_path}")
        except ImportError:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        try:
            import typer

            raise typer.BadParameter(f"Error parsing YAML config: {e}")
        except ImportError:
            raise

    if config is None:
        return {}

    return config
