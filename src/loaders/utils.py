"""Pre-flight checks shared by the inference entry points.

``validate_inputs`` asserts that the data directory, checkpoint and
hyperparameter file all exist before a run starts, so a long job fails
immediately and with a readable message rather than midway through.
"""

from pathlib import Path

import typer


def validate_inputs(data_path: Path, model_path: Path, hyperparams_path: Path) -> None:
    """Validate that required input files exist.

    Works for both directory-based (test_data/) and file-based (HDF5) data
    sources.

    Args:
        data_path: Path to the data source (directory or HDF5 file).
        model_path: Path to the model weights file.
        hyperparams_path: Path to the hyperparameters JSON file.

    Raises:
        typer.BadParameter: If any required path is missing.
    """
    if not data_path.exists():
        raise typer.BadParameter(f"Data source not found: {data_path}")
    if not model_path.exists():
        raise typer.BadParameter(f"Model file not found: {model_path}")
    if not hyperparams_path.exists():
        raise typer.BadParameter(f"Hyperparams file not found: {hyperparams_path}")
