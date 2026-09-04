"""Run-directory layout and the reproducibility manifest.

Flow:
1. :func:`setup_training_run_directory` creates the timestamped run directory
   and its fixed subdirectories (checkpoints, attention, failures, validation).
2. :func:`write_manifest` records what produced the run: git commit and dirty
   state, package versions, GPU model, dataset fingerprint, resolved config.
3. :func:`save_resolved_config` writes the fully-merged config back as YAML.
4. :class:`MetricsLog` appends one JSON object per epoch, flushed each write, so
   a SIGKILL truncates at most the last line rather than corrupting history.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)

# ── Run directory layout ────────────────────────────────────────────────────


RUN_SUBDIRS = ("checkpoints", "attention", "failures", "validation")


def setup_training_run_directory(
    runs_dir: Path,
    config_name: str,
) -> Path:
    """Create the standardized training run directory.

    Layout::

        runs/train_<timestamp>_<config_name>/
            ├─ checkpoints/
            ├─ attention/
            ├─ failures/
            └─ validation/

    Args:
        runs_dir: Base directory under which all training runs live.
        config_name: Short identifier from the YAML config; appended to
            the timestamp so a directory listing is easy to scan.

    Returns:
        Path to the freshly created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in config_name)
    run_dir = runs_dir / f"train_{timestamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in RUN_SUBDIRS:
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir


# ── Reproducibility manifest ────────────────────────────────────────────────


def _git_info() -> Dict[str, Optional[Any]]:
    """Return current commit hash and dirty flag, or ``None`` on failure."""
    info: Dict[str, Optional[Any]] = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        info["dirty"] = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    """Cheap fingerprint of a file (size + mtime + sha256 of first 1 MiB)."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256_first_mib": h.hexdigest(),
    }


def _gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "count": torch.cuda.device_count(),
        "current": torch.cuda.current_device(),
        "name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "capability": torch.cuda.get_device_capability(torch.cuda.current_device()),
    }


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Best-effort serialization of a dataclass or plain object."""
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"repr": repr(config)}


def write_manifest(
    run_dir: Path,
    config: Any,
    dataset_path: Path,
    excluded_indexes_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``manifest.json`` capturing the run's environment and inputs.

    Args:
        run_dir: Destination directory.
        config: Resolved training config (a dataclass instance or any
            object with a ``__dict__``).
        dataset_path: Path to the H5 dataset file used.
        excluded_indexes_path: Optional path to the excluded-indexes file.
        extra: Optional additional fields (e.g. train/val split counts)
            merged into the manifest.

    Returns:
        Path to the written manifest.
    """
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": _git_info(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "gpu": _gpu_info(),
        "argv": sys.argv,
        "config": _config_to_dict(config),
        "dataset": _file_fingerprint(Path(dataset_path)),
    }
    if excluded_indexes_path is not None:
        manifest["excluded_indexes"] = _file_fingerprint(Path(excluded_indexes_path))
    if extra:
        manifest.update(extra)

    out = run_dir / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder, default=str)
    return out


def save_resolved_config(config: Any, path: Path) -> None:
    """Dump the resolved config to YAML for human inspection."""
    with open(path, "w") as f:
        yaml.safe_dump(_config_to_dict(config), f, default_flow_style=False, sort_keys=False)


# ── Streaming metrics log ───────────────────────────────────────────────────


class MetricsLog:
    """Append-only JSONL writer for per-epoch metrics.

    One line per epoch; the file survives mid-write SIGKILL because
    each line is flushed and ``fsync``'d independently.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def log(self, record: Dict[str, Any]) -> None:
        """Append a record (a dict of JSON-serializable values)."""
        line = json.dumps(record, cls=NumpyEncoder, default=str)
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
