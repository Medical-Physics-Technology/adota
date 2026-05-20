"""Run-lifecycle helpers for ADoTA training.

Everything that persists state across a training run lives here:

- :func:`setup_training_run_directory` builds the standardized layout.
- :func:`setup_training_logging` installs a phase-tagged, relative-time
  log formatter on both the run-dir log file and the console.
- :func:`silence_pymedphys` mutes pymedphys's verbose gamma logger so
  it doesn't pollute the training log.
- :func:`log_banner` / :func:`log_section` / :func:`log_phase` emit
  ASCII-only structured log output.
- :func:`write_manifest` records reproducibility info (git, versions, GPU,
  dataset fingerprint, resolved config).
- :class:`MetricsLog` streams per-epoch metrics into a JSONL file so that
  a SIGKILL never corrupts the history.
- :class:`CheckpointManager` enforces the "best + last + every-N" retention
  policy and provides full resume snapshots (model, optimizer, scheduler,
  RNG, training counters, loss balancer).
- :class:`GracefulShutdown` traps SIGINT/SIGTERM and lets the training
  loop finish the current epoch before exiting.
- :func:`dump_nan_context` saves the offending batch + diagnostic info
  when a non-finite loss appears.
- :func:`compute_grad_norm` / :func:`compute_param_norm` are cheap
  diagnostics tracked once per epoch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import signal
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from src.training.losses import TwoObjectiveBalancer
from src.utils.serialization import NumpyEncoder

logger = logging.getLogger(__name__)


# ── Logging: phase-tagged formatter ─────────────────────────────────────────


BANNER_WIDTH = 80
PHASE_FIELD_WIDTH = 5  # widest tag we use (TRAIN, EPOCH, ERROR, ...)
_DEFAULT_PHASE = "LOG"


class RelativeTimeFormatter(logging.Formatter):
    """Format log records as ``HH:MM:SS [PHASE] message``.

    The time component is the elapsed wall time since training started,
    not an absolute timestamp -- the run directory name already records
    the start date, and relative timing makes a long training log much
    easier to skim.

    The phase tag is taken from ``record.phase`` if the call site passed
    ``extra={"phase": "TRAIN"}``; otherwise it falls back to
    :data:`_DEFAULT_PHASE`.

    Args:
        start_time: ``time.time()`` value captured at run start.
        phase_width: Width to left-justify the phase tag inside the
            brackets (defaults to :data:`PHASE_FIELD_WIDTH`).
    """

    def __init__(self, start_time: float, phase_width: int = PHASE_FIELD_WIDTH):
        super().__init__()
        self.start_time = start_time
        self.phase_width = phase_width

    def format(self, record: logging.LogRecord) -> str:
        elapsed = max(0.0, record.created - self.start_time)
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        phase = getattr(record, "phase", _DEFAULT_PHASE)
        return f"{time_str} [{phase:<{self.phase_width}}] {record.getMessage()}"


def setup_training_logging(
    run_dir: Path,
    start_time: float,
    verbose: bool = False,
    log_filename: str = "training.log",
) -> Path:
    """Install :class:`RelativeTimeFormatter` on the root logger.

    Wipes any handlers configured by an earlier ``setup_logging`` call so
    the training script gets a clean stream into both stdout and the
    run-dir log file. Returns the log-file path.
    """
    log_file = run_dir / log_filename
    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = RelativeTimeFormatter(start_time=start_time)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_file


def silence_pymedphys() -> None:
    """Suppress pymedphys's INFO-level gamma logging.

    pymedphys emits five INFO lines per ``gamma(...)`` call describing
    its normalisation choices; for a 20-sample GPR subset that's 100
    lines we don't want in the training log. We push its whole logger
    tree to WARNING.
    """
    for name in ("pymedphys", "pymedphys._gamma", "pymedphys.gamma"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Logging: structured helpers ─────────────────────────────────────────────


def log_phase(
    phase: str,
    message: str,
    *,
    level: int = logging.INFO,
    target_logger: Optional[logging.Logger] = None,
) -> None:
    """Emit one log line tagged with ``[phase]``.

    Args:
        phase: Short tag (e.g. ``"TRAIN"``, ``"GPR"``). Kept under
            :data:`PHASE_FIELD_WIDTH` characters for column alignment.
        message: The text.
        level: Logging level (default INFO).
        target_logger: Logger to emit through. Defaults to the root
            logger so any module can call this helper.
    """
    target = target_logger if target_logger is not None else logging.getLogger()
    target.log(level, message, extra={"phase": phase})


def log_banner(title: str, *, char: str = "=") -> None:
    """Emit an 80-char banner with a centered title.

    Three lines: a separator, the title (centered), another separator.
    Banner lines are raw (no time / phase prefix) so they stand out.
    """
    root = logging.getLogger()
    # We bypass the formatter for banners so the separators are full-width.
    for handler in root.handlers:
        handler.stream.write(char * BANNER_WIDTH + "\n")
        handler.stream.write(title.center(BANNER_WIDTH) + "\n")
        handler.stream.write(char * BANNER_WIDTH + "\n")
        handler.stream.flush()


def log_section(title: str, *, char: str = "=") -> None:
    """Emit a section separator with the title inline.

    Two lines: an 80-char rule above and below a single line containing
    the title (indented two spaces).
    """
    root = logging.getLogger()
    rule = char * BANNER_WIDTH
    for handler in root.handlers:
        handler.stream.write(rule + "\n")
        handler.stream.write(f"  {title}\n")
        handler.stream.write(rule + "\n")
        handler.stream.flush()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {sec:02d}s"


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


# ── Checkpoint manager ──────────────────────────────────────────────────────


def _rng_state() -> Dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])


class CheckpointManager:
    """Save and load full training snapshots.

    Each snapshot contains everything needed to deterministically resume:
    model + optimizer + scheduler state dicts, epoch counter, best val
    loss, early-stopping patience counter, RNG state for all relevant
    generators, and the loss balancer's running state.

    Retention policy: ``best.pth`` and ``last.pth`` are always rewritten;
    additionally an ``epoch_NNNN.pth`` snapshot is kept every
    ``save_every_n_epochs`` epochs (1-indexed).
    """

    def __init__(self, checkpoint_dir: Path, save_every_n_epochs: int):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        best_val_loss: float,
        patience_counter: int,
        balancer: Optional[TwoObjectiveBalancer] = None,
        is_best: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist a training snapshot. Returns the ``last.pth`` path."""
        state: Dict[str, Any] = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "rng": _rng_state(),
            "balancer_running_weights": (
                balancer.running_weights.detach().cpu()
                if balancer is not None and balancer.running_weights is not None
                else None
            ),
        }
        if extra:
            state["extra"] = extra

        last_path = self.dir / "last.pth"
        torch.save(state, last_path)

        if is_best:
            torch.save(state, self.dir / "best.pth")

        if (epoch + 1) % self.save_every_n_epochs == 0:
            torch.save(state, self.dir / f"epoch_{epoch:04d}.pth")

        return last_path

    @staticmethod
    def load(
        path: Path,
        *,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        balancer: Optional[TwoObjectiveBalancer] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Restore a snapshot in-place; returns the bookkeeping fields."""
        map_location = device if device is not None else "cpu"
        # weights_only=False is required: the snapshot stores NumPy/Python
        # RNG state (and a balancer tensor) alongside the state dicts.
        state = torch.load(path, map_location=map_location, weights_only=False)

        model.load_state_dict(state["model"])
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if balancer is not None and state.get("balancer_running_weights") is not None:
            balancer.running_weights = state["balancer_running_weights"]
        if state.get("rng") is not None:
            _restore_rng_state(state["rng"])

        return {
            "epoch": state["epoch"],
            "best_val_loss": state["best_val_loss"],
            "patience_counter": state["patience_counter"],
            "extra": state.get("extra", {}),
        }


# ── Signal handling ─────────────────────────────────────────────────────────


class GracefulShutdown:
    """Trap SIGINT / SIGTERM so the training loop can exit cleanly.

    Usage::

        shutdown = GracefulShutdown()
        for epoch in range(...):
            if shutdown.requested:
                break
            ...
    """

    def __init__(self) -> None:
        self.requested: bool = False
        self._previous_int = signal.signal(signal.SIGINT, self._handler)
        self._previous_term = signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, frame: Optional[FrameType]) -> None:
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning(
            "Received %s; will finish current epoch then shut down cleanly.", name
        )
        self.requested = True

    def restore(self) -> None:
        """Restore original handlers (call at end of training)."""
        signal.signal(signal.SIGINT, self._previous_int)
        signal.signal(signal.SIGTERM, self._previous_term)


# ── NaN / Inf context dump ──────────────────────────────────────────────────


def dump_nan_context(
    run_dir: Path,
    epoch: int,
    batch_idx: int,
    x: torch.Tensor,
    energy: torch.Tensor,
    y: torch.Tensor,
    outputs: torch.Tensor,
    loss_components: Dict[str, float],
    weights: Dict[str, float],
    grad_norm: Optional[float] = None,
    sample_ids: Optional[List[str]] = None,
) -> Path:
    """Save tensors + diagnostic context for a non-finite batch.

    Files written under ``run_dir/failures/epoch_NNNN_batch_NNNN/``:
    ``x.npy``, ``energy.npy``, ``y.npy``, ``outputs.npy``,
    ``context.json``. Returns the directory path.
    """
    fail_dir = run_dir / "failures" / f"epoch_{epoch:04d}_batch_{batch_idx:04d}"
    fail_dir.mkdir(parents=True, exist_ok=True)

    np.save(fail_dir / "x.npy", x.detach().cpu().numpy())
    np.save(fail_dir / "energy.npy", energy.detach().cpu().numpy())
    np.save(fail_dir / "y.npy", y.detach().cpu().numpy())
    np.save(fail_dir / "outputs.npy", outputs.detach().cpu().numpy())

    context = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "loss_components": loss_components,
        "weights": weights,
        "grad_norm": grad_norm,
        "sample_ids": sample_ids,
        "x_min": float(x.min().item()),
        "x_max": float(x.max().item()),
        "y_min": float(y.min().item()),
        "y_max": float(y.max().item()),
        "energy_min": float(energy.min().item()),
        "energy_max": float(energy.max().item()),
    }
    with open(fail_dir / "context.json", "w") as f:
        json.dump(context, f, indent=2, cls=NumpyEncoder)

    return fail_dir


# ── Training diagnostics ────────────────────────────────────────────────────


def compute_grad_norm(model: torch.nn.Module) -> float:
    """L2 norm of all gradients currently attached to model parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
    return float(total**0.5)


def compute_param_norm(model: torch.nn.Module) -> float:
    """L2 norm of all model parameters (trainable or not)."""
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().norm(2).item()) ** 2
    return float(total**0.5)
