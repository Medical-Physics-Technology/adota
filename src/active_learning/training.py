"""The retraining step of a cycle: a real training run, driven as a subprocess.

A cycle does not reimplement training. It writes a training YAML that points at the
accumulated beamlet sources and runs ``scripts/train_adota.py``, so every cycle gets
the run directory, reproducibility manifest, metrics log, checkpoints with RNG state
and validation artifacts that any other training run gets, and a cycle can be
inspected, resumed or rerun on its own.

A subprocess rather than an import: ``torch.compile`` caches and CUDA context are
released when the process exits, so the loop can score candidates and run Monte Carlo
between cycles without holding GPU memory it is not using.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


def cycle_training_config(base_config: Path, *, run_name: str, sources_csv: Path,
                          overrides: Optional[Dict] = None) -> dict:
    """The base training config, pointed at this cycle's union training set."""
    config = yaml.safe_load(Path(base_config).read_text())
    config["config_name"] = run_name
    config["al_dir_sources"] = [str(sources_csv)]
    config.update(overrides or {})
    return config


def run_training(
    config: dict,
    *,
    config_path: Path,
    runs_dir: Path,
    init_checkpoint: Path,
    device_index: int,
    log_path: Path,
    weights_only: bool = True,
    extra_args: Optional[List[str]] = None,
) -> Path:
    """Run one training job to completion; return its run directory.

    Args:
        config: The resolved cycle config, written verbatim to ``config_path``.
        config_path: Where to write it (inside the loop's run directory).
        runs_dir: Base directory the training run creates its own directory under.
        init_checkpoint: Weights to start from -- the deployed model for the first
            cycle, the previous cycle's best afterwards.
        device_index: CUDA device for this arm.
        log_path: File the training job's stdout and stderr are written to.
        weights_only: Load only the weights, with a fresh optimizer and schedule.
            True for every cycle: a cycle is a fine-tune of the previous cycle's
            model on a training set that just changed, not a continuation of its
            schedule.
        extra_args: Further CLI flags, appended verbatim.

    Raises:
        RuntimeError: If training exits non-zero, or writes no run directory.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    runs_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "scripts/train_adota.py",
        "--config", str(config_path),
        "--runs-dir", str(runs_dir),
        "--resume-from", str(init_checkpoint),
        "--device-index", str(device_index),
    ]
    if weights_only:
        command.append("--weights-only")
    command.extend(extra_args or [])

    logger.info("training: %s", " ".join(command))
    logger.info("training log -> %s", log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"training exited {result.returncode}; see {log_path}")

    runs = sorted(runs_dir.glob("train_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise RuntimeError(f"training wrote no run directory under {runs_dir}")
    return runs[-1]


def cycle_checkpoint(train_run_dir: Path, selection: str = "last") -> Path:
    """The checkpoint a cycle hands to the next one.

    ``last`` is the default, and it is the right default here even though ``best`` is
    the usual choice. ``best`` is chosen by the loss on the HDF5 validation split,
    which is the distribution the model already fits; a cycle that trains on newly
    bought beamlets can raise that loss while getting better at exactly the geometry
    it just bought, and selecting on it would carry the pre-cycle weights forward and
    make the loop measure nothing. A cycle is a fixed budget of optimizer steps, so
    what it produced is what it hands on. Use ``best`` only to reproduce a
    model-selection regime deliberately.
    """
    checkpoints = train_run_dir / "checkpoints"
    order = ["last.pth", "best.pth"] if selection == "last" else ["best.pth", "last.pth"]
    for name in order:
        if (checkpoints / name).exists():
            if name != order[0]:
                logger.warning("no %s in %s; carrying %s forward", order[0], checkpoints, name)
            return checkpoints / name
    raise FileNotFoundError(f"no checkpoint under {checkpoints}")
