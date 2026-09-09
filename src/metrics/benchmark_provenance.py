# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the MIT License. See the repository LICENSE file.

"""Provenance capture for timing benchmarks.

A timing number is only as reproducible as the record of what produced it. This
module writes, beside every benchmark run, the things that decide a wall time
and are otherwise lost the moment the process exits: thread counts and CPU
affinity, library and driver versions, the GPU's identity and whether anything
else was running on it, the exact git state of the code including any
uncommitted diff, and SHA-256 hashes of every input the run consumed.

Nothing here is specific to gamma; it is kept under ``src/metrics`` because the
gamma harnesses are its only callers today.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "THREAD_ENV_VARS",
    "sha256_file",
    "environment_manifest",
    "gpu_processes",
    "git_state",
    "capture_system_dumps",
    "write_manifest",
    "timestamped_run_dir",
]

# The environment variables that bound the thread pools a CPU gamma run can use.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)


def _run(command: Sequence[str], timeout: float = 60.0) -> str:
    """Run a command and return its stdout, or an error marker; never raise."""
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"<unavailable: {error}>"
    if completed.returncode != 0:
        return f"<exit {completed.returncode}>\n{completed.stdout}{completed.stderr}"
    return completed.stdout


def sha256_file(path: Path, chunk_bytes: int = 1 << 24) -> str:
    """SHA-256 of a file's contents, streamed so large datasets fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamped_run_dir(root: Path, label: str) -> Path:
    """``<root>/<label>_<UTC timestamp>``, created; never reuses a directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(root) / f"{label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _gpu_query(fields: str, extra: Sequence[str] = ()) -> List[List[str]]:
    """Rows of ``nvidia-smi --query-gpu=<fields>`` as lists of strings."""
    text = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", *extra])
    if text.startswith("<"):
        return []
    return [[cell.strip() for cell in line.split(",")] for line in text.strip().splitlines() if line]


def gpu_processes() -> List[Dict[str, str]]:
    """Every compute process currently resident on any GPU, per ``nvidia-smi``."""
    text = _run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    )
    if text.startswith("<"):
        return [{"error": text}]
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = [cell.strip() for cell in line.split(",", 3)]
        rows.append({"gpu_uuid": uuid, "pid": pid, "process_name": name, "used_memory_mib": memory})
    return rows


def _torch_section(device: Optional[str]) -> Dict[str, Any]:
    """Torch, CUDA and GPU identity, tolerant of a CPU-only machine."""
    try:
        import torch
    except ImportError:
        return {"torch": None}
    section: Dict[str, Any] = {
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(index)
        section.update(
            {
                "device": str(device),
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_uuid_torch": str(getattr(properties, "uuid", "")),
            }
        )
    return section


def _numba_section() -> Dict[str, Any]:
    try:
        import numba

        return {
            "numba": numba.__version__,
            "numba_num_threads": int(numba.config.NUMBA_NUM_THREADS),
            "numba_threading_layer": str(numba.config.THREADING_LAYER),
        }
    except ImportError:
        return {"numba": None}


def environment_manifest(device: Optional[str] = None) -> Dict[str, Any]:
    """Everything about the machine and the process that can move a wall time.

    Args:
        device: The torch device the run uses, so its GPU is identified.

    Returns:
        A JSON-serialisable dict.
    """
    import numpy

    memory_kib = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    memory_kib = int(line.split()[1])
                    break
    except OSError:
        pass

    gpus = [
        {"index": row[0], "uuid": row[1], "name": row[2], "driver": row[3]}
        for row in _gpu_query("index,uuid,name,driver_version")
        if len(row) >= 4
    ]
    try:
        import pymedphys

        pymedphys_version = pymedphys.__version__
    except ImportError:
        pymedphys_version = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "cpu_model": next(
            (line.split(":", 1)[1].strip() for line in _run(["lscpu"]).splitlines() if line.startswith("Model name")),
            None,
        ),
        "cpu_logical": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "memory_total_kib": memory_kib,
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "numpy": numpy.__version__,
        "pymedphys": pymedphys_version,
        **_numba_section(),
        **_torch_section(device),
        "gpus": gpus,
        "gpu_processes": gpu_processes(),
    }


def git_state(repo: Path, out_dir: Optional[Path] = None, label: str = "repo") -> Dict[str, Any]:
    """Commit, branch and dirtiness of a checkout, with the diff saved if dirty.

    A dirty worktree described by its commit alone is not reproducible, so the
    full ``git diff --binary`` is written beside the manifest and hashed.
    """
    repo = Path(repo)
    commit = _run(["git", "-C", str(repo), "rev-parse", "HEAD"]).strip()
    branch = _run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    status = _run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"]).strip()
    state: Dict[str, Any] = {"path": str(repo), "commit": commit, "branch": branch, "dirty": bool(status)}
    if status and out_dir is not None:
        diff = _run(["git", "-C", str(repo), "diff", "--binary", "HEAD"], timeout=120)
        diff_path = Path(out_dir) / f"git_diff_{label}.patch"
        diff_path.write_text(diff)
        state["diff_path"] = str(diff_path)
        state["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        state["status"] = status
    return state


def capture_system_dumps(out_dir: Path) -> Dict[str, str]:
    """Save the raw tool outputs the manifest summarises, for later inspection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dumps = {
        "nvidia_smi_q.txt": _run(["nvidia-smi", "-q"]),
        "lscpu.txt": _run(["lscpu"]),
        "uname.txt": _run(["uname", "-a"]),
        "pip_freeze.txt": _run(["uv", "pip", "freeze"], timeout=180),
    }
    try:
        import torch

        dumps["torch_config.txt"] = torch.__config__.show()
    except ImportError:
        dumps["torch_config.txt"] = "<torch unavailable>"
    written = {}
    for name, text in dumps.items():
        path = out_dir / name
        path.write_text(text)
        written[name] = str(path)
    return written


def write_manifest(
    run_dir: Path,
    *,
    device: Optional[str],
    repos: Dict[str, Path],
    inputs: Dict[str, Path],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``manifest.json`` plus the raw dumps into ``run_dir``.

    Args:
        run_dir: The run's directory.
        device: Torch device string of the run.
        repos: ``{label: path}`` of git checkouts whose state to record.
        inputs: ``{label: path}`` of input files to hash.
        extra: Anything run-specific to include verbatim.

    Returns:
        The manifest path.
    """
    run_dir = Path(run_dir)
    dumps_dir = run_dir / "provenance"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "environment": environment_manifest(device),
        "git": {label: git_state(path, dumps_dir, label) for label, path in repos.items()},
        "inputs": {
            label: {"path": str(path), "sha256": sha256_file(path), "bytes": Path(path).stat().st_size}
            for label, path in inputs.items()
        },
        "dumps": capture_system_dumps(dumps_dir),
        "extra": extra or {},
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    logger.info("Wrote %s", path)
    return path
