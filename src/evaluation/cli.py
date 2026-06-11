"""Shared CLI / config / device helpers for the evaluation scripts.

The scripts already share logging and run-dir helpers from
:mod:`src.adota.config`; those are re-exported here so a script has one import
site. The genuinely new helpers are:

* :func:`resolve_device` -- a single device entrypoint that selects a GPU when
  available, falls back to CPU when CUDA is absent or the requested index is out
  of range, and logs the choice; and
* :func:`merge_config` -- the generic ``CLI > YAML > default`` merge that
  replaces the hand-wired per-field merge blocks in every ``main``.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import torch

# Re-exported so scripts import logging / run-dir / YAML helpers from one place.
from src.adota.config import (  # noqa: F401
    copy_config,
    load_yaml_config,
    setup_logging,
    setup_run_directory,
)

logger = logging.getLogger(__name__)


def resolve_device(device_index: Optional[int] = None) -> torch.device:
    """Resolve the torch device, preferring GPU and falling back to CPU.

    Args:
        device_index: CUDA device ordinal. ``-1`` forces CPU. ``None`` means
            "auto": the first CUDA device when available, otherwise CPU. A
            non-negative index that is unavailable (no CUDA, or out of range)
            falls back to CPU with a warning.

    Returns:
        The resolved :class:`torch.device`.
    """
    cuda_available = torch.cuda.is_available()

    if device_index is not None and device_index < 0:
        return torch.device("cpu")

    if not cuda_available:
        if device_index is not None and device_index >= 0:
            logger.warning(
                "CUDA not available; requested device index %d falls back to CPU.",
                device_index,
            )
        return torch.device("cpu")

    if device_index is None:
        return torch.device("cuda:0")

    n_devices = torch.cuda.device_count()
    if device_index >= n_devices:
        logger.warning(
            "CUDA device index %d out of range (%d visible); falling back to CPU.",
            device_index,
            n_devices,
        )
        return torch.device("cpu")

    return torch.device(f"cuda:{device_index}")


def merge_config(
    cli_overrides: Mapping[str, Any],
    yaml_config: Mapping[str, Any],
    defaults: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Merge configuration sources with ``CLI > YAML > default`` precedence.

    A CLI override counts only when it is not ``None`` (typer leaves unset
    options as ``None``), matching the per-field ``cli_value or yaml.get(...)``
    pattern the scripts currently spell out by hand.

    Args:
        cli_overrides: CLI option values; ``None`` means "unset".
        yaml_config: Parsed YAML config.
        defaults: Fallback values used when neither CLI nor YAML provides a key.

    Returns:
        A plain dict keyed by the union of all provided keys.
    """
    defaults = defaults or {}
    keys = set(yaml_config) | set(defaults)
    keys |= {k for k, v in cli_overrides.items() if v is not None}

    merged: dict[str, Any] = {}
    for key in keys:
        cli_value = cli_overrides.get(key)
        if cli_value is not None:
            merged[key] = cli_value
        elif key in yaml_config:
            merged[key] = yaml_config[key]
        else:
            merged[key] = defaults.get(key)
    return merged
