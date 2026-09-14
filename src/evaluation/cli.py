"""Shared CLI / config / device helpers for the evaluation scripts.

The scripts already share logging and run-dir helpers from
:mod:`src.adota.config`; those are re-exported here so a script has one import
site. The genuinely new helpers are:

* :func:`resolve_device` -- a single device entrypoint that selects a GPU when
  available, falls back to CPU when CUDA is absent or the requested index is out
  of range, and logs the choice; and
* :func:`merge_config` -- the generic ``CLI > YAML > default`` merge that
  replaces the hand-wired per-field merge blocks in every ``main``; and
* :func:`apply_set_overrides` -- the generic, repeatable ``--set key=value``
  option, so a variant of a config (a smoke test, a scale-up) is a command line
  rather than another YAML file.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Mapping, Optional, Sequence

import torch
import yaml

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


SET_OVERRIDE_HELP = (
    "Override one config key: KEY=VALUE, repeatable. Dotted keys reach nested "
    "blocks (training.compile=false, scorer.n_workers=8); the value is parsed as "
    "YAML (false, null, 8, 0.3, [80.0,105.0], quoted strings). Applied to the YAML "
    "before validation, so precedence is per-field option > --set > YAML > default."
)


def apply_set_overrides(config: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``--set KEY=VALUE`` overrides to a parsed YAML config.

    The command line is where a config variant belongs (a smoke test, a scale-up),
    not a second YAML file that repeats the first one. Each entry is ``key=value``:
    ``key`` may be dotted to reach a nested mapping (``training.compile``), and
    intermediate mappings that do not exist yet are created. ``value`` is parsed
    as a YAML scalar (``yaml.safe_load``), so ``false``, ``null``, ``8``, ``0.3``,
    flow lists such as ``[80.0,105.0]`` and quoted strings all become what they
    would be in the file; an empty value means ``null``.

    Args:
        config: The parsed YAML mapping. It is not modified.
        overrides: ``KEY=VALUE`` strings, applied in order.

    Returns:
        A deep copy of ``config`` with the overrides applied.

    Raises:
        ValueError: On an entry without ``=``, an empty key or key segment, or a
            dotted path that walks through an existing non-mapping value.
    """
    result: dict[str, Any] = copy.deepcopy(dict(config))
    for entry in overrides:
        key, sep, raw_value = entry.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {entry!r}")
        segments = key.split(".")
        if any(not segment for segment in segments):
            raise ValueError(f"--set key {key!r} has an empty segment")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(f"--set {key}: cannot parse value {raw_value!r} as YAML: {exc}") from exc

        node: dict[str, Any] = result
        for depth, segment in enumerate(segments[:-1]):
            child = node.get(segment)
            if child is None:
                child = node[segment] = {}
            elif not isinstance(child, dict):
                path = ".".join(segments[: depth + 1])
                raise ValueError(
                    f"--set {key}: {path!r} is a {type(child).__name__}, not a mapping")
            node = child
        node[segments[-1]] = value
    return result
