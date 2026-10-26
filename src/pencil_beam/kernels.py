"""Load DoseCUDA's compiled CUDA kernels (the ``dose_kernels`` extension).

Only the extension is used, never DoseCUDA's Python package: that package pins
``numpy < 2``, reads its tables through ``pkg_resources`` from its own install
directory, and would have to be importable from adota's venv. The kernels are
built for adota's interpreter by ``scripts/pencil_beam/provision_dosecuda.sh`` and
loaded here by absolute path, the way the MCsquare binary is referenced.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

__all__ = ["load_dose_kernels"]


@lru_cache(maxsize=None)
def _load(path: str) -> ModuleType:
    loader = importlib.machinery.ExtensionFileLoader("dose_kernels", path)
    spec = importlib.util.spec_from_file_location("dose_kernels", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def load_dose_kernels(path: str | Path) -> ModuleType:
    """Return the ``dose_kernels`` module compiled at ``path``.

    Args:
        path: The ``dose_kernels.so`` written by ``provision_dosecuda.sh``
            (``/home/mstryja/tools/dosecuda/current/dose_kernels.so`` by default).

    Raises:
        FileNotFoundError: If nothing is built there, naming the script to run.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DoseCUDA kernels not found at {path}; build them with "
            "scripts/pencil_beam/provision_dosecuda.sh (see docs/pencil_beam.md)")
    return _load(str(path.resolve()))
