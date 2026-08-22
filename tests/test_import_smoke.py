"""Import every module under ``src/`` and assert none of them fails.

Flow:
1. Walk ``src/`` on the filesystem (not via ``pkgutil``, so implicit namespace
   packages without ``__init__.py`` are covered too).
2. Translate each path into its dotted module name.
3. Import it in a parametrized test case.

This is the guard for module splits: when a file is broken into several modules
behind a re-exporting ``__init__.py``, a symbol left behind fails here first.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "src"


def _module_names() -> list[str]:
    """Dotted names for every importable module under ``src/``."""
    names: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(PROJECT_ROOT)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        names.append(".".join(parts))
    return names


MODULE_NAMES = _module_names()


def test_module_inventory_is_not_empty() -> None:
    """A silent glob failure would make every other case vacuously pass."""
    assert len(MODULE_NAMES) > 80


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
