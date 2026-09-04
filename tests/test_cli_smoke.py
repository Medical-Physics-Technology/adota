"""Run ``--help`` on every Typer script and assert it exits cleanly.

Flow:
1. Discover the scripts under ``scripts/`` that import ``typer``.
2. Run each one as a subprocess with ``--help``.
3. Require exit code 0 and non-empty output.

This catches import-time breakage in ``scripts/`` -- a moved symbol, a stale
import path -- without needing the dataset, a checkpoint or a GPU, none of
which ``--help`` touches.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
HELP_TIMEOUT_SECONDS = 180


def _typer_scripts() -> list[str]:
    """Paths (relative to the repository root) of every Typer-based script."""
    found = [
        path
        for path in sorted(SCRIPTS_ROOT.rglob("*.py"))
        if "import typer" in path.read_text(encoding="utf-8")
    ]
    return [str(path.relative_to(PROJECT_ROOT)) for path in found]


TYPER_SCRIPTS = _typer_scripts()


def test_script_inventory_is_not_empty() -> None:
    """A silent glob failure would make every other case vacuously pass."""
    assert len(TYPER_SCRIPTS) > 20


@pytest.mark.parametrize("script", TYPER_SCRIPTS)
def test_help_exits_cleanly(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=HELP_TIMEOUT_SECONDS,
    )
    assert completed.returncode == 0, (
        f"{script} --help exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip(), f"{script} --help produced no output"
