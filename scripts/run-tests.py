"""Repository test runner.

Flow:
1. Pick a marker expression from the sub-command (`unit`, `integration`, `e2e`).
2. Run `uv run pytest` from the repository root with that expression.
3. Report pass/fail/skip per suite and exit non-zero if any suite failed.

Usage:
    uv run python scripts/run-tests.py unit
    uv run python scripts/run-tests.py unit --fast     # skip the slow perf suite
    uv run python scripts/run-tests.py integration     # needs the dataset + checkpoint
    uv run python scripts/run-tests.py e2e
    uv run python scripts/run-tests.py all

`adota` is a single Python package rooted at this repository, so there is one
pytest invocation per suite rather than one per component.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIT_EXPRESSION = "not integration and not e2e"
UNIT_FAST_EXPRESSION = "not integration and not e2e and not slow"

# Unknown options are forwarded to pytest verbatim, so `run-tests.py unit -q -x`
# works the way the surrounding docs promise.
CONTEXT_SETTINGS = {"allow_extra_args": True, "ignore_unknown_options": True}

app = typer.Typer(
    help="Run the adota test suites.",
    add_completion=False,
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
)


def _run(marker: str, label: str, pytest_args: List[str]) -> int:
    """Run one suite and return pytest's exit code."""
    command = ["uv", "run", "pytest", "-m", marker, *pytest_args]
    typer.secho(f"\n=== {label}  (-m {marker!r}) ===", fg=typer.colors.CYAN, bold=True)
    typer.echo("$ " + " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return completed.returncode


def _report(results: List[tuple], ) -> int:
    """Print a per-suite summary; return the process exit code."""
    typer.secho("\n=== summary ===", bold=True)
    failed = False
    for label, code in results:
        # pytest exit code 5 means "no tests collected", which is a normal
        # outcome for an opt-in suite that has no tests yet.
        if code == 0:
            status, color = "passed", typer.colors.GREEN
        elif code == 5:
            status, color = "no tests collected", typer.colors.YELLOW
        else:
            status, color = f"FAILED (exit {code})", typer.colors.RED
            failed = True
        typer.secho(f"  {label:<12} {status}", fg=color)
    return 1 if failed else 0


@app.command(context_settings=CONTEXT_SETTINGS)
def unit(
    fast: bool = typer.Option(False, "--fast", help="Also skip the slow performance suite."),
    pytest_args: Optional[List[str]] = typer.Argument(None, help="Extra pytest arguments."),
) -> None:
    """Fast suite: no dataset, no checkpoint, no GPU, no network."""
    marker = UNIT_FAST_EXPRESSION if fast else UNIT_EXPRESSION
    code = _run(marker, "unit", list(pytest_args or []))
    raise typer.Exit(_report([("unit", code)]))


@app.command(context_settings=CONTEXT_SETTINGS)
def integration(
    pytest_args: Optional[List[str]] = typer.Argument(None, help="Extra pytest arguments."),
) -> None:
    """Dependency-backed suite: needs the HDF5 dataset and a trained checkpoint."""
    code = _run("integration", "integration", list(pytest_args or []))
    raise typer.Exit(_report([("integration", code)]))


@app.command(context_settings=CONTEXT_SETTINGS)
def e2e(
    pytest_args: Optional[List[str]] = typer.Argument(None, help="Extra pytest arguments."),
) -> None:
    """Full-workflow suite; opt-in."""
    code = _run("e2e", "e2e", list(pytest_args or []))
    raise typer.Exit(_report([("e2e", code)]))


@app.command(context_settings=CONTEXT_SETTINGS)
def all(
    pytest_args: Optional[List[str]] = typer.Argument(None, help="Extra pytest arguments."),
) -> None:
    """Every suite in turn; keeps going so one failure does not hide the rest."""
    extra = list(pytest_args or [])
    results = [
        ("unit", _run(UNIT_EXPRESSION, "unit", extra)),
        ("integration", _run("integration", "integration", extra)),
        ("e2e", _run("e2e", "e2e", extra)),
    ]
    raise typer.Exit(_report(results))


if __name__ == "__main__":
    sys.exit(app())
