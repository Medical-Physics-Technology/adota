"""Module-size ratchet.

CLAUDE.md caps a module at 500 lines, split by role rather than at an arbitrary
cut. The repository does not satisfy that today: two `src/` modules and sixteen
`scripts/` are over, and splitting them is planned work, not a merge blocker. So
this check is a ratchet rather than an absolute limit -- a file listed in the
baseline may stay over the cap but must not grow, and any file not listed must
stay under it.

Shrinking a baselined file below its recorded size is fine and expected; rerun
with ``--update`` after a split to record the new number.

Usage:
    uv run --no-project --with typer python ci/check_module_size.py
    uv run --no-project --with typer python ci/check_module_size.py --update
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, Dict, List

import typer

LIMIT = 500
ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).with_name("module_size_baseline.txt")


def tracked_python_files() -> List[Path]:
    """Every tracked .py file under src/ and scripts/ (git, so no build junk)."""
    out = subprocess.run(["git", "ls-files", "src/**/*.py", "scripts/**/*.py",
                          "src/*.py", "scripts/*.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines() if line]


def line_counts() -> Dict[str, int]:
    """Line count per repository-relative path."""
    return {str(path.relative_to(ROOT)): len(path.read_text(encoding="utf-8").splitlines())
            for path in tracked_python_files()}


def read_baseline() -> Dict[str, int]:
    """The recorded size of each known offender."""
    if not BASELINE.exists():
        return {}
    entries = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            count, name = line.split(None, 1)
            entries[name.strip()] = int(count)
    return entries


def write_baseline(counts: Dict[str, int]) -> None:
    """Rewrite the baseline from the current sizes."""
    over = sorted(((n, f) for f, n in counts.items() if n > LIMIT), reverse=True)
    lines = ["# Modules that exceed the 500-line rule today, with their current size.",
             "# New files must stay under the limit; these must not grow. See",
             "# docs/scripts_refactor_plan.md for the planned splits.", ""]
    lines += [f"{count} {name}" for count, name in over]
    BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def violations(counts: Dict[str, int], baseline: Dict[str, int]) -> List[str]:
    """Every new or growing offender, as messages."""
    messages = []
    for name, count in sorted(counts.items()):
        if count <= LIMIT:
            continue
        if name not in baseline:
            messages.append(f"{name}: {count} lines, over the {LIMIT}-line limit "
                            f"(split it by role, or add it to the baseline deliberately)")
        elif count > baseline[name]:
            messages.append(f"{name}: grew from {baseline[name]} to {count} lines "
                            f"while already over the limit")
    return messages


def main(
    update: Annotated[bool, typer.Option("--update",
                                         help="Rewrite the baseline instead of checking.")] = False,
) -> None:
    """Fail when a module is newly over the line limit, or grew while already over."""
    counts = line_counts()
    if update:
        write_baseline(counts)
        typer.echo(f"baseline rewritten: {sum(c > LIMIT for c in counts.values())} entries")
        return

    baseline = read_baseline()
    for name in sorted(set(baseline) - set(counts)):
        typer.echo(f"note: {name} is in the baseline but no longer exists; rerun with --update")

    if messages := violations(counts, baseline):
        typer.secho("module-size ratchet failed:", fg=typer.colors.RED, bold=True)
        for message in messages:
            typer.echo(f"  - {message}")
        raise typer.Exit(code=1)
    typer.secho(f"module-size ratchet passed ({len(baseline)} known offenders, none grew)",
                fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.run(main)
