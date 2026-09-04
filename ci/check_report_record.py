"""Every code change carries a report record.

The records live in a private submodule mounted at `reports/`, so this check
never reads their content: it only asks whether the submodule pointer moved in
this pull request. That is enough, because a new or edited record is a commit in
the records repository, which advances the pointer.

Usage:
    uv run --no-project --with typer python ci/check_report_record.py --base <sha>

Passes when no code changed, when the pointer moved, when the pull request title
says `[skip-report]`, or when the submodule is not wired up yet.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, List, Optional, Sequence, Tuple

import typer

ROOT = Path(__file__).resolve().parent.parent
CODE_PREFIXES = ("src/", "scripts/", "tests/", "ci/", ".github/")
SUBMODULE = "reports"
SKIP_TOKEN = "[skip-report]"

MISSING_RECORD = f"""this pull request changes {{n}} code files but adds no report record.
Write one with:
  uv run python scripts/report.py new change "<title>"
then commit it in the `{SUBMODULE}` submodule and commit the pointer here.
For a change that genuinely needs no record, put {SKIP_TOKEN} in the title."""


def changed_files(base: str) -> List[str]:
    """Files this branch changes relative to ``base``."""
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def submodule_configured(root: Path = ROOT) -> bool:
    """Whether the records submodule is wired into this checkout."""
    modules = root / ".gitmodules"
    return modules.exists() and SUBMODULE in modules.read_text(encoding="utf-8")


def decide(files: Sequence[str], title: str = "",
           has_submodule: bool = True) -> Tuple[int, str]:
    """``(exit_code, message)`` for one pull request. Pure, so it is testable."""
    if not has_submodule:
        return 0, f"note: no `{SUBMODULE}` submodule yet, skipping the report check"
    if SKIP_TOKEN in title:
        return 0, f"note: {SKIP_TOKEN} in the pull request title, skipping"
    code = [f for f in files if f.startswith(CODE_PREFIXES)]
    if not code:
        return 0, "no code changed, no record required"
    if SUBMODULE in files:
        return 0, f"record present: the `{SUBMODULE}` pointer moved in this pull request"
    return 1, MISSING_RECORD.format(n=len(code))


def main(
    base: Annotated[str, typer.Option(help="Base commit of the pull request.")],
    title: Annotated[Optional[str], typer.Option(envvar="PR_TITLE",
                                                 help="Pull request title, searched "
                                                      "for the skip token.")] = "",
) -> None:
    """Fail when a pull request changes code without adding a report record."""
    code, message = decide(changed_files(base), title or "", submodule_configured())
    typer.secho(message, fg=typer.colors.RED if code else None)
    if code:
        raise typer.Exit(code=code)


if __name__ == "__main__":
    typer.run(main)
