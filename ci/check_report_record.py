"""Every code change carries a report record.

The records live in a private submodule mounted at `reports/`, so this check
never reads their content: it only asks whether the submodule pointer moved in
this pull request. That is enough, because a new or edited record is a commit in
the records repository, which advances the pointer.

    python ci/check_report_record.py --base <sha>

Passes when no code changed, when the pointer moved, when the pull request says
`[skip-report]`, or when the submodule is not wired up yet.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE_PREFIXES = ("src/", "scripts/", "tests/", "ci/", ".github/")
SUBMODULE = "reports"
SKIP_TOKEN = "[skip-report]"


def changed_files(base: str) -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base commit of the pull request.")
    parser.add_argument("--title", default=os.environ.get("PR_TITLE", ""),
                        help="Pull request title, searched for the skip token.")
    args = parser.parse_args()

    if not (ROOT / ".gitmodules").exists() or SUBMODULE not in \
            (ROOT / ".gitmodules").read_text(encoding="utf-8"):
        print(f"note: no `{SUBMODULE}` submodule yet, skipping the report check")
        return 0
    if SKIP_TOKEN in args.title:
        print(f"note: {SKIP_TOKEN} in the pull request title, skipping")
        return 0

    files = changed_files(args.base)
    code = [f for f in files if f.startswith(CODE_PREFIXES)]
    if not code:
        print("no code changed, no record required")
        return 0
    if SUBMODULE in files:
        print(f"record present: the `{SUBMODULE}` pointer moved in this pull request")
        return 0

    print(f"this pull request changes {len(code)} code files but adds no report record.")
    print("Write one with:")
    print("  uv run python scripts/report.py new change \"<title>\"")
    print(f"then commit it in the `{SUBMODULE}` submodule and commit the pointer here.")
    print(f"For a change that genuinely needs no record, put {SKIP_TOKEN} in the title.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
