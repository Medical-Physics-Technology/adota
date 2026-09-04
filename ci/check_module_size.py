"""Module-size ratchet.

CLAUDE.md caps a module at 500 lines, split by role rather than at an arbitrary
cut. The repository does not satisfy that today: two `src/` modules and sixteen
`scripts/` are over, and splitting them is planned work, not a merge blocker. So
this check is a ratchet rather than an absolute limit -- a file listed in the
baseline may stay over the cap but must not grow, and any file not listed must
stay under it.

Shrinking a baselined file below its recorded size is fine and expected; run
this script with --update after a split to record the new number.

    python ci/check_module_size.py            # fails on a new or growing offender
    python ci/check_module_size.py --update   # rewrite the baseline
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LIMIT = 500
ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).with_name("module_size_baseline.txt")


def tracked_python_files() -> list[Path]:
    """Every tracked .py file under src/ and scripts/ (git, so no build junk)."""
    out = subprocess.run(["git", "ls-files", "src/**/*.py", "scripts/**/*.py",
                          "src/*.py", "scripts/*.py"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines() if line]


def line_counts() -> dict[str, int]:
    counts = {}
    for path in tracked_python_files():
        counts[str(path.relative_to(ROOT))] = len(path.read_text(encoding="utf-8").splitlines())
    return counts


def read_baseline() -> dict[str, int]:
    if not BASELINE.exists():
        return {}
    entries = {}
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            count, name = line.split(None, 1)
            entries[name.strip()] = int(count)
    return entries


def write_baseline(counts: dict[str, int]) -> None:
    over = sorted(((n, f) for f, n in counts.items() if n > LIMIT), reverse=True)
    lines = ["# Modules that exceed the 500-line rule today, with their current size.",
             "# New files must stay under the limit; these must not grow. See",
             "# docs/scripts_refactor_plan.md for the planned splits.", ""]
    lines += [f"{count} {name}" for count, name in over]
    BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    counts = line_counts()
    if "--update" in sys.argv:
        write_baseline(counts)
        print(f"baseline rewritten: {sum(c > LIMIT for c in counts.values())} entries")
        return 0

    baseline = read_baseline()
    failures = []
    for name, count in sorted(counts.items()):
        if count <= LIMIT:
            continue
        if name not in baseline:
            failures.append(f"{name}: {count} lines, over the {LIMIT}-line limit "
                            f"(split it by role, or add it to the baseline deliberately)")
        elif count > baseline[name]:
            failures.append(f"{name}: grew from {baseline[name]} to {count} lines "
                            f"while already over the limit")
    for name in sorted(set(baseline) - set(counts)):
        print(f"note: {name} is in the baseline but no longer exists; run --update")

    if failures:
        print("module-size ratchet failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"module-size ratchet passed ({len(baseline)} known offenders, none grew)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
