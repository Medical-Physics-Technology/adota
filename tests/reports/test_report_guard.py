"""The CI guard that requires a report record for every code change.

The guard lives in ``ci/`` rather than in a package, so it is loaded by path.
Its four decision paths are what CI depends on, so they are pinned here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

import pytest

GUARD_PATH = Path(__file__).resolve().parents[2] / "ci" / "check_report_record.py"


def _guard(changed: List[str]):
    """Load the guard with its git lookup replaced by a fixed file list."""
    spec = importlib.util.spec_from_file_location("report_guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.changed_files = lambda base: changed
    return module


def _run(monkeypatch, changed: List[str], title: str = "") -> int:
    module = _guard(changed)
    monkeypatch.setattr("sys.argv", ["guard", "--base", "HEAD", "--title", title])
    return module.main()


@pytest.mark.parametrize("changed, title, expected", [
    (["src/foo.py", "tests/test_foo.py"], "", 1),      # code, no record: blocked
    (["src/foo.py", "reports"], "", 0),                # pointer moved: allowed
    (["scripts/foo.py"], "Tidy up [skip-report]", 0),  # explicit opt-out
    (["README.md", "docs/x.md"], "", 0),               # no code changed
    ([], "", 0),                                       # empty diff
])
def test_guard_decisions(monkeypatch, changed, title, expected):
    assert _run(monkeypatch, changed, title) == expected


def test_guard_is_inert_without_the_submodule(monkeypatch, tmp_path):
    """A checkout with no `reports` submodule must not block anything."""
    module = _guard(["src/foo.py"])
    monkeypatch.setattr(module, "ROOT", tmp_path)          # no .gitmodules here
    monkeypatch.setattr("sys.argv", ["guard", "--base", "HEAD"])
    assert module.main() == 0
