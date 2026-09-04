"""The CI guard that requires a report record for every code change.

The guard lives in ``ci/`` rather than in a package, so it is loaded by path.
Its decision logic is a pure function, so the tests exercise that rather than
the typer wrapper; the four paths are what CI depends on.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

GUARD_PATH = Path(__file__).resolve().parents[2] / "ci" / "check_report_record.py"


def _guard():
    spec = importlib.util.spec_from_file_location("report_guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changed, title, expected", [
    (["src/foo.py", "tests/test_foo.py"], "", 1),      # code, no record: blocked
    (["src/foo.py", "reports"], "", 0),                # pointer moved: allowed
    (["scripts/foo.py"], "Tidy up [skip-report]", 0),  # explicit opt-out
    (["README.md", "docs/x.md"], "", 0),               # no code changed
    ([], "", 0),                                       # empty diff
])
def test_guard_decisions(changed, title, expected):
    code, message = _guard().decide(changed, title)
    assert code == expected, message


def test_blocked_message_names_the_way_out():
    _, message = _guard().decide(["src/foo.py"])
    assert "scripts/report.py new change" in message and "[skip-report]" in message


def test_guard_is_inert_without_the_submodule():
    """A checkout with no `reports` submodule must not block anything."""
    assert _guard().decide(["src/foo.py"], has_submodule=False)[0] == 0


def test_submodule_detection(tmp_path: Path):
    guard = _guard()
    assert guard.submodule_configured(tmp_path) is False
    (tmp_path / ".gitmodules").write_text('[submodule "reports"]\n\tpath = reports\n')
    assert guard.submodule_configured(tmp_path) is True
