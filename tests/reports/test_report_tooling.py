"""The report record tooling, exercised on synthetic records.

The real records live in a private submodule that may not be checked out, so
every test here builds its own tree in a tmp directory. That keeps the tooling
covered by the ordinary unit suite on any machine.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.reports import (
    build_index,
    load_records,
    load_schema,
    next_id,
    validate_tree,
)
from src.reports.records import parse_record

SCHEMA = load_schema()


def _body(**overrides: str) -> str:
    """A body with every required section filled, unless overridden."""
    sections = {heading: f"Text for {heading}." for heading in SCHEMA["sections"]["required"]}
    sections.update(overrides)
    return "\n\n".join(f"## {heading}\n\n{text}" for heading, text in sections.items()) + "\n"


def _experiment(record_id: str = "EXP-0001", **front: str) -> str:
    fields = {
        "id": record_id, "title": "A test experiment", "kind": "experiment",
        "date": "2026-09-04", "status": "complete", "branch": "exp/test",
        "hypothesis": "Something is true.",
    }
    fields.update(front)
    lines = [f"{key}: {value}" for key, value in fields.items()]
    lines += ["data:", "  - {name: reference-run, n: 100}",
              "metrics:", "  - {name: pearson, value: 0.85, split: test, n: 42}"]
    return "---\n" + "\n".join(lines) + "\n---\n\n" + _body()


def _change(record_id: str = "CHG-0001", **front: str) -> str:
    fields = {"id": record_id, "title": "A test change", "kind": "change",
              "date": "2026-09-04", "status": "complete", "branch": "chore/test"}
    fields.update(front)
    return "---\n" + "\n".join(f"{k}: {v}" for k, v in fields.items()) + "\n---\n\n" + _body()


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """A records repository holding one valid record of each kind."""
    (tmp_path / "experiments").mkdir()
    (tmp_path / "changes").mkdir()
    (tmp_path / "experiments" / "EXP-0001-a-test.md").write_text(_experiment())
    (tmp_path / "changes" / "CHG-0001-a-test.md").write_text(_change())
    return tmp_path


def test_valid_tree_has_no_problems(tree: Path):
    assert validate_tree(tree) == {}
    assert [record.id for record in load_records(tree)] == ["CHG-0001", "EXP-0001"]


def test_missing_section_is_reported(tree: Path):
    path = tree / "experiments" / "EXP-0001-a-test.md"
    path.write_text(path.read_text().replace("## What went wrong\n\nText for What went wrong.", ""))
    messages = validate_tree(tree)[str(path)]
    assert any("What went wrong" in message for message in messages)


def test_placeholder_section_is_rejected(tree: Path):
    path = tree / "experiments" / "EXP-0001-a-test.md"
    path.write_text(path.read_text().replace("Text for Results: negative.", "TODO"))
    assert any("placeholder" in m for m in validate_tree(tree)[str(path)])


def test_explicit_none_is_accepted(tree: Path):
    """'None.' is a real answer; the validator must not force invention."""
    path = tree / "changes" / "CHG-0001-a-test.md"
    path.write_text(path.read_text().replace(
        "Text for Results: negative.", "None. This change is a refactor with no results."))
    assert validate_tree(tree) == {}


def test_missing_required_field_is_reported(tree: Path):
    path = tree / "experiments" / "EXP-0001-a-test.md"
    path.write_text(path.read_text().replace("hypothesis: Something is true.\n", ""))
    assert any("hypothesis" in m for m in validate_tree(tree)[str(path)])


def test_bad_status_enum_is_reported(tree: Path):
    path = tree / "changes" / "CHG-0001-a-test.md"
    path.write_text(path.read_text().replace("status: complete", "status: nearly"))
    assert any("not one of" in m for m in validate_tree(tree)[str(path)])


def test_wrong_directory_is_reported(tmp_path: Path):
    (tmp_path / "changes").mkdir()
    (tmp_path / "experiments").mkdir()
    path = tmp_path / "changes" / "EXP-0001-misfiled.md"
    path.write_text(_experiment())
    assert any("belongs in experiments/" in m for m in validate_tree(tmp_path)[str(path)])


def test_duplicate_ids_are_reported(tree: Path):
    twin = tree / "experiments" / "EXP-0001-twin.md"
    twin.write_text(_experiment())
    problems = validate_tree(tree)
    assert any("duplicate id" in m for messages in problems.values() for m in messages)


def test_unresolved_link_is_reported(tree: Path):
    path = tree / "changes" / "CHG-0001-a-test.md"
    path.write_text(_change(related="[EXP-9999]"))
    assert any("unknown record" in m for m in validate_tree(tree)[str(path)])


def test_metric_row_needs_a_value(tree: Path):
    path = tree / "experiments" / "EXP-0001-a-test.md"
    path.write_text(path.read_text().replace(
        "  - {name: pearson, value: 0.85, split: test, n: 42}", "  - {name: pearson}"))
    assert any("needs `name` and `value`" in m for m in validate_tree(tree)[str(path)])


def test_next_id_increments(tree: Path):
    assert next_id(tree, "experiment", SCHEMA) == "EXP-0002"
    assert next_id(tree, "change", SCHEMA) == "CHG-0002"


def test_index_flattens_records_and_metrics(tree: Path, tmp_path: Path):
    out = tmp_path / "out"
    written = build_index(load_records(tree), out, out / "index.db")

    rows = list(csv.DictReader(written["index"].open()))
    assert {row["id"] for row in rows} == {"CHG-0001", "EXP-0001"}
    assert next(r for r in rows if r["id"] == "EXP-0001")["n_metrics"] == "1"

    # Records outside the output directory keep an absolute path; the ones the
    # index sits beside (the normal case) are recorded relative to it.
    relative = build_index(load_records(tree), tree)
    paths = [row["path"] for row in csv.DictReader(relative["index"].open())]
    assert sorted(paths) == ["changes/CHG-0001-a-test.md", "experiments/EXP-0001-a-test.md"]

    metrics = list(csv.DictReader(written["metrics"].open()))
    assert metrics == [{"record_id": "EXP-0001", "date": "2026-09-04", "name": "pearson",
                        "value": "0.85", "unit": "", "split": "test", "n": "42"}]
    assert written["sqlite"].exists()


def test_sections_are_split_in_file_order():
    record = parse_record("---\nid: EXP-0001\nkind: experiment\n---\n\n"
                          "## First\n\nOne.\n\n## Second\n\nTwo.\n")
    assert list(record.sections) == ["First", "Second"]
    assert record.sections["First"] == "One."
