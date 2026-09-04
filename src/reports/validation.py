"""Schema validation for report records.

Returns messages rather than raising, so one run reports every problem in a
records tree instead of stopping at the first. :func:`validate_tree` is what the
CLI and the tests call; the two halves are per-record checks (fields, types,
sections) and collection-level checks (unique ids, links that resolve).
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, Iterable, List

from src.reports.records import Record, load_records, load_schema

_SCALARS = {"str": str, "int": int, "float": (int, float), "list": list, "dict": dict}


def _type_error(field: str, value, spec) -> str:
    """Message if ``value`` does not match ``spec``, else an empty string."""
    if isinstance(spec, dict) and "enum" in spec:
        return "" if value in spec["enum"] else \
            f"{field}: {value!r} is not one of {spec['enum']}"
    if spec == "date":
        if isinstance(value, (dt.date, dt.datetime)):
            return ""
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return ""
        return f"{field}: {value!r} is not an ISO date (YYYY-MM-DD)"
    expected = _SCALARS.get(str(spec))
    if expected is None:
        return ""
    if isinstance(value, bool) and expected is not bool:
        return f"{field}: expected {spec}, got a boolean"
    return "" if isinstance(value, expected) else f"{field}: expected {spec}, got {type(value).__name__}"


def _check_fields(record: Record, spec: dict) -> List[str]:
    """Required-field presence and type checks for one field-spec block."""
    messages = []
    for field, field_spec in (spec.get("required") or {}).items():
        if record.front.get(field) in (None, "", [], {}):
            messages.append(f"missing required field: {field}")
            continue
        if message := _type_error(field, record.front[field], field_spec):
            messages.append(message)
    for field, field_spec in (spec.get("optional") or {}).items():
        if field in record.front and record.front[field] is not None:
            if message := _type_error(field, record.front[field], field_spec):
                messages.append(message)
    return messages


def _check_entries(record: Record) -> List[str]:
    """`data` entries need a name; `metrics` rows need a name and a value."""
    messages = []
    for entry in record.front.get("data") or []:
        if not isinstance(entry, dict) or not entry.get("name"):
            messages.append(f"data entry needs a `name`: {entry!r}")
    for row in record.front.get("metrics") or []:
        if not isinstance(row, dict) or not row.get("name") or "value" not in row:
            messages.append(f"metric row needs `name` and `value`: {row!r}")
    return messages


def _check_sections(record: Record, schema: dict) -> List[str]:
    """Every required heading present, non-trivial and not a placeholder."""
    rules = schema["sections"]
    messages = []
    for heading in rules["required"]:
        if heading not in record.sections:
            messages.append(f"missing required section: ## {heading}")
            continue
        body = record.sections[heading].strip()
        # Placeholder first: "TODO" is also too short, but naming it a placeholder
        # is the more useful message.
        if body.lower().strip(" .*_") in rules["placeholders"]:
            messages.append(f"section '{heading}' is a placeholder ({body!r})")
        elif len(body) < rules["min_chars"]:
            messages.append(f"section '{heading}' is empty or too short "
                            f"(write 'None.' if that is the answer)")
    return messages


def validate_record(record: Record, schema: dict) -> List[str]:
    """Every schema violation in one record."""
    kind = record.kind
    if kind not in schema["id_pattern"]:
        return [f"unknown kind: {kind!r} (expected one of {sorted(schema['id_pattern'])})"]

    messages = _check_fields(record, schema["frontmatter"]["common"])
    messages += _check_fields(record, schema["frontmatter"][kind])
    messages += _check_entries(record)
    messages += _check_sections(record, schema)

    if not re.match(schema["id_pattern"][kind], record.id):
        messages.append(f"id {record.id!r} does not match {schema['id_pattern'][kind]}")
    expected_dir = schema["directories"][kind]
    if record.path.parent.name != expected_dir:
        messages.append(f"a {kind} record belongs in {expected_dir}/, not "
                        f"{record.path.parent.name}/")
    if not record.path.name.startswith(record.id):
        messages.append(f"filename {record.path.name!r} should start with the id {record.id}")
    return messages


def validate_collection(records: Iterable[Record]) -> Dict[str, List[str]]:
    """Cross-record checks: ids unique, `related` and `supersedes` resolve."""
    records = list(records)
    known = {record.id for record in records}
    seen: Dict[str, Path] = {}
    problems: Dict[str, List[str]] = {}

    for record in records:
        messages = []
        if record.id in seen:
            messages.append(f"duplicate id, already used by {seen[record.id].name}")
        else:
            seen[record.id] = record.path
        links = list(record.front.get("related") or [])
        if supersedes := record.front.get("supersedes"):
            links.append(supersedes)
        for link in links:
            if str(link) not in known:
                messages.append(f"link to unknown record: {link}")
        if messages:
            problems[str(record.path)] = messages
    return problems


def validate_tree(root: Path) -> Dict[str, List[str]]:
    """Validate a whole records repository; ``{file: [messages]}``, empty if clean."""
    schema = load_schema()
    records = load_records(root, schema)
    problems: Dict[str, List[str]] = {}
    for record in records:
        if messages := validate_record(record, schema):
            problems[str(record.path)] = messages
    for path, messages in validate_collection(records).items():
        problems.setdefault(path, []).extend(messages)
    return problems
