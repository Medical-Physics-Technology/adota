"""Parsing and writing of report records.

A record is one Markdown file with a YAML frontmatter block::

    ---
    id: EXP-0007
    ...
    ---

    ## What was done
    ...

The frontmatter carries everything machine-readable (ids, data sources, metric
rows, artifact paths); the body carries the narrative under fixed headings. This
module knows only the file format -- what the fields *mean* is
:mod:`src.reports.schema`, and the flattening into tables is
:mod:`src.reports.index`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

SCHEMA_PATH = Path(__file__).with_name("schema.yaml")
TEMPLATE_DIR = Path(__file__).parent / "templates"

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


class RecordError(ValueError):
    """A record file that cannot be parsed at all (as opposed to one that
    parses but violates the schema, which the validator reports as a message)."""


@dataclass(frozen=True)
class Record:
    """One parsed record file."""

    path: Path
    front: dict
    sections: Dict[str, str]

    @property
    def id(self) -> str:
        return str(self.front.get("id", ""))

    @property
    def kind(self) -> str:
        return str(self.front.get("kind", ""))

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.id} ({self.path.name})"


def split_sections(body: str) -> Dict[str, str]:
    """Map ``## Heading`` to the text beneath it, in file order."""
    sections: Dict[str, str] = {}
    matches = list(_HEADING.finditer(body))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[match.group(1)] = body[match.end():end].strip()
    return sections


def parse_record(text: str, path: Optional[Path] = None) -> Record:
    """Parse one record from its file contents."""
    match = _FRONTMATTER.match(text)
    if match is None:
        raise RecordError(f"{path or '<string>'}: no YAML frontmatter block")
    try:
        front = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise RecordError(f"{path or '<string>'}: frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(front, dict):
        raise RecordError(f"{path or '<string>'}: frontmatter is not a mapping")
    return Record(path=path or Path("<string>"), front=front,
                  sections=split_sections(match.group(2)))


def load_record(path: Path) -> Record:
    """Parse the record stored at ``path``."""
    return parse_record(path.read_text(encoding="utf-8"), path)


def load_records(root: Path, schema: Optional[dict] = None) -> List[Record]:
    """Load every record under ``root``, sorted by id.

    Only the directories the schema names are searched, so a records repository
    can hold a README, figures or scratch files without confusing the loader.
    """
    directories = (schema or load_schema())["directories"].values()
    records: List[Record] = []
    for directory in directories:
        for path in sorted((root / directory).glob("*.md")):
            records.append(load_record(path))
    return sorted(records, key=lambda r: r.id)


def load_schema(path: Path = SCHEMA_PATH) -> dict:
    """Load the record schema."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def next_id(root: Path, kind: str, schema: Optional[dict] = None) -> str:
    """The next free id for ``kind`` (``EXP-0008`` after ``EXP-0007``)."""
    schema = schema or load_schema()
    prefix = {"change": "CHG", "experiment": "EXP"}[kind]
    directory = root / schema["directories"][kind]
    used = [int(m.group(1)) for path in directory.glob("*.md")
            if (m := re.match(rf"{prefix}-(\d{{4}})", path.name))]
    return f"{prefix}-{max(used, default=0) + 1:04d}"


def slugify(title: str) -> str:
    """``"Retrospective pool AL" -> "retrospective-pool-al"``."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60].rstrip("-")
