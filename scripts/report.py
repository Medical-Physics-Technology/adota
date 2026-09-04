"""Research report records: scaffold, validate, index and export.

The records themselves live in the private repository mounted at ``reports/``;
this CLI is the public tooling around them (:mod:`src.reports`).

Usage:
    uv run python scripts/report.py new experiment "Retrospective pool AL"
    uv run python scripts/report.py validate
    uv run python scripts/report.py index          # index.csv + metrics.csv + .db
    uv run python scripts/report.py export --tag active-learning -o chapter.md

`validate` is what CI runs, and what `tests/reports/` exercises on fixtures.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from string import Template
from typing import Annotated, List, Optional

import typer

from src.reports import build_index, load_records, load_schema, next_id, slugify, validate_tree
from src.reports.records import TEMPLATE_DIR

DEFAULT_ROOT = Path(os.environ.get("ADOTA_REPORTS_DIR", "reports"))

app = typer.Typer(help="Manage the research report records.", add_completion=False,
                  no_args_is_help=True)

RootOption = Annotated[Path, typer.Option("--root", help="Records repository root.")]


def _require_root(root: Path) -> Path:
    """Fail with an actionable message when the records repo is not checked out."""
    if not root.exists():
        raise typer.BadParameter(
            f"{root} does not exist. The records live in a private submodule: run "
            "`git submodule update --init reports`, or point --root elsewhere.")
    return root


def _current_branch() -> str:
    head = Path(".git/HEAD")
    if head.exists() and (text := head.read_text().strip()).startswith("ref: refs/heads/"):
        return text.split("refs/heads/", 1)[1]
    return ""


@app.command()
def new(
    kind: Annotated[str, typer.Argument(help="change | experiment")],
    title: Annotated[str, typer.Argument(help="Human-readable title.")],
    root: RootOption = DEFAULT_ROOT,
) -> None:
    """Scaffold a new record with the next free id."""
    schema = load_schema()
    if kind not in schema["directories"]:
        raise typer.BadParameter(f"kind must be one of {sorted(schema['directories'])}")
    _require_root(root)

    record_id = next_id(root, kind, schema)
    directory = root / schema["directories"][kind]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record_id}-{slugify(title)}.md"
    if path.exists():
        raise typer.BadParameter(f"{path} already exists")

    template = Template((TEMPLATE_DIR / f"{kind}.md").read_text(encoding="utf-8"))
    path.write_text(template.substitute(id=record_id, title=title, date=date.today().isoformat(),
                                        branch=_current_branch()), encoding="utf-8")
    typer.secho(f"created {path}", fg=typer.colors.GREEN)


@app.command()
def validate(root: RootOption = DEFAULT_ROOT) -> None:
    """Check every record against the schema; exit non-zero on any violation."""
    _require_root(root)
    problems = validate_tree(root)
    n_records = len(load_records(root))
    if not problems:
        typer.secho(f"{n_records} records, all valid", fg=typer.colors.GREEN)
        return
    for path, messages in sorted(problems.items()):
        typer.secho(path, fg=typer.colors.RED, bold=True)
        for message in messages:
            typer.echo(f"  - {message}")
    raise typer.Exit(code=1)


@app.command()
def index(
    root: RootOption = DEFAULT_ROOT,
    sqlite: Annotated[bool, typer.Option(help="Also write index.db.")] = True,
) -> None:
    """Rebuild index.csv, metrics.csv and (optionally) the SQLite view."""
    _require_root(root)
    records = load_records(root)
    written = build_index(records, root, root / "index.db" if sqlite else None)
    for label, path in written.items():
        typer.echo(f"{label:8s} -> {path}")
    typer.secho(f"{len(records)} records indexed", fg=typer.colors.GREEN)


@app.command()
def export(
    out: Annotated[Path, typer.Option("-o", "--out", help="Output Markdown file.")],
    root: RootOption = DEFAULT_ROOT,
    kind: Annotated[Optional[str], typer.Option(help="Only this kind.")] = None,
    tag: Annotated[Optional[List[str]], typer.Option(help="Only records with this tag.")] = None,
    status: Annotated[Optional[str], typer.Option(help="Only this status.")] = None,
) -> None:
    """Concatenate the matching records into one document, for paper writing."""
    _require_root(root)
    selected = []
    for record in load_records(root):
        tags = {str(t) for t in (record.front.get("tags") or [])}
        if kind and record.kind != kind:
            continue
        if status and record.front.get("status") != status:
            continue
        if tag and not set(tag) & tags:
            continue
        selected.append(record)

    lines = ["# Report export", "",
             f"{len(selected)} records: " + ", ".join(r.id for r in selected) or "none", ""]
    for record in selected:
        lines += [f"## {record.id} {record.front.get('title', '')}", "",
                  f"*{record.front.get('date', '')} | {record.front.get('status', '')} | "
                  f"branch `{record.front.get('branch', '')}`*", ""]
        for heading, body in record.sections.items():
            lines += [f"### {heading}", "", body, ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    typer.secho(f"wrote {out} ({len(selected)} records)", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
