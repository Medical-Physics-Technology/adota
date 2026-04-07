"""Add one or more record IDs to an exclusion index file."""

from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(help="Add record IDs to an exclusion index file.")


@app.command()
def main(
    excluded_indexes_file: Path = typer.Option(
        ..., help="Path to the exclusion index file"
    ),
    id: Optional[str] = typer.Option(
        None, help="Single record ID to add (e.g. 004bf6eb-…)"
    ),
    ids: Optional[List[str]] = typer.Option(
        None, help="Multiple record IDs to add (repeat --ids for each)"
    ),
) -> None:
    """Append the given ID(s) to the exclusion file if not already present.

    Specify a single ID with --id or multiple IDs by repeating --ids::

        uv run python scripts/add_excluded_index.py \\
            --excluded-indexes-file path/to/file.txt \\
            --id 004bf6eb-ca00-4fb5-b84f-4f43f45760d4

        uv run python scripts/add_excluded_index.py \\
            --excluded-indexes-file path/to/file.txt \\
            --ids aaa-... --ids bbb-... --ids ccc-...
    """
    # Collect all requested IDs
    all_ids: list[str] = []
    if id is not None:
        all_ids.append(id)
    if ids:
        all_ids.extend(ids)

    if not all_ids:
        typer.echo("Error: provide at least one ID via --id or --ids")
        raise typer.Exit(code=1)

    # Load existing entries
    existing: list[str] = []
    if excluded_indexes_file.exists():
        existing = excluded_indexes_file.read_text().splitlines()
    else:
        excluded_indexes_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine which IDs are new
    already_present = [rid for rid in all_ids if rid in existing]
    to_add = [rid for rid in all_ids if rid not in existing]

    for rid in already_present:
        typer.echo(f"ID '{rid}' is already in {excluded_indexes_file} – skipped")

    if not to_add:
        typer.echo("Nothing to add – all IDs already present")
        raise typer.Exit(code=1)

    # Append new IDs (ensuring trailing newline)
    with open(excluded_indexes_file, "a+") as f:
        f.seek(0, 2)  # move to end
        if f.tell() > 0:
            f.seek(f.tell() - 1)
            last_char = f.read(1)
            if last_char != "\n":
                f.write("\n")
        for rid in to_add:
            f.write(f"{rid}\n")

    for rid in to_add:
        typer.echo(f"Added '{rid}' to {excluded_indexes_file}")
    typer.echo(f"Total added: {len(to_add)}")


if __name__ == "__main__":
    app()
