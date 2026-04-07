"""Remove a single record from an HDF5 file by its ID."""

import sys
from pathlib import Path

import h5py
import typer

app = typer.Typer(help="Remove a record from an HDF5 file by ID.")


@app.command()
def main(
    path_to_h5py: Path = typer.Option(..., help="Path to the HDF5 file"),
    id: str = typer.Option(
        ..., help="Record ID to remove (e.g. 004bf6eb-ca00-4fb5-b84f-4f43f45760d4)"
    ),
) -> None:
    """Delete the record with the given ID from the HDF5 file."""
    if not path_to_h5py.exists():
        typer.echo(f"Error: file not found: {path_to_h5py}", err=True)
        raise typer.Exit(code=1)

    with h5py.File(path_to_h5py, "a") as f:
        if id not in f:
            typer.echo(f"Error: ID '{id}' not found in {path_to_h5py}")
            raise typer.Exit(code=1)

        del f[id]
        typer.echo(f"Removed '{id}' from {path_to_h5py}")


if __name__ == "__main__":
    app()
