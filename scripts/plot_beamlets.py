"""Plot constructed beamlet inputs (CT crop + flux) for correctness checks.

Reads the per-spot ``{id}_ct.npy`` / ``{id}_flux.npy`` / ``{id}_sim_res.json``
files produced by the extraction stage and renders a CT/flux axial+sagittal
mosaic per spot via ``src.figures.single_beam.beamlet_input_figure``.

Usage:
    uv run python scripts/plot_beamlets.py \\
        --beamlets-dir /scratch/.../adota_beamlets --n 5
"""

import json
import sys
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.figures.single_beam import beamlet_input_figure

app = typer.Typer(help="Render CT-crop + flux figures for extracted beamlets.")


def _discover_spot_ids(beamlets_dir: Path) -> List[str]:
    """Return the spot ids present in a beamlets directory, sorted."""
    return sorted(p.name[: -len("_ct.npy")] for p in beamlets_dir.glob("*_ct.npy"))


@app.command()
def main(
    beamlets_dir: Annotated[
        Path,
        typer.Option(help="Directory with {id}_ct.npy / _flux.npy / _sim_res.json."),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Output dir (default: <beamlets_dir>/figures)."),
    ] = None,
    spot_ids: Annotated[
        Optional[str], typer.Option(help="Comma-separated spot ids to plot.")
    ] = None,
    n: Annotated[
        Optional[int], typer.Option(help="Plot only the first N discovered spots.")
    ] = None,
    ct_window: Annotated[
        Optional[str],
        typer.Option(help="CT HU window as 'vmin,vmax' (default: per-crop min/max)."),
    ] = None,
) -> None:
    """Render a CT/flux mosaic per spot."""
    beamlets_dir = beamlets_dir if beamlets_dir.is_absolute() else PROJECT_ROOT / beamlets_dir
    if not beamlets_dir.is_dir():
        raise typer.BadParameter(f"Beamlets directory not found: {beamlets_dir}")

    out_dir = output_dir or (beamlets_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if spot_ids is not None:
        ids = [s.strip() for s in spot_ids.split(",") if s.strip()]
    else:
        ids = _discover_spot_ids(beamlets_dir)
    if n is not None:
        ids = ids[:n]
    if not ids:
        raise typer.BadParameter(f"No beamlet files found in {beamlets_dir}")

    window = None
    if ct_window is not None:
        lo, hi = (float(v) for v in ct_window.split(","))
        window = (lo, hi)

    for spot_id in ids:
        ct = np.load(beamlets_dir / f"{spot_id}_ct.npy")
        flux = np.load(beamlets_dir / f"{spot_id}_flux.npy")
        meta_path = beamlets_dir / f"{spot_id}_sim_res.json"
        energy = None
        angles = None
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            energy = meta.get("initial_energy")
            angles = meta.get("simulation_log", {}).get("beamlet_angles")
            if angles is not None:
                angles = tuple(angles)

        paths = beamlet_input_figure(
            ct,
            flux,
            str(out_dir / f"{spot_id}_input"),
            initial_energy=energy,
            beamlet_angles=angles,
            spot_id=spot_id,
            ct_window=window,
        )
        typer.echo(f"{spot_id}: {paths[-1]}")

    typer.echo(f"Wrote {len(ids)} figure(s) to {out_dir}")


if __name__ == "__main__":
    app()
