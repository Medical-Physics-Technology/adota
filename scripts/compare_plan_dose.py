"""Visual comparison of two completed plan dose maps: ADoTA vs MCsquare.

Loads the CT, the accumulated ``Dose_ADoTA.mhd`` and the MCsquare ``Dose.mhd``
from a plan directory and renders an orthogonal-view + depth-profile comparison
via ``src.figures.plan_comparison.plan_dose_comparison``.

Usage:
    uv run python scripts/compare_plan_dose.py --plan-dir /scratch/.../<plan>
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

import SimpleITK as sitk
import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.dose_scaling import load_dose_gy
from src.beamlets.structures import load_oriented_structures
from src.figures.dvh_comparison import dvh_comparison_figure, write_dvh_metrics_json
from src.figures.plan_comparison import plan_dose_comparison
from src.loaders.plan_directory import load_plan_directory

app = typer.Typer(help="Compare ADoTA vs MCsquare plan dose maps.")


@app.command()
def main(
    plan_dir: Annotated[
        Path, typer.Option(help="Plan directory with CT.mhd, Dose_ADoTA.mhd, Dose.mhd.")
    ],
    adota_dose: Annotated[
        str, typer.Option(help="ADoTA dose filename in the plan dir.")
    ] = "Dose_ADoTA.mhd",
    mc_dose: Annotated[
        str, typer.Option(help="MCsquare dose filename in the plan dir.")
    ] = "Dose.mhd",
    output: Annotated[
        Optional[Path], typer.Option(help="Output figure path (default: <plan_dir>/dose_comparison).")
    ] = None,
    slice_zyx: Annotated[
        Optional[str], typer.Option(help="Slice index 'z,y,x' (default: MC peak voxel).")
    ] = None,
    ct_window: Annotated[
        str, typer.Option(help="CT HU window 'vmin,vmax'.")
    ] = "-200,400",
) -> None:
    """Render the ADoTA vs MCsquare dose comparison figure (doses in Gy)."""
    plan_dir = plan_dir if plan_dir.is_absolute() else PROJECT_ROOT / plan_dir

    # The plan + BDL are needed to convert eV/g/proton -> Gy (OpenTPS method).
    plan_directory = load_plan_directory(plan_dir)
    bdl = BeamDataLibrary.from_file(plan_directory.bdl_path)

    ct = sitk.GetArrayFromImage(plan_directory.ct)
    dose_a = sitk.GetArrayFromImage(
        load_dose_gy(plan_dir / adota_dose, plan_directory.plan, bdl)
    )
    dose_b = sitk.GetArrayFromImage(
        load_dose_gy(plan_dir / mc_dose, plan_directory.plan, bdl)
    )

    slice_index = None
    if slice_zyx is not None:
        slice_index = tuple(int(v) for v in slice_zyx.split(","))
    lo, hi = (float(v) for v in ct_window.split(","))

    out = output or (plan_dir / "dose_comparison")
    paths = plan_dose_comparison(
        ct,
        dose_a,  # ADoTA, in Gy
        dose_b,  # MCsquare, in Gy
        str(out),
        labels=("ADoTA", "MCsquare"),
        slice_zyx=slice_index,
        ct_window=(lo, hi),
        dose_unit="Gy",
    )
    typer.echo(f"Wrote dose comparison: {plan_dir / 'dose_comparison.png'}")

    # DVH comparison (structures oriented onto the dose grid).
    try:
        structures, _flips = load_oriented_structures(plan_directory)
        spacing = plan_directory.ct.GetSpacing()
        dvh_comparison_figure(
            structures, dose_a, dose_b, spacing,
            str(plan_dir / "dvh_comparison"), labels=("ADoTA", "MCsquare"),
        )
        write_dvh_metrics_json(
            plan_dir / "dvh_metrics.json", structures, dose_a, dose_b, spacing,
            labels=("ADoTA", "MCsquare"),
        )
        typer.echo(f"Wrote DVH comparison: {plan_dir / 'dvh_comparison.png'}")
        typer.echo(f"Wrote DVH metrics:    {plan_dir / 'dvh_metrics.json'}")
    except ValueError as exc:
        typer.echo(f"Skipping DVH comparison: {exc}")


if __name__ == "__main__":
    app()
