"""Runtime against problem size on nested plan crops (EXP-0008, experiment D).

Cuts nested sub-volumes out of one or more plan dose pairs, centred on the
high-dose region, and times rungs 1, 3 and 4 on each under one identical gamma
configuration and one timing protocol. The normalisation dose is pinned to the
full plan's reference maximum so the dose tolerance and cutoff do not move with
the crop. See :mod:`src.metrics.gamma_scaling_benchmark`.

Example::

    uv run python scripts/gamma_scaling_benchmark.py \\
        --plans LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4 \\
        --targets 144000,1000000,8000000,32000000 --repeats 5 --device cuda:0 \\
        --out /scratch/mstryja/<run>/D/scaling.json --provenance
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.benchmark_provenance import write_manifest  # noqa: E402
from src.metrics.gamma_beamlet_benchmark import RUNGS, GammaCase, environment_stamp  # noqa: E402
from src.metrics.gamma_benchmark import corpus_dir, corpus_skip_reason, load_plan_case  # noqa: E402
from src.metrics.gamma_scaling_benchmark import high_dose_centre, nested_crops, time_crop  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(help="Gamma runtime against problem size on nested plan crops.", add_completion=False)

DEFAULT_PLANS = "LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4"


@app.command()
def main(
    out: Path = typer.Option(...),
    plans: str = typer.Option(DEFAULT_PLANS, help="Comma-separated plan directory names."),
    targets: str = typer.Option("144000,1000000,8000000,32000000", help="Voxel counts; the full grid is added."),
    criteria: str = typer.Option("2/2", help='Comma-separated "percent/mm" criteria.'),
    cutoff: float = typer.Option(10.0),
    interp_fraction: int = typer.Option(10),
    max_gamma: float = typer.Option(2.0),
    rungs: str = typer.Option("rung1,rung3,rung4"),
    repeats: int = typer.Option(5),
    include_full: bool = typer.Option(True, help="Add the full grid as the largest crop."),
    device: str = typer.Option("cuda:0"),
    provenance: bool = typer.Option(False),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Time every (plan, crop, criterion, rung) combination."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    plan_names = [name.strip() for name in plans.split(",") if name.strip()]
    reason = corpus_skip_reason(plan_names)
    if reason:
        raise typer.BadParameter(reason)
    selected = [RUNGS[name.strip()] for name in rungs.split(",") if name.strip()]
    sizes = [int(token) for token in targets.split(",") if token.strip()]
    cases = []
    for token in criteria.split(","):
        percent, _, millimetres = token.strip().partition("/")
        cases.append(GammaCase(float(percent), float(millimetres), cutoff, interp_fraction, max_gamma))
    out.parent.mkdir(parents=True, exist_ok=True)
    if provenance:
        write_manifest(
            out.parent,
            device=device,
            repos={"adota": PROJECT_ROOT, "reports": PROJECT_ROOT / "reports"},
            inputs={f"{name}/{file}": corpus_dir() / name / file
                    for name in plan_names for file in ("Dose.raw", "Dose_ADoTA.raw")},
            extra={"result": str(out), "plans": plan_names, "targets": sizes, "repeats": repeats,
                   "rungs": [r.label for r in selected], "criteria": [c.as_params() for c in cases]},
        )

    rows: List[dict] = []
    crops_described: List[dict] = []
    for name in plan_names:
        case = load_plan_case(corpus_dir() / name)
        normalisation = float(case.dose_ref.max())
        centre = high_dose_centre(case.dose_ref)
        crops = nested_crops(case.dose_ref, case.dose_eval, sizes, centre)
        if not include_full:
            crops = [crop for crop in crops if crop.voxels != case.n_voxels]
        logger.info("[%s] centre zyx %s, %d crops: %s", name, centre, len(crops), [c.shape for c in crops])
        for crop in crops:
            for gamma_case in cases:
                cutoff_dose = gamma_case.lower_percent_dose_cutoff / 100.0 * normalisation
                description = {"plan": name, "centre_zyx": list(centre), "spacing_zyx": list(case.spacing_zyx),
                               "global_normalisation": normalisation, **crop.describe(cutoff_dose)}
                crops_described.append(description)
                for rung in selected:
                    row = time_crop(crop, case.spacing_zyx, gamma_case, rung, normalisation,
                                    device=device, repeats=repeats)
                    row.update({"plan": name, "criterion": f"{gamma_case.dose_percent_threshold:g}%/"
                                f"{gamma_case.distance_mm_threshold:g}mm/{cutoff:g}%",
                                "target_voxels": crop.target_voxels, "bounds_zyx": description["bounds_zyx"],
                                "n_above_cutoff": description["n_above_cutoff"]})
                    rows.append(row)
                    logger.info("[%s] %s voxels %.3g rung %d: median %.3f s (evaluated %d)", name,
                                row["criterion"], crop.voxels, rung.rung, row["seconds_median"], row["n_evaluated"])
                    payload = {"environment": environment_stamp(), "crops": crops_described, "rows": rows}
                    out.write_text(json.dumps(payload, indent=1) + "\n")
        del case, crops
    typer.echo(f"Wrote {len(rows)} scaling rows to {out}")


if __name__ == "__main__":
    app()
