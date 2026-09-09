"""Plan-scale gamma map agreement, compared in memory before any cast.

The plan-level ladder of CHG-0005 persisted every gamma map as float32 and
compared the stored files, so its voxel-level parity claim covers stored values,
not the float64 outputs the double-precision rungs actually produced. This
script recomputes the maps for the two plans that ladder used, under each plan's
own recorded gamma recipe, and compares them in RAM in their native dtype. The
maps are not written to disk: two 100 million-voxel float64 maps are 1.5 GiB,
which fits in memory but not sensibly in a run directory times 24. What is
saved is a SHA-256 digest of every map's raw bytes plus the comparison
statistics, so the result can be verified by anyone who recomputes a map.

Example::

    uv run python scripts/gamma_plan_map_agreement.py \\
        --plans LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4 \\
        --criteria 1/1/10,2/2/10,3/3/10 --device cuda:0 \\
        --out /scratch/mstryja/<run>/plan_maps.json --provenance
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import List

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.gamma_beamlet_benchmark import RUNGS, environment_stamp  # noqa: E402
from src.metrics.gamma_benchmark import corpus_dir, corpus_skip_reason, load_plan_case  # noqa: E402
from src.metrics.gamma_map_agreement import (  # noqa: E402
    check_gates,
    compare_maps,
    map_digest,
    pass_rate_pct,
    plan_gamma_map,
)
from src.metrics.plan_gamma import criterion_label, parse_criteria  # noqa: E402

logger = logging.getLogger(__name__)
app = typer.Typer(help="Plan-scale gamma map agreement in native precision.", add_completion=False)

DEFAULT_PLANS = "LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4"


@app.command()
def main(
    out: Path = typer.Option(..., help="Destination JSON."),
    plans: str = typer.Option(DEFAULT_PLANS, help="Comma-separated plan directory names."),
    criteria: str = typer.Option("1/1/10,2/2/10,3/3/10", help='Comma-separated "percent/mm/cutoff" criteria.'),
    device: str = typer.Option("cuda:0", help="GPU device for rungs 3 and 4."),
    rungs: str = typer.Option("rung1,rung2,rung3,rung4", help="Rungs to compute; rung1 must be first."),
    provenance: bool = typer.Option(False, help="Write manifest.json and system dumps beside --out."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compute the maps rung by rung and compare each against rung 1 in memory."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    plan_names = [name.strip() for name in plans.split(",") if name.strip()]
    reason = corpus_skip_reason(plan_names)
    if reason:
        raise typer.BadParameter(reason)
    selected = [RUNGS[name.strip()] for name in rungs.split(",") if name.strip()]
    if selected[0].rung != 1:
        raise typer.BadParameter("rung1 must be first: it is the reference.")
    parsed = parse_criteria(
        [tuple(float(v) for v in token.split("/")) for token in criteria.split(",") if token.strip()]
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    inputs = {}
    for name in plan_names:
        inputs[f"{name}/Dose.mhd"] = corpus_dir() / name / "Dose.raw"
        inputs[f"{name}/Dose_ADoTA.mhd"] = corpus_dir() / name / "Dose_ADoTA.raw"
    if provenance:
        from src.metrics.benchmark_provenance import write_manifest

        write_manifest(
            out.parent,
            device=device,
            repos={"adota": PROJECT_ROOT, "reports": PROJECT_ROOT / "reports"},
            inputs={k: v for k, v in inputs.items() if v.is_file()},
            extra={"result": str(out), "plans": plan_names, "criteria": [list(c) for c in parsed]},
        )

    rows: List[dict] = []
    for name in plan_names:
        case = load_plan_case(corpus_dir() / name)
        for criterion in parsed:
            label = criterion_label(criterion)
            computed = {}
            elapsed = {}
            for rung in selected:
                started = perf_counter()
                gamma_map = plan_gamma_map(
                    case.dose_ref, case.dose_eval, case.spacing_zyx, criterion, case.gamma_params_base, rung, device
                )
                elapsed[rung.rung] = perf_counter() - started
                computed[rung.rung] = gamma_map
                logger.info(
                    "[%s] %s rung %d: %.1f s, GPR %.4f%%",
                    name, label, rung.rung, elapsed[rung.rung], pass_rate_pct(gamma_map),
                )
            comparisons = [(r.rung, 1) for r in selected if r.rung != 1]
            if 3 in computed and 2 in computed:
                comparisons.append((3, 2))
            for tested, baseline in comparisons:
                comparison = compare_maps(computed[baseline], computed[tested])
                rows.append(
                    {
                        "plan": name,
                        "criterion": label,
                        "gamma_params_base": case.gamma_params_base,
                        "grid_zyx": list(case.dose_ref.shape),
                        "spacing_zyx": list(case.spacing_zyx),
                        "tested_rung": tested,
                        "baseline_rung": baseline,
                        "elapsed_s": {str(k): v for k, v in elapsed.items()},
                        **comparison,
                        **check_gates(comparison, str(computed[tested].dtype)),
                        "rung_pass_rates_pct": {str(k): pass_rate_pct(v) for k, v in computed.items()},
                        "rung_digests": {str(k): map_digest(v) for k, v in computed.items()},
                    }
                )
            del computed
            out.write_text(json.dumps({"environment": environment_stamp(), "rows": rows}, indent=2) + "\n")
    typer.echo(f"Wrote {len(rows)} plan map comparisons to {out}")


if __name__ == "__main__":
    app()
