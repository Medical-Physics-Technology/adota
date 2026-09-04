"""Deviation-ladder + performance harness for the plan-level gamma index.

Thin CLI over :mod:`src.metrics.gamma_benchmark`. Three sub-commands:

``cpu``
    Rung 1 -- re-run the recorded pymedphys path on this machine, per plan and
    criterion, and write the results as JSON. Optionally persists each gamma map
    to ``--maps-dir`` for the voxel-level comparison.
``torch``
    Rungs 2-4 -- the same criteria through :mod:`src.metrics.gamma_torch`,
    selected by ``--device`` and ``--dtype``.
``report``
    Assemble the rung JSONs (plus rung 0, read back from each plan's own
    ``gamma_metrics.json``) into the deviation table, the performance table and
    the voxel-level table, as JSON and markdown.

Unlike the pipeline scripts this one takes no YAML config: it has no per-run
knobs beyond the plan list and the device, and the gamma recipe deliberately
comes from each plan's ``gamma_metrics.json`` rather than from adota's defaults.

Examples::

    uv run python scripts/gamma_benchmark.py cpu \\
        --plans LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4 \\
        --out /scratch/mstryja/gamma_gpu/rung1.json \\
        --maps-dir /scratch/mstryja/gamma_gpu/maps

    uv run python scripts/gamma_benchmark.py torch --device cuda:0 --dtype float64 \\
        --out /scratch/mstryja/gamma_gpu/rung3.json

    uv run python scripts/gamma_benchmark.py report \\
        --rung /scratch/mstryja/gamma_gpu/rung1.json \\
        --rung /scratch/mstryja/gamma_gpu/rung2.json \\
        --out-dir docs/gamma_gpu
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.gamma_benchmark import (  # noqa: E402
    BENCHMARK_PLANS,
    corpus_dir,
    corpus_skip_reason,
    load_plan_case,
    write_rung_json,
)
from src.metrics.gamma_report import build_report, write_report  # noqa: E402
from src.metrics.gamma_rungs import run_cpu_case, run_torch_case  # noqa: E402

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Gamma deviation ladder and GPU benchmark over the OpenTPS plan corpus.",
    add_completion=False,
    no_args_is_help=True,
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _selected_plans(plans: Optional[str]) -> List[str]:
    """Resolve the ``--plans`` CSV, defaulting to the full eight-plan corpus."""
    if not plans:
        return list(BENCHMARK_PLANS)
    return [name.strip() for name in plans.split(",") if name.strip()]


def _require_corpus(names: List[str]) -> None:
    reason = corpus_skip_reason(names)
    if reason is not None:
        typer.secho(reason, fg=typer.colors.RED)
        raise typer.Exit(code=2)


@app.command()
def cpu(
    plans: Optional[str] = typer.Option(
        None, help="Comma-separated plan directory names (default: all eight)."
    ),
    out: Path = typer.Option(..., help="Where to write the rung JSON."),
    maps_dir: Optional[Path] = typer.Option(
        None, help="Persist each gamma map here as float32 .npy (~400 MB each)."
    ),
    rung: str = typer.Option("1", help="Rung label recorded in the JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rung 1: re-run the recorded pymedphys CPU path on this machine."""
    _configure_logging(verbose)
    names = _selected_plans(plans)
    _require_corpus(names)

    results = {}
    for name in names:
        case = load_plan_case(corpus_dir() / name)
        results[name] = run_cpu_case(case, maps_dir=maps_dir, tag=f"rung{rung}")
        del case
    write_rung_json(out, rung, "pymedphys", "cpu", None, results)


@app.command()
def torch(
    plans: Optional[str] = typer.Option(
        None, help="Comma-separated plan directory names (default: all eight)."
    ),
    out: Path = typer.Option(..., help="Where to write the rung JSON."),
    device: str = typer.Option("cuda:0", help="'cpu' or 'cuda:<index>'."),
    dtype: str = typer.Option("float64", help="'float32' or 'float64'."),
    maps_dir: Optional[Path] = typer.Option(
        None, help="Persist each gamma map here as float32 .npy (~400 MB each)."
    ),
    rung: Optional[str] = typer.Option(
        None, help="Rung label; inferred from device/dtype when omitted."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rungs 2-4: the same criteria through the torch gamma kernel."""
    _configure_logging(verbose)
    names = _selected_plans(plans)
    _require_corpus(names)

    if rung is None:
        if device == "cpu":
            rung = "2"
        else:
            rung = "3" if dtype == "float64" else "4"

    results = {}
    for name in names:
        case = load_plan_case(corpus_dir() / name)
        results[name] = run_torch_case(
            case, device=device, dtype=dtype, maps_dir=maps_dir, tag=f"rung{rung}"
        )
        del case
    write_rung_json(out, rung, "gamma_torch", device, dtype, results)


@app.command()
def report(
    rung: List[Path] = typer.Option(
        ..., "--rung", help="Rung JSON from `cpu` or `torch`; repeat per rung."
    ),
    out_dir: Path = typer.Option(..., help="Directory for the JSON + markdown."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Assemble the deviation, performance and voxel-level tables."""
    _configure_logging(verbose)
    report_data = build_report([Path(p) for p in rung])
    paths = write_report(report_data, Path(out_dir))
    for path in paths:
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
