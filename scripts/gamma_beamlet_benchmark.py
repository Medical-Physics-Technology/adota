"""Beamlet-scale gamma parity and timing benchmark.

Thin CLI over :mod:`src.metrics.gamma_beamlet_benchmark` and
:mod:`src.metrics.gamma_beamlet_report`. The plan-level sibling is
``scripts/gamma_benchmark.py``; this one measures the same four rungs on the
160 x 30 x 30 beamlet grid the model is trained on, which is the size that
decides whether the gamma pass rate is affordable during training.

Three sub-commands:

``pairs``
    Run a checkpoint over ``--count`` records of the beamlet test set and cache
    the (Monte Carlo, ADoTA) dose pairs to an ``.npz``. Needs a GPU and the
    dataset; everything after it does not need the model again.
``sweep``
    Time every (beamlet, criterion, rung, entry point) combination over a cached
    pair file and write the raw rows as JSON.
``report``
    Reduce one or more sweep JSONs into the parity, timing and throughput tables,
    as JSON, markdown and CSV.

Like ``scripts/gamma_benchmark.py`` this script takes no YAML config: its only
inputs are a dataset, a checkpoint, a device and the criteria, and all four are
better read from the command line than from a file that would drift from it.

Examples::

    uv run python scripts/gamma_beamlet_benchmark.py pairs \\
        --h5 /scratch/mstryja/DoTA_dataset_v2/testset_downsampled_v0_all_SingleGaussian.h5 \\
        --run-dir /scratch/mstryja/adota_runs/train_20260519_231135_baseline \\
        --count 32 --device cuda:0 \\
        --out /scratch/mstryja/gamma_beamlet/pairs.npz

    uv run python scripts/gamma_beamlet_benchmark.py sweep \\
        --pairs /scratch/mstryja/gamma_beamlet/pairs.npz \\
        --device cuda:0 --repeats 3 \\
        --out /scratch/mstryja/gamma_beamlet/sweep_criteria.json

    uv run python scripts/gamma_beamlet_benchmark.py report \\
        --sweep /scratch/mstryja/gamma_beamlet/sweep_criteria.json \\
        --out-dir docs/gamma_beamlet
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.gamma_beamlet_benchmark import (  # noqa: E402
    RUNGS,
    GammaCase,
    build_pairs,
    default_cases,
    environment_stamp,
    load_pairs,
    save_pairs,
    sweep,
)
from src.metrics.gamma_beamlet_report import build_report, write_report  # noqa: E402
from src.metrics.gamma_map_agreement import (  # noqa: E402
    beamlet_gamma_map,
    check_gates,
    compare_maps,
    map_digest,
    pass_rate_pct,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Beamlet-scale gamma index parity and timing benchmark.",
    add_completion=False,
    no_args_is_help=True,
)

# The training scale the baseline run was fitted with. It is a property of the
# dataset rather than of this benchmark, so it is read from the run directory
# when present and falls back to these values otherwise.
FALLBACK_SCALE = {
    "min_ds": 0.0,
    "max_ds": 25277028.0,
    "min_ct": -1024.0,
    "max_ct": 3071.0,
    "min_energy": 70.0,
    "max_energy": 270.0,
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _scale_from_run(run_dir: Path) -> dict:
    """Read the training scale out of a run's ``config.yaml``, or fall back."""
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        import yaml

        config = yaml.safe_load(config_path.read_text()) or {}
        scale = config.get("scale")
        if scale:
            logger.info("Read scale from %s", config_path)
            return {key: float(value) for key, value in scale.items()}
    logger.warning("No scale in %s; using the built-in fallback", run_dir)
    return dict(FALLBACK_SCALE)


def _parse_rungs(rungs: str) -> list:
    """Resolve a comma-separated rung list against :data:`RUNGS`."""
    selected = []
    for name in rungs.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in RUNGS:
            raise typer.BadParameter(f"Unknown rung {name!r}; known: {', '.join(RUNGS)}")
        selected.append(RUNGS[name])
    return selected


def _write_provenance(out: Path, device: str, pairs_path: Path, extra: dict) -> None:
    """Write a manifest and raw system dumps beside a result file."""
    from src.metrics.benchmark_provenance import write_manifest

    write_manifest(
        out.parent,
        device=device,
        repos={"adota": PROJECT_ROOT, "reports": PROJECT_ROOT / "reports"},
        inputs={"pairs": pairs_path},
        extra={"result": str(out), **extra},
    )


def _parse_criteria(criteria: str, interp_fraction: int, cutoff: float) -> List[GammaCase]:
    """Parse ``"1/1,2/2,3/3"`` into gamma cases, or fall back to the defaults."""
    if not criteria:
        return [
            GammaCase(
                case.dose_percent_threshold,
                case.distance_mm_threshold,
                cutoff,
                interp_fraction,
            )
            for case in default_cases()
        ]
    cases: List[GammaCase] = []
    for token in criteria.split(","):
        token = token.strip()
        if not token:
            continue
        percent, _, millimetres = token.partition("/")
        cases.append(
            GammaCase(float(percent), float(millimetres), cutoff, interp_fraction)
        )
    return cases


@app.command()
def pairs(
    h5: Path = typer.Option(..., help="Beamlet HDF5 test set."),
    run_dir: Path = typer.Option(..., help="Training run directory with hyperparams.json and checkpoints/."),
    out: Path = typer.Option(..., help="Destination .npz for the cached dose pairs."),
    count: int = typer.Option(32, help="Number of beamlets to draw."),
    device: str = typer.Option("cuda:0", help="Device to run inference on."),
    checkpoint: str = typer.Option("best.pth", help="Checkpoint file under run_dir/checkpoints."),
    seed: int = typer.Option(1234, help="Seed for the record draw."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build and cache the (Monte Carlo, ADoTA) beamlet dose pairs."""
    import torch

    _configure_logging(verbose)
    scale = _scale_from_run(run_dir)
    built = build_pairs(
        h5_path=h5,
        run_dir=run_dir,
        count=count,
        device=torch.device(device),
        scale=scale,
        seed=seed,
        checkpoint_name=checkpoint,
    )
    save_pairs(out, built, scale)
    typer.echo(f"Wrote {len(built)} pairs of shape {built[0].shape} to {out}")


def sweep_command(
    pairs_path: Path = typer.Option(..., "--pairs", help="Cached .npz from the pairs command."),
    out: Path = typer.Option(..., help="Destination JSON for the raw timing rows."),
    device: str = typer.Option("cuda:0", help="GPU device for rungs 3 and 4."),
    rungs: str = typer.Option("rung1,rung2,rung3,rung4", help="Comma-separated rung names."),
    criteria: str = typer.Option("", help='Comma-separated "percent/mm" criteria; empty means 1/1,2/2,3/3.'),
    cutoff: float = typer.Option(10.0, help="Lower percent dose cutoff."),
    interp_fraction: int = typer.Option(10, help="Steps the distance threshold is divided into."),
    entry_points: str = typer.Option("array", "--paths", help='Comma-separated: "array", "tensor".'),
    repeats: int = typer.Option(3, help="Timed repetitions per combination."),
    limit: Optional[int] = typer.Option(None, help="Use only the first N cached beamlets."),
    rotate: bool = typer.Option(True, help="Rotate the rung execution order per beamlet."),
    provenance: bool = typer.Option(False, help="Write manifest.json and system dumps beside --out."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Time every (beamlet, criterion, rung, entry point) combination."""
    _configure_logging(verbose)
    loaded, scale = load_pairs(pairs_path)
    if limit is not None:
        loaded = loaded[:limit]

    selected = _parse_rungs(rungs)
    cases = _parse_criteria(criteria, interp_fraction, cutoff)
    entries = tuple(token.strip() for token in entry_points.split(",") if token.strip())
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "rungs": [rung.label for rung in selected],
        "criteria": [case.as_params() for case in cases],
        "paths": list(entries),
        "repeats": repeats,
        "rotate": rotate,
        "device": device,
        "pairs": str(pairs_path),
        "n_pairs": len(loaded),
    }
    if provenance:
        _write_provenance(out, device, pairs_path, settings)

    rows = sweep(loaded, cases, selected, scale, device=device, repeats=repeats, paths=entries, rotate=rotate)
    out.write_text(
        json.dumps({"environment": environment_stamp(), "settings": settings, "rows": rows}, indent=2) + "\n"
    )
    typer.echo(f"Wrote {len(rows)} timing rows to {out}")


@app.command()
def maps(
    pairs_path: Path = typer.Option(..., "--pairs", help="Cached .npz from the pairs command."),
    out: Path = typer.Option(..., help="Destination JSON for the map comparisons."),
    device: str = typer.Option("cuda:0", help="GPU device for rungs 3 and 4."),
    rungs: str = typer.Option("rung1,rung2,rung3,rung4", help="Comma-separated rung names; rung1 is required."),
    criteria: str = typer.Option("", help='Comma-separated "percent/mm" criteria; empty means 1/1,2/2,3/3.'),
    cutoff: float = typer.Option(10.0, help="Lower percent dose cutoff."),
    interp_fraction: int = typer.Option(10, help="Steps the distance threshold is divided into."),
    limit: Optional[int] = typer.Option(None, help="Use only the first N cached beamlets."),
    save_maps: bool = typer.Option(True, help="Save every raw map in its native dtype under <out dir>/maps."),
    provenance: bool = typer.Option(False, help="Write manifest.json and system dumps beside --out."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compare raw gamma maps between rungs in memory, before any cast."""
    import numpy as np

    _configure_logging(verbose)
    loaded, scale = load_pairs(pairs_path)
    if limit is not None:
        loaded = loaded[:limit]
    selected = _parse_rungs(rungs)
    if not selected or selected[0].rung != 1:
        raise typer.BadParameter("rung1 must be first: it is the reference every other rung is compared against.")
    cases = _parse_criteria(criteria, interp_fraction, cutoff)
    out.parent.mkdir(parents=True, exist_ok=True)
    maps_dir = out.parent / "maps"
    if save_maps:
        maps_dir.mkdir(exist_ok=True)
    settings = {"rungs": [r.label for r in selected], "criteria": [c.as_params() for c in cases], "device": device}
    if provenance:
        _write_provenance(out, device, pairs_path, settings)

    # Every rung is compared against rung 1, and rung 3 additionally against
    # rung 2, which is the comparison that isolates the device from the kernel.
    comparisons = [(rung.rung, 1) for rung in selected if rung.rung != 1]
    if any(r.rung == 3 for r in selected) and any(r.rung == 2 for r in selected):
        comparisons.append((3, 2))

    rows: List[dict] = []
    for index, pair in enumerate(loaded):
        for case in cases:
            computed = {}
            for rung in selected:
                gamma_map = beamlet_gamma_map(pair, case, rung, scale, device)
                computed[rung.rung] = gamma_map
                if save_maps:
                    label = f"{pair.sample_id}__{case.dose_percent_threshold:g}_{case.distance_mm_threshold:g}"
                    np.save(maps_dir / f"{label}__rung{rung.rung}.npy", gamma_map)
            for tested, baseline in comparisons:
                comparison = compare_maps(computed[baseline], computed[tested])
                dtype = str(computed[tested].dtype)
                rows.append(
                    {
                        "sample_id": pair.sample_id,
                        "criterion": f"{case.dose_percent_threshold:g}%/{case.distance_mm_threshold:g}mm/{cutoff:g}%",
                        "interp_fraction": interp_fraction,
                        "tested_rung": tested,
                        "baseline_rung": baseline,
                        **comparison,
                        **check_gates(comparison, dtype),
                        "rung_pass_rates_pct": {str(k): pass_rate_pct(v) for k, v in computed.items()},
                        "rung_digests": {str(k): map_digest(v) for k, v in computed.items()},
                    }
                )
        logger.info("pair %d/%d (%s): %d comparison rows", index + 1, len(loaded), pair.sample_id[:8], len(rows))

    payload = {"environment": environment_stamp(), "settings": settings, "rows": rows}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    typer.echo(f"Wrote {len(rows)} map comparisons to {out}")


@app.command()
def report(
    sweeps: List[Path] = typer.Option(..., "--sweep", help="Sweep JSON; repeat for several."),
    out_dir: Path = typer.Option(..., help="Directory for the JSON, markdown and CSV output."),
    stem: str = typer.Option("gamma_beamlet_results", help="Basename shared by the outputs."),
    pool_sizes: str = typer.Option("20,200,2000", help="Validation pool sizes to extrapolate to."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Reduce the sweep rows into the parity, timing and throughput tables."""
    _configure_logging(verbose)
    rows: List[dict] = []
    environment: dict = {}
    for path in sweeps:
        payload = json.loads(path.read_text())
        rows.extend(payload["rows"])
        environment = payload.get("environment", environment)

    sizes = tuple(int(token) for token in pool_sizes.split(",") if token.strip())
    written = write_report(build_report(rows, environment, sizes), out_dir, stem)
    for path in written:
        typer.echo(f"Wrote {path}")


# Registered explicitly rather than with a decorator: the command is "sweep",
# but the function cannot be, because this module already imports a `sweep`
# helper from src.metrics.gamma_beamlet_benchmark.
app.command(name="sweep")(sweep_command)


if __name__ == "__main__":
    app()
