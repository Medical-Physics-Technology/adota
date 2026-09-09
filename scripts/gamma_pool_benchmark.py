"""Directly measured gamma cost over validation pools (EXP-0008, experiment C).

Three sub-commands, run in order:

``pools``
    Define the pools by record id, run the checkpoint over them once, and cache
    the dose pairs with their ids and hashes. Three pools are built: a seeded
    200-record subset of the held-out test set, the complete held-out test set,
    and a seeded subset of the training run's own validation split, which is
    the set the training loop monitors.
``time``
    Cached-pair passes: one warm-up, then ``--passes`` complete timed passes
    per configuration, every beamlet timed individually.
``integrated``
    Read, infer and score every record of a pool, inference and gamma timed
    separately, for the configurations a training loop would actually use.

Example::

    uv run python scripts/gamma_pool_benchmark.py pools --out-dir /scratch/mstryja/<run>/C
    uv run python scripts/gamma_pool_benchmark.py time --pool /scratch/mstryja/<run>/C/pool_test200.npz \\
        --out /scratch/mstryja/<run>/C/time_test200.json --provenance
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

from src.metrics.benchmark_provenance import sha256_file, write_manifest  # noqa: E402
from src.metrics.gamma_beamlet_benchmark import RUNGS, GammaCase, environment_stamp  # noqa: E402
from src.metrics.gamma_beamlet_pairs import (  # noqa: E402
    build_pairs,
    load_baseline_model,
    load_pairs,
    open_beamlet_dataset,
    save_pairs,
)
from src.metrics.gamma_pool_benchmark import (  # noqa: E402
    cached_passes,
    integrated_passes,
    test_pool_ids,
    validation_split_pool_ids,
)

logger = logging.getLogger(__name__)
app = typer.Typer(help="Directly measured gamma cost over beamlet pools.", add_completion=False, no_args_is_help=True)

TEST_H5 = Path("/scratch/mstryja/DoTA_dataset_v2/testset_downsampled_v0_all_SingleGaussian.h5")
TRAIN_H5 = Path(
    "/scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
)
EXCLUDED = PROJECT_ROOT / (
    "data/excluded_indexes/IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt"
)
BASELINE_RUN = Path("/scratch/mstryja/adota_runs/train_20260519_231135_baseline")

# The production monitoring criterion: what the training configs use.
PRODUCTION = GammaCase(2.0, 2.0, 10.0, interp_fraction=10, max_gamma=2.0)

# (rung name, entry point) pairs, as the report names them.
CACHED_CONFIGS = [("rung1", "array"), ("rung3", "array"), ("rung4", "array"), ("rung3", "tensor"), ("rung4", "tensor")]
INTEGRATED_CONFIGS = [("rung1", "array"), ("rung3", "tensor"), ("rung4", "tensor")]


def _logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )


def _scale(run_dir: Path) -> dict:
    import yaml

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    return {key: float(value) for key, value in config["scale"].items()}


def _configs(spec: str, default: list) -> list:
    if not spec:
        return default
    return [tuple(token.split(":")) for token in spec.split(",") if token.strip()]


@app.command()
def pools(
    out_dir: Path = typer.Option(..., help="Directory for the pool pair files and id lists."),
    test_h5: Path = typer.Option(TEST_H5),
    train_h5: Path = typer.Option(TRAIN_H5),
    excluded: Path = typer.Option(EXCLUDED),
    run_dir: Path = typer.Option(BASELINE_RUN, help="Training run with the checkpoint and scale."),
    device: str = typer.Option("cuda:0"),
    subset_size: int = typer.Option(200),
    split_pool_size: int = typer.Option(2000),
    seed: int = typer.Option(1234),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Define the three pools and cache their dose pairs."""
    import torch

    _logging(verbose)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = _scale(run_dir)
    split = validation_split_pool_ids(train_h5, excluded, 0.2, 42, split_pool_size, seed)
    definitions = {
        f"test{subset_size}": {
            "h5": test_h5,
            "ids": test_pool_ids(test_h5, subset_size, seed),
            "source": "held-out test set, seeded subset",
        },
        "testall": {"h5": test_h5, "ids": test_pool_ids(test_h5, None, seed), "source": "held-out test set, complete"},
        f"valsplit{split_pool_size}": {
            "h5": train_h5,
            "ids": split["ids"],
            "source": (
                f"training-run validation split ({split['n_validation_split']} of {split['n_usable']} usable "
                "records), seeded subset"
            ),
        },
    }
    index = {}
    for name, definition in definitions.items():
        pairs = build_pairs(definition["h5"], run_dir, 0, torch.device(device), scale, record_ids=definition["ids"])
        path = out_dir / f"pool_{name}.npz"
        save_pairs(path, pairs, scale)
        (out_dir / f"pool_{name}_ids.json").write_text(json.dumps(definition["ids"], indent=0) + "\n")
        index[name] = {
            "pairs": str(path),
            "pairs_sha256": sha256_file(path),
            "h5": str(definition["h5"]),
            "n": len(definition["ids"]),
            "source": definition["source"],
            "seed": seed,
        }
        typer.echo(f"{name}: {len(pairs)} pairs -> {path}")
    (out_dir / "pools.json").write_text(json.dumps({"checkpoint_run": str(run_dir), "pools": index}, indent=2) + "\n")


@app.command()
def time(
    pool: Path = typer.Option(..., help="Cached pool .npz from the pools command."),
    out: Path = typer.Option(...),
    device: str = typer.Option("cuda:0"),
    configs: str = typer.Option("", help='Comma-separated "rungN:path"; default is the five report configurations.'),
    passes: int = typer.Option(3),
    limit: Optional[int] = typer.Option(None, help="Use only the first N beamlets (smoke tests)."),
    provenance: bool = typer.Option(False),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cached-pair timed passes over one pool."""
    _logging(verbose)
    pairs, scale = load_pairs(pool)
    if limit is not None:
        pairs = pairs[:limit]
    selected = _configs(configs, CACHED_CONFIGS)
    out.parent.mkdir(parents=True, exist_ok=True)
    if provenance:
        write_manifest(
            out.parent,
            device=device,
            repos={"adota": PROJECT_ROOT, "reports": PROJECT_ROOT / "reports"},
            inputs={"pool": pool},
            extra={"result": str(out), "configs": selected, "passes": passes},
        )
    results: List[dict] = []
    for rung_name, path in selected:
        results.append(
            cached_passes(pairs, PRODUCTION, RUNGS[rung_name], scale, device=device, path=path, passes=passes)
        )
        payload = {"environment": environment_stamp(), "pool": str(pool), "results": results}
        out.write_text(json.dumps(payload, indent=1) + "\n")
    typer.echo(f"Wrote {len(results)} configurations to {out}")


@app.command()
def integrated(
    pool_ids: Path = typer.Option(..., help="Pool id list JSON from the pools command."),
    h5: Path = typer.Option(..., help="The dataset the ids belong to."),
    out: Path = typer.Option(...),
    run_dir: Path = typer.Option(BASELINE_RUN),
    device: str = typer.Option("cuda:0"),
    configs: str = typer.Option("", help='Comma-separated "rungN:path"; default: training-loop configurations.'),
    passes: int = typer.Option(1),
    limit: Optional[int] = typer.Option(None, help="Use only the first N records (smoke tests)."),
    provenance: bool = typer.Option(False),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Read, infer and score every record of a pool; inference and gamma timed apart."""
    import torch

    _logging(verbose)
    ids = json.loads(pool_ids.read_text())
    if limit is not None:
        ids = ids[:limit]
    scale = _scale(run_dir)
    dataset = open_beamlet_dataset(h5, ids)
    model = load_baseline_model(run_dir, torch.device(device))
    selected = _configs(configs, INTEGRATED_CONFIGS)
    out.parent.mkdir(parents=True, exist_ok=True)
    if provenance:
        write_manifest(
            out.parent,
            device=device,
            repos={"adota": PROJECT_ROOT, "reports": PROJECT_ROOT / "reports"},
            inputs={"pool_ids": pool_ids, "checkpoint": run_dir / "checkpoints" / "best.pth"},
            extra={"result": str(out), "configs": selected, "passes": passes, "h5": str(h5)},
        )
    results: List[dict] = []
    for rung_name, path in selected:
        results.append(
            integrated_passes(
                dataset, model, PRODUCTION, RUNGS[rung_name], scale, device=device, path=path, passes=passes
            )
        )
        payload = {"environment": environment_stamp(), "pool_ids": str(pool_ids), "results": results}
        out.write_text(json.dumps(payload, indent=1) + "\n")
    typer.echo(f"Wrote {len(results)} configurations to {out}")


if __name__ == "__main__":
    app()
