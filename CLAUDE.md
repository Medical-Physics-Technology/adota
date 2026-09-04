# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup

```bash
uv sync                    # creates .venv from the pinned lockfile, installs src/ editable
cp .env.example .env       # optional; every variable has a working default
```

Everything runs through the synced environment: `uv run python ...`, never a bare `python`.

### Tests

```bash
uv run python scripts/run-tests.py unit          # default: CPU only, no dataset, no network
uv run python scripts/run-tests.py unit --fast   # also skips the slow perf suite
uv run python scripts/run-tests.py integration   # needs the HDF5 dataset and a checkpoint
uv run python scripts/run-tests.py e2e
uv run python scripts/run-tests.py all

uv run pytest tests/test_training_losses.py -v            # one file
uv run pytest tests/test_training_losses.py -k lmse       # one test
uv run python scripts/run-tests.py unit -x -q             # extra args pass through to pytest
```

Suites are marker-selected (`integration`, `e2e`, `gpu`, `slow`), declared in [pyproject.toml](pyproject.toml). A test that needs something the machine lacks must **skip with a reason naming what to install or where the data belongs**, never fail.

### Lint

```bash
uv run ruff check .                 # required by CI
uv run ruff format path/to/file.py  # a file you are already touching, never `.`
```

`ruff check` is enforced; `ruff format` is not. The repository is not formatted
to ruff's taste, so `ruff format .` would rewrite 218 of 273 files and their
blame. Never run it repository-wide.

There is no type checker and no pre-commit config in this repo.

## Workflow

**Branches.** `main` is the protected release branch and receives merges only from
`dev`; `dev` is the integration branch and receives merges only through pull
requests with green CI. Never commit to either directly. Work happens on a branch
off `dev`, named by type: `feat/`, `fix/`, `exp/`, `refactor/`, `docs/`, `chore/`
plus a short slug, for example `exp/al-retrospective-pool`.

**Pull requests.** One coherent change per pull request, into `dev`. CI runs three
required checks: `lint` (`ruff check .`), `unit` (`scripts/run-tests.py unit
--fast`), and `guards` (the module-size ratchet in `ci/check_module_size.py` plus
the report-record check). Run all three locally before opening it; none of them
needs the dataset or a GPU. Update [CHANGELOG.md](CHANGELOG.md) when a public
import path or an output format changes.

**Formatting is not enforced.** `ruff check` is required, `ruff format` is not:
the repository is not ruff-formatted and reformatting it would rewrite 218 files
and their blame. Match the surrounding style instead.

**The module-size ratchet.** The 500-line rule is enforced as a ratchet against
`ci/module_size_baseline.txt`: a new module must stay under the limit, and the 18
existing offenders must not grow. After splitting one, rerun `python
ci/check_module_size.py --update`.

## Reporting

Every pull request into `dev` carries a **report record**, and every experiment
gets one of its own. The records live in the private `reports/` submodule; the
schema, templates and tooling are public. Full guide:
[docs/reporting.md](docs/reporting.md).

```bash
uv run python scripts/report.py new change "What this PR does"
uv run python scripts/report.py new experiment "What this run tested"
uv run python scripts/report.py validate
```

A record carries structured frontmatter (data sources with sample counts and a
leakage statement, metric rows, artifact paths, Monte Carlo seconds, the
publication target) and nine required sections: what was done, methodology, data
used, what precisely changed, results positive, results negative, what went well,
what went wrong, next steps. **`Results: negative` is mandatory**; write `None.`
when there is nothing, never `TODO`. These records are the source material for
the papers and the thesis, so write them for a reader who was not there.

## Architecture

### Two layers: importable `src/`, config-driven `scripts/`

`src/` is the library, `scripts/` are thin entry points. The importable package root is `src/` itself, so `import src.foo` resolves from any working directory via the editable install. `pythonpath = ["."]` in pyproject additionally exposes `scripts.*` to tests that import a script directly.

Every script is a [typer](https://typer.tiangolo.com/) CLI paired with a YAML config, with one fixed precedence: **CLI arguments > YAML config > built-in defaults**. Do not hand-wire per-field merge blocks in a new script; use `merge_config` from [src/evaluation/cli.py](src/evaluation/cli.py), which also re-exports the logging, run-directory, and YAML helpers so a script has a single import site. `resolve_device` there is the only device entry point: it falls back to CPU with a warning rather than raising when CUDA or the requested index is unavailable.

Each script has a dedicated guide in [scripts/docs/](scripts/docs/); [scripts/README.md](scripts/README.md) is the index. Design rationale for the current layout lives in [docs/scripts_refactor_plan.md](docs/scripts_refactor_plan.md) and [docs/baseline_alignment_refactor_plan.md](docs/baseline_alignment_refactor_plan.md).

### The four pipelines

Understanding these four flows explains most of `src/`:

**1. Per-beamlet training.** [scripts/train_adota.py](scripts/train_adota.py) trains `DoTA3D_v3` ([src/adota/models.py](src/adota/models.py), a 3D U-Net with optional transformer encoder layers) on an HDF5 beamlet dataset. Input is `(B, 2, D, H, W)`, a CT channel plus an analytical flux channel, plus a scalar energy. `src/training/` holds the loop, losses, checkpointing, validation, and diagnostics as separate role-named modules. Every run writes a timestamped `runs/train_*/` directory with `manifest.json` (git hash, GPU, dataset fingerprint, resolved config), append-only `metrics.jsonl`, and checkpoints carrying full RNG state for deterministic resume.

**2. Shared inference evaluation.** `src/evaluation/` is one `evaluate(...)` engine ([engine.py](src/evaluation/engine.py)) fed by pluggable sample sources ([sources.py](src/evaluation/sources.py): `DirSource` for numpy directories, `H5Source` for the HDF5 dataset). It exists because four scripts previously reimplemented the same load, infer, `inverse_minmax`, metrics loop. New evaluation scripts extend the source or the metric toggles, they do not re-copy the loop.

**3. Plan-level dose.** [scripts/run_plan_opentps.py](scripts/run_plan_opentps.py) turns an OpenTPS plan directory into a full ADoTA plan dose and validates it against MCsquare. Stages are comma-separated: `extract` (per-spot beam's-eye-view CT crop plus flux, rotating the CT around each field's isocenter), `infer`, `accumulate` (deposit and de-rotate beamlets onto the patient grid), `gamma`. A fused `stream` mode does the same work disk-free at roughly 2x. Implementation is in `src/beamlets/`. Full guide: [scripts/docs/run_plan_opentps.md](scripts/docs/run_plan_opentps.md).

**4. Monte Carlo ground-truth generation.** `src/mc_generation/` drives the MCsquare engine to produce per-beamlet ground truth over angle, energy, and gantry sweeps, with `scripts/mc/` as the entry points. `sweep.py` defines the angle lattice and geometry screening, `robustness.py` orchestrates, `config_writer.py` and `mcsquare_runner.py` are the engine boundary, and `angle_robustness_analysis.py` plus [scripts/mc/plot_angle_robustness.py](scripts/mc/plot_angle_robustness.py) score gamma pass rates and render the panels.

### Conventions that are not obvious from any single file

- **Python 3.10 is the floor.** `requires-python = ">=3.10"` and ruff targets `py310`, raised from 3.9 in 1.5.0 so `pymedphys` could move to 0.41. `torch` is deliberately pinned `<2.9`: the machines run a 535.x driver that a torch 2.11 CUDA build will not load. The existing modules are written in the `typing.Optional` / `List` style under a `from __future__ import annotations` header; match them rather than mixing styles, and keep `src/metrics/gamma_torch.py` 3.9-portable, since it is destined for upstream PyMedPhys.
- **500 lines per module.** Modules that outgrow it get split by role rather than at an arbitrary cut. This is **enforced in `src/`** (the six offenders were split in 1.4.0) and is a **known backlog in `scripts/`**, where 16 files still exceed it and the split is planned in [docs/scripts_refactor_plan.md](docs/scripts_refactor_plan.md). Do not add a new `src/` module over the limit. Check with:
  ```bash
  find src scripts -name '*.py' | xargs wc -l | awk '$1>500 && $2!="total"'
  ```
- **Ruff selects `E`, `F`, `I` only** at line-length 120. Import order is enforced: standard library, then third-party, then local. `E402` is per-file-ignored only where an import is deliberately deferred behind setup code (BLAS thread pinning, CUDA preloading); each such entry in [pyproject.toml](pyproject.toml) carries a comment saying why.
- **Large outputs never live in the repo or in `$HOME`.** Datasets, MC working directories, and figure trees go on `/scratch`. Home storage is size-limited on the training machines.
- **The MCsquare engine is an external, uncommitted install** referenced by absolute path from the YAML `mcsquare_install` key. It resolves `Materials/`, `Scanners/`, and `BDL/` relative to the current working directory, so the runner builds a per-run `/scratch` working dir with symlinks. See [docs/mcsquare_engine.md](docs/mcsquare_engine.md); provision with [scripts/mc/provision_mcsquare.sh](scripts/mc/provision_mcsquare.sh).
- **BLAS threads are pinned to 1** in the scripts that fan gamma computation across a process pool, otherwise the pool oversubscribes the cores. This is why those scripts import numpy and torch after setting the environment, and why they carry the `E402` ignore.
- **`angles_to_spot_position` in [src/beamlets/bdl.py:285](src/beamlets/bdl.py#L285) has crossed argument names.** Its first parameter `theta_y` drives the **z** spot component via `d_smy`, and it returns `(y, z)`. Read the body, not the names, before using it.
- **[CHANGELOG.md](CHANGELOG.md) is maintained** under semantic versioning, with entries naming moved symbols and their new import paths. Update it for anything that changes a public import path or an output format.

### Golden tests

`tests/golden/` compares against reference CSVs stored outside the repository at `$ADOTA_GOLDEN_DIR`, and is marked `integration`. Set `ADOTA_GOLDEN_UPDATE=1` to re-capture rather than compare, and leave it unset in normal use. Shared, importable test helpers live in [tests/utils/](tests/utils/) rather than in fixtures.
