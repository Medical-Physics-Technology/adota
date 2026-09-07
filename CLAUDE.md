# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Phantom & Geometry Conventions

- **Never surround a phantom with air on every face.** `PhantomSpec.air_layer_depth` in
  [src/datasets/phantom.py](src/datasets/phantom.py) puts an air shell on all six faces and causes
  lateral grazing, which surfaces as corner-beamlet artifacts in the dose maps. Use `air_front_mm`
  instead: an air slab only in front of the beam, full width. The all-faces field is kept so old runs
  reproduce, not for new geometries.
- **Screen the geometry before generating.** `sweep_fits_ct`, `sweep_z_half_extent_mm` and
  `max_theta_x_deg` in [src/mc_generation/sweep.py](src/mc_generation/sweep.py) decide whether the
  whole angle lattice fits inside the CT; `dose_in_body_fraction` in
  [src/mc_generation/geometry.py](src/mc_generation/geometry.py) says how much of a beamlet's dose
  landed in the patient. Run both on a few beamlets before committing to a long sweep. A beamlet
  leaving the body is sometimes real physics rather than a bug, so say which one it is.
- **Body masks come from `body_mask`** in
  [src/mc_generation/geometry.py:96](src/mc_generation/geometry.py#L96), never from a fresh
  implementation. Per axial slice it labels external air as the air connected to the slice border and
  takes the complement, which keeps lung inside the patient. Do **not** swap it for
  `scipy.ndimage.binary_fill_holes`: hole-filling drops the lung wherever it joins external air
  through the airways, and that is the exact bug the current implementation was written to fix.

## Before Launching Long MC Jobs

- **Smoke test first.** Run a single beamlet at a low particle count and print the dose statistics
  (min, max, mean, NaN count) together with the in-body dose fraction before submitting a full sweep.
  A geometry mistake costs seconds there and hours in the sweep.
- **Hand back the log, do not narrate it.** Once a long job is launched, report the exact command,
  the process or job id, and the full path of the log file, then stop. Do not tail the log, summarize
  progress, or estimate remaining runtime unless explicitly asked. The user checks status themselves,
  and polling only burns context.

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

- **typer is the only CLI library.** Every command-line entry point in this repository is a typer CLI, with no exceptions: not `argparse`, not bare `sys.argv` parsing. That includes the CI guards under `ci/`, which are not part of the package and therefore run as `uv run --no-project --with typer python ci/<name>.py` so they cost a second rather than a full project sync. Keep the decision logic in a pure function and let the typer command be a thin wrapper around it, so tests exercise the logic without going through the CLI.
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

## Analysis & Figure Output

- **Reuse the figure code that already exists.** `src/figures/` is the publication figure layer
  (plan, beamlet, gamma, DVH, CT, robustness-grid, transect). Every new figure either calls a function
  there or adds one to it. Do not open a one-off matplotlib block inside a script: the drift is already
  measurable, seven of the nine `scripts/analysis/plot_*.py` call `fig.savefig` directly and five of
  them import nothing from `src/figures/` at all, which is why the same quantity comes out looking
  different from run to run. Before writing any plotting code, state which `src/figures/` function is
  being reused or extended.
- **Saving is `save_figure_as_publication_formats`** from
  [src/figures/axes_utils.py:68](src/figures/axes_utils.py#L68). It writes SVG, PDF and PNG side by
  side at 300 dpi with `bbox_inches="tight"`. Do not call `fig.savefig` directly, and do not invent a
  different dpi or a different format set.
- **Colorbars use `aligned_colorbar`** from the same module, so the bar height matches the image axis.
- **Write the numbers beside the figure.** Any figure reporting metrics gets a CSV or JSON of the
  underlying values in the same output directory, so a panel can always be traced back to the run that
  produced it.
- **Figures go to `/scratch`** through the config's `output_dir` key, never into the repository. This
  is the large-outputs convention above.
- **Gamma pass rates default to 3%/3mm with a 10% dose cutoff** unless told otherwise, and the criteria
  belong in both the filename and the caption. The plan pipeline evaluates a list of criteria
  (`gamma_criteria` in [scripts/config_run_plan_opentps.yaml](scripts/config_run_plan_opentps.yaml))
  and the training and ablation configs default to 2%/2mm, so 3%/3mm is the headline number rather
  than the only one computed.
