# Baseline-alignment refactor plan

Status: **awaiting approval**.

Goal: bring the repository into compliance with the baselines in
`an_instructions/` (referenced from `AGENTS.md`) without changing any numeric
behaviour. Every phase below is gated by a check that runs **on this machine**,
with no dataset, no checkpoint, no GPU and no server access. Work that can only
be proven on the server is isolated into Phase 6 and is not started here.

Decisions taken with the user before writing this plan:

1. **Scope:** tooling + test organisation + the six oversized `src/` modules are
   executed now. The sixteen oversized `scripts/` are **planned only** (Phase 5)
   and approved separately, because their proof of equivalence needs the real
   dataset.
2. **Ruff rule set:** `E`, `F`, `I`. Import ordering is enforced
   (PYTHON_CODE_STYLE §1); `UP`/`B` are deliberately not enabled this round.
3. **Python floor stays `>=3.9`.** PEP 604 (`str | None`) is obtained through
   `from __future__ import annotations` where new code needs it. The 30 files
   still using `Optional[...]` are left alone; see "Known deviations" at the end.

---

## 0. Measured baseline (recorded 2026-08-22, this machine)

Everything below is re-measured after every phase and must not regress.

| Signal | Baseline |
| --- | --- |
| `uv run pytest -q` | **346 passed, 7 failed, 29 skipped**, ~175 s |
| `uv run ruff check .` | **508 errors** (238 `E402`, 164 `E702`, 74 `F401`, 13 `F841`, 4 `E401`, 4 `E701`, 4 `E731`, 4 `F541`, 2 `E741`, 1 `E703`) |
| Files over the 500-line limit | **22** (6 in `src/`, 16 in `scripts/`) |
| `sys.path.insert` bootstraps | **40 files** |

The 7 baseline failures are environment-caused, not code defects:

- `tests/mc_generation/test_config_writer.py` (4 tests) — `ModuleNotFoundError:
  datagenerator`. Parity tests against the external OpenTPS-side package, which
  is not installed locally.
- `tests/test_plan_gamma.py` (3 tests) — the local `pymedphys` 0.40 install
  resolves its extras file to `.venv/lib/python3.9/lib/pymedphys/dependency-extra.txt`,
  which does not exist. A packaging/interpreter-layout fault in the dependency,
  not in our code.

TESTS.md is explicit that a test may skip only when its dependency is genuinely
unavailable, and that the skip reason must be actionable. Both groups qualify,
so Phase 2 converts them from failures into guarded skips. **Target end state:
0 failed.**

---

## Phase 1 — Safety net first

Nothing is refactored until there is a check that catches a broken split. All of
this runs with no data.

**1.1 `tests/test_import_smoke.py`** — walks `src/` with `pkgutil` and imports
every module; asserts zero import errors. This is the single most valuable guard
for Phase 4, where modules are split and re-exported: a missed symbol in an
`__init__.py` fails here immediately.

**1.2 `tests/test_public_api.py`** — for each module that Phase 4 splits, assert
that every name currently importable from it is still importable from the same
path afterwards. The expected name lists are generated from the *current* code
and committed, so they pin today's public surface before anything moves.

**1.3 `tests/test_cli_smoke.py`** — invokes `--help` on every `typer` script via
`subprocess` and asserts exit code 0. Catches import-time breakage in `scripts/`
without needing data or a checkpoint. Scripts that cannot import without their
external environment (`run_plan_opentps.py`, `inference_worker.py`) are
`skipif`-gated with an actionable reason.

**Gate:** all three files green against unmodified code. This is the proof that
the safety net measures the right thing.

---

## Phase 2 — Tooling and test organisation

Addresses DEV_SETUP.md, TESTS.md and TECH_STACK.md.

**2.1 `pyproject.toml`.** Four additions:

- `[build-system]` with `hatchling` and `[tool.hatch.build.targets.wheel]
  packages = ["src"]`. **Verified as a live probe before writing this plan:**
  `uv sync` then `uv run --project <repo> python -c "import src.adota.config"`
  from `/tmp` succeeded. This is what makes 2.2 possible.
- `[tool.ruff]` — `target-version = "py39"`, `line-length = 100`,
  `lint.select = ["E", "F", "I"]`, with a narrow per-file ignore for
  `scripts/inference_worker.py` if its deliberate dependency-minimal shape needs
  one.
- `[tool.pytest.ini_options]` — register the markers `integration`, `e2e`,
  `gpu`, `slow` so TESTS.md's marker scheme is real rather than aspirational,
  and add `pythonpath = ["."]`.
- Move the `[dependency-groups] dev` comment intact; no dependency changes. No
  new package is added anywhere in this refactor (TECH_STACK.md "Adding
  Dependencies": `hatchling` is a build backend, not a runtime dependency).

**2.2 Delete all 40 `sys.path.insert` bootstraps** (37 in `scripts/`, plus the
one in `tests/conftest.py`). With 2.1 in place `src.*` resolves from the editable
install, so the imports that were forced below the bootstrap move to the top of
the file. This alone clears **238 of the 508 ruff errors** and removes the
awkward split-import blocks visible in e.g. `scripts/run_model.py:26-58`, where
six imports sit below `logger = logging.getLogger(__name__)`.

**2.3 Clear the remaining ruff errors.** `ruff check --fix` handles the 83
mechanical ones (`F401`, `E401`, `F541`, `E703`) plus `I` import ordering. The
rest are hand-edited, largest first: 164 `E702` semicolon-chained statements
(47 of them in `scripts/analysis/plot_metric_families.py`, 12 in
`src/figures/mc_beamlet_qc.py`), 13 `F841` unused variables, 4 `E731` lambda
assignments, 4 `E701`, 2 `E741`. Each is a formatting change with no behavioural
effect; the test suite is the guard.

**2.4 Add the two missing `__init__.py`** (`src/training/`, `src/adota/`). Both
are currently implicit namespace packages, which is why `pkgutil` walking in 1.1
needs care and why editable-install resolution is fragile.

**2.5 Module docstrings** for the 21 files that lack one (18 in `src/`, 3 in
`scripts/`), following PYTHON_CODE_STYLE §2: purpose, 3-5 step flow, output
shape where relevant. Listed in an appendix at the bottom of this document.

**2.6 `scripts/run-tests.py`** — the repository test runner TESTS.md requires, as
a `typer` CLI with `unit`, `integration`, `e2e`, `all`. `unit` runs
`-m "not integration and not e2e"`, reports pass/fail/skip, returns non-zero on
any failure. This becomes the documented daily command.

**2.7 Mark the dependency-backed tests.** `tests/golden/` and `tests/perf/`
become `@pytest.mark.integration` (they need the ~196 GB HDF5 dataset and a
checkpoint); `tests/beamlets/test_flux_gpu.py` gets `@pytest.mark.gpu`. Their
existing `skipif` guards stay — the marker is what keeps the default suite fast,
the `skipif` is what makes it degrade cleanly.

**2.8 Convert the 7 baseline failures into guarded skips.** A module-level
`pytest.importorskip("datagenerator", reason=...)` for the config-writer parity
tests, and an equivalent guard for the `pymedphys` extras fault in
`tests/test_plan_gamma.py`. Both reasons name the missing dependency and where
it comes from, per TESTS.md.

**2.9 `tests/utils/`** — TESTS.md asks for shared test helpers in an importable
module rather than duplicated across files. `tests/golden/_goldenlib.py` moves to
`tests/utils/golden.py` and the four golden tests import from there. No external
service manager is created: this repository has no Docker Compose stack, so that
section of TESTS.md does not apply.

**2.10 `.env.example`** — names and safe local defaults only, for the four
environment variables actually read by the code: `ADOTA_GOLDEN_DIR`,
`ADOTA_GOLDEN_UPDATE`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`.

**Gate:** `uv run ruff check .` reports **0 errors**; `uv run pytest -q` reports
**0 failed** with the pass count at or above 346; `python scripts/run-tests.py
unit` green; Phase 1 smoke tests green.

---

## Phase 3 — Verify the net before splitting

Re-run the full baseline table. Any pass-count change from Phase 2 is
investigated and explained before a single module is split. Phase 2 is pure
plumbing; a changed test result there means something was misunderstood.

---

## Phase 4 — The six oversized `src/` modules

Every split below preserves the existing import path exactly, so no call site
changes — except `src/training/run.py`, which is the one deliberate rename and is
called out. All six are covered by tests that pass on this machine today.

### 4.1 `src/adota/layers.py` (617) → `src/adota/layers/` package

Ten `nn.Module` classes in one file. Becomes a package whose `__init__.py`
re-exports all ten, so `from src.adota.layers import ConvBlock3D_v2` keeps
working for `src/adota/models.py` and the three tests that import it.

| New module | Contents | ≈ lines |
| --- | --- | --- |
| `conv.py` | `Conv3D`, `ConvBlock3D_v2` | 190 |
| `encoder_decoder.py` | `ConvEncoder3D`, `ConvDecoder3D` | 230 |
| `transformer.py` | `TransformerEncoderLayerDoTA`, `PositionalEmbedding`, `LinearProj` | 165 |
| `tensor_ops.py` | `Permute`, `ReshapeLayer`, `CroppingLayer` | 80 |

Covered by `tests/test_conv_block.py`, `tests/test_conv_regularization.py`,
`tests/test_model_hyperparams.py` — all CPU, all currently green.

### 4.2 `src/training/run.py` (629) → four role-named siblings

`src/training/` is already a flat capability package, and the name `run.py` is
misleading — it holds runtime *infrastructure*, while the actual entry point is
`scripts/train_adota.py`. This is the one split that changes import paths, in
exchange for names that say what the modules are:

| New module | Contents | ≈ lines |
| --- | --- | --- |
| `logging_utils.py` | `RelativeTimeFormatter`, `setup_training_logging`, `silence_pymedphys`, `log_phase`, `log_banner`, `log_section`, `format_duration` | 170 |
| `run_dir.py` | `setup_training_run_directory`, `write_manifest`, `save_resolved_config`, `MetricsLog`, `_git_info`, `_file_fingerprint`, `_gpu_info`, `_config_to_dict` | 190 |
| `checkpoints.py` | `CheckpointManager`, `_rng_state`, `_restore_rng_state`, `_unwrap_compiled` | 170 |
| `diagnostics.py` | `GracefulShutdown`, `dump_nan_context`, `compute_grad_norm`, `compute_param_norm` | 120 |

Eight import sites are updated: `scripts/train_adota.py`, `src/training/loop.py`,
`src/training/gpr_pool.py`, `src/training/validation.py`,
`tests/test_checkpoint_manager.py`, `tests/training/test_compile_equivalence.py`
and `tests/test_timing_report.py`. `run.py` is deleted rather than left as a
re-export shim. Covered by `tests/test_checkpoint_manager.py` and
`tests/test_timing_report.py`.

### 4.3 `src/beamlets/extraction.py` (617) → `src/beamlets/extraction/` package

Per PYTHON_CODE_STYLE §1 ("prefer `equipment/extractor.py` over
`equipment/equipment_extractor.py`"), the sub-modules get role names inside the
package rather than an `extraction_` prefix. `__init__.py` re-exports
`run_extraction`, `run_extraction_pooled` and `ExtractionConfig`.

| New module | Contents | ≈ lines |
| --- | --- | --- |
| `__init__.py` | `ExtractionConfig`, `_FieldTiming`, `_union_seconds`, `run_extraction`, `run_extraction_pooled`, `_extract_impl` | 330 |
| `spot.py` | `_process_spot`, `_build_sim_res` | 140 |
| `io.py` | `_prepare_output_dir`, `_save_spot`, `_build_manifest`, `_save_field_overlay` | 155 |

Covered by `tests/beamlets/test_extraction.py`,
`tests/beamlets/test_extraction_pooled.py`.

### 4.4 `src/figures/single_beam.py` (802) → three modules

| Module | Contents | ≈ lines |
| --- | --- | --- |
| `figures/axes_utils.py` (new) | `identify_axes`, `aligned_colorbar`, `save_figure_as_publication_formats` | 75 |
| `figures/input_comparison.py` (new) | `compare_two_inputs` | 170 |
| `figures/beamlet_input.py` (new) | `beamlet_input_figure` | 120 |
| `figures/single_beam.py` | `publication_figure` | ~480 |

`single_beam.py` lands at roughly 480 lines — under the limit, but with little
margin, and `publication_figure` is a single 462-line function that also breaches
PYTHON_CODE_STYLE §5. Its per-row panel builders and colourbar helpers are
extracted into `axes_utils.py` in the same step, targeting ~380 lines for
`single_beam.py`. Covered by `tests/test_beamlet_input_figure.py`; the eight
other `src/figures/` modules that import from it are updated.

### 4.5 `src/figures/ct_visualizations.py` (600) → three modules

| Module | Contents | ≈ lines |
| --- | --- | --- |
| `figures/ct_segmentation.py` (new) | `smooth_ct`, `segment_hu` | 65 |
| `figures/bp_diagnostic.py` (new) | `plot_bp_estimation_diagnostic` | 180 |
| `figures/ct_visualizations.py` | `plot_ct_with_segmentation` | ~380 |

`plot_ct_with_segmentation` is 345 lines in one function; its slice-panel and
overlay blocks are extracted into named helpers in the same step (§5).

### 4.6 `src/training/validation.py` (512) → three modules

| Module | Contents | ≈ lines |
| --- | --- | --- |
| `training/binning.py` (new) | `_bin_by_fixed_edges`, `_bin_by_quantile`, `_worst_k_records` | 90 |
| `training/attention.py` (new) | `save_attention_snapshot` | 60 |
| `training/validation.py` | `_SampleMetrics`, `_gamma_pass_rate`, `pick_canary`, `pick_gpr_subset`, `evaluate_validation` | ~380 |

`evaluate_validation` is 274 lines; the per-energy breakdown and worst-K logging
blocks become named helpers (§5). Covered by `tests/test_validation_adota.py`.

**Gate for every sub-step of Phase 4, run individually:** `uv run pytest -q`
with 0 failed and the pass count unchanged, `uv run ruff check .` clean, the
Phase 1 public-API test green, and `find src -name '*.py' -exec wc -l {} \;
| awk '$1>500'` empty. One module is split, verified and committed before the
next is touched.

---

## Phase 5 — `scripts/` plan only, no execution

The sixteen oversized scripts total ~14 000 lines. `docs/scripts_refactor_plan.md`
and `docs/scripts_refactor_phase1_plan.md` already own this work, and
`src/evaluation/` (its Part 1 target architecture) exists and is adopted by
`run_model.py` and `run_model_h5py.py`. This phase produces **a written per-file
split proposal appended to `docs/scripts_refactor_plan.md`** — no code changes —
covering:

| Script | Lines |
| --- | --- |
| `training_set_analysis_advanced_metrics.py` | 1811 |
| `analysis_texture_with_inference.py` | 1589 |
| `training_set_analysis.py` | 1460 |
| `run_model.py` | 1386 |
| `run_plan_opentps.py` | 1162 |
| `training_set_vlm_based_quantification.py` | 1040 |
| `rotation_performance_analysis.py` | 986 |
| `beamlet_timing_comparison.py` | 935 |
| `beamlet_bev_rotation_timing.py` | 864 |
| `threshold_sweep.py` | 851 |
| `bragg_peak_estimation.py` | 847 |
| `train_adota.py` | 662 |
| `range_analysis.py` | 658 |
| `run_model_h5py.py` | 651 |
| `validation_adota.py` | 631 |
| `reinterp_gpu_benchmark.py` | 509 |

Note that Phase 2 already reduces every one of these by the 5-25 lines of
`sys.path` bootstrap it deletes, and `reinterp_gpu_benchmark.py` (509) may fall
under the limit from that alone — it is re-measured at the Phase 3 gate.

Nothing in this phase is executed until you approve it separately.

---

## Phase 6 — Server-only verification

Deferred, listed so it is not forgotten. On the machine with the dataset,
checkpoints and GPU:

- `python scripts/run-tests.py integration` — the four golden tests and
  `tests/perf/`, which pin the numeric output of `run_model.py`,
  `run_model_h5py.py` and the two `training_set_analysis` scripts.
- `tests/beamlets/test_flux_gpu.py` and the other CUDA-gated tests.
- `tests/mc_generation/test_config_writer.py` in the OpenTPS environment where
  `datagenerator` is importable, confirming the Phase 2.8 skip guard does not
  mask a real regression there.
- One end-to-end run of `train_adota.py --smoke-test` to confirm the 4.2 rename
  did not break the training entry point.

Phase 4 touches `src/training/` and `src/figures/`, both of which the golden
tests exercise, so this verification is required before the branch merges even
though it cannot run here.

---

## Out of scope, with reasons

- **FRONTEND.md and `an_artifacts/frontend-testing/`** — this repository has no
  frontend. There is no `package.json` anywhere outside `.venv`; the `slides/`
  HTML files are static reveal.js decks, not an application.
- **CLOUD_PRINCIPLES.md** — no `cloud/` directory, no CDK, no AWS resources, no
  boto3 usage. Nothing to align.
- **PROJ_STRUCTURE.md's multi-service layout** — this is a single Python package
  with one root `pyproject.toml`, which that document explicitly permits. No
  `service_a/`/`common/` decomposition is warranted, and none is proposed.
- **Moving existing `src/` modules into capability subpackages** —
  PYTHON_CODE_STYLE §1 says the subpackage rule applies *prospectively* and that
  existing modules must not be moved to comply. The two new subpackages in 4.1
  and 4.3 are splits of a single oversized file, not migrations.
- **`Optional[...]` → `X | None` across 30 files** — deferred with the Python
  floor, per the decision above.
- **The 22 `pytest.mark.parametrize`-free, docstring-free test modules** — TESTS.md
  sets no docstring requirement for tests.

## Known deviations recorded, not fixed

- **`requires-python = ">=3.9"` vs PYTHON_CODE_STYLE §4 (`str | None`, 3.10+).**
  Kept at 3.9 by decision; revisit when the server environment moves.
- **`main.py`** at the repository root is a two-line `print("Hello from adota!")`
  stub from `uv init`. It is dead code and PROJ_STRUCTURE.md does not sanction
  it. Recommend deletion; flagged rather than assumed because it costs nothing
  to keep and something may reference it.
- **Nine top-level directories** (`data/`, `models/`, `registry/`, `materials/`,
  `publications/`, `slides/`, `research/`, `guides/`, `graphify-out/`) exceed
  PROJ_STRUCTURE.md's "add a top-level directory only when it represents a clear
  deployable component, shared package, infrastructure boundary, or operational
  concern". All are gitignored except for a README, a PDF or a small CSV, so
  they cost the repository almost nothing. Consolidating them into one `assets/`
  or `paper/` tree would churn a lot of documentation links for little gain;
  recommend leaving them and noting the deviation.

## Documentation updates, in the final commit of the executed phases

- `README.md` — repository-structure map (new `src/` modules), and the Tests
  section rewritten around `scripts/run-tests.py`.
- `scripts/README.md` — add `run-tests.py`.
- `CHANGELOG.md` — one entry under a new version; version bump in
  `pyproject.toml` per repository convention.
- `docs/scripts_refactor_plan.md` — Phase 5's appendix.

---

## Appendix — the 21 modules missing a docstring (Phase 2.5)

`src/`: `metrics/gamma_pass_rate.py`, `metrics/classic.py`,
`augmentation/geo_augmenations.py`, `tables/results.py`, `dcm/load_data.py`,
`processing/plan_pencil.py`, `adota/models.py`, `adota/utils.py`,
`adota/layers.py`, `utils/unit_conversions.py`, `utils/dose_grid_utils.py`,
`utils/scallers.py`, `figures/single_beam.py`, `loaders/dir_based.py`,
`loaders/generator.py`, `loaders/utils.py`,
`image_processing/homogeneity_scores.py`, `image_processing/edge_detection.py`.

`scripts/`: `rotation_performance_analysis.py`, `beamlet_timing_comparison.py`,
`analysis/plot_sparsity_path.py`.

Note `augmentation/geo_augmenations.py` and `utils/scallers.py` are misspelled
(`augmenations`, `scallers`). Renaming them is a one-line-per-call-site change
and is **not** included; flagged for a separate decision.
