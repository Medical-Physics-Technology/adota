# Changelog

All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

The difficulty score becomes computable for a beamlet that has not been
simulated. The score of `research/acquisition_function_final_summary.md` was
fitted on features that used the Monte Carlo dose to locate the Bragg peak (and,
for the dose-weighted edge metrics, as the quantity itself); `src/acquisition/`
supplies an analytic dose from the CT, the flux and the energy instead, and
exposes the score as one call on a full CT grid for the active-learning loop.
Model behaviour is **unchanged**; the reference study's numbers are unchanged.

### Added

- **`src/metrics/gamma_beamlet_benchmark.py`, `src/metrics/gamma_beamlet_report.py`
  and `scripts/gamma_beamlet_benchmark.py`**: the four-rung gamma deviation
  ladder at *beamlet* scale (160 x 30 x 30), where the plan-level speed-ups of
  1.5.0 do not carry over: the GPU is a median 2-7x faster on a beamlet against
  18-33x on a plan, and float32 buys nothing over float64 because the kernels
  are no longer the bottleneck. Times both the NumPy entry point and the
  device-resident one, the latter being the training path. See
  `scripts/docs/gamma_beamlet_benchmark.md` and `docs/gamma_beamlet/`.
- **EXP-0008 evidence harnesses** (`src/metrics/gamma_pool_benchmark.py`,
  `gamma_scaling_benchmark.py`, `gamma_map_agreement.py`,
  `benchmark_provenance.py`; `scripts/gamma_pool_benchmark.py`,
  `gamma_scaling_benchmark.py`, `gamma_profile_beamlet.py`,
  `gamma_plan_map_agreement.py`) and their reductions
  (`src/metrics/gamma_evidence_summaries.py`, `gamma_evidence_latex.py`,
  `src/figures/gamma_scaling.py`, `scripts/analysis/audit_gamma_report.py`).
  `gamma_beamlet_report.paired_speedups` raises `MatchedSetError` on unmatched
  case sets; the ratio-of-medians `speedup` is now withheld there too.
  `gamma_beamlet_benchmark.sweep` retains every repetition, rotates the rung
  order per beamlet and records the resolved configuration. The rung and
  criterion specs moved to `src/metrics/gamma_beamlet_specs.py` and pair
  construction to `src/metrics/gamma_beamlet_pairs.py`; both are re-exported.
  See `scripts/docs/gamma_evidence.md`.
- **`src/figures/gamma_backend_performance.py`**:
  `gamma_backend_performance_figure`, the two-scale timing and speed-up panels.
- **`scripts/analysis/report_gamma_acceleration.py`**: assembles both gamma
  benchmarks into the LaTeX tables, figure and CSVs of the gamma-acceleration
  technical report under `reports/technical-reports/gamma-pass-rate/`.
- **`src/acquisition/`**: `bragg_curve` (Bortfeld 1997, straggled with the HPTC
  beam model's energy spread vendored under `data/hptc_energy_spread.csv`),
  `surrogate` (per-ray WEPL times the Bragg curve times the flux; the
  loader-equivalent peak index and crop; the `peak_inside_crop` validity gate),
  `features` (the thirty metrics behind `compute_features(ct, flux, energy, dose)`
  and `FeatureConfig`, whose defaults are the reference run's settings),
  `scorer` (`DifficultyScorer`, the frozen weighted-percentile score),
  `candidates` (`BeamletCandidate`, `score_candidates(ct_image, candidates, bdl,
  scorers)`), and `reference` (the study's record path without the model).
- **`scripts/analysis/acquisition_input_only_{features,compare,refit}.py`**: the
  validation of the above against the reference set (see
  `scripts/docs/acquisition_input_only.md`).
- `data/excluded_indexes/`: the training exclusion list, previously read from
  another repository's checkout.

### Changed

- **`analyse_density_regions` and `compute_advanced_metrics` moved** from
  `scripts/training_set_analysis_advanced_metrics.py` to
  `src.acquisition.features`, verbatim; the script imports them. Import from
  the new location.
- `src.mc_generation.robustness`: `_FieldGeometry` / `_field_geometry` are now
  public as `FieldGeometry` / `field_geometry`; the underscore names remain as
  aliases.

## [1.5.0] - 2026-09-03

A GPU gamma index. `pymedphys.gamma` dominated gamma pass rate evaluation --
it is why `src/training/gpr_pool.py` exists at all, computing GPR on a frozen
*subset* of the validation set. `src/metrics/gamma_torch/` replaces the kernel
with a torch implementation that keeps both dose grids on the device. **The
reported metric is unchanged**: the backend is opt-in and defaults to
pymedphys, and the `gamma_values -> gamma_pass_rate` arithmetic is shared
verbatim between the two paths, quirky denominator included.

### Changed

- **Python floor raised to 3.10** (`requires-python`, `.python-version`, ruff
  `target-version`), which moves `pymedphys` from 0.40 to 0.41 and its
  interpolation from the econforge `interpolation` package to an in-house numba
  kernel. `numba` is now an explicit dependency: pymedphys ships it only as an
  optional extra and `pymedphys.gamma` raises without it.
- **`torch` pinned `>=2.8.0,<2.9`.** The 3.10 resolution would otherwise take
  torch 2.11, whose CUDA build does not load on the 535.x driver these machines
  run.
- The pymedphys 0.40 -> 0.41 interpolator change **moves published gamma pass
  rates**, by up to 0.056 pp in absolute value across the eight-plan benchmark
  corpus, in both directions (mean 0.0075 pp over the 40 plan x criterion
  pairs). This affects previously reported numbers regardless of the GPU work.
  It also makes the CPU path about 4.9x faster on its own. See
  [`docs/gamma_gpu/`](docs/gamma_gpu/) for the full 40-row comparison.
- `gamma_index`, `gamma_index_torch` and `plan_gamma` take `backend`
  (`"pymedphys"` by default, or `"torch"`) and `backend_options`. Existing
  callers are unaffected. Under the torch backend `gamma_index_torch`
  de-normalises and thresholds on the tensors' own device instead of copying
  two full volumes to the host first.

### Added

- **`src/metrics/gamma_torch/`** -- the gamma shell method in torch, for 1D, 2D
  and 3D, supporting `local_gamma`, `max_gamma`, `skip_once_passed`,
  `lower_percent_dose_cutoff`, `global_normalisation`, `interp_fraction` and
  `random_subset`. Scalar thresholds only; the sequence form that
  `pymedphys.gamma` answers with a dict raises `NotImplementedError`.
  Apache-2.0 rather than the repository's MIT, and free of every `src.` import,
  because it is written for contribution back to PyMedPhys.
- **`scripts/gamma_benchmark.py`** plus `src/metrics/gamma_benchmark.py`,
  `gamma_rungs.py` and `gamma_report.py` -- the deviation ladder and the
  performance benchmark over the OpenTPS plan corpus at `$ADOTA_GAMMA_CORPUS`.
  Guide: [`scripts/docs/gamma_benchmark.md`](scripts/docs/gamma_benchmark.md).
- [`docs/gamma_gpu/`](docs/gamma_gpu/) -- the recorded results: the 40-row
  deviation ladder, the voxel-level parity table and the performance table,
  plus a `README.md` reading them.
- `tests/test_gamma_torch.py` (synthetic parity against `pymedphys.gamma`) and
  `tests/test_gamma_torch_corpus.py` (the same at plan scale; `integration`,
  `slow`, `gpu`, and skips when the corpus is absent).

### Measured

Across all 40 (plan x criterion) pairs of the benchmark corpus, on one A40:

| comparison | tolerance | max abs delta | verdict |
|---|---|---|---|
| rung 2 (torch-CPU float64) vs rung 1 (pymedphys) | 0.01 pp | 0.000000 pp | PASS |
| rung 3 (GPU float64) vs rung 2 | 0.001 pp | 0.000000 pp | PASS |
| rung 4 (GPU float32) vs rung 2 | 0.1 pp | 0.047677 pp | PASS |
| rung 1 vs rung 0 (recorded) | reported | 0.056401 pp | not gated |

The float64 backend reproduces the pymedphys pass rate exactly on every pair,
on CPU and GPU alike, with zero gamma = 1 boundary crossings over the 9.3 M
evaluated voxels whose maps were compared.

Wall time for the whole corpus, same machine, warm-up excluded: pymedphys
3016.0 s, torch-CPU 1767.9 s, **one A40 159.0 s in float64 (19.0x) and 111.3 s
in float32**. Peak device memory 2.1 to 2.8 GiB. The 4.13 h in the recorded
JSONs is provenance from another machine, not a controlled measurement, and no
speedup is quoted against it.

### Known gaps

- `gamma_torch` requires uniformly spaced axes on both grids, which it checks
  on entry.
- Sequence thresholds are not implemented.
- One deliberate numerical difference from pymedphys: it stores the per-shell
  minimum relative dose difference in an array shaped like the reference dose,
  so a float32 dose quantises that minimum. `gamma_torch` keeps it at the
  working dtype. On float64 input the two agree to 1e-14; on the float32 doses
  the plan pipeline uses, this accounts for a ~1e-7 difference in gamma, far
  below the 0.01 pp pass-rate gate. Reproducing it would defeat the float64
  mode.

## [1.4.0] - 2026-08-22

Repository alignment with the `an_instructions/` baselines. **No behaviour
changes**: this release moves code, adds tooling and adds tests. Every dose
number, metric and output format is unchanged. Planned and tracked in
[`docs/baseline_alignment_refactor_plan.md`](docs/baseline_alignment_refactor_plan.md).

### Changed

- **`src/training/run.py` is gone**, split into four role-named modules. The old
  name described runtime infrastructure, not an entry point (that is
  `scripts/train_adota.py`). Update imports:
  `CheckpointManager` -> `src.training.checkpoints`;
  `GracefulShutdown`, `dump_nan_context`, `compute_grad_norm`,
  `compute_param_norm` -> `src.training.diagnostics`;
  `setup_training_logging`, `log_phase`, `log_banner`, `log_section`,
  `format_duration`, `silence_pymedphys` -> `src.training.logging_utils`;
  `setup_training_run_directory`, `write_manifest`, `save_resolved_config`,
  `MetricsLog` -> `src.training.run_dir`.
- `src.figures.single_beam` kept `publication_figure`; its shared helpers moved
  to `src.figures.axes_utils` (`aligned_colorbar`, `identify_axes`,
  `save_figure_as_publication_formats`), with `compare_two_inputs` in
  `src.figures.input_comparison` and `beamlet_input_figure` in
  `src.figures.beamlet_input`.
- `plot_bp_estimation_diagnostic` moved to `src.figures.bp_diagnostic`.
- `save_attention_snapshot` moved to `src.training.attention`; the energy-binning
  and worst-K helpers to `src.training.binning`.
- `src/adota/layers.py` and `src/beamlets/extraction.py` became packages. Their
  import paths are unchanged -- every name is re-exported.
- Version in `pyproject.toml` corrected from a stale `1.0.0` to match this file.

### Added

- **`scripts/run-tests.py`** -- the repository test runner:
  `unit` / `integration` / `e2e` / `all`, with per-suite reporting and a
  non-zero exit on failure. Extra arguments pass through to pytest.
- **Registered pytest markers** (`integration`, `e2e`, `gpu`, `slow`). The
  golden characterization suite is `integration`; the performance suite `slow`.
- **`tests/utils/`** -- importable shared test helpers: `golden.py` (moved from
  `tests/golden/_goldenlib.py`), `bdl.py` (one synthetic beam-data-library
  builder replacing a fixture copied across eight modules), `deps.py`
  (optional-dependency probes).
- **Three data-free guard suites**: `test_import_smoke.py` (every module under
  `src/` imports), `test_public_api.py` (the split modules still expose every
  name they used to), `test_cli_smoke.py` (`--help` exits 0 on all 29 Typer
  scripts).
- **Committed ruff configuration** (`E`, `F`, `I`; py39; line length 120) and a
  hatchling build backend, so the project installs editable and `src.*` resolves
  from any directory.
- **`.env.example`** documenting the five environment variables the code reads.
- Module docstrings for the 21 modules and 11 packages that had none.

### Fixed

- Removed all 40 `sys.path` bootstraps, **including ten that hardcoded
  `/home/mstryja/projects/adota`**, which made those scripts unrunnable from any
  other checkout.
- `ruff check .`: 508 errors to 0. Beyond formatting, this fixed 13 unused
  variables and two ambiguous `l` identifiers.
- The 7 pre-existing test failures now skip with actionable reasons instead of
  failing: four need the external `datagenerator` package (set
  `ADOTA_DATAGENERATOR_ROOT`), three need pymedphys's optional econforge
  `interpolation` dependency.
- Deleted `src/processing/plan_pencil.py`, an empty placeholder nothing imported.

### Notes

- Every file under `src/` is now within the mandatory 500-line limit. Sixteen
  files under `scripts/` are not; the per-file proposal for those is Appendix A
  of [`docs/scripts_refactor_plan.md`](docs/scripts_refactor_plan.md) and is not
  yet executed.
- Unit suite: 571 passed, 29 skipped, 0 failed (was 346 passed, 7 failed).

## [1.3.0] - 2026-08-07

Two additions: a single unified physical model behind every heterogeneity metric,
and a complete input-only difficulty-score study for active-learning beamlet
selection (report plus reproducible analysis code). The per-beamlet dose model is
**unchanged**.

### New
- **`src/processing/mcsquare_calibration.py`** - self-contained HU to relative
  stopping power (RSP) using the MCsquare `default` scanner tables (the ones the
  DoTA data generation used): `RSP = rho(HU) * SP_material(HU,E) / SP_water(E)`.
  Validated against OpenTPS to six decimals (water SP at 100 MeV = 7.256284), no
  runtime OpenTPS dependency. Scanner and Geant4 material stopping-power tables
  live in `src/processing/data/mcsquare_default/`.
- **`src/processing/range_energy.py`** - Grevillot energy-to-R80 range fits, so the
  Bragg-peak depth is derivable from the beam energy alone (input-only).
- **[`research/acquisition_function_final_summary.md`](research/acquisition_function_final_summary.md)**
  (with rendered [PDF](research/acquisition_function_final_summary.pdf)) - the
  difficulty-score study. An input-only score (weighted sum of percentile-normalized
  physics metrics) predicts the relative dose error on unseen patients to Pearson
  0.85 / Spearman 0.86, with an interpretable 14-metric version at 0.79 / 0.82;
  frozen-test validated on 7 held-out patients. The gamma pass rate is predictable
  only to 0.71. Design and math framework in
  [`research/acquisition_function_design.md`](research/acquisition_function_design.md).
- **`scripts/analysis/`** - reproducible analysis behind the report:
  `acquisition_regression_study.py` and `acquisition_dev_analysis.py` (achievability),
  `acquisition_target_comparison.py` and `extract_mape.py` (target/transform),
  `acquisition_single_metric_corr.py` (single-metric floor), `acquisition_frozen_test.py`
  and `acquisition_rde_finalize.py` (final scorer plus frozen test),
  `acquisition_ranking_benchmark.py` (tail-lift), `build_beamlet_provenance.py`
  (per-beamlet anatomy/patient map), and the `plot_*` figure generators. Figures under
  `research/figures/acquisition/`.

### Changed
- **Unified physical model** - `src/processing/rsp.py`,
  `src/processing/tissue_decomposition.py`, `scripts/bragg_peak_estimation.py` and
  `src/processing/pflugfelder_hi.py` now delegate their density and RSP values to
  `mcsquare_calibration`, so WEPL, ISI, range and Bragg-peak estimation share one
  physically consistent model (verified bit-identical across the entry points).

### Fixed
- **WEPL used a density ratio instead of stopping power** - the previous `hu_to_rsp`
  defaulted to `rho/rho_water`, overestimating bone RSP by about 25 percent at
  HU 1500. `compute_wepl_map` and the Pflugfelder index now use the MCsquare
  stopping-power calibration.

### Added (metric, not wired into the pipeline)
- **`compute_parallel_beam_wepl_diff`** in `pflugfelder_hi.py` - a parallel-to-beam
  WEPL split ("half bone, half air"). Kept and smoke-tested, but found redundant
  with `wepl_std` (Spearman 0.92) and deliberately not wired into extraction.

### Tests
- `tests/test_pflugfelder_hi.py` updated for the stopping-power RSP (HU 0 now gives
  soft-tissue RSP about 1.017, not pure water; added water-SP and bone-correction
  checks). Golden-test baseline refreshed.

### Dependencies
- Added `scikit-learn>=1.6.1` (used by the acquisition analysis scripts).

### Docs
- `guides/` - the supervisor's publication guidelines that the report follows.

## [1.2.0] - 2026-06-16

Plan-level dose pipeline: a new `src/beamlets/` package and
`scripts/run_plan_opentps.py` take an OpenTPS plan directory all the way to an
accumulated ADoTA plan dose, validated against the MCsquare reference with
comparison figures, DVHs, and a gamma analysis. The per-beamlet model code is
**unchanged** — this is a wrapper around it — and every new speed/quality switch
is opt-in with defaults that preserve the reference behavior.

### New
- **`scripts/run_plan_opentps.py`** — end-to-end, config-driven plan pipeline with
  comma-separated stages `extract,infer,accumulate,gamma`. Writes `Dose_ADoTA.mhd`
  (on the original CT/dose grid), `dose_comparison.*`, `dvh_comparison.*` +
  `dvh_metrics.json`, `gamma_comparison.*` + `gamma_metrics.json`, and
  `pipeline_timing.json`, all next to the plan.
- **`src/beamlets/` package** — `extraction`, `rotation`, `isocenter`, `cropping`,
  `flux`, `inference`, `accumulation`, `dose_scaling`, `structures`, `dvh`,
  `plan_spots`, `bdl`, `geometry`; plus `src/loaders/plan_directory.py` for the
  OpenTPS plan loader/parser.
- **Geometry-correct extraction/accumulation** — the CT is rotated around the
  physical isocenter into a **grid-expanded** frame so the off-isocenter rotation
  clips no patient tissue, with the plan→CT isocenter x-flip handled in one place
  (`src/beamlets/isocenter.py`); accumulation de-rotates each field back onto the
  original grid, so the output matches `Dose.mhd` size exactly (required for DVH).
- **Plan gamma stage** — `src/metrics/plan_gamma.py` (gamma pass rate over several
  `[dose%, distance_mm, cutoff%]` criteria, reusing the per-beamlet `gamma_index`)
  and `src/figures/gamma_comparison.py` (3 views × N criteria gamma maps at the
  isocenter).
- **Plan dose metrics** — `src/metrics/plan_metrics.py`: MAPE and RMSE over a
  high-dose mask (voxels > 10% of the dose's 99th percentile) plus whole-grid
  relative dose error.

### Performance (opt-in; defaults unchanged)
- **GPU flux projection** — `flux_projection_gpu`, a Torch twin of the NumPy
  `flux_projection` (float64, numerically identical — proven bit-identical at the
  stored float32 precision by `tests/beamlets/test_flux_gpu.py`). Enabled via
  `flux_on_gpu`.
- **Parallel extraction** — `run_extraction_pooled`, a thread-pooled twin of
  `run_extraction` that overlaps the per-spot crop / flux / disk-write while
  sharing the rotated CT and CUDA context zero-copy. Output is **byte-identical**
  to the serial reference; selected via `extraction_parallel` / `extraction_workers`.
- **Inference down-sample on the GPU** — `get_single_record_no_gt` takes a
  `device` so the CT/flux resize to the `160x30x30` ADoTA grid runs on the GPU
  (the up-sample already did); default `device=None` keeps every other caller on CPU.
- **Honest timing report** — the inference breakdown splits the old combined
  "record load (trilinear)" into file-read / down-sample / up-sample / write; the
  extraction breakdown reports the **real wall-clock time** each step was active
  (union of concurrent intervals) instead of a thread-sum, so pooled runs are no
  longer misread.

### Quality (opt-in)
- **Dose calibration** — `AccumulationConfig.calibration_factor` multiplies the
  accumulated dose before writing (`dose_calibration_enabled` /
  `dose_calibration_factor`, default off / `1.0`), to correct the model's measured
  ~2.8% systematic per-beamlet under-prediction.

### Tests
- New beamlet/plan tests: flux CPU-vs-GPU equivalence, serial-vs-pooled extraction
  byte-identity (`_union_seconds` included), plan gamma, plan metrics, gamma figure,
  and accumulation calibration; plus the inference timing-split report tests.

### Docs
- Added a `run_plan_opentps.py` section to `scripts/README.md` (stages, config
  keys, outputs, performance notes) and linked the plan pipeline from the main
  `README.md` (overview, repo structure, a dedicated section).

## [1.1.0] - 2026-06-11

Scripts refactor (part 1): the duplicated inference-evaluation pipeline is moved
into a shared `src/evaluation/` package, the scripts become thin CLIs over it,
device handling is unified, and two latent metric bugs are corrected. Behavior
is pinned by new characterization (golden) tests on a fixed slice of the real
dataset; the two intentional changes below are the only differences in output.
This is correctness-only work — performance optimizations (batching, on-GPU
metrics, persistent H5 handle) are deferred to part 2.

### New
- **`src/evaluation/` package** shared by the inference scripts:
  - `cli.py` — `resolve_device` (prefers GPU, falls back to CPU on absent CUDA
    or an out-of-range index) and `merge_config` (generic CLI > YAML > default
    merge);
  - `sources.py` — `DirSource` / `H5Source`, a uniform per-sample iterator over
    the directory and HDF5 layouts;
  - `engine.py` — `evaluate(...)`, the shared per-sample loop (device move,
    `no_grad` inference, timing, skip protocol) with a per-script callback;
  - `outputs.py` — a shared results-CSV writer driven by explicit per-script
    column specs, so column order and float precision are preserved exactly.
- Characterization (golden) tests for `run_model.py`, `run_model_h5py.py`,
  `training_set_analysis.py`, and `training_set_analysis_advanced_metrics.py`
  (reference CSVs stored under `/scratch`, not the repo), plus unit tests for the
  new package.

### Changed (intentional behavior changes)
- **MAPE corrected** in `run_model_h5py.py` and `training_set_analysis.py` to the
  canonical `run_model.py` form (mask on the ground truth; arguments in
  `(prediction, reference)` order). This changes only the `mape_pct` column of
  those two scripts; all other columns are unchanged.
- **Removed the leftover `i == 169` debug skip** in `run_model.py`, which had
  silently dropped one sample.

### Internal (no output change)
- `run_model.py`, `run_model_h5py.py`, and `training_set_analysis.py` now run on
  the shared engine and shared CSV writer; their per-sample CSV output is
  byte-stable against the goldens (timing columns aside).
- Unified device handling via `resolve_device` across `run_model.py`,
  `run_model_h5py.py`, `training_set_analysis.py`,
  `training_set_analysis_advanced_metrics.py`, `beamlet_timing_comparison.py`,
  and `train_adota.py`.
- Renamed `scripts/rotatation_performance_analysis.py` →
  `scripts/rotation_performance_analysis.py` (typo fix); README and docs updated.

### Docs
- Added a `train_adota.py` section to `scripts/README.md` (usage, options,
  examples including a 1%-of-data / 3-epoch run, YAML, output layout).

## [1.0.0] - 2026-06-10

Model refactor of `DoTA3D_v3` and the layers, focused on new ablation options,
correctness fixes, performance, and cleanup. **All changes are backward
compatible**: every new option defaults to the original behavior, and existing
checkpoints and configs load unchanged. Verified by the full test suite (106
tests), including loading a real released checkpoint and a short end-to-end
training run on real data.

### New options (opt-in; defaults unchanged)
- **Residual ablations** — `transformer_residual` and `conv_residual` flags to
  turn off the transformer residual connections and the encoder–decoder skip
  connections independently.
- **Feed-forward width** — `dim_feedforward` is now a real, configurable
  hyperparameter for the transformer layers.
- **Convolution regularization** — `weight_standardization` (on/off),
  `norm_layer` (`batch` / `group` / `none`), and `weight_init`
  (`default` / `kaiming` / `xavier`).

All of the above are settable from the training config and saved with the model.

### Fixes
- Running with `zero_padding=False` no longer crashes during the forward pass.
- The attention output size is now derived from the input depth instead of a
  hardcoded value, so it is correct for any input size.
- Fixed a latent bug in `inference_worker.py` that passed the model's
  `(dose, attention)` output straight into prediction saving.

### Performance
- The causal attention mask and positional indices are now computed once and
  reused instead of rebuilt on every forward pass. Faster transformer step,
  with identical numerical results and no change to saved checkpoints.

### Cleanup
- The model now always returns a consistent `(dose, attention)` pair (previously
  the return shape differed between training and inference).
- Removed dead code, replaced a wildcard import with explicit imports, and
  corrected a few misleading type hints / comments.

### Tooling
- Moved `pytest` and `ruff` to a development dependency group so they are no
  longer installed for plain runtime use.
