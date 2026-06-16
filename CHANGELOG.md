# Changelog

All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org).

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
