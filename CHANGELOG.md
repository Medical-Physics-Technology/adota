# Changelog

All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org).

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
