# Changelog

All notable changes to this project are documented in this file. This project
adheres to [Semantic Versioning](https://semver.org).

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
