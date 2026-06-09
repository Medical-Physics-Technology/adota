# Refactor / performance backlog: models.py & layers.py

Captured 2026-06-09 while adding the `transformer_residual` / `conv_residual`
ablation flags. Revisit **after** the two ablation runs are evaluated.

Scope: `src/adota/models.py` and `src/adota/layers.py`.

## Correctness / latent bugs

1. **[DONE]** **`zero_padding=False` crashes forward** — the output-crop step in
   `models.py` is now guarded with `if self.zero_padding:`, so the unpadded path
   runs.
2. **[DONE]** **`dim_feedforward` is now a real hyperparameter** —
   `TransformerEncoderLayerDoTA` reads `dim_feedforward` (defaults to
   `embeded_dim`); `DoTA3D_v3` resolves it (defaults to `token_size`), passes it
   through, records it in `to_dict()`, and it is exposed via `TrainingConfig`
   (`dim_feedforward: Optional[int] = None`). Default reproduces the original FF
   shapes, so checkpoints stay compatible.
3. **[DONE]** **Hardcoded magic number `161`** — the eval `num_transformers == 0`
   attention placeholder now uses `sequence_length = x.shape[2] + 1` (input
   depth + energy token), derived from the input tensor.

Backward compatibility verified: full suite (91 tests) green, incl. real
released-checkpoint load + inference. New coverage in
`tests/test_model_hyperparams.py`.

## Dead code

4. **`self.mask` / `generate_subsequent_mask`** — `models.py`. Already marked
   "Deprecate this attribute"; `self.mask` is computed but never used (the
   causal mask is recomputed inside the layer). Safe to delete.
5. **`Conv3D` class (weight-standardized conv)** — `layers.py`. Defined but
   only referenced in a comment. Dead.
6. **`ConvBlock3D` (v1)** — `layers.py`. Superseded by `ConvBlock3D_v2`, never
   instantiated. Dead (~80 lines).
7. **Wildcard import** — `models.py` `from src.adota.layers import *`. Hides
   what is used; replace with explicit imports.

## Code clarity

8. **train/eval return-type divergence** — both `forward`s return a bare tensor
   in training and a `(tensor, attn_weights)` tuple in eval. The branching
   ripples through the model loop and is error-prone. Consider always returning
   attn weights (or `None`). Most invasive item; do separately.
9. **Misleading type hint** — `ConvBlock3D_v2` annotates `token_size: int` but
   it is a tuple.
10. **`token_size` stored as a one-shot generator** — `layers.py`
    `self.token_size = (ts // 2 for ts in self.token_size)` is a generator
    consumed by the `LayerNorm` construction; a second access yields empty.
    Use a tuple/list.

## Performance

11. **[DONE]** **Causal mask rebuilt every forward, every layer, every step** —
    `layers.py` `_causal_mask` built on CPU and `.to(device)` each call. Now
    split into `_build_causal_mask` (builds on-device) + `_causal_mask` (lazy
    cache keyed by `(seq_len, device)`, plain dict so state_dict is unchanged).
    Measured: mask provision ~47x; transformer block fwd ~2.7x (eval) / ~1.2x
    (fwd+bwd) on a single record. Full-model step is conv-dominated (no visible
    change). Covered by `tests/perf/test_perf_refactor.py`.
12. **[DONE]** **Positional indices rebuilt every forward** — `layers.py`
    `PositionalEmbedding` now registers `positions` as a `persistent=False`
    buffer (kept out of the state_dict) instead of rebuilding `torch.arange`.
13. **[DONE]** **`np.prod` returns a NumPy scalar** — `models.py`
    `self.token_size = int(np.prod(...))`.

Backward compatibility verified: full suite (83 tests, incl. real released
checkpoint load + inference) green. All A/B equivalence checks bit-exact
(`max|orig-opt| = 0`).

## Suggested first pass (high value / low risk)

1, 11, 4, 7, 13. Items 5/6 are the biggest line reductions (pure deletion).
Item 8 is the most invasive; schedule on its own.
