# Sequence mixers: the swappable backbone between encoder and decoder

ADoTA encodes the (CT, flux) volume into one token per depth slice, puts the
projected beam energy in front as an extra token, mixes that sequence, and
decodes it back into dose. The **sequence mixer** is the part that lets slices
exchange information along the beam. It is chosen by one config key, so the
encoder, the decoder, the loss and the inputs stay the same whichever mixer
runs. That is what EXP-0014 needs to vary one thing at a time.

```yaml
sequence_mixer: transformer   # transformer | mlstm | none
num_transformers: 1           # number of mixer blocks, whatever the mixer (historical name)
num_heads: 4                  # attention heads, or mLSTM memory heads
num_forward: 2                # look-ahead in slices, honoured by every mixer
causal: true                  # must stay true for mlstm
mlstm_proj_factor: 2.0        # mlstm only: inner width / token size
mlstm_conv_kernel: 4          # mlstm only: causal depthwise convolution width
```

The same keys are `TrainingConfig` fields and `DoTA3D_v3` keyword arguments,
and `DoTA3D_v3.to_dict()` writes them to `hyperparams.json`. A hyperparameter
file without `sequence_mixer` builds the transformer, so every existing model
directory and checkpoint loads unchanged (v11's checkpoint is tested to load
strictly).

## The mixers

| Name | Block | What it sees |
|---|---|---|
| `transformer` | `TransformerEncoderLayerDoTA`: multi-head attention + feed-forward, post-norm | slices up to `t + num_forward` (causal mask) |
| `mlstm` | `MLSTMBlockDoTA` (Beck et al. 2024): up projection, causal depthwise convolution, the mLSTM matrix memory with exponential input and sigmoid forget gates, per-head norm, gated down projection, residual, layer norm | the state after reading slice `t + num_forward`, from the energy token onward |
| `none` | nothing; a plain 3D U-Net over the token grid | its own slice only (plus the convolutional receptive field) |

The mLSTM is implemented in pure PyTorch (`src/adota/layers/mlstm.py`), not
with the `xlstm` package: its fast kernels need Python 3.11, it would add a
dependency against the `torch<2.9` pin, and an in-house cell can be exported to
ONNX and tested against the recurrence. The cell runs in its parallel form,
which is exact; `tests/test_sequence_mixer.py` checks it against a
step-by-step float64 recurrence, including large gate values that exercise the
stabiliser. In eval mode a mixer without attention returns a zero attention
placeholder, as the no-transformer model always did.

## Size and speed at the v11 configuration

One A40, batch 16, float32, forward + backward + AdamW step, median of 10
steps (2026-10-19):

| Mixer | Parameters | vs v11 | In the mixer | Step | Peak memory |
|---|---|---|---|---|---|
| transformer | 14,927,651 | 1.000 | 99,584 | 135 ms | 4.43 GB |
| mlstm | 15,133,099 | 1.014 | 305,032 | 137 ms | 4.48 GB |
| none | 14,828,067 | 0.993 | 0 | 134 ms | 4.40 GB |

The convolutional encoder and decoder hold over 98 percent of the parameters
and set the step time, so at this configuration the mixers differ in what they
compute, not in capacity or cost.

## Adding a mixer

Add a block with the `(B, L, d) -> ((B, L, d), attention or None)` interface
under `src/adota/layers/`, a branch in `build_sequence_mixer` and a name in
`SEQUENCE_MIXERS` (`src/adota/layers/mixers.py`), and parametrise the tests in
`tests/test_sequence_mixer.py` over it. Mamba is the planned next one.
