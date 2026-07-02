# train_adota.py

> **Reproducibility / portability.** The paths in the examples and config
> snippets below are the original development environment's locations — replace
> them with your own. Nothing is hard-coded to a specific machine: point the
> script at your data via the config or CLI, and keep large outputs in a
> directory of your choice (off your home if space-constrained).

---

The ADoTA training entry point. Loads a YAML config into a `TrainingConfig` (CLI flags override), builds train/val `H5PYGenerator` datasets, trains `DoTA3D_v3` with AdamW + `ReduceLROnPlateau`, validates every epoch (RMSE / MAPE / RDE on the de-normalised dose, with periodic gamma pass rate and attention snapshots), and writes checkpoints with full RNG/optimizer/scheduler state for deterministic resume.

The script is **config-driven**: there are no positional arguments. Point it at a YAML config and override individual fields from the CLI as needed.

### Usage

```bash
# Run from a YAML config (the normal case)
uv run python scripts/train_adota.py --config scripts/config_train_adota.yaml

# YAML config with CLI overrides (CLI takes precedence)
uv run python scripts/train_adota.py --config scripts/config_train_adota.yaml \
    --device-index 0 --num-epochs 50

# Resume from a checkpoint
uv run python scripts/train_adota.py --config scripts/config_train_adota.yaml \
    --resume-from /path/to/adota_runs/<run>/checkpoints/last.pth
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | – | Path to the YAML training config |
| `--resume-from` | path | – | Path to a previous checkpoint `.pth` to resume from |
| `--device-index` | int | `1` (from YAML) | CUDA device index; `-1` for CPU |
| `--max-records` | int | all | Subsample the record list to N samples **before** the train/val split. For tiny end-to-end runs |
| `--num-epochs` | int | `400` (from YAML) | Override the number of epochs |
| `--max-hours` | float | – | Wall-time budget in hours; the loop exits cleanly when exceeded |
| `--config-name` | string | `dota_v3_baseline` (from YAML) | Label used in the run-directory name |
| `--runs-dir` | string | `/path/to/adota_runs` | Base directory under which the run directory is created |
| `--smoke-test` | flag | `False` | CI sanity check: 2 epochs, 4 train/val batches each, then exit |
| `--verbose` | flag | `False` | Enable DEBUG-level logging |
| `--help` | – | – | Show help message and exit |

### Examples

**Tiny end-to-end run on 1% of the dataset for 3 epochs** (exercises every artifact path quickly). The dataset has ~70k usable records, so 1% ≈ 701:
```bash
uv run python scripts/train_adota.py \
    --config scripts/config_train_adota.yaml \
    --max-records 701 \
    --num-epochs 3 \
    --config-name dota_v3_smoke_1pct \
    --runs-dir /path/to/adota_runs
```

**CI smoke test** (fixed 2 epochs, 4 batches each — not a real training run):
```bash
uv run python scripts/train_adota.py --config scripts/config_train_adota.yaml --smoke-test
```

**Full training run on a specific GPU:**
```bash
uv run python scripts/train_adota.py --config scripts/config_train_adota.yaml --device-index 0
```

**Show help:**
```bash
uv run python scripts/train_adota.py --help
```

### YAML Configuration

All training settings live in `scripts/config_train_adota.yaml` (dataset path, train/val split, batch size, optimizer, scheduler, loss balancing, validation cadence, checkpoint retention, and the full model hyperparameters). Key fields:

```yaml
config_name: dota_v3_baseline
dataset_path: /path/to/DoTA_dataset_v2/...SingleGaussian.h5
excluded_indexes_file: /path/to/auxiliary_files/IndexesExclude_...txt
train_test_split: 0.2
batch_size: 56
num_epochs: 400
learning_rate: 0.0005
device_index: 1
seed: 1234
```

**Precedence order:** CLI arguments > YAML config > built-in defaults.

#### Acceleration (opt-in)

Two optional flags speed up training; both default to `false`, leaving the FP32 eager path unchanged.

```yaml
compile: false          # wrap the model with torch.compile (fused forward/backward)
compile_mode: default   # default | reduce-overhead | max-autotune
allow_tf32: false       # enable TF32 matmul/conv kernels (Ampere+)
```

Notes:
- With `compile: true`, the **first epoch is slower** (one-time compilation warmup); steady-state epochs are faster. Compare the logged `samples/s` from epoch 2 onward, not epoch 1.
- `allow_tf32` is a small numeric change for a large throughput gain on Ampere+ GPUs.
- Either flag relaxes cuDNN determinism (`benchmark=True`) so kernel autotuning is effective; with both off, runs stay bit-for-bit deterministic.
- Checkpoints are format-identical whether or not compile is enabled, so a model trained with `compile: true` resumes into a `compile: false` run (and vice versa) and stays loadable by the inference scripts.

For reproducibility, the input config is copied to the run directory as `config_input.yaml`, and the fully-resolved config is written as `config.yaml` alongside a manifest (git commit, library versions, GPU info, dataset fingerprint).

### Output

Each run creates a directory under `--runs-dir` (default `/path/to/adota_runs`, which keeps large checkpoints off the home directory) named `train_<YYYYMMDD_HHMMSS>_<config_name>`:

```
<runs-dir>/
└── train_20260610_143022_dota_v3_baseline/
    ├── config_input.yaml      # Copy of the provided --config
    ├── config.yaml            # Fully-resolved TrainingConfig
    ├── manifest.json          # Reproducibility manifest
    ├── hyperparams.json       # Model hyperparameters (for inference scripts)
    ├── training.log           # Full training log
    ├── metrics.jsonl          # Streaming per-epoch metrics
    ├── checkpoints/
    │   ├── best.pth           # Lowest val loss
    │   ├── last.pth           # Most recent epoch
    │   └── epoch_NNNN.pth     # Every checkpoint_every_n_epochs
    ├── attention/             # Periodic attention snapshots (canary sample)
    ├── validation/            # Per-epoch validation artifacts
    └── failures/              # NaN/Inf dumps, if any
```

The training loop honours SIGINT / SIGTERM (graceful shutdown after the current epoch) and the optional `--max-hours` wall-time budget.

### Requirements

- An HDF5 training dataset (`dataset_path` in the config) produced by the DoTA data pipeline; each record holds a 2-channel input (CT + flux), the beam energy, and a 3-D Monte-Carlo dose grid.
- An optional exclusion file listing known-bad record IDs.
