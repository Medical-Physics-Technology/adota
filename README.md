# ADoTA: Adaptive Dose Targeting with Transformers

ADoTA is a deep-learning framework for per-beamlet proton dose prediction using a 3D U-Net architecture augmented with transformer attention. Given a CT volume and an analytical flux projection for a single pencil beam, the model predicts the Monte Carlo dose distribution directly, enabling fast and accurate dose estimation for proton therapy treatment planning.

## Overview

| Component | Description |
|---|---|
| Model | `DoTA3D_v3` -- 3D U-Net with optional transformer encoder layers |
| Input | `(B, 2, D, H, W)` = CT channel + analytical flux channel + scalar energy |
| Output | `(B, 1, D, H, W)` predicted dose volume |
| Dataset | HDF5 dataset of beamlet records (pelvis, downsampled to 160x30x30 mm grid) |
| Training losses | Normalised MSE (`LMSE`) + Integral Depth Dose (`LPS`), balanced adaptively |
| Validation metrics | RMSE, MAPE, RDE, Gamma Pass Rate (2%/2mm), per-energy breakdowns |

---

## Repository Structure

```
adota/
├── data/
│   ├── example_inputs/       # Small numpy arrays for unit tests (ct, flux, dose, pred)
│   └── proton_tables/        # Schneider HU-to-stopping-power conversion tables
│
├── models/                   # Saved model checkpoints (not tracked in git)
│
├── runs/                     # Training run output directories (not tracked in git)
│
├── scripts/
│   ├── train_adota.py              # Main training entry point
│   ├── config_train_adota.yaml     # Production training config
│   ├── run_ablation.sh             # Full 2x2 ablation study launcher
│   │
│   ├── ablation/                   # Ablation study configs and aggregation
│   │   ├── config_A_analytical_mse_idd.yaml
│   │   ├── config_B_angle_broadcast_mse_idd.yaml
│   │   ├── config_C_analytical_mse_only.yaml
│   │   ├── config_D_angle_broadcast_mse_only.yaml
│   │   └── aggregate_results.py
│   │
│   ├── run_model.py                # Inference on directory-based samples
│   ├── run_model_h5py.py           # Inference on HDF5 dataset
│   ├── training_set_analysis.py    # Tissue-interface prevalence + GPR split
│   ├── training_set_analysis_advanced_metrics.py
│   ├── analysis_texture_with_inference.py
│   ├── ct_texture_analysis.py
│   └── analysis/
│       └── hu_to_sp.py             # HU-to-stopping-power conversion
│
├── src/
│   ├── adota/
│   │   ├── models.py               # DoTA3D_v3 model definition
│   │   ├── layers.py               # Custom 3D conv/transformer layers
│   │   ├── config.py               # Shared constants, device setup, logging
│   │   └── utils.py
│   │
│   ├── loaders/
│   │   └── generator.py            # H5PYGenerator -- PyTorch Dataset for HDF5
│   │
│   ├── training/
│   │   ├── losses.py               # LMSE, LPS, LossLPD, TwoObjectiveBalancer
│   │   ├── run.py                  # CheckpointManager, logging utilities
│   │   ├── utils.py                # validate_tensor_ranges, get_lr
│   │   └── validation.py           # Per-epoch validation loop (RMSE/MAPE/RDE/GPR)
│   │
│   ├── metrics/
│   │   ├── classic.py              # RMSE, MAPE, RDE implementations
│   │   ├── gamma_pass_rate.py      # GPR wrapper around pymedphys
│   │   └── sobel.py                # Flux-weighted Sobel edge metrics
│   │
│   ├── processing/
│   │   ├── interface_severity.py   # ISI: tissue-interface severity index
│   │   ├── pflugfelder_hi.py       # Pflugfelder heterogeneity index
│   │   ├── rsp.py                  # Relative stopping power utilities
│   │   └── tissue_decomposition.py
│   │
│   ├── schemas/
│   │   ├── configs.py              # TrainingConfig, EvaluationConfig, etc.
│   │   ├── analysis.py
│   │   └── results.py
│   │
│   ├── figures/                    # Plotting helpers
│   ├── image_processing/           # Texture, heterogeneity, GLCM
│   ├── tables/                     # ASCII result table formatting
│   └── utils/
│       ├── serialization.py        # NumpyEncoder for JSON serialization
│       └── dose_grid_utils.py
│
└── tests/
    ├── conftest.py
    ├── test_training_losses.py     # 43 tests for LMSE, LPS, TwoObjectiveBalancer
    ├── test_checkpoint_manager.py  # 5 tests for CheckpointManager save/load
    ├── test_interface_severity.py
    └── test_pflugfelder_hi.py
```

---

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. Python 3.9 or later is required.

**1. Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone and enter the repository**

```bash
git clone <repo-url>
cd adota
```

**3. Create the virtual environment and install dependencies**

```bash
uv sync
```

This installs all dependencies declared in [pyproject.toml](pyproject.toml) into `.venv/`.

**4. Verify the setup**

```bash
uv run pytest tests/ -v
```

All tests should pass. The loss and checkpoint tests use only CPU and the example data in [data/example_inputs/](data/example_inputs/).

---

## Training

### Quick start

Copy and edit the production config, then launch:

```bash
cp scripts/config_train_adota.yaml scripts/config_my_run.yaml
# Edit dataset_path, device_index, num_epochs, etc.
uv run python scripts/train_adota.py --config scripts/config_my_run.yaml
```

### Smoke test (2 epochs, 30 samples)

```bash
uv run python scripts/train_adota.py \
    --config scripts/config_train_adota.yaml \
    --smoke-test \
    --max-records 30
```

### Background run

```bash
nohup uv run python scripts/train_adota.py \
    --config scripts/config_train_adota.yaml \
    > /tmp/train.out 2>&1 & echo "PID: $!"

tail -f /tmp/train.out
```

### Run directory layout

Every run creates a timestamped directory under `runs/`:

```
runs/train_YYYYMMDD_HHMMSS_<config_name>/
├── manifest.json        # Git hash, GPU info, dataset fingerprint, full config
├── metrics.jsonl        # Append-only per-epoch metrics (RMSE, MAPE, RDE, GPR, ...)
├── training.log         # Structured HH:MM:SS [PHASE] log
├── config.yaml          # Resolved config as actually used
├── checkpoints/
│   ├── best.pth         # Best val-loss checkpoint
│   ├── last.pth         # Most recent checkpoint
│   └── epoch_NNNN.pth   # Periodic snapshots (every N epochs)
├── validation/
│   └── epoch_NNNN/
│       ├── summary.json
│       └── per_sample.csv
└── attention/
    └── epoch_NNNN.npy   # Attention maps from fixed canary sample
```

### Key config fields

| Field | Default | Description |
|---|---|---|
| `dataset_path` | | Path to the HDF5 training dataset |
| `num_epochs` | `400` | Maximum training epochs |
| `patience` | `100` | Early-stopping patience |
| `batch_size` | `56` | Samples per batch |
| `device_index` | `0` | CUDA device index |
| `loss_mode` | `mse_idd` | `mse_idd` (LMSE + LPS) or `mse_only` |
| `flux_mode` | `analytical` | `analytical` or `angle_broadcast` (ablation) |
| `gpr_every_n_epochs` | `25` | GPR evaluation cadence (expensive) |
| `smoke_test` | `false` | 2 epochs, 4 batches/epoch |
| `resume_from` | `null` | Path to a `.pth` checkpoint to resume from |

Full reference: [scripts/config_train_adota.yaml](scripts/config_train_adota.yaml) and [src/schemas/configs.py](src/schemas/configs.py).

---

## Inference

### From an HDF5 dataset

```bash
uv run python scripts/run_model_h5py.py \
    --config scripts/config_run_model_h5py.yaml
```

### From a directory of numpy arrays

```bash
uv run python scripts/run_model.py \
    --config scripts/config_run_model.yaml
```

Each sample is identified by a UUID and expects `{uuid}_ct.npy`, `{uuid}_flux.npy` files. See [data/example_inputs/](data/example_inputs/) for the format.

---

## Analysis Scripts

All analysis scripts follow the same pattern: edit a YAML config and run.

| Script | Config | Description |
|---|---|---|
| [training_set_analysis.py](scripts/training_set_analysis.py) | `config_training_set_analysis.yaml` | Tissue-interface prevalence and GPR split |
| [training_set_analysis_advanced_metrics.py](scripts/training_set_analysis_advanced_metrics.py) | `config_analysis_advanced_metrics.yaml` | Per-beamlet Sobel, ISI, and Pflugfelder HI metrics |
| [analysis_texture_with_inference.py](scripts/analysis_texture_with_inference.py) | `config_analysis_texture_with_inference.yaml` | CT texture metrics correlated with model error |
| [ct_texture_analysis.py](scripts/ct_texture_analysis.py) | `config_texture_analysis.yaml` | GLCM and heterogeneity metrics only (no inference) |
| [bragg_peak_estimation.py](scripts/bragg_peak_estimation.py) | `config_bp_estimation.yaml` | Bragg peak range estimation from dose volumes |

---

## Ablation Study

The 2x2 factorial ablation isolates two design choices:

| Variant | `flux_mode` | `loss_mode` | Description |
|---|---|---|---|
| A | `analytical` | `mse_idd` | Baseline (full model) |
| B | `angle_broadcast` | `mse_idd` | No spatial flux structure |
| C | `analytical` | `mse_only` | No IDD loss |
| D | `angle_broadcast` | `mse_only` | Both ablations active |

In the `angle_broadcast` mode the flux channel is replaced by a spatially-uniform volume whose value is `sqrt(theta_x^2 + theta_y^2)` -- the beam-deflection magnitude -- removing all spatial structure while preserving beam-direction information.

**Run the full ablation:**

```bash
nohup bash scripts/run_ablation.sh > /tmp/ablation.out 2>&1 & echo "PID: $!"
```

**Aggregate results after runs complete:**

```bash
uv run python scripts/ablation/aggregate_results.py 'runs/*ablation_*/'
```

Or use the path printed at the end of `run_ablation.sh`:

```bash
cat runs/ablation_<timestamp>/results_summary.json
```

Configs: [scripts/ablation/](scripts/ablation/)

---

## Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific file
uv run pytest tests/test_training_losses.py -v

# Run with coverage
uv run pytest tests/ --tb=short
```

Tests use only CPU and the small numpy arrays in [data/example_inputs/](data/example_inputs/). No GPU or full HDF5 dataset is required.

| Test file | Coverage |
|---|---|
| [test_training_losses.py](tests/test_training_losses.py) | LMSE, LPS, LossLPD, TwoObjectiveBalancer (43 tests) |
| [test_checkpoint_manager.py](tests/test_checkpoint_manager.py) | Round-trip fidelity, RNG state, retention policy, partial restore (5 tests) |
| [test_interface_severity.py](tests/test_interface_severity.py) | ISI metric computation |
| [test_pflugfelder_hi.py](tests/test_pflugfelder_hi.py) | Pflugfelder heterogeneity index |

---

## Data Format

Training data is stored in HDF5 files. Each beamlet record is a group with:

| Key | Shape | Description |
|---|---|---|
| `ct` | `(H, W, D)` | CT volume in HU |
| `flux` | `(H, W, D)` | Analytical flux projection |
| `dose` | `(H, W, D)` | Monte Carlo dose (ground truth) |
| `attrs["initial_energy"]` | scalar | Beam energy in MeV (70--270) |
| `attrs["beamlet_angles"]` | `(2,)` | In-plane beam angles `(theta_x, theta_y)` |
| `attrs["gantry_angle"]` | scalar | Gantry angle in degrees |

The loader crops and pads all volumes to `(160, 30, 30)` (depth x lateral x lateral) centred on the Bragg peak. See [src/loaders/generator.py](src/loaders/generator.py).

---

## Reproducibility

Every training run writes a `manifest.json` containing:
- Git commit hash and dirty-tree flag
- PyTorch and CUDA versions
- GPU name and compute capability
- SHA-256 of the first 1 MiB of the dataset file
- Full resolved config
- Hostname, PID, and command-line arguments

Checkpoints include the complete RNG state (PyTorch, CUDA, NumPy, Python) so runs can be resumed deterministically.
