# ADoTA: Adaptive Dose Targeting with Transformers

ADoTA is a deep-learning framework for per-beamlet proton dose prediction using a 3D U-Net architecture augmented with transformer attention. Given a CT volume and an analytical flux projection for a single pencil beam, the model predicts the Monte Carlo dose distribution directly, enabling fast and accurate dose estimation for proton therapy treatment planning.

Beyond single beamlets, ADoTA assembles these per-spot predictions into a **full treatment-plan dose**: [scripts/run_plan_opentps.py](scripts/run_plan_opentps.py) takes an OpenTPS plan directory, extracts per-spot inputs, runs inference, accumulates the dose on the patient grid, and validates it against the MCsquare reference with DVH and gamma analysis (see [Plan-level dose pipeline](#plan-level-dose-pipeline)).

## What this repo offers

- **Train** the per-beamlet dose model on an HDF5 dataset ([Training](#training)).
- **Infer** single-beamlet doses from an HDF5 dataset or a directory of numpy arrays ([Inference](#inference)).
- **Predict whole-plan doses** end to end from an OpenTPS plan and validate them against MCsquare (DVH, gamma) ([Plan-level dose pipeline](#plan-level-dose-pipeline)).
- **Analyse** model behaviour vs CT texture / tissue interfaces, and benchmark preprocessing ([Analysis Scripts](#analysis-scripts)).

## Overview

| Component | Description |
|---|---|
| Model | `DoTA3D_v3` -- 3D U-Net with optional transformer encoder layers |
| Input | `(B, 2, D, H, W)` = CT channel + analytical flux channel + scalar energy |
| Output | `(B, 1, D, H, W)` predicted dose volume |
| Dataset | HDF5 dataset of beamlet records (pelvis, downsampled to 160x30x30 mm grid) |
| Training losses | Normalised MSE (`LMSE`) + Integral Depth Dose (`LPS`), balanced adaptively |
| Validation metrics | RMSE, MAPE, RDE, Gamma Pass Rate (2%/2mm), per-energy breakdowns |
| Plan pipeline | `scripts/run_plan_opentps.py` (`src/beamlets/`) -- OpenTPS plan to accumulated ADoTA dose + DVH / gamma vs MCsquare |

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
│   ├── run_plan_opentps.py         # End-to-end plan-level dose pipeline
│   ├── config_run_plan_opentps.yaml
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
│   │   ├── layers/                 # The nn.Module building blocks
│   │   │   ├── conv.py             # Conv3D (weight-standardised), ConvBlock3D_v2
│   │   │   ├── encoder_decoder.py  # ConvEncoder3D / ConvDecoder3D + skip tensors
│   │   │   ├── transformer.py      # Causal-masked attention, positional embedding
│   │   │   └── tensor_ops.py       # Permute / Reshape / Cropping shape adapters
│   │   ├── config.py               # Shared constants, device setup, logging
│   │   └── utils.py
│   │
│   ├── beamlets/                  # Plan-level pipeline: OpenTPS plan -> ADoTA dose
│   │   ├── extraction/             # Per-spot BEV CT crop + flux (serial + pooled)
│   │   │   ├── __init__.py         # Config, per-field orchestration, manifest
│   │   │   ├── spot.py             # One spot: crop + flux projection
│   │   │   ├── io.py               # Output tree and per-spot writes
│   │   │   └── overlay.py          # Per-field sanity-check PNG
│   │   ├── rotation.py             # CT rotation around the isocenter (grid-expanding)
│   │   ├── isocenter.py            # Plan->CT isocenter convention (x-flip)
│   │   ├── cropping.py             # Air-padded depth-from-entrance ROI crop
│   │   ├── flux.py                 # Analytical flux projection (NumPy + GPU twin)
│   │   ├── inference.py            # Batched ADoTA inference over extracted beamlets
│   │   ├── accumulation.py         # Deposit + de-rotate beamlets -> full-grid dose
│   │   ├── dose_scaling.py         # MCsquare/ADoTA dose -> Gy conversion
│   │   └── structures.py / dvh.py  # Oriented structure masks + DVH
│   │
│   ├── loaders/
│   │   ├── generator.py            # H5PYGenerator -- PyTorch Dataset for HDF5
│   │   ├── dir_based.py            # Per-spot record loader/saver (inference path)
│   │   └── plan_directory.py       # OpenTPS plan-directory loader/parser
│   │
│   ├── training/
│   │   ├── losses.py               # LMSE, LPS, LossLPD, TwoObjectiveBalancer
│   │   ├── loop.py                 # train_one_epoch
│   │   ├── logging_utils.py        # Phase-tagged, relative-time log formatting
│   │   ├── run_dir.py              # Run directory, manifest, MetricsLog
│   │   ├── checkpoints.py          # CheckpointManager, RNG snapshot/restore
│   │   ├── diagnostics.py          # GracefulShutdown, NaN dumps, grad/param norms
│   │   ├── validation.py           # Per-epoch validation loop (RMSE/MAPE/RDE/GPR)
│   │   ├── binning.py              # Per-sample records, energy bins, worst-K
│   │   ├── attention.py            # Canary attention snapshots
│   │   └── utils.py                # validate_tensor_ranges, get_lr
│   │
│   ├── metrics/
│   │   ├── classic.py              # RMSE, MAPE, RDE implementations
│   │   ├── gamma_pass_rate.py      # GPR wrapper around pymedphys
│   │   ├── plan_gamma.py           # Plan gamma over multiple criteria
│   │   ├── plan_metrics.py         # Plan MAPE/RMSE (high-dose mask) + RDE
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
│   ├── evaluation/                 # Shared inference-evaluation engine
│   │   ├── cli.py                  # resolve_device, merge_config, run setup
│   │   ├── sources.py              # DirSource / H5Source sample iteration
│   │   ├── engine.py               # The shared evaluate(...) loop
│   │   └── outputs.py              # CSV writer, summary printer
│   │
│   ├── figures/                    # Plotting helpers
│   │   ├── single_beam.py          # The main per-beamlet publication figure
│   │   ├── axes_utils.py           # Aligned colorbars, multi-format saving
│   │   ├── input_comparison.py     # Before/after rotation QC overlays
│   │   ├── beamlet_input.py        # Model inputs only (CT + flux)
│   │   ├── ct_visualizations.py    # HU segmentation figures
│   │   └── bp_diagnostic.py        # Bragg-peak estimator diagnostic
│   ├── image_processing/           # Texture, heterogeneity, GLCM
│   ├── tables/                     # ASCII result table formatting
│   └── utils/
│       ├── serialization.py        # NumpyEncoder for JSON serialization
│       └── dose_grid_utils.py
│
└── tests/
    ├── utils/                      # Shared test helpers (importable, not fixtures)
    │   ├── golden.py               # Golden-CSV comparison policy
    │   ├── bdl.py                  # Synthetic beam-data-library builder
    │   └── deps.py                 # Optional-dependency probes
    ├── test_import_smoke.py        # Every module under src/ imports
    ├── test_public_api.py          # Pins the importable surface of split modules
    ├── test_cli_smoke.py           # `--help` on every Typer script
    ├── test_training_losses.py     # 43 tests for LMSE, LPS, TwoObjectiveBalancer
    ├── test_checkpoint_manager.py  # 5 tests for CheckpointManager save/load
    ├── golden/                     # Characterization tests (marked `integration`)
    └── perf/                       # Performance A/B tests (marked `slow`)
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

This installs all dependencies declared in [pyproject.toml](pyproject.toml) into
`.venv/`, and installs the project itself in editable mode so `src.*` imports
resolve from any working directory.

**4. Configure the environment (optional)**

```bash
cp .env.example .env       # local values only; every variable has a working default
```

**5. Verify the setup**

```bash
uv run python scripts/run-tests.py unit
uv run ruff check .
```

The unit suite should report 0 failures. It uses only CPU and the example data
in [data/example_inputs/](data/example_inputs/); anything needing the full
dataset, a checkpoint or a GPU skips with a reason.

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

## Plan-level dose pipeline

[scripts/run_plan_opentps.py](scripts/run_plan_opentps.py) turns an OpenTPS plan
directory into a full **ADoTA plan dose** and validates it against MCsquare. It
runs as comma-separated stages:

`extract` (per-spot BEV CT crop + flux, rotating the CT around each field's
isocenter) → `infer` (batched ADoTA inference) → `accumulate` (deposit and
de-rotate the predicted beamlets onto the patient grid → `Dose_ADoTA.mhd`, plus
dose-comparison and DVH figures) → `gamma` (plan gamma pass rate per criterion +
MAPE/RMSE/RDE + gamma-map figure). It also runs **fused and disk-free** (`stream`,
~2× faster, identical dose), optionally on a **2 mm field grid** (`grid_factor: 2`,
~2.6× faster with the dose preserved).

```bash
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir /path/to/your/plan --stages stream,gamma --overwrite
```

Optional speed and quality switches (all opt-in, defaults preserve the reference
behaviour): GPU flux projection (`flux_on_gpu`), thread-pooled extraction
(`extraction_parallel`), field-level 2 mm resampling (`grid_factor`), and a measured
dose calibration (`dose_calibration_*`). The model code is unchanged — this is a
wrapper around it.

**Full guide** — what plan data you need and where to put it (DICOM reader planned),
the execution modes, the `grid_factor` 2 mm mode, the complete config/CLI reference,
outputs, and batch helpers:
[scripts/docs/run_plan_opentps.md](scripts/docs/run_plan_opentps.md).

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

One runner drives every suite:

```bash
uv run python scripts/run-tests.py unit          # default: fast, no data needed
uv run python scripts/run-tests.py unit --fast   # also skips the slow perf suite
uv run python scripts/run-tests.py integration   # needs the HDF5 dataset + a checkpoint
uv run python scripts/run-tests.py e2e           # opt-in, full workflow
uv run python scripts/run-tests.py all
```

Extra arguments are forwarded to pytest, so `run-tests.py unit -x -q` works. To
run one file directly:

```bash
uv run pytest tests/test_training_losses.py -v
```

The **unit** suite uses only CPU and the small numpy arrays in
[data/example_inputs/](data/example_inputs/) -- no GPU, no HDF5 dataset, no
network. Tests needing something the machine does not have skip with a reason
that says what to install or where the data belongs.

Suites are selected by marker: `integration` (real dataset or an external
package), `e2e`, `gpu`, `slow`. They are declared in
[pyproject.toml](pyproject.toml).

Lint with the committed ruff configuration:

```bash
uv run ruff check .
```

| Test file | Coverage |
|---|---|
| [test_training_losses.py](tests/test_training_losses.py) | LMSE, LPS, LossLPD, TwoObjectiveBalancer (43 tests) |
| [test_checkpoint_manager.py](tests/test_checkpoint_manager.py) | Round-trip fidelity, RNG state, retention policy, partial restore (5 tests) |
| [test_interface_severity.py](tests/test_interface_severity.py) | ISI metric computation |
| [test_pflugfelder_hi.py](tests/test_pflugfelder_hi.py) | Pflugfelder heterogeneity index |
| [test_import_smoke.py](tests/test_import_smoke.py) | Every module under `src/` imports cleanly |
| [test_public_api.py](tests/test_public_api.py) | The split modules still expose every name they used to |
| [test_cli_smoke.py](tests/test_cli_smoke.py) | `--help` exits 0 on every Typer script |

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
