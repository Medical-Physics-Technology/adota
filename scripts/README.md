# Scripts

This directory contains utility scripts for running and evaluating DoTA models.

## train_adota.py

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
    --resume-from /scratch/mstryja/adota_runs/<run>/checkpoints/last.pth
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
| `--runs-dir` | string | `/scratch/mstryja/adota_runs` | Base directory under which the run directory is created |
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
    --runs-dir /scratch/mstryja/adota_runs
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
dataset_path: /scratch/mstryja/DoTA_dataset_v2/...SingleGaussian.h5
excluded_indexes_file: /home/mstryja/projects/dota_pytorch/auxilary_files/IndexesExclude_...txt
train_test_split: 0.2
batch_size: 56
num_epochs: 400
learning_rate: 0.0005
device_index: 1
seed: 1234
```

**Precedence order:** CLI arguments > YAML config > built-in defaults.

For reproducibility, the input config is copied to the run directory as `config_input.yaml`, and the fully-resolved config is written as `config.yaml` alongside a manifest (git commit, library versions, GPU info, dataset fingerprint).

### Output

Each run creates a directory under `--runs-dir` (default `/scratch/mstryja/adota_runs`, which keeps large checkpoints off the home directory) named `train_<YYYYMMDD_HHMMSS>_<config_name>`:

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

---

## run_model.py

A command-line tool for running DoTA model inference on dose prediction tasks.

### Usage

```bash
# Run with CLI arguments
uv run python scripts/run_model.py <MODEL_NAME> <TEST_DATA> [OPTIONS]

# Run from YAML config file
uv run python scripts/run_model.py --config scripts/config_run_model.yaml

# Mix: YAML config with CLI overrides (CLI takes precedence)
uv run python scripts/run_model.py --config scripts/config_run_model.yaml --dose-threshold 3.0
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `MODEL_NAME` | string | Yes | Name of the model directory (located in `models/`) |
| `TEST_DATA` | path | Yes | Path to the directory with input data to evaluate |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | - | Path to YAML configuration file |
| `--downsampling-method` | string | `interpolation` | Downsampling method: `interpolation` or `avg_pooling` |
| `--model-fname` | string | `best_model.pth` | Model filename within the model directory |
| `--device-index` | int | `0` | CUDA device index (-1 for CPU) |
| `--dose-threshold` | float | `2.0` | Dose percent threshold for gamma calculation |
| `--distance-threshold` | float | `2.0` | Distance threshold (mm) for gamma calculation |
| `--no-progress` | flag | `False` | Disable progress bar |
| `--verbose` | flag | `False` | Enable verbose/debug output |
| `--help` | - | - | Show help message and exit |

### Examples

**Basic usage with example data:**
```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs
```

**Specify a different downsampling method:**
```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs --downsampling-method avg_pooling
```

**Use a specific model checkpoint:**
```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs --model-fname checkpoint_epoch_100.pth
```

**Run on CPU instead of GPU:**
```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs --device-index -1
```

**Custom gamma parameters:**
```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs \
    --dose-threshold 3.0 \
    --distance-threshold 3.0
```

**Combine multiple options:**
```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs \
    --downsampling-method interpolation \
    --model-fname best_model.pth \
    --device-index 0 \
    --verbose
```

**Show help:**
```bash
uv run python scripts/run_model.py --help
```

### YAML Configuration

Instead of passing all options via the CLI, you can use a YAML config file. A default configuration file is provided at `scripts/config_run_model.yaml`:

```yaml
# --- Required Arguments ---
model_name: DoTA_v3_grid_search_v11
test_data: data/example_inputs

# --- Optional Settings ---
downsampling_method: interpolation
model_fname: best_model.pth
device_index: 0
dose_threshold: 2.0
distance_threshold: 2.0
no_progress: false
verbose: false
```

**Run entirely from YAML:**
```bash
uv run python scripts/run_model.py --config scripts/config_run_model.yaml
```

**Override specific YAML values with CLI flags:**
```bash
uv run python scripts/run_model.py --config scripts/config_run_model.yaml \
    --dose-threshold 3.0 --distance-threshold 3.0 --verbose
```

**Precedence order:** CLI arguments > YAML config > built-in defaults.

For reproducibility, the YAML config file is automatically copied to the run output directory (`runs/<YYYYMMDD_HHMMSS>/config_run_model.yaml`).

### Input Data Format

The `TEST_DATA` directory should contain input files with the following naming convention:
- `<uuid>_ct.npy` - CT scan data
- `<uuid>_ds.npy` - Dose distribution (ground truth)
- `<uuid>_ds_pred.npy` - Predicted dose distribution (generated by script)
- `<uuid>_flux.npy` - Flux data
- `<uuid>_sim_res.json` - Simulation results

The script will automatically detect unique sample IDs from the filenames.

### Output

Each run creates a timestamped directory in `runs/` with the format `YYYYMMDD_HHMMSS`:

```
runs/
└── 20260203_143022/
    ├── config_run_model.yaml   # Copy of config (if --config was used)
    ├── evaluation.log          # Complete log with results table
    └── figures/
        ├── gpr_vs_energy.png   # Plot of Gamma Pass Rate vs Energy
        ├── Best_E<energy>MeV.svg
        ├── Worst_E<energy>MeV.svg
        └── Closest_to_Mean_E<energy>MeV.svg
```

Additionally, predicted dose files (`<uuid>_ds_pred.npy`) are saved in the test data directory.

### Requirements

The model directory must contain:
- Model weights file (default: `best_model.pth`)
- `hyperparams.json` - Model hyperparameters configuration file

---

## rotation_performance_analysis.py

Benchmarks 3-D rotation of CT, dose, and target-mask volumes around a plan-derived pivot point. The script compares SciPy, CuPy, and PyTorch implementations, checks numerical agreement against SciPy, and writes a comparison plot plus a CSV timing table.

### Quick Smoke Test

Use the smoke mode to verify the script without external OpenTPS plan data. This uses small synthetic CT, dose, and target volumes and runs only the dependencies already declared in the project.

```bash
uv run --no-sync python scripts/rotation_performance_analysis.py \
    --smoke \
    --device cpu \
    --skip-framework cupy \
    --repeats 1 \
    --output-dir /tmp/adota_rotation_performance_smoke
```

Expected behavior:
- the script prints timings for SciPy and Torch;
- the correctness check reports `allclose=True` for CT, dose, and target;
- outputs are saved to `/tmp/adota_rotation_performance_smoke/rotation_comparison.png` and `/tmp/adota_rotation_performance_smoke/results_table.csv`.

`--no-sync` is recommended when you want to use the existing project virtual environment exactly as-is and avoid installing or updating packages.

### Run On Plan Data

To run on an OpenTPS plan folder, provide the root directory containing plan folders and the plan folder name:

```bash
uv run --no-sync python scripts/rotation_performance_analysis.py \
    --plans-root /scratch/mstryja/opentps_plans \
    --plan-name Prostate-AEC-120_100M_bilateral_test_1_review \
    --angle 30 \
    --device cuda:0 \
    --repeats 3 \
    --output-dir results/rotation_performance_analysis
```

The plan folder is expected to contain:
- `PlanPencil.txt` for the isocenter/pivot;
- `CT.mhd` for the CT grid;
- `Dose.mhd` for the dose grid, unless `--no-dose` is used;
- `target.mhd` for the target mask, unless `--no-target` is used.

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--smoke` | `False` | Use synthetic data instead of loading a plan folder. |
| `--plans-root` | `/scratch/mstryja/opentps_plans/` | Directory containing OpenTPS plan folders. |
| `--plan-name` | `Prostate-AEC-120_100M_bilateral_test_1_review` | Plan folder name inside `--plans-root`. |
| `--angle` | `30.0` | Rotation angle in degrees. |
| `--device` | CUDA if available, otherwise CPU | Device for Torch and CuPy, for example `cpu`, `cuda`, or `cuda:0`. |
| `--repeats` | `3` | Number of timed runs per framework and volume. The median is reported. |
| `--skip-framework` | none | Framework to skip. Repeatable; useful values are `scipy`, `cupy`, and `torch`. |
| `--no-dose` | `False` | Do not load or rotate `Dose.mhd`. |
| `--no-target` | `False` | Do not load or rotate `target.mhd`. |
| `--output-dir` | `results/rotation_performance_analysis` | Directory for the plot and CSV output. |
| `--show` | `False` | Display plots interactively in addition to saving them. |

### CuPy Notes

CuPy is optional and is not listed in the default project dependencies. If CuPy is not installed in the active `.venv`, run with `--skip-framework cupy`.

To benchmark CuPy, install the CuPy package that matches the CUDA runtime on the machine, then run without `--skip-framework cupy`. Do not use `uv sync` or install new packages on a shared environment unless you intentionally want to update the virtual environment.

### Outputs

For each run with `--output-dir` set, the script writes:
- `rotation_comparison.png` - a 2x2 visual comparison of original, SciPy, CuPy, and Torch rotations;
- `results_table.csv` - per-framework, per-volume timings and any errors.

The console output also includes a formatted timing table and a correctness check versus SciPy.

---

## beamlet_timing_comparison.py

Loads ADoTA test samples, extracts `beamlet_angles` with `get_single_record(..., beamlet_angle=True)`, and applies the lateral-axis correction rotation to the raw full-resolution CT, flux, and ground-truth dose volumes. CT, flux, and dose are always rotated and timed separately; the CSV also reports the combined all-volume rotation time.

The rotation order is Y/H first and X/W second:

```text
rotation_y_deg = -ba1
rotation_x_deg = ba0
```

### Run A 10-Figure CPU Test

`--max-samples` is applied per dataset. With the default two datasets, `--max-samples 5` produces 10 figures.

```bash
uv run --no-sync python scripts/beamlet_timing_comparison.py \
    --device-index -1 \
    --rotation-backend scipy \
    --max-samples 5 \
    --repeats 1 \
    --output-dir runs/beamlet_timing_10figures_all_volumes_scipy
```

### Run The Full Timing Study

Use `--full` to process all samples from each dataset. Full runs skip the per-sample CT/flux/dose validation figures automatically, but still generate the timing summary figures. If `--output-dir` is omitted, the run directory follows `runs/beamlet_timing_rotvsproj_<YYYYMMDD>_<HHMMSS>`.

```bash
uv run --no-sync python scripts/beamlet_timing_comparison.py \
    --device-index -1 \
    --rotation-backend scipy \
    --repeats 1 \
    --full
```

`--no-sync` is recommended to use the current project environment without installing or updating packages.

### Outputs

Each run writes:
- `per_sample_rotation_timing.csv` - one row per sample with `ct_rotation_*`, `flux_rotation_*`, `dose_rotation_*`, and `all_rotation_total_s` timing columns;
- `per_sample_branch_comparison.csv` - per-sample comparison of fixed ADoTA projection time versus measured CT/flux/dose rotation time, both with the shared crop/ray-tracer time included;
- `timing_summary.csv` and `timing_summary.json` - overall and per-dataset mean/median/min/max/std summaries, including branch speedup ratios;
- `rotated_previews/*.npz` - original and rotated CT, flux, and dose arrays;
- `figures/preprocessing_times_per_beamlet_rotvsproj.png` and `figures/preprocessing_times_per_beamlet_rotvsproj.pdf` - publication-style two-panel bar figure comparing reinterpolation against ADoTA projection;
- `figures/*.png` - optional five-row per-sample comparison figures, plus `timing_branch_comparison.png` and `timing_volume_breakdown.png` summary plots;
- `config.json` and `run.log` for reproducibility.

The legacy `--rotation-volume` option is ignored for timing; it remains accepted only for compatibility. All three volumes are rotated in every run.

---

## analysis_texture_with_inference.py

Combines DoTA model evaluation with CT heterogeneity / texture analysis and computes **Pearson correlations** between model performance (MAPE, GPR) and CT complexity scores (G_φ, R, H_φ, GLCM homogeneity).

For each sample the script:
1. Runs DoTA model inference and computes standard metrics (GPR, RMSE, MAPE, RDE).
2. Loads the **original** (un-normalised) CT and fluence grids, then computes beam-aligned heterogeneity metrics and GLCM homogeneity.
3. Prints a combined results table.
4. Computes Pearson correlations with significance levels.
5. Generates scatter plots with trend lines.
6. Saves everything (JSON, CSV, correlation plots) in a timestamped run directory.

### Usage

```bash
# Run from YAML config file (recommended)
uv run python scripts/analysis_texture_with_inference.py \
    --config scripts/config_analysis_texture_with_inference.yaml

# Run with CLI arguments
uv run python scripts/analysis_texture_with_inference.py \
    DoTA_v3_grid_search_v11 /path/to/test_data [OPTIONS]

# Mix: YAML config with CLI overrides (CLI takes precedence)
uv run python scripts/analysis_texture_with_inference.py \
    --config scripts/config_analysis_texture_with_inference.yaml \
    --device-index 0 --verbose
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `MODEL_NAME` | string | Yes | Name of the model directory (located in `models/`) |
| `TEST_DATA` | path | Yes | Path to the directory with input data to evaluate |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config`, `-c` | path | – | Path to YAML configuration file |
| `--downsampling-method` | string | `interpolation` | Downsampling method: `interpolation` or `avg_pooling` |
| `--model-fname` | string | `best_model.pth` | Model filename within the model directory |
| `--device-index` | int | `0` | CUDA device index (-1 for CPU) |
| `--enable-heterogeneity` | bool | `True` | Enable beam-aligned heterogeneity metrics (G_φ, R, H_φ) |
| `--enable-glcm` | bool | `True` | Enable GLCM homogeneity metrics |
| `--enable-intensity` | bool | `True` | Enable global intensity heterogeneity (entropy, uniformity, IQR, MAD, etc.) |
| `--no-progress` | flag | `False` | Disable progress bar |
| `--verbose`, `-v` | flag | `False` | Enable verbose/debug output |
| `--help` | – | – | Show help message and exit |

### Examples

**Run with the example data on GPU 1:**
```bash
uv run python scripts/analysis_texture_with_inference.py \
    DoTA_v3_grid_search_v11 data/example_inputs \
    --device-index 1
```

**Run entirely from YAML config:**
```bash
uv run python scripts/analysis_texture_with_inference.py \
    --config scripts/config_analysis_texture_with_inference.yaml
```

**Override YAML values from the CLI:**
```bash
uv run python scripts/analysis_texture_with_inference.py \
    --config scripts/config_analysis_texture_with_inference.yaml \
    --device-index -1 --verbose
```

**Run only beam-aligned heterogeneity (disable GLCM and intensity):**
```bash
uv run python scripts/analysis_texture_with_inference.py \
    --config scripts/config_analysis_texture_with_inference.yaml \
    --enable-heterogeneity --no-enable-glcm --no-enable-intensity
```

**Show help:**
```bash
uv run python scripts/analysis_texture_with_inference.py --help
```

### YAML Configuration

A default configuration file is provided at `scripts/config_analysis_texture_with_inference.yaml`:

```yaml
# --- Model Settings ---
model_name: DoTA_v3_grid_search_v11
test_data: /scratch/mstryja/DoTA_dataset_v2/beamlet_angle_robustness_Lung-PET-CT-Dx_791_e135_v2
downsampling_method: interpolation
model_fname: best_model.pth
device_index: 1

# --- Gamma Parameters ---
gamma:
  dose_percent_threshold: 3.0
  distance_mm_threshold: 3.0
  lower_percent_dose_cutoff: 10

# --- Scaling Parameters ---
scale:
  min_ds: 0.0
  max_ds: 25277028.0
  min_ct: -1024
  max_ct: 3071
  min_energy: 70.0
  max_energy: 270.0

# --- Heterogeneity Metrics ---
heterogeneity:
  spacing_zyx: [2.0, 2.0, 2.0]
  beam_axis: 0
  mask_threshold: -900

# --- GLCM Homogeneity ---
glcm:
  levels: 64
  variant: idm
  distances: [1]
  angles_deg: [0, 45, 90, 135]
  symmetric: true
  n_sample_slices: 10

# --- Global Intensity Heterogeneity ---
intensity:
  hu_range: null        # fixed range [min, max] or null for auto
  bin_width_hu: 25.0

# --- Metrics Toggle ---
metrics:
  heterogeneity: true
  glcm: true
  intensity: true

# --- Output ---
runs_dir: runs
verbose: false
```

**Precedence order:** CLI arguments > YAML config > built-in defaults.

For reproducibility, the YAML config file is automatically copied to the run output directory.

### Computed Metrics

#### Model Performance Metrics

| Metric | Description |
|--------|-------------|
| **GPR** | Gamma Pass Rate (%) |
| **RMSE** | Root Mean Square Error (Gy) |
| **MAPE** | Mean Absolute Percentage Error (%) |
| **RDE** | Relative Dose Error (%) |

#### CT Texture / Heterogeneity Metrics

| Metric | Symbol | Description |
|--------|--------|-------------|
| Beam-weighted gradient | **G_φ** | Weighted average of ‖∇I‖₂ using beamlet fluence weights. Quantifies tissue-boundary variation as seen by the beam. |
| Beam-axis roughness | **R** | Mean absolute difference of lateral-mean HU between consecutive depth slices. Captures depth-wise tissue composition changes. |
| Combined heterogeneity | **H_φ** | Product G_φ × R — a single scalar combining gradient and roughness information. |
| GLCM homogeneity | **IDM** | Inverse Difference Moment from the Gray-Level Co-occurrence Matrix, averaged over sampled axial slices. |
| Intensity entropy | **entropy** | Shannon entropy of the HU histogram (nats). Higher values indicate more diverse tissue composition. |
| Intensity uniformity | **uniformity** | Energy / uniformity of the HU histogram (Σ p²). Higher values indicate more homogeneous tissue. |
| Inter-quartile range | **IQR** | Q75 − Q25 of the HU distribution. Robust dispersion measure. |
| Median absolute deviation | **MAD** | Median of |HU − median(HU)|. Robust alternative to standard deviation. |
| Skewness | **skewness** | Fisher-Pearson standardised third moment. Describes asymmetry of HU distribution. |
| Excess kurtosis | **kurtosis** | Fourth standardised moment − 3. Describes tail heaviness relative to Gaussian. |

#### Correlation Analysis

Pearson correlation coefficients (r, p-value) are computed between:
- **MAPE** vs {G_φ, R, H_φ, GLCM homogeneity, entropy, uniformity, IQR, MAD}
- **GPR** vs {G_φ, R, H_φ, GLCM homogeneity, entropy, uniformity, IQR, MAD}

Only the enabled metric families are included in the correlation analysis.

### Output

Each run creates a timestamped directory in `runs/` with the format `analysis_YYYYMMDD_HHMMSS`:

```
runs/
└── analysis_20260226_143022/
    ├── config_analysis_texture_with_inference.yaml  # Copy of config
    ├── analysis.log                                 # Complete log with tables
    ├── results.csv                                  # Per-sample results (spreadsheet-friendly)
    ├── results.json                                 # Full results + correlations
    └── figures/
        ├── correlation_MAPE.png                     # MAPE vs texture metrics scatter plots
        └── correlation_GPR.png                      # GPR vs texture metrics scatter plots
```

### Input Data Format

Same as `run_model.py` — the `TEST_DATA` directory should contain:
- `<uuid>_ct.npy` — CT scan data
- `<uuid>_ds.npy` — Dose distribution (ground truth)
- `<uuid>_flux.npy` — Flux data
- `<uuid>_sim_res.json` — Simulation results metadata

### Requirements

The model directory must contain:
- Model weights file (default: `best_model.pth`)
- `hyperparams.json` — Model hyperparameters configuration file

---

## training_set_analysis.py

Quantifies how many beamlets in the HDF5 training set have their Bragg peak at a tissue-density interface and compares ADoTA's performance on **interface** vs **homogeneous** cases.

For each beamlet the script:
1. Locates the **Bragg peak** from the Monte-Carlo ground-truth dose.
2. Extracts CT density in a **spherical neighbourhood** (configurable radius) around the BP.
3. Computes three heterogeneity metrics inside that sphere:
   - **σ_HU** — standard deviation of HU values (used as the classification criterion).
   - **TV** — Total Variation: Σ|v_{i+1} − v_i|, capturing roughness of tissue composition.
   - **CV** — Coefficient of Variation: σ(v) / |μ(v)|, capturing relative density spread.
4. Classifies the beamlet as *interface* (σ_HU > threshold) or *homogeneous*.
5. Runs ADoTA inference and computes per-beamlet metrics (GPR, RMSE, MAPE, RDE).
6. Reports prevalence + per-group performance statistics; generates figures.

### Usage

```bash
# Run from YAML config file (recommended)
uv run python scripts/training_set_analysis.py \
    --config scripts/config_training_set_analysis.yaml

# Run with CLI arguments
uv run python scripts/training_set_analysis.py <MODEL_NAME> <H5_PATH> [OPTIONS]

# Mix: YAML config with CLI overrides (CLI takes precedence)
uv run python scripts/training_set_analysis.py \
    --config scripts/config_training_set_analysis.yaml \
    --sigma-hu-threshold 200 --n-samples 500
```

### Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `MODEL_NAME` | string | Yes | Name of the model directory (located in `models/`) |
| `H5_PATH` | path | Yes | Path to the HDF5 dataset file |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | – | Path to YAML configuration file |
| `--excluded-indexes-file` | path | – | Path to file listing record IDs to exclude |
| `--model-fname` | string | `best_model.pth` | Model filename within the model directory |
| `--device-index` | int | `0` | CUDA device index (-1 for CPU) |
| `--sigma-hu-threshold` | float | `150.0` | σ_HU threshold [HU] to classify interface beamlets |
| `--bp-radius-mm` | float | `5.0` | Radius of spherical neighbourhood at Bragg peak [mm] |
| `--n-samples` | int | all | Limit evaluation to the first N samples |
| `--no-gpr` | flag | `False` | Skip gamma pass rate calculation (faster experiments) |
| `--no-progress` | flag | `False` | Disable progress bar |
| `--verbose` | flag | `False` | Enable verbose/debug output |
| `--help` | – | – | Show help message and exit |

### Examples

**Run entirely from YAML config:**
```bash
uv run python scripts/training_set_analysis.py \
    --config scripts/config_training_set_analysis.yaml
```

**Fast experiment (500 samples, no GPR):**
```bash
uv run python scripts/training_set_analysis.py \
    --config scripts/config_training_set_analysis.yaml \
    --n-samples 500 --no-gpr
```

**Override classification threshold:**
```bash
uv run python scripts/training_set_analysis.py \
    --config scripts/config_training_set_analysis.yaml \
    --sigma-hu-threshold 200 --bp-radius-mm 7.0
```

**Run with CLI arguments only:**
```bash
uv run python scripts/training_set_analysis.py \
    DoTA_v3_grid_search_v11 /scratch/mstryja/DoTA_dataset_v2/dataset.h5 \
    --device-index 1 --n-samples 1000 --verbose
```

**Show help:**
```bash
uv run python scripts/training_set_analysis.py --help
```

### YAML Configuration

A default configuration file is provided at `scripts/config_training_set_analysis.yaml`:

```yaml
# ── Required ────────────────────────────────────────────────
model_name: DoTA_v3_grid_search_v11
h5_path: /scratch/mstryja/DoTA_dataset_v2/dataset.h5

# ── Optional ────────────────────────────────────────────────
model_fname: best_model.pth
excluded_indexes_file: /path/to/IndexesExclude.txt
device_index: 1

# ── Analysis parameters ────────────────────────────────────
sigma_hu_threshold: 150.0   # σ_HU [HU] for interface classification
bp_radius_mm: 5.0           # Sphere radius at Bragg peak [mm]
                             # (used for σ_HU, TV, and CV)

# ── Sampling ────────────────────────────────────────────────
# n_samples: 500             # Limit to first N (omit for all)

# ── Flags ───────────────────────────────────────────────────
no_progress: false
verbose: false
# no_gpr: true               # Skip GPR for speed
```

**Precedence order:** CLI arguments > YAML config > built-in defaults.

For reproducibility, the YAML config file is automatically copied to the run output directory.

### Computed Metrics

#### Bragg-Peak Heterogeneity Metrics (within sphere)

| Metric | Symbol | Description |
|--------|--------|-------------|
| Standard deviation | **σ_HU** | Std. dev. of HU values in the spherical neighbourhood. Used as classification criterion. |
| Total Variation | **TV** | Sum of absolute differences between consecutive HU voxels: Σ\|v_{i+1} − v_i\|. Captures roughness. |
| Coefficient of Variation | **CV** | Ratio σ(v) / \|μ(v)\|. Captures relative density spread. |

#### Model Performance Metrics

| Metric | Description |
|--------|-------------|
| **GPR** | Gamma Pass Rate (2%/2mm) [%] — optionally skipped with `--no-gpr` |
| **RMSE** | Root Mean Square Error [Gy] |
| **MAPE** | Mean Absolute Percentage Error [%] |
| **RDE** | Relative Dose Error [%] |

### Output

Each run creates a timestamped directory in `runs/` with the format `YYYYMMDD_HHMMSS`:

```
runs/
└── 20260330_143022/
    ├── config_training_set_analysis.yaml   # Copy of config (if --config was used)
    ├── evaluation.log                      # Complete log with tables & summaries
    ├── results.csv                         # Per-beamlet results (all metrics)
    └── figures/
        ├── sigma_hu_histogram.png          # Distribution of σ_HU with threshold
        ├── interface_vs_homogeneous_violin.png  # Violin+box plots per group
        ├── sigma_hu_vs_gpr_scatter.png     # σ_HU vs GPR scatter (when GPR computed)
        └── publication/
            ├── homogeneous_Best_*.svg      # Best/Worst/Mean RDE per group
            ├── homogeneous_Worst_*.svg
            ├── homogeneous_Closest_to_Mean_*.svg
            ├── interface_Best_*.svg
            ├── interface_Worst_*.svg
            └── interface_Closest_to_Mean_*.svg
```

### Input Data Format

The script reads from an **HDF5 dataset** produced by the DoTA data pipeline. Each record contains a 2-channel input tensor (CT + flux), beam energy, and a 3-D Monte-Carlo dose grid. An optional exclusion file can be provided to skip known bad records.

### Requirements

The model directory must contain:
- Model weights file (default: `best_model.pth`)
- `hyperparams.json` — Model hyperparameters configuration file
