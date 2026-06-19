# training_set_analysis.py

> **Reproducibility / portability.** The paths in the examples and config
> snippets below are the original development environment's locations — replace
> them with your own. Nothing is hard-coded to a specific machine: point the
> script at your data via the config or CLI, and keep large outputs in a
> directory of your choice (off your home if space-constrained).

---

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
    DoTA_v3_grid_search_v11 /path/to/DoTA_dataset_v2/dataset.h5 \
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
h5_path: /path/to/DoTA_dataset_v2/dataset.h5

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
