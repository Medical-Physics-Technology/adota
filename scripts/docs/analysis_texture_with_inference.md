# analysis_texture_with_inference.py

> **Reproducibility / portability.** The paths in the examples and config
> snippets below are the original development environment's locations — replace
> them with your own. Nothing is hard-coded to a specific machine: point the
> script at your data via the config or CLI, and keep large outputs in a
> directory of your choice (off your home if space-constrained).

---

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
test_data: /path/to/DoTA_dataset_v2/beamlet_angle_robustness_Lung-PET-CT-Dx_791_e135_v2
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
