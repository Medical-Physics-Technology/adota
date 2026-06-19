# Scripts

Utility scripts for **training**, **running**, and **evaluating** ADoTA / `DoTA3D_v3`
models. Each script is **config-driven** (a YAML config; CLI flags override) and
shares the same precedence: **CLI arguments > YAML config > built-in defaults**. Run
everything through the synced environment with `uv run python scripts/<script>.py`.

Every script has a **dedicated guide in [`docs/`](docs/)** — start there for usage,
options, config reference, outputs, and requirements.

> **Reproducibility / portability.** The paths shown in the docs are example
> locations from the development environment — replace them with your own. Nothing is
> hard-coded to a specific machine: point each script at your data via the config or
> CLI, and keep large outputs in a directory of your choice (off your home if
> space-constrained). Begin with `uv sync` to create `.venv` from the pinned lockfile.

---

## Pipelines

| Script | Config | What it does | Guide |
|---|---|---|---|
| [`run_plan_opentps.py`](run_plan_opentps.py) | `config_run_plan_opentps.yaml` | **End-to-end plan-level dose pipeline**: plan directory → per-spot inputs → inference → accumulated plan dose → validation (DVH, gamma) vs Monte-Carlo. Staged or fused (`stream`), optional 2 mm field grid (`grid_factor`). | [docs/run_plan_opentps.md](docs/run_plan_opentps.md) |
| [`train_adota.py`](train_adota.py) | `config_train_adota.yaml` | Train the per-beamlet `DoTA3D_v3` model on an HDF5 dataset (AdamW + `ReduceLROnPlateau`, deterministic resume, full reproducibility manifest). | [docs/train_adota.md](docs/train_adota.md) |
| [`run_model.py`](run_model.py) | `config_run_model.yaml` | Single-beamlet inference + gamma evaluation on a directory of numpy samples. | [docs/run_model.md](docs/run_model.md) |

## Analysis & benchmarks

| Script | Config | What it does | Guide |
|---|---|---|---|
| [`analysis_texture_with_inference.py`](analysis_texture_with_inference.py) | `config_analysis_texture_with_inference.yaml` | Correlate model error (MAPE, GPR) with CT texture / heterogeneity metrics. | [docs/analysis_texture_with_inference.md](docs/analysis_texture_with_inference.md) |
| [`training_set_analysis.py`](training_set_analysis.py) | `config_training_set_analysis.yaml` | Tissue-interface prevalence at the Bragg peak + interface-vs-homogeneous performance split. | [docs/training_set_analysis.md](docs/training_set_analysis.md) |
| [`rotation_performance_analysis.py`](rotation_performance_analysis.py) | CLI flags | Benchmark 3-D rotation (SciPy / CuPy / PyTorch) around a plan pivot; correctness + timings. | [docs/rotation_performance_analysis.md](docs/rotation_performance_analysis.md) |
| [`beamlet_timing_comparison.py`](beamlet_timing_comparison.py) | CLI flags | Compare per-beamlet reinterpolation vs ADoTA projection timing on real samples. | [docs/beamlet_timing_comparison.md](docs/beamlet_timing_comparison.md) |

## Batch runners

| Script | What it does |
|---|---|
| [`run_all_plans.sh`](run_all_plans.sh) | Run `stream,gamma` (2 mm field grid) over a list of plans sequentially; logs in `run_logs/`. |
| [`run_grid_factor_ab.sh`](run_grid_factor_ab.sh) | A/B harness: `grid_factor` 1 vs 2 per plan, archived for a go/no-go comparison. |
| [`run_ablation.sh`](run_ablation.sh) | Launch the 2×2 training ablation study (see [`ablation/`](ablation/)). |

Both plan runners are documented in the [plan-pipeline guide](docs/run_plan_opentps.md#batch--reproducibility-helpers);
**edit their `PLANS=( ... )` array to your own plan directories before running.**

## Other scripts

Smaller / supporting utilities (run with `--help` for usage): `run_model_h5py.py`
(inference on an HDF5 dataset), `training_set_analysis_advanced_metrics.py`,
`ct_texture_analysis.py`, `bragg_peak_estimation.py`, `multi_radius_analysis.py`,
`threshold_sweep.py`, `compare_plan_dose.py`, `inference_worker.py`,
`plot_beamlets.py`, and the dataset helpers `add_excluded_index.py` /
`remove_h5_record.py`.

---

For the project overview and the per-beamlet model itself, see the
[repository README](../README.md).
