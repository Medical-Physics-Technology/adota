# rotation_performance_analysis.py

> **Reproducibility / portability.** The paths in the examples and config
> snippets below are the original development environment's locations — replace
> them with your own. Nothing is hard-coded to a specific machine: point the
> script at your data via the config or CLI, and keep large outputs in a
> directory of your choice (off your home if space-constrained).

---

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
    --plans-root /path/to/opentps_plans \
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
| `--plans-root` | `/path/to/opentps_plans/` | Directory containing OpenTPS plan folders. |
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
