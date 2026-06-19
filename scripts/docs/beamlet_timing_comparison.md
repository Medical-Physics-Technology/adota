# beamlet_timing_comparison.py

> **Reproducibility / portability.** The paths in the examples and config
> snippets below are the original development environment's locations — replace
> them with your own. Nothing is hard-coded to a specific machine: point the
> script at your data via the config or CLI, and keep large outputs in a
> directory of your choice (off your home if space-constrained).

---

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
