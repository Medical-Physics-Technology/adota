# Scripts refactor plan (`scripts/`)

Status: **draft, awaiting decisions** (see "Open questions" at the end).

Goal: keep the granular, one-scenario-per-script layout (easy to control and run
individual scenarios) while sharing as much code as possible between scripts.
Maintain all current functionality. Keep `typer` as the CLI. As part of the
work, speed up the pipelines — especially pre/post-processing.

---

## 1. Analysis of the current state

**Inventory:** 17 scripts, ~14,000 lines total.

Three scripts are special cases and are largely out of scope:
- `run_model_hptc.py` — runs in the OpenTPS environment, different stack.
- `inference_worker.py` — intentionally minimal subprocess worker; keep
  dependency-light.
- `train_adota.py` — already refactored (shares `src/training/`).

The remaining analysis/evaluation scripts reimplement the same pipeline.

### Duplication map (verified by inspection)

1. **The inference-evaluation pipeline (largest).** `run_model.py`,
   `run_model_h5py.py`, `analysis_texture_with_inference.py`,
   `training_set_analysis.py` each reimplement `evaluate_single_sample` +
   `evaluate_samples` (~100 lines each). Identical skeleton: load sample →
   `model(...)` → `inverse_minmax` → RMSE / MAPE / RDE (± GPR, ± TV/CV) →
   result dataclass. They differ only in **data source** (directory vs H5) and
   **which metrics are toggled on**.

2. **CLI/config boilerplate.** Every script repeats ~40–60 lines: `sys.path`
   insert, typer app, YAML load, then *manual per-field* `cli_value or
   yaml.get(...)` merging. ~500 lines of copy-paste; each new option must be
   hand-wired in every script.

3. **Output layer.** `save_results_csv` ×7, `print_summary` ×5,
   `generate_publication_figures` ×4, `setup_logging` / `setup_run_dir` ×4 —
   near-identical with minor field-name drift.

4. **Local dataclasses.** `run_model.py`, `beamlet_timing_comparison.py`,
   `rotation_performance_analysis.py` define result dataclasses locally even
   though `src/schemas/results.py` exists.

5. **Bragg-peak logic** appears in 7 scripts; some import
   `src/utils/dose_grid_utils.py`, others inline their own.

### Performance hotspots (pre/post-processing)

- **P1 — H5 file opened per sample.** `src/loaders/generator.py` (`__getitem__`)
  calls `h5py.File(...)` for *every* record; the analysis scripts iterate tens
  of thousands of samples over a ~196 GB file. A persistent (per-process)
  handle removes a per-sample constant cost.
- **P2 — batch=1 inference everywhere.** All analysis scripts run the model one
  sample at a time. A batched evaluation engine (configurable `batch_size`) is
  the largest GPU-throughput win.
- **P3 — CPU round-trips in post-processing.** Tensors go GPU→CPU→NumPy per
  sample, then `inverse_minmax` + three metric passes over full volumes;
  `to_gy(...)` is recomputed on the same array. Computing denormalization +
  RMSE/MAPE/RDE on-GPU in torch (batched), transferring only scalars, removes
  most of this.
- **P4 — `torch.no_grad()` → `torch.inference_mode()`** (cheaper).
- **P5 — GPR (gamma)** is the dominant per-sample cost where enabled; keep it
  opt-in / subset-based (as some scripts already do).

---

## 2. Target architecture

Scripts stay granular (one scenario = one thin typer CLI + its YAML). All shared
logic moves into a new `src/evaluation/` package:

```
src/evaluation/
├── sources.py    # SampleSource protocol: H5Source (persistent handle), DirSource
│                 #   -> uniform iteration over (id, x, energy, y), optional batching
├── engine.py     # evaluate(model, source, metrics=..., batch_size=..., device=...)
│                 #   -> batched inference_mode loop, on-GPU denorm + metrics,
│                 #      returns list[EvaluationResult]; per-metric toggles
│                 #      (rmse, mape / thresholded-mape, rde, gpr, tv_cv)
├── outputs.py    # save_results_csv (superset schema, per-script fieldname list),
│                 #   print_summary, figure-selection helper (best/worst/mean)
└── cli.py        # merge_yaml_cli(config_path, overrides) generic merger,
                  #   setup_logging, setup_run_dir, resolve_device
```

Plus: consolidate stray local dataclasses into `src/schemas/results.py`, and
route all Bragg-peak uses through `src/utils/dose_grid_utils.py`.

Each script keeps: its `typer` signature (same flags), its YAML, and only its
scenario-specific logic (texture metrics, VLM scoring, threshold-sweep loop, …).

Expected reduction: roughly 2.5–3.5k of the 14k lines; new scenarios become
~100-line scripts.

---

## 3. Phasing

Each phase is independently verifiable and preserves functionality.

- **Phase 0 — characterization tests (safety net, before touching anything).**
  Run `run_model_h5py.py` and `training_set_analysis.py` on a small fixed slice
  of the real dataset, capture the produced CSV rows as golden values in-repo,
  assert numeric equality (tolerance ~1e-6). Pins today's behavior.
- **Phase 1 — shared CLI infra** (`cli.py`): generic YAML+CLI merge, logging,
  run-dir, device resolve. Adopt in 2 pilot scripts; behavior identical.
- **Phase 2 — data sources + P1**: `H5Source` with a persistent handle (fix
  `H5PYGenerator` to lazily cache the handle per process, close on `__del__`);
  `DirSource` wrapping `get_single_record`. Benchmark per-sample load before/after.
- **Phase 3 — evaluation engine + P2/P3/P4**: batched `evaluate(...)` with
  on-GPU metrics; adopt in `run_model.py` + `run_model_h5py.py` first (purest
  duplicates). Characterization tests stay green; a perf test (extending
  `tests/perf/`) reports samples/s before vs after with a strict "not slower"
  gate and an expected multi-× win.
- **Phase 4 — adopt in the analysis scripts** one at a time:
  `analysis_texture_with_inference`, `training_set_analysis`,
  `training_set_analysis_advanced_metrics`, `threshold_sweep`,
  `training_set_vlm_based_quantification`, `bragg_peak_estimation`,
  `multi_radius_analysis`. Each keeps its CLI/flags; each diff reviewed against
  its golden CSV.
- **Phase 5 — outputs consolidation**: shared CSV writer (exact per-script field
  order preserved via explicit fieldname lists — no format drift), summary
  printer, figure helpers.
- **Phase 6 — cleanup**: delete now-dead local copies, changelog entry, version
  bump.

**Out of scope (deliberate):** `run_model_hptc.py` (different venv);
`inference_worker.py` (kept dependency-minimal — only swap to `inference_mode`);
`rotation_performance_analysis.py` and `ct_texture_analysis.py` (self-contained,
low duplication — Phase 1 infra only).

---

## 4. Risks & mitigations

- **CSV format drift breaking downstream analysis** → explicit fieldname lists
  per script + golden-CSV characterization tests.
- **Numeric drift from GPU metrics** (float32 sum order) → goldens compared with
  tolerance; if any metric is sensitive, keep that one on CPU float64.
- **H5 handle + DataLoader workers** → open lazily per process (standard
  pattern); never share a handle across forks.
- **Long-running ablation trainings** → nothing here touches `train_adota.py`
  or the model.

---

## 5. Open questions (need clear instructions)

1. **CSV byte-stability vs numeric equality.** Must output CSVs be
   byte-for-byte identical to today (column order, header names, float
   formatting), or is numeric equality enough (allowing column rename/reorder)?
   Recommendation: preserve exact formats — cheap insurance for downstream
   consumers. *Need: confirm.*

2. **Phase order — perf first or correctness-infra first?** Plan does
   safety-net + CLI infra before the perf engine. Do you want P1–P3 (perf)
   pulled forward instead? Recommendation: keep characterization tests first,
   then perf. *Need: confirm or reorder.*

3. **On-GPU metrics.** OK to move RMSE/MAPE/RDE to batched torch on-GPU (scalars
   transferred only), accepting tiny float32 reordering differences within
   tolerance? Or must metric values match the current NumPy path exactly?
   *Need: tolerance policy.*

4. **Golden-test data dependency.** Characterization tests need the real dataset
   on `/scratch`. OK to `skipif`-gate them (like the existing smoke test), so
   they run locally but skip in environments without the data? *Need: confirm.*

5. **Batched data source shapes.** Records can have variable native shape
   (cropped to (160,30,30)). For batching, all samples must share a shape after
   crop/interp — confirm every analysis path already enforces the (160,30,30)
   grid (so a batch is well-formed), or should the engine fall back to batch=1
   for variable-shape sources?

6. **`H5PYGenerator` persistent-handle change.** P1 modifies a core loader used
   by `train_adota.py` too. OK to change it (carefully, lazily per process), or
   should the persistent handle live only in the new `H5Source` wrapper and
   leave `H5PYGenerator` untouched? Recommendation: implement in `H5Source`
   first, leave the generator alone unless training also benefits. *Need: confirm.*

7. **Scope confirmation.** Agreed that `run_model_hptc.py`,
   `inference_worker.py`, `rotation_performance_analysis.py`, and
   `ct_texture_analysis.py` stay out of the deep refactor (infra-only)?

8. **`src/evaluation/` package name/location.** Acceptable, or do you prefer the
   shared engine to live under an existing package (e.g. `src/training/` or a
   new `src/inference/`)? *Need: confirm naming.*

---

# Appendix A — Per-file split proposal for the 500-line limit

Added by the baseline-alignment refactor (see
[baseline_alignment_refactor_plan.md](baseline_alignment_refactor_plan.md),
Phase 5). **Nothing here has been executed.** It needs separate approval, and
its proof of equivalence is the golden CSVs, which require the real dataset.

## A.1 Where things stand

Every file under `src/` is now within the mandatory 500-line limit. Sixteen
files under `scripts/` are not:

| Script | Lines | Tier |
| --- | --- | --- |
| `training_set_analysis_advanced_metrics.py` | 1804 | 1 |
| `analysis_texture_with_inference.py` | 1577 | 1 |
| `training_set_analysis.py` | 1448 | 1 |
| `run_model.py` | 1379 | 1 |
| `run_plan_opentps.py` | 1162 | 3 |
| `training_set_vlm_based_quantification.py` | 1030 | 1 |
| `rotation_performance_analysis.py` | 1007 | 2 |
| `beamlet_timing_comparison.py` | 960 | 2 |
| `beamlet_bev_rotation_timing.py` | 861 | 2 |
| `threshold_sweep.py` | 841 | 1 |
| `bragg_peak_estimation.py` | 839 | 1 |
| `train_adota.py` | 657 | 3 |
| `range_analysis.py` | 656 | 1 |
| `run_model_h5py.py` | 640 | 1 |
| `validation_adota.py` | 627 | 3 |
| `reinterp_gpu_benchmark.py` | 507 | 2 |

Tier 1 = adopts the shared `src/evaluation/` engine (the work Part 1 of this
document began). Tier 2 = self-contained benchmark, split in place. Tier 3 =
orchestration entry point, split in place.

## A.2 Duplication, re-measured

An AST scan over `scripts/` finds **~1784 lines of redundant copies** across 19
duplicated names. The largest:

| Name | Copies | Total lines |
| --- | --- | --- |
| `generate_publication_figures` | 4 | 512 |
| `_make_per_sample_fn` | 5 | 499 |
| `save_results_csv` | 7 | 251 |
| `print_summary` | 5 | 214 |
| `generate_correlation_analysis` | 2 | 158 |
| `analyse_density_regions` | 2 | 140 |
| `evaluate_samples` | 3 | 134 |
| `_build_timing_report` | 2 | 124 |
| `plot_energy_stratified` | 2 | 96 |
| `plot_worst_idd_overlays` | 2 | 81 |

This confirms the duplication map in section 1 and refines it: `_make_per_sample_fn`
has spread to five scripts since that map was written, and `save_results_csv`
to seven. **Extracting the duplicates is what brings most scripts under the
limit** — a mechanical split into `_figures.py` / `_io.py` siblings would meet
the letter of the rule while leaving the duplication in place, and is not what
is proposed here.

## A.3 Proposed destinations

New shared modules, extending the existing `src/evaluation/` package:

```
src/evaluation/
├── metrics_fn.py     # the per-sample metric callables the 5 _make_per_sample_fn
│                     #   copies collapse into, parameterized by which metrics
│                     #   are on (rmse/mape/rde/gpr/tv/cv/sigma_hu/bp)
├── correlation.py    # generate_correlation_analysis + _correlate_target +
│                     #   _partial_correlation + the metric/target vocabulary
└── figures.py        # the shared parts of the 4 generate_publication_figures
                      #   copies (best/worst/mean selection, panel layout)
```

`src/figures/` gains the plot functions that are currently duplicated in
scripts: `plot_energy_stratified`, `plot_worst_idd_overlays`, `generate_gpr_plot`,
the violin/scatter/histogram family from `training_set_analysis.py`.
`src/schemas/results.py` absorbs the three local `TestDataset` dataclasses and
`normalize_test_data_config` / `discover_sample_ids`.

Per-script, after that extraction:

| Script | Stays in the script | Moves out |
| --- | --- | --- |
| `training_set_analysis_advanced_metrics.py` | CLI, `extract_all_samples`, `read_angle_map` | density-region + advanced metrics -> `src/processing/`; correlation and energy-stratified analysis -> `src/evaluation/correlation.py`; angle-performance maps and clustermap -> `src/figures/` |
| `analysis_texture_with_inference.py` | CLI, texture-metric selection | heterogeneity/GLCM/intensity metric wrappers -> `src/image_processing/`; correlation tables + plots -> `src/evaluation/correlation.py`; `generate_metrics_description` -> `src/tables/` |
| `training_set_analysis.py` | CLI, prevalence report | BP sigma/TV/CV -> `src/processing/`; the six figure generators -> `src/figures/` |
| `run_model.py` | CLI, anatomical-site summary | `density_variability_vs_gpr` + `advanced_metrics_and_figures` -> `src/figures/`; CSV/summary -> `src/evaluation/outputs.py` |
| `training_set_vlm_based_quantification.py` | CLI, VLM prompt + parsing + voting | `render_review_panel` -> `src/figures/`; `analyse_density_regions` / `estimate_bp_range` -> shared (duplicated with the advanced-metrics script) |
| `threshold_sweep.py` | CLI, `run_sweep` | the three plot functions -> `src/figures/` |
| `bragg_peak_estimation.py` | CLI | the five estimator classes -> `src/processing/bp_estimators/`; its local `setup_logging`/`setup_run_directory`/`load_yaml_config`/`denormalize_energy` are re-implementations of `src.adota.config` and should just be deleted |
| `range_analysis.py` | CLI | plots -> `src/figures/`; `TestDataset` plumbing -> `src/schemas/` |
| `run_model_h5py.py` | CLI | figures -> `src/figures/`; CSV/summary -> `src/evaluation/outputs.py` |
| `rotation_performance_analysis.py` | CLI | the three backend implementations (scipy/cupy/torch) -> `src/image_processing/rotation_backends/`; table + plot -> `src/figures/` |
| `beamlet_timing_comparison.py` | CLI | `write_publication_preprocessing_figure` (194 lines) -> `src/figures/`; timing summary -> `src/evaluation/outputs.py` |
| `beamlet_bev_rotation_timing.py` | CLI | validation figures -> `src/figures/`; `_build_timing_report` / `_format_timing_table` -> shared with `run_plan_opentps.py` (duplicated today) |
| `reinterp_gpu_benchmark.py` | CLI | timing/table/figure helpers -> shared benchmark utilities |
| `run_plan_opentps.py` | CLI, stage dispatch | each `_run_*_stage` -> `src/beamlets/stages/`; timing report -> shared |
| `train_adota.py` | CLI | its 555-line `main` is an orchestration flow; extract the epoch loop body into `src/training/` helpers |
| `validation_adota.py` | CLI | `_run_logs_mode` / `_run_inference_mode` -> `src/evaluation/` |

## A.4 Sequencing and proof

Do it one script at a time, in the Part 1 order (`run_model.py` and
`run_model_h5py.py` first, since `src/evaluation/` already serves them), and
after each one:

1. `uv run ruff check .` clean, file under 500 lines;
2. `tests/test_import_smoke.py`, `tests/test_public_api.py`,
   `tests/test_cli_smoke.py` green — these run anywhere and catch a broken
   extraction immediately;
3. **on the machine with the dataset**, `python scripts/run-tests.py integration`
   — the golden CSV for that script must be unchanged. This is the only check
   that proves numeric equivalence, and it cannot run on a laptop.

Extend `tests/golden/` to cover each Tier-1 script before that script is
touched; today it covers four of them (`run_model`, `run_model_h5py`,
`training_set_analysis`, `training_set_analysis_advanced_metrics`).

## A.5 Note on two functions that will still be too large

Outside `scripts/`, two figure functions survived the `src/` phase intact and
breach PYTHON_CODE_STYLE section 5 (small, single-responsibility helpers) even
though their files are now compliant:

- `src/figures/single_beam.py::publication_figure` — 460 lines, leaving the file
  at 482 of the permitted 500.
- `src/figures/ct_visualizations.py::plot_ct_with_segmentation` — 336 lines.

Both are single multi-panel matplotlib layouts. Decomposing them is safe only
with a figure-level regression check (render to PNG, compare against a stored
reference within tolerance); there is no such check today, and the existing
tests only assert that the call completes. Recommend adding that check first,
then splitting each into per-panel helpers.
