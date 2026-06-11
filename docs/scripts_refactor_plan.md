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
