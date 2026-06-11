# Scripts refactor: Part 1 action plan (architecture + functional tests)

Status: **ready to execute**. Scope: migrate the inference-evaluation pipeline
onto the target `src/evaluation/` architecture **without changing behavior**, and
prove equivalence with functional (golden) tests. Performance work (batching,
on-GPU classic metrics, persistent H5 handle, `inference_mode`) is **explicitly
deferred to Part 2**.

## Decisions locked in (from review)

1. **CSV stability:** preserve exact current formats (column order, header
   names, float formatting). Per-script explicit fieldname lists. No drift.
   Verification is structural + numeric, not literal bytes, because the CSVs
   contain non-deterministic timing columns (`calc_time_s`, `extract_time_s`)
   and timing-derived summary rows. The golden check: header line matches
   exactly (locks column order, names, and `f"{...:.Nf}"` precision),
   non-numeric columns match exactly, metric columns match within numeric
   tolerance, timing columns and their summary stats are ignored.
2. **Order:** correctness first. This whole document is correctness only. Every
   step preserves numerics; no optimization lands here.
3. **Device management:** a single `resolve_device` chooses GPU when available,
   else CPU, and falls back to CPU if a requested GPU index is unavailable. All
   model and input tensors follow the resolved device; tensors are moved back to
   CPU for the NumPy metric path exactly as today. The "classic metrics on GPU"
   option is scaffolded behind a flag that defaults to the current CPU/NumPy
   path; the actual GPU metric kernels and their tolerance validation land in
   Part 2. GPR (`gamma_index_torch`) already runs on the resolved device and
   stays as-is.
4. **Package name:** `src/evaluation/` (confirmed).
5. **H5 persistent handle:** not in Part 1. `H5Source` wraps the existing
   `H5PYGenerator` with identical per-sample semantics. The persistent-handle
   optimization is Part 2 and will live in the wrapper, leaving `H5PYGenerator`
   untouched.
6. **Golden tests:** `skipif`-gated on dataset availability, like the existing
   smoke test. Golden artifacts (reference CSVs, captured run outputs) are stored
   under `/scratch/mstryja/tmp_adota`, NOT in the repo or home directory, because
   home storage is limited. Tests read the golden path from an env var
   (default `/scratch/mstryja/tmp_adota`) so the location is configurable.
7. **Canonical MAPE:** `run_model.py`'s thresholded-MAPE formula is the correct
   one (gt-based mask, `calculate_pure_mape(prediction[mask], ground_truth[mask])`).
   The shared engine adopts it. `run_model_h5py.py`'s MAPE (swapped argument
   order, prediction-based mask) is a bug and will be corrected to the canonical
   formula. This intentionally changes the h5py MAPE column versus today, so its
   golden is captured AFTER the fix, and the change is called out explicitly at
   the h5py adoption step.
8. **The `i == 169` skip is removed.** Confirmed leftover debug for a
   problematic sample in one database. The engine evaluates every sample; no
   skip logic is carried.

## Working rule: one modification at a time

The user wants full control. Each modification (file create/edit) is proposed
with its rationale and exact intended change, then implemented only after
explicit approval. No batching of unrelated edits.

## Scope (confirmed this iteration)

Exactly seven scripts are in scope; everything else stays untouched (including
`run_model_hptc.py`, deferred to a later day).

Tier 1 (deep: adopt the shared `src/evaluation/` engine). These four reimplement
the same load -> infer -> `inverse_minmax` -> metrics -> result-dataclass -> CSV
skeleton and are the heart of the refactor:
- `run_model.py` (DirSource; gpr + beamlet_angles + thresholded MAPE)
- `run_model_h5py.py` (H5Source; rmse/mape/rde + tv/cv, no gpr)
- `training_set_analysis.py` (H5Source; BeamletResult: BP sigma_HU/tv/cv +
  rmse/mape/rde + optional gpr; per-sample skip on zero-flux / energy threshold)
- `training_set_analysis_advanced_metrics.py` (H5Source; SampleRecord: large
  metric set + rde/gpr; per-sample skip guards)

Tier 2 (light: shared infra + device management + correctness; no engine).
These are self-contained, with low duplication:
- `train_adota.py` already shares `src/training/`; align device resolution with
  `resolve_device` and adopt shared logging/run-dir only where it does not
  change training behavior. Its own `_build_config_from_yaml` already does a
  generic merge, so leave config plumbing as is.
- `rotation_performance_analysis.py` (RENAMED via `git mv` from the former
  misspelling `rotatation_...`; README + docs updated). Self-contained
  benchmark; align its `_resolve_device` with the shared `resolve_device`. No
  engine.
- `beamlet_timing_comparison.py` self-contained timing study; adopt shared
  `resolve_device` / `setup_logging` / `setup_run_directory`; optionally route
  its `get_single_record` load through `DirSource`. Timing outputs are
  non-deterministic, so it gets a smoke check (runs + produces same-structure
  outputs), not a numeric golden.

Golden numeric tests apply to the four Tier-1 scripts (deterministic metric
columns). Tier-2 scripts are guarded by smoke runs:
`rotation_performance_analysis.py` and `beamlet_timing_comparison.py` already
have `--smoke` / small-sample modes; `train_adota.py` has
`tests/test_train_pipeline_smoke.py`.

Per-sample skip guards (zero-flux, energy-threshold in the two
training_set_analysis scripts) are real behavior and are preserved exactly. The
only intentional removals/changes are the two already agreed: the `i == 169`
skip in `run_model.py` and the h5py MAPE argument-order fix.

## What is preserved vs. intentionally changed

The two pilots differ in structure; the engine takes explicit per-metric
callables and per-script result factories rather than assuming one shape.

Preserved exactly (golden byte-for-byte):
- `run_model.py` carries `gpr` + `beamlet_angles` + thresholded MAPE; its CSV,
  summary, and figures stay identical.
- `run_model_h5py.py` carries `tv` + `cv` and no GPR; that structure stays.
- The canonical thresholded-MAPE formula from `run_model.py`
  (`calculate_pure_mape(prediction[mask], ground_truth[mask])`, gt-based mask).

Intentionally changed (decided with the user, not preserved):
- `run_model_h5py.py` MAPE is corrected from the swapped-argument,
  prediction-masked form to the canonical formula. Its golden is captured AFTER
  this fix; the change is called out at the adoption step.
- The `run_model.py:285` `if i == 169: continue` skip is removed. The engine
  evaluates every sample; no skip logic exists. (Because this changes which
  samples appear in `run_model.py` output, its golden is captured AFTER removal.)

---

## Step 0 — Characterization (golden) tests: the safety net

Goal: pin today's exact output before touching any pipeline code.

0.1 Add `tests/golden/` with a `skipif` guard mirroring the smoke test
(`tests/test_train_pipeline_smoke.py`): skip when the dataset path / checkpoint
is absent.

0.2 Write `tests/golden/test_run_model_golden.py` and
`tests/golden/test_run_model_h5py_golden.py`. Each:
- runs the **current, unmodified** script entrypoint (call `main` via typer's
  runner, or import and call `evaluate_samples`) on a small fixed slice
  (e.g. first N record ids / a tiny dir fixture) with a fixed checkpoint,
- captures the produced per-sample CSV rows and the summary rows,
- writes them to a committed golden file (`tests/golden/data/*.csv`) on first
  run, and on subsequent runs asserts equality.

0.3 Equality policy (per decision 1, timing-aware):
- Header line: exact string match (locks column order, names, precision).
- Non-numeric columns (`sample_id`, `label`, ...): exact match.
- Metric columns (rmse, mape, rde, gpr, tv, cv, sigma_hu, ...): numeric match
  within tolerance (default rel/abs `1e-6`; loosen per-metric only if GPU
  nondeterminism demands it, and record which).
- Timing columns (`calc_time_s`, `extract_time_s`) and their summary rows:
  ignored.
- The harness sets torch deterministic flags and a fixed device to minimize
  run-to-run noise.

0.4 Run, commit goldens. **Gate:** both tests green against current code.

Verification: `pytest tests/golden -v` (with data present) green; skips cleanly
without data.

---

## Step 1 — `src/evaluation/` scaffold + CLI merge helper

Goal: create the package and the one genuinely missing shared helper. Do not
move pipeline logic yet.

1.1 Create `src/evaluation/__init__.py`.

1.2 Create `src/evaluation/cli.py` with:
- `resolve_device(device_index: int | None = None) -> torch.device`: wraps the
  existing `get_device` semantics, adds CPU fallback when CUDA is unavailable or
  the requested index is out of range (logs the fallback). This is the single
  device entrypoint going forward.
- `merge_config(config_path, cli_overrides: dict, defaults: dict) -> dict`:
  generic `cli_value if not None else yaml.get(key, default)` merge that
  replaces the ~20-line manual per-field blocks in each `main`. Returns a plain
  dict; each script maps it into its existing `EvaluationConfig`.
- Re-export `setup_logging`, `setup_run_directory`, `load_yaml_config` from
  `src.adota.config` so scripts have one import site. (No reimplementation.)

1.3 Unit test `tests/test_evaluation_cli.py`:
- `merge_config` precedence (CLI > YAML > default), missing keys, None handling.
- `resolve_device`: CPU fallback path (monkeypatch `torch.cuda.is_available`),
  index clamping.

**Gate:** new unit tests green. No script changed yet; goldens still green
(unchanged).

---

## Step 2 — `src/evaluation/sources.py` (behavior-preserving wrappers)

Goal: one uniform iteration interface over both data sources, with **identical
per-sample semantics** to today. No persistent handle, no batching.

2.1 Define a `SampleSource` protocol yielding
`(sample_id, x, energy, y, extra)` where `extra` carries source-specific fields
(e.g. `beamlet_angles` for dir, `None` for H5). Shapes/dtypes/normalization
exactly as the current loaders return.

2.2 `DirSource`: thin wrapper around `get_single_record(...)` with the same
args currently passed in `run_model.evaluate_single_sample`
(`scale`, `normalize_flux`, `downsampling_method`, `beamlet_angle=True`).

2.3 `H5Source`: wraps an existing `H5PYGenerator` instance, iterating by index
exactly as `run_model_h5py.evaluate_samples` does (`dataset[i]`, paired with
`record_ids[i]`). Per-sample `h5py.File` open stays as-is (Part 2 changes this).

2.4 Unit test `tests/test_evaluation_sources.py`: assert each source yields
tensors equal to a direct call of the underlying loader for a few ids/indices
(skipif-gated on data, or on tiny synthetic fixtures where feasible).

**Gate:** source unit tests green. Goldens still green (scripts unchanged).

---

## Step 3 — `src/evaluation/engine.py` (faithful extraction, batch=1)

Goal: extract the shared `evaluate_single_sample` / `evaluate_samples` skeleton
into one engine that **reproduces both pilots exactly**, parameterized by source
and per-metric callables. Still single-sample, still NumPy classic metrics.

3.1 `evaluate(model, source, *, device, metrics, result_factory, show_progress,
skip_indices=()) -> list`:
- `with torch.no_grad():` (kept as-is in Part 1; `inference_mode` is Part 2),
- per sample: move tensors to `device`, `model(x.unsqueeze(0), e.unsqueeze(0))`,
  `inverse_minmax` on CPU NumPy exactly as today, then call the caller-supplied
  `metrics` to produce the metric dict, then `result_factory(...)` to build the
  schema dataclass,
- cache `prediction/ground_truth/input_data` on CPU exactly as today,
- `skip_indices` makes the 169 quirk explicit if the user chooses to keep it.

3.2 Define the two metric callables (capturing the divergent formulas from the
invariant section) in the engine or alongside each script:
- `run_model_metrics`: thresholded MAPE (gt-mask, pred/gt arg order) + RMSE +
  RDE + GPR via `gamma_index_torch`.
- `run_model_h5py_metrics`: single MAPE (pred-mask, swapped arg order) + RMSE +
  RDE + TV/CV from the CT channel.

3.3 The device-metrics option: `metrics` callables accept the resolved `device`;
in Part 1 they ignore it for the classic NumPy metrics (default path) and only
GPR uses it (as today). Leave a clearly-commented seam for the Part 2 GPU path.

**Gate:** new engine has unit coverage on a synthetic record asserting it
returns the same dataclass fields as a direct call to the old per-sample
function (import the old function in the test before it is deleted).

---

## Step 4 — Adopt the engine in the two pilots

Goal: replace the in-script `evaluate_single_sample` / `evaluate_samples` with
calls into `src/evaluation/`, keeping each script's typer signature, YAML, CSV,
and summary identical.

4.1 `run_model.py`: replace its eval functions with `DirSource` + `evaluate(...,
metrics=run_model_metrics, result_factory=EvaluationResult, skip_indices={169}
if preserving)`. Swap its manual config merge for `merge_config`. Leave figures,
anatomical-site summary, CSV writer untouched in this step.

4.2 `run_model_h5py.py`: same with `H5Source` +
`metrics=run_model_h5py_metrics`, `result_factory=H5EvaluationResult`.

4.3 **Gate (the core proof):** rerun `tests/golden` for both scripts. Output
CSVs must be byte-for-byte identical to Step 0 goldens. Review each diff (expect
none). If any byte differs, the extraction is not faithful: fix before
proceeding.

---

## Step 5 — Consolidate the output layer (no format drift)

Goal: collapse the duplicated `save_results_csv` / `print_summary` into
`src/evaluation/outputs.py`, **driven by explicit per-script fieldname lists** so
formatting is preserved exactly.

5.1 `outputs.save_results_csv(results, output_path, fieldnames, row_fn,
summary_stats=...)`: the writer is generic; the **fieldname list and the
per-field formatting (`f"{...:.9f}"` etc.) are passed in** from each script,
copied verbatim from the current writers. Same sort key, same blank-row +
mean/std/min/max trailer.

5.2 `outputs.print_summary(...)`: shared, parameterized by the fields each script
prints.

5.3 Point both pilots at `outputs.*`, passing their existing fieldname lists.

5.4 **Gate:** `tests/golden` byte-identical again. This is the step most likely
to drift; the golden CSVs are the guard.

---

## Step 6 — Part 1 wrap-up

6.1 Remove now-dead in-script copies (the extracted eval/CSV/summary functions in
the two pilots only). Do not touch the other analysis scripts yet (that is Part 1
continuation / a later batch) and do not touch out-of-scope scripts.

6.2 Changelog entry + version bump per repo convention.

6.3 Final gate: full `pytest` green (unit + golden, with data present; clean
skips without). Manually run both pilots end-to-end on the real slice once and
eyeball the run dir (CSV, log, figures) against a pre-refactor run.

---

## Out of scope for Part 1

- All performance items P1 to P5 (persistent H5 handle, batching, on-GPU classic
  metrics, `inference_mode`, GPR throughput). Deferred to Part 2.
- The other analysis scripts (`analysis_texture_with_inference`,
  `training_set_analysis`, `*_advanced_metrics`, `threshold_sweep`,
  `*_vlm_based_quantification`, `bragg_peak_estimation`, `multi_radius_analysis`):
  adopted one-at-a-time **after** the two pilots prove the engine, each against
  its own golden CSV. Can be a Part 1b batch once the pattern is trusted.
- `run_model_hptc.py`, `inference_worker.py`,
  `rotation_performance_analysis.py`, `ct_texture_analysis.py`: untouched
  (infra-only at most).

## Definition of done (Part 1)

- `src/evaluation/{__init__,cli,sources,engine,outputs}.py` exist and are used by
  `run_model.py` and `run_model_h5py.py`.
- Both scripts' CSV outputs are byte-for-byte identical to the Step 0 goldens.
- `resolve_device` is the single device entrypoint and correctly falls back to
  CPU; tensor placement is consistent across both pilots.
- Manual per-field config merge is gone from both pilots (uses `merge_config`).
- Full test suite green with data, clean skips without.
- The `i == 169` quirk has been surfaced to the user with a keep/remove decision
  recorded.
