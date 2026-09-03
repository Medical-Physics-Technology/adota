# `run_plan_opentps.py` — end-to-end plan-level dose pipeline

The flagship reproducibility entry point. Given a **plan directory** it turns the
per-beamlet ADoTA model into a full **treatment-plan dose** and validates it against
a Monte-Carlo reference (dose comparison, DVH, gamma).

It is **config-driven**: point it at a YAML config and override individual fields
from the CLI (CLI > YAML > built-in defaults). The model code is untouched — this
is an orchestration wrapper around `src/beamlets/`.

- Script: [`scripts/run_plan_opentps.py`](../run_plan_opentps.py)
- Config: [`scripts/config_run_plan_opentps.yaml`](../config_run_plan_opentps.yaml)

---

## What you need before running

The pipeline works on **your own machine and your own directory layout** — nothing
is hard-coded. You point it at three things via the config (or CLI):

1. **The synced Python environment.** `uv sync` creates `.venv` from the pinned
   lockfile. Run everything with `uv run ...`.

2. **A model directory** under `models/<model_name>/` containing the weights
   (e.g. `best_model.pth`) and `hyperparams.json`. Set `model_name` to that folder.

3. **A plan directory** — a folder *you choose, anywhere on your system* that holds
   one plan's data. Set `plan_dir` to it. It must currently contain an **OpenTPS**
   export:

   ```
   <your-plan-dir>/
   ├── CT.mhd / .raw          # the patient CT
   ├── PlanPencil.txt         # the spot/energy/geometry plan
   ├── bdl.txt                # beam data library (machine model); plan-local
   ├── target.mhd, OAR_*.mhd  # structure masks (for DVH)
   └── Dose.mhd / .raw        # the MCsquare Monte-Carlo reference dose (for validation)
   ```

   > **Where does this come from / roadmap.** Today the pipeline reads an **OpenTPS
   > plan directory** in the layout above. A **direct DICOM reader** (RT-Plan +
   > RT-Struct + CT series → this layout) is planned, so you will be able to point
   > `plan_dir` straight at a DICOM export. Until then, produce the OpenTPS export
   > first and point `plan_dir` at it.

All outputs are written **into your `plan_dir`** (see [Outputs](#outputs)) — so the
result location follows wherever you put the plan; there is no assumed scratch or
home path. Large intermediate beamlets can optionally be redirected, but by default
nothing is written outside `plan_dir`.

---

## Quickstart (reproduce one plan)

```bash
# 1. Sync the environment (creates .venv from the pinned lockfile)
uv sync

# 2. Point the run at YOUR locations. Either edit the three keys in
#    scripts/config_run_plan_opentps.yaml (plan_dir, model_name, device_index),
#    or pass them on the CLI as below. Use a shell variable for clarity:
PLAN_DIR=/path/to/your/plan          # the folder described above

uv run python scripts/run_plan_opentps.py \
    --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" \
    --model-name <your_model_dir> \
    --device-index 0 \
    --stages stream,gamma --overwrite
```

(`--device-index -1` runs on CPU.) When it finishes you should see, **inside your
`plan_dir`**:

```
<your-plan-dir>/
├── Dose_ADoTA.mhd / .raw          # the accumulated ADoTA plan dose (same grid as Dose.mhd)
├── dose_comparison.{png,pdf,svg}  # ADoTA vs MC slices + integral depth dose
├── dvh_comparison.{png,pdf,svg}   # DVH overlay  (+ dvh_metrics.json)
├── gamma_comparison.{png,pdf,svg} # gamma maps    (+ gamma_metrics.json: GPR per criterion)
└── pipeline_timing.json           # per-stage timing report
```

`gamma_metrics.json` holds the **gamma pass rate vs the Monte-Carlo reference** per
criterion — the headline number that tells you the plan dose is correct.

---

## Two execution modes

The pipeline computes the same dose two ways. Pick with `--stages`:

| Mode | Stages | Writes per-beamlet files? | Speed | Use when |
|---|---|---|---|---|
| **Staged** | `extract,infer,accumulate` | Yes (`adota_beamlets/`) | baseline | you want to inspect/keep per-spot crops, flux, and predictions, or restart a single stage |
| **Stream** | `stream` | **No** (in-memory, bounded) | ~2× faster | the normal path — you just want the plan dose |

The **stream** path fuses crop → flux → infer → deposit per field in one pass and
never touches disk for beamlets. It reuses the *same* crop/flux, the *same*
`prepare_input_from_arrays` / `postprocess_prediction`, and the *same* `deposit_crop`
as the staged path, so the accumulated dose is **numerically identical** (verified
by `tests/beamlets/test_streaming.py` and, on a real plan with the production model,
bit-identical to the staged dose).

`gamma` and `dvh` run after either mode and both reuse `Dose_ADoTA.mhd`, so
`--stages gamma` and `--stages dvh` work standalone (no re-run, no model needed).

---

## Field-level 2 mm resampling (`grid_factor`)

By default every beamlet is cropped on the native **1 mm** grid `(60,60,320)` and
trilinearly resized to the model grid `(160,30,30)` — then the prediction is
up-sampled back. `grid_factor: 2` instead rotates/crops/projects-flux/deposits on a
**2 mm field grid** so a crop is *already* the model grid: the per-beamlet
down/up-sample is replaced by a single per-field rotate / de-rotate.

- **Optional and off by default.** `grid_factor: 1` is byte-identical to the
  historical pipeline (guarded by the test suite).
- **Available in both modes** — stream *and* staged (`extract,infer,accumulate`).
  The two are numerically identical at `grid_factor=2` as well.
- **Faster.** On a representative lung field (2675 spots) streaming dropped from
  **~48 s → ~18 s (≈2.6×)**; the per-beamlet resize rows (crop / flux / prep /
  post / deposit) all shrink and the model forward becomes the floor.
- **Dose preserved.** Against the 1 mm path on the same plan: voxel correlation
  **>0.999**, dose-weighted centroid shift **<0.1 mm**, integral ratio within
  ~0.3 %, and the gamma-vs-MC pass rate does **not** regress (validate per plan via
  the A/B harness below before adopting it for a new cohort).

Enable it:

```bash
# config: set `grid_factor: 2`   — or override on the CLI:
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" --stages stream,gamma --grid-factor 2 --overwrite
```

The mode is surfaced in `pipeline_timing.json` and the printed table as a distinct
label (`Streaming (fused, 1mm)` vs `Streaming (fused, 2mm field)`), so the two are
directly comparable.

> **Reproducibility note.** When comparing a 2 mm dose to a 1 mm dose, compare on
> the **de-rotated patient grid** (e.g. via gamma vs MC, or `compare_plan_dose.py`),
> never in per-beamlet array-index space — the 1 mm and 2 mm crops round their crop
> windows to different sub-voxel positions, which manufactures a phantom shift in
> index space that vanishes once each beamlet sits at its true physical position.

---

## Batched host↔device staging (`flux_batched` / `batched_prep`)

At `grid_factor: 2` the per-beamlet resize is already gone, so what was left
dominating the stream stage was **transfer overhead**, not compute: the pipeline
moved two arrays per beamlet across the PCIe bus one at a time, and built the flux
one spot at a time. Ported from the optimized-GPU reinterpretation benchmark
([`reinterp_gpu_benchmark.py`](../reinterp_gpu_benchmark.py)), two options move each
batch as a single contiguous block instead:

- **`batched_prep`** stages the batch's CT crops into one contiguous pinned block
  and copies them host→device **once** instead of `batch_size` times, then
  normalizes / permutes / resizes over the batch
  (`prepare_inputs_from_arrays_batched`).
- **`flux_batched`** builds the whole batch's flux in one
  `flux_projection_gpu_batched` call and — with `batched_prep` — hands the result
  straight to the model input as a device tensor, so the flux never makes a
  device→host→device round trip.

Measured on `LUNG1-062_Publication_Plan_1` (285 spots, 2 mm, fp16, batch 120):

| Step | per-spot | batched | |
|---|---:|---:|---|
| flux projection | 1.502 s | 0.019 s | 79× |
| input prep (H→D) | 4.983 s | 0.097 s | 51× |
| **stream compute** | **10.77 s** | **2.86 s** | **3.8×** |
| **wall clock** | **13.33 s** | **5.19 s** | **2.6×** |

`batch_size` was swept at 56 / 120 / 240 / 360 → 3.28 / 2.86 / 3.28 / 2.97 s of
compute: **120 is the sweet spot**, and bigger is not better (the forward stops
getting cheaper and the pinned staging buffers just grow).

### What this does to the dose

The pipeline is bit-reproducible (running the same config twice gives *zero*
differing voxels), so these differences are real signal, not run-to-run noise:

| Variant vs the per-spot path | fp32 | fp16 (production) |
|---|---:|---:|
| `batched_prep` alone | **0** (bit-identical) | **0** (bit-identical) |
| `flux_batched` (float64) | 1.5e-5 of peak | 1.8e-4 of peak |

`batched_prep` only changes *how* the batch reaches the GPU, so it is bit-identical
end to end. `flux_batched` is a **reassociation** of the same float64 math (batched
matmul instead of per-spot), which lands within one float32 ulp on the normalized
flux channel the model consumes; the fp16 forward then amplifies that to 1.8e-4
(0.018 %) of the peak dose — 16× below the ~0.3 % the fp16 forward itself already
costs, with **gamma pass rates vs MCsquare unchanged**. `flux_batched_dtype:
float32` is ~2× faster on the flux alone but an order of magnitude looser; it is
not used for publication runs.

Guarded by `tests/beamlets/test_flux_gpu.py` (batched flux vs the per-spot NumPy
and GPU paths), `tests/loaders/test_dir_based_batched_prep.py` (batched prep is
bit-identical to the per-spot loop, CPU and CUDA) and
`tests/beamlets/test_streaming.py` (whole-pipeline equivalence at both grid
factors).

## Dose-comparison figure style (`dose_render`)

The ADoTA-vs-reference **`dose_comparison.*`** figure (written by the `accumulate`
and `stream` stages) can be rendered two ways, selected by `dose_render`:

| `dose_render` | Dose columns | Look |
|---|---|---|
| **`image`** (default) | filled `jet` overlay on the CT (`ax.imshow`), alpha-masked above a low threshold | smooth dose wash |
| **`contour`** | filled **isodose contours** on the CT (`ax.contourf`) at **10 / 30 / 50 / 70 / 90 / 95 / 100 %** of the shared dose peak, with thin **labeled isodose lines** on top | the clinical isodose view |

In **both** modes the difference column (ADoTA − reference) stays an `imshow`
heatmap, the shared colorbar reflects the dose scale (the isodose levels in
`contour` mode), and the three orthogonal views + integrated-depth-dose profile are
unchanged. The `.png` / `.pdf` / `.svg` triple and the `dose_comparison_caption.txt`
sidecar are produced identically (the caption text adapts to the chosen style). Only
the dose-comparison figure is affected — DVH and gamma figures are unchanged.

Isodose contours are common in clinical review, so `contour` gives a familiar
read of target coverage and high-/low-dose spill. Enable it:

```bash
# config: set `dose_render: contour`   — or override on the CLI:
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" --stages stream --dose-render contour --overwrite
```

`dose_render` and `grid_factor` are independent and can be combined freely (e.g.
`--grid-factor 2 --dose-render contour`).

---

## Stage reference (`stages:`)

Run any comma-separated subset, in order:

| Stage | Does | Writes (in `plan_dir`) |
|---|---|---|
| `extract` | Rotate the CT around each field's isocenter; per spot, crop the BEV CT and build the flux projection (on the 1 mm or 2 mm grid per `grid_factor`) | `adota_beamlets/{id}_ct.npy`, `_flux.npy`, `_sim_res.json` |
| `infer` | Batched ADoTA inference over the extracted beamlets | `adota_beamlets/{id}_ds_pred.npy` |
| `accumulate` | Deposit predicted beamlets back onto the full CT grid (de-rotating each field); auto-generates the dose-comparison + DVH figures | `Dose_ADoTA.mhd`, `dose_comparison.*`, `dvh_comparison.*`, `dvh_metrics.json` |
| `stream` | **Fused, disk-free** alternative to `extract,infer,accumulate`: crop → flux → infer → deposit per field in one pass. Same `Dose_ADoTA.mhd` + figures, no per-beamlet files | `Dose_ADoTA.mhd`, `dose_comparison.*`, `dvh_comparison.*` |
| `dvh` | **Regenerate only the DVH** figure + metrics from an existing `Dose_ADoTA.mhd` (standalone, like `gamma`; no re-run). Optionally renames structures to anatomical names via `structure_names.json` (see below). Touches nothing else | `dvh_comparison.*`, `dvh_metrics.json` |
| `gamma` | Plan gamma pass rate per criterion + MAPE / RMSE / RDE + gamma-map figure (reuses `Dose_ADoTA.mhd`, so it can run standalone) | `gamma_comparison.*`, `gamma_metrics.json` |
| `beamlets` | **Per-spot** ADoTA vs MCsquare comparison against the MC per-beamlet matrix (`beamlets_*/`); reuses the existing `adota_beamlets/` predictions (run `extract,infer` first). Standalone, like `gamma`. | `beamlet_metrics.csv`, `beamlet_metrics.json`, `beamlet_analysis.*` |

### Per-beamlet analysis (`beamlets` stage)

Compares ADoTA and MCsquare **spot by spot** (not just at plan level), to see which
pencil beams ADoTA reproduces well as a function of energy. It needs two things in
the plan dir:

- the MCsquare **per-beamlet** matrix `beamlets_<primaries>/` (from OpenTPS'
  `reconstructPlanFromPencil.py`; format in [`docs/mcsquare_beamlet_format.md`](../../docs/mcsquare_beamlet_format.md)), and
- the ADoTA **per-spot predictions** `adota_beamlets/{id}_ds_pred.npy` — so run
  `--stages extract,infer` once first (the `stream` path writes no per-spot files).

It matches spot `i` (matrix column) to ADoTA record `b{beam}_l{layer}_s{spot}`
(asserted), scales each to Gy on the same footing as the accumulated dose, and
compares **in the ADoTA BEV crop frame** (the MC beamlet is rotated+cropped into
each spot's crop). Metrics per spot (reusing `src/metrics/`): local gamma 2%/2mm &
3%/3mm, MAPE/RMSE on high-dose voxels, and the R80 range difference. Everything is
restricted to the beamlet's support; the sparse matrix is never densified. At
`1e5` primaries the MC beamlets are noisy, so `max()` is never used and the
low-dose noise floor is reported.

Outputs: `beamlet_metrics.csv` (one row per spot: indices, `energy_mev`, `mu`,
`mu_fraction`, the metrics), `beamlet_metrics.json` (aggregates: overall and
stratified above/below 150 MeV, unweighted and MU-weighted; records the
`beamlet_dir` + `primaries_per_beamlet`), `beamlet_analysis.*` (metric-vs-energy
scatter, point colour/size = MU fraction, 150 MeV reference line), and
`gpr_per_layers_<%>pct_<mm>mm_cut<cutoff>.*` — one per gamma criterion: the plain
**per-energy-layer mean** GPR vs plan energy layer (error bars = within-layer std),
with the out-of-distribution layers (≥ 150 MeV, above the thoracic training limit)
shaded and drawn as distinct markers. This is the single publication panel for the
energy-layer trend.

```bash
# generate per-spot predictions once, then analyse against the MC beamlets
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" --stages extract,infer --overwrite
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" --stages beamlets            # --beamlet-dir to pick a specific beamlets_* set
```

### Regenerating just the DVH (`dvh` stage)

`--stages dvh` re-renders `dvh_comparison.{png,pdf,svg}` and `dvh_metrics.json` from the
**already-accumulated** `Dose_ADoTA.mhd` and the MC `Dose.mhd` — no extraction,
inference, accumulation, dose-comparison or gamma. It needs no model, so it runs
without `--model-name`. Use it to refresh DVH figures (e.g. after changing structure
labels) without recomputing the plan dose.

#### Anatomical structure names (`structure_names.json`)

By default the DVH labels structures by their mask keys (`target`, `OAR_1`, `OAR_2`, …).
Drop a **`structure_names.json`** in the plan directory to relabel them to anatomical
names in both the figure legend and `dvh_metrics.json`:

```json
{ "target": "Target", "OAR_1": "Spinal-Cord", "OAR_2": "Lung-Right" }
```

Keep the target generic (`"Target"`) rather than a plan-specific GTV/Prostate name;
it stays classified as the target regardless of the label.

Only the `dvh` stage reads it (accumulate/stream are unchanged). Renaming is a pure
relabel: the DVH curves and metric **values are identical**, and the renamed target is
still classified as the target (keeps `D95`/`D98`). Unmapped keys are left as-is; with
no file present, behaviour is unchanged. A `structure_names` mapping in the YAML config
is used as a fallback when the per-plan file is absent.

Legend labels are formatted for display (`Femur_Head_L` → `Femur Head L`,
`Lungs-Total` → `Lungs (Total)`), and each structure uses a **fixed anatomy-aware
colour** so the same organ keeps the same colour across every DVH panel.

**Axis and multi-panel options (dvh stage):**

| Option (CLI / config) | Effect |
|---|---|
| `--dvh-max-dose` / `dvh_max_dose` | Fixed DVH x-axis upper limit in Gy (e.g. **70** prostate, **80** thoracic) for a shared, comparable scale. Default: a robust dose percentile, so a single outlier voxel can't stretch the axis to the raw maximum. |
| `--dvh-compact` / `dvh_compact` | Compact panel for multi-panel composition: **no title, no ADoTA/MCsquare (solid/dashed) legend**; keeps only the per-panel Structure legend + axes. Default off. |

```bash
# thoracic panel for the paper (capped 0-80 Gy, compact for tiling)
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" --stages dvh --dvh-max-dose 80 --dvh-compact --overwrite
```

---

## Configuration reference

All settings live in [`config_run_plan_opentps.yaml`](../config_run_plan_opentps.yaml).
A copy is saved next to the run for reproducibility. **Paths are examples — set them
to your own locations.**

| Key | Default | Description |
|---|---|---|
| `plan_dir` | – | Your plan directory (see [What you need](#what-you-need-before-running)) |
| `model_name` | – | Model directory under `models/` (required for `infer` / `stream`) |
| `model_fname` | `best_model.pth` | Weights filename within the model directory |
| `bdl_path` | `null` | Override the beam data library (default: plan-local `bdl.txt`) |
| `device_index` | `0` | CUDA device index (`-1` for CPU) |
| `runs_dir` | example path | Base for auxiliary run outputs — set to any directory you like (keep large outputs off your home if space-constrained) |
| `stages` | `extract` | Comma-separated stage list (see table above; e.g. `dvh`, `gamma`) |
| `structure_names` | `null` | Fallback `{mask_key: display_name}` map for the `dvh` stage when a plan has no `structure_names.json` (per-plan file takes precedence) |
| `beamlet_dir` | `null` | `beamlets` stage: MCsquare per-beamlet directory (default: newest `beamlets_*` in the plan dir) |
| `beamlet_energy_split_mev` / `beamlet_high_dose_frac` | `150` / `0.5` | `beamlets` stage: energy stratification split (MeV) and the high-dose mask fraction of the robust peak |
| `n_spots` / `beams` | `null` | Subset controls for cheap runs (first N spots / specific field indices) |
| `overwrite` | `false` | Allow (re)writing into a non-empty `adota_beamlets/` |
| `no_overlays` | `true` | Skip the per-field overlay PNGs |
| `flux_on_gpu` | `false` | Compute the flux projection on the GPU (`flux_projection_gpu`); numerically identical to NumPy, a speed option |
| `extraction_parallel` | `false` | `false` → serial `run_extraction`; `true` → thread-pooled `run_extraction_pooled` (bit-identical output) |
| `extraction_workers` | `0` | Thread count when parallel (`0` = auto, `min(32, os.cpu_count())`) |
| `batch_size` | `56` | Spots per GPU forward pass (inference / stream). With batched I/O on, `120` is the measured sweet spot; beyond it the forward stops getting cheaper and the pinned buffers just grow |
| `flux_batched` | `false` | `stream`: build the whole batch's flux in one `flux_projection_gpu_batched` call and (with `batched_prep`) keep it resident on the device instead of a device→host→device round trip |
| `batched_prep` | `false` | `stream`: stage the batch's CT crops into one contiguous pinned block and copy them host→device **once** instead of `batch_size` times, then normalize/permute/resize over the batch. **Bit-identical** end to end |
| `flux_batched_dtype` | `float64` | Compute dtype of the batched flux. `float64` is the same math as the per-spot `flux_projection_gpu`, agreeing to round-off; `float32` is ~2× faster on the flux alone but an order of magnitude looser |
| **`grid_factor`** | **`1`** | **Field-level resampling: `1` = 1 mm per-beamlet (byte-identical); `2` = 2 mm field grid (see above). Applies to stream and staged.** |
| `dose_render` | `image` | Dose-comparison figure style: `image` (filled jet overlay) or `contour` (clinical filled isodose contours at 10/30/50/70/90/95/100 % of peak + labeled lines; the difference panel stays a heatmap) |
| `dose_source` | `null` | `prediction` (model dose) or `flux` (stand-in); auto-selected when `infer` ran |
| `dose_calibration_enabled` / `dose_calibration_factor` | `false` / `1.029` | Optional multiplicative dose calibration at accumulation (off by default) |
| `gamma_criteria` | 5 criteria | List of `[dose%, distance_mm, dose_cutoff%]` for the gamma stage |
| `gamma_params` | `interp_fraction: 5, ...` | Extra pymedphys gamma parameters |
| `verbose` | `false` | DEBUG-level logging |

---

## CLI reference

CLI flags override the YAML values.

| Option | Description |
|---|---|
| `--config` | Path to the YAML config |
| `--plan-dir` | Your plan directory |
| `--model-name` / `--model-fname` | Model directory / weights filename |
| `--bdl-path` | Beam data library path |
| `--device-index` | CUDA device index (`-1` = CPU) |
| `--stages` | Comma-separated stages (e.g. `stream,gamma`) |
| `--grid-factor` | `1` (1 mm per-beamlet) or `2` (2 mm field grid) |
| `--dose-render` | `image` (filled overlay) or `contour` (clinical isodose contours) |
| `--n-spots` / `--beams` | Subset controls |
| `--overwrite` / `--no-overlays` / `--verbose` | Flags |
| `--help` | Full help |

```bash
# Staged, GPU 0, extract+infer+accumulate only
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" --stages extract,infer,accumulate --device-index 0 --overwrite

# Gamma only, reusing an already-accumulated Dose_ADoTA.mhd
uv run python scripts/run_plan_opentps.py --config scripts/config_run_plan_opentps.yaml \
    --plan-dir "$PLAN_DIR" --stages gamma
```

---

## Outputs

Everything is written **inside your `plan_dir`** — the result location follows
wherever your plan lives:

```
<your-plan-dir>/
├── adota_beamlets/            # staged only: per-spot CT crops, flux, predictions (removable)
├── Dose_ADoTA.mhd / .raw      # accumulated ADoTA plan dose (same grid as Dose.mhd)
├── dose_comparison.{png,pdf,svg}    # ADoTA vs MC (axial/coronal/sagittal + IDD)
├── dvh_comparison.{png,pdf,svg}     # DVH overlay; dvh_metrics.json (Dmean/D95/D98/...)
├── gamma_comparison.{png,pdf,svg}   # gamma maps (3 views × N criteria at isocenter)
├── gamma_metrics.json        # per-criterion GPR + MAPE/RMSE/RDE
└── pipeline_timing.json      # per-stage timing report
```

The timing report records **real wall-clock time per step** — under the extraction
thread pool the per-step rows are the union of concurrent intervals (real active
wall time), not a thread-sum. The streaming stage is labelled by mode
(`1mm` / `2mm field`), and the JSON carries `grid_factor` + `grid_mode` so the two
modes are directly comparable.

---

## Batch / reproducibility helpers

Both scripts contain a `PLANS=( ... )` array — **edit it to your plan directories**
(and adjust `CONFIG` if needed) before running.

| Script | What it does |
|---|---|
| [`run_all_plans.sh`](../run_all_plans.sh) | Runs `stream,gamma` on the 2 mm field grid (`--grid-factor 2`) over the listed plans, sequentially; per-plan logs in `run_logs/`. |
| [`run_publication_plans.sh`](../run_publication_plans.sh) | Timing run over the 8 publication plans (`stream`, 2 mm, fp16, batched I/O). Archives each plan's previous `pipeline_timing.json` first so the merge cannot carry stale stages into the measurement, then calls `summarize_publication_timing.py` for a combined table. |
| [`run_grid_factor_ab.sh`](../run_grid_factor_ab.sh) | The **A/B harness**: for each plan runs `stream,gamma` twice (`grid_factor` 1 then 2) and archives each mode's dose + `gamma_metrics.json` + `pipeline_timing.json` into `<plan>/grid_ab/{1mm,2mm}/` for a direct go/no-go comparison. |

```bash
# Run detached; one failing plan does not abort the rest
nohup bash scripts/run_all_plans.sh      > run_logs/run_all_plans.out 2>&1 &
nohup bash scripts/run_grid_factor_ab.sh > run_logs/grid_factor_ab.out 2>&1 &
```

---

## Requirements recap

- The synced environment (`uv sync`).
- A **model directory** under `models/` with the weights + `hyperparams.json`
  (needed for `infer` / `stream`).
- A **plan directory** in the OpenTPS layout above (DICOM reader planned). All
  outputs land inside it.
