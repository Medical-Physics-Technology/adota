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

`gamma` runs after either mode (it reuses `Dose_ADoTA.mhd`, so `--stages gamma`
works standalone).

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

## Stage reference (`stages:`)

Run any comma-separated subset, in order:

| Stage | Does | Writes (in `plan_dir`) |
|---|---|---|
| `extract` | Rotate the CT around each field's isocenter; per spot, crop the BEV CT and build the flux projection (on the 1 mm or 2 mm grid per `grid_factor`) | `adota_beamlets/{id}_ct.npy`, `_flux.npy`, `_sim_res.json` |
| `infer` | Batched ADoTA inference over the extracted beamlets | `adota_beamlets/{id}_ds_pred.npy` |
| `accumulate` | Deposit predicted beamlets back onto the full CT grid (de-rotating each field); auto-generates the dose-comparison + DVH figures | `Dose_ADoTA.mhd`, `dose_comparison.*`, `dvh_comparison.*`, `dvh_metrics.json` |
| `stream` | **Fused, disk-free** alternative to `extract,infer,accumulate`: crop → flux → infer → deposit per field in one pass. Same `Dose_ADoTA.mhd` + figures, no per-beamlet files | `Dose_ADoTA.mhd`, `dose_comparison.*`, `dvh_comparison.*` |
| `gamma` | Plan gamma pass rate per criterion + MAPE / RMSE / RDE + gamma-map figure (reuses `Dose_ADoTA.mhd`, so it can run standalone) | `gamma_comparison.*`, `gamma_metrics.json` |

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
| `stages` | `extract` | Comma-separated stage list (see table above) |
| `n_spots` / `beams` | `null` | Subset controls for cheap runs (first N spots / specific field indices) |
| `overwrite` | `false` | Allow (re)writing into a non-empty `adota_beamlets/` |
| `no_overlays` | `true` | Skip the per-field overlay PNGs |
| `flux_on_gpu` | `false` | Compute the flux projection on the GPU (`flux_projection_gpu`); numerically identical to NumPy, a speed option |
| `extraction_parallel` | `false` | `false` → serial `run_extraction`; `true` → thread-pooled `run_extraction_pooled` (bit-identical output) |
| `extraction_workers` | `0` | Thread count when parallel (`0` = auto, `min(32, os.cpu_count())`) |
| `batch_size` | `56` | Spots per GPU forward pass (inference / stream) |
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
