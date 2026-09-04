# gamma_benchmark.py

> **Reproducibility / portability.** The paths in the examples below are the
> original development environment's locations -- replace them with your own. The
> plan corpus is found through `$ADOTA_GAMMA_CORPUS` (default
> `/scratch/mstryja/opentps_plans`), and every output path is a CLI argument, so
> nothing is hard-coded to a specific machine. Keep the gamma maps off your home
> directory: they are ~400 MB each.

---

Validates and benchmarks the GPU gamma backend
([`src/metrics/gamma_torch/`](../../src/metrics/gamma_torch/)) against the
`pymedphys.gamma` path it is meant to replace, over a corpus of OpenTPS plan
directories that each carry a `gamma_metrics.json` from a previous CPU run.

Unlike the pipeline scripts this one takes **no YAML config**. It has no per-run
knobs beyond the plan list and the device, and the gamma recipe deliberately
comes from each plan's own `gamma_metrics.json` rather than from
`DEFAULT_GAMMA_PARAMS`: those defaults have drifted (`interp_fraction` is 10
there and was 5 in the recorded runs), and the JSON is the record of what
actually ran.

## The deviation ladder

A single before/after comparison cannot say *which* change moved a pass rate, so
each rung is measured against the one above it:

| Rung | What it is | Isolates |
|---|---|---|
| 0 | the recorded `gamma_metrics.json` (pymedphys 0.40, Python 3.9) | n/a |
| 1 | a CPU re-run on this machine at the current pymedphys | the interpolator |
| 2 | `gamma_torch` on torch-CPU, float64 | the implementation |
| 3 | `gamma_torch` on one GPU, float64 | the device |
| 4 | `gamma_torch` on one GPU, float32 | precision |

Acceptance, on `pass_rate_pct` across every (plan x criterion) pair:

* rung 2 vs rung 1: max abs delta <= 0.01 pp (the correctness gate)
* rung 3 vs rung 2: max abs delta <= 0.001 pp
* rung 4 vs rung 2: max abs delta <= 0.1 pp
* rung 1 vs rung 0 is **reported, not gated**: that shift belongs to pymedphys.

Rung 3 vs rung 1 is reported as well. Rungs 2 and 3 run the same code at the same
precision, so where rung 2 was not measured it carries the same evidence as the
rung-2 gate, and is held to the same 0.01 pp.

### Usage

```bash
# Rung 1 -- the CPU baseline. Persist the gamma maps only for the plans you
# intend to compare voxel by voxel; they are ~400 MB per criterion.
uv run python scripts/gamma_benchmark.py cpu \
    --plans LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4 \
    --out /scratch/mstryja/gamma_gpu/rung1.json \
    --maps-dir /scratch/mstryja/gamma_gpu/maps

# Rungs 2-4 -- the torch backend. The rung label is inferred from device/dtype.
uv run python scripts/gamma_benchmark.py torch --device cpu    --dtype float64 \
    --out /scratch/mstryja/gamma_gpu/rung2.json
uv run python scripts/gamma_benchmark.py torch --device cuda:0 --dtype float64 \
    --out /scratch/mstryja/gamma_gpu/rung3.json
uv run python scripts/gamma_benchmark.py torch --device cuda:0 --dtype float32 \
    --out /scratch/mstryja/gamma_gpu/rung4.json

# The tables. Repeat --rung once per JSON; several JSONs may share a rung label
# (a long baseline is usually run as a few batches) and their plans are merged.
uv run python scripts/gamma_benchmark.py report \
    --rung /scratch/mstryja/gamma_gpu/rung1.json \
    --rung /scratch/mstryja/gamma_gpu/rung3.json \
    --out-dir docs/gamma_gpu
```

### Options

| Option | Sub-command | Meaning |
|---|---|---|
| `--plans` | `cpu`, `torch` | Comma-separated plan directory names. Defaults to the eight-plan benchmark corpus. |
| `--out` | `cpu`, `torch` | Where to write the rung JSON. |
| `--maps-dir` | `cpu`, `torch` | Persist every gamma map there as a float32 `.npy`, for the voxel-level table. Omit unless you need it. |
| `--device` | `torch` | `cpu` or `cuda:<index>`. One GPU only. |
| `--dtype` | `torch` | `float32` or `float64`. |
| `--rung` | `cpu`, `torch` | Override the rung label written into the JSON. |
| `--rung` | `report` | A rung JSON to include; repeat per rung. |
| `--out-dir` | `report` | Directory for `gamma_gpu_results.json` and `.md`. |

### Outputs

* **`<out>.json`** per rung -- grid size, the recipe used, the recorded rung-0 pass
  rates, and per criterion the pass rate, wall time, evaluated-voxel count,
  interpolated-sample count, peak GPU memory and (optionally) the gamma-map path.
* **`gamma_gpu_results.md` / `.json`** from `report` -- the deviation table, the
  acceptance verdicts, the performance table with totals, and the voxel-level
  table for whichever plans have persisted maps on both sides.

### Requirements

* The plan corpus at `$ADOTA_GAMMA_CORPUS`. Each plan directory needs `CT.mhd`,
  `PlanPencil.txt`, `config.txt`, `bdl.txt`, `Dose.mhd` (MCsquare), and
  `Dose_ADoTA.mhd` plus `gamma_metrics.json` from a previous pipeline run.
* A CUDA device for rungs 3 and 4. Peak device memory is 2.1 to 2.8 GiB for the
  67.5 M to 100.5 M-voxel plans in this corpus, in float64; lower `tile_elements`
  if that does not fit.
* Disk for the maps: a 500 x 500 x 402 float32 map is 402 MB, so five criteria on
  two plans is about 3 GB.

### Related

* [`src/metrics/gamma_torch/`](../../src/metrics/gamma_torch/) -- the kernel.
  Apache-2.0 and free of adota imports, because it is written for contribution
  back to PyMedPhys.
* `tests/test_gamma_torch.py` -- synthetic parity against `pymedphys.gamma`.
* `tests/test_gamma_torch_corpus.py` -- the same at plan scale; marked
  `integration`, `slow`, `gpu`, and skips when the corpus is absent.
