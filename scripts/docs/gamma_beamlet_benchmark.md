# gamma_beamlet_benchmark.py

> **Reproducibility / portability.** The paths in the examples below are the
> original development environment's locations -- replace them with your own.
> Every input and output is a CLI argument, so nothing is hard-coded to a
> specific machine. Keep the cached pair file off your home directory: 32
> beamlets are about 20 MB, and a larger draw grows linearly.

---

Beamlet-scale counterpart to [`gamma_benchmark.py`](gamma_benchmark.md). That
script measures the gamma backends on 67-100 million-voxel patient plans; this
one measures the same four rungs on the 160 x 30 x 30 beamlet grid the model is
trained on, which is the size that decides whether the gamma pass rate is
affordable as a training signal.

The two studies are not interchangeable. A beamlet is three orders of magnitude
smaller than a plan, so the fixed costs -- CUDA kernel launches, host-device
transfers, the Python-side convergence loop -- are a large fraction of the
total, and the plan-scale speed-ups do not carry over. The GPU rungs are a
median 2 to 7 times faster than `pymedphys.gamma` on a beamlet against 18 to 33
times on a plan, and float32 buys nothing over float64 at this size because the
kernels are no longer the bottleneck.

## The rungs

Identical to the plan-level ladder, minus rung 0, which is a historical
recording with no beamlet-scale analogue:

| Rung | What it is | Isolates |
|---|---|---|
| 1 | `pymedphys.gamma` on the host | the baseline |
| 2 | `gamma_torch` on torch-CPU, float64 | the implementation |
| 3 | `gamma_torch` on one GPU, float64 | the device |
| 4 | `gamma_torch` on one GPU, float32 | precision |

`--device cuda:N` moves rungs 3 and 4 only. Rung 2 is the torch-CPU rung by
definition and stays on the CPU whatever `--device` says; if it did not, rungs 2
and 3 would be the same measurement and the ladder would prove nothing.

## The two entry points

`--paths` selects which of them is timed, and both answer a different question.

`array`
: NumPy in, NumPy out, through `src.metrics.gamma_pass_rate.gamma_index`. The
  offline analysis path, and the one comparable with the plan-level table.

`tensor`
: Volumes already resident on the GPU, through `gamma_index_torch`. The training
  path: under the pymedphys backend it must copy two full volumes to the host
  and back, and under the torch backend it does not. That difference is part of
  what is being measured, which is why the two paths are timed separately rather
  than averaged.

## Usage

The three sub-commands are meant to be run in order; each one's output is the
next one's input, so a sweep can be repeated without re-running inference and a
report can be rebuilt without re-running a sweep.

```bash
# 1. Cache the (Monte Carlo, ADoTA) dose pairs. Needs the dataset, a checkpoint
#    and a GPU; nothing after this step needs the model again.
uv run python scripts/gamma_beamlet_benchmark.py pairs \
    --h5 /scratch/mstryja/DoTA_dataset_v2/testset_downsampled_v0_all_SingleGaussian.h5 \
    --run-dir /scratch/mstryja/adota_runs/train_20260519_231135_baseline \
    --count 32 --device cuda:0 \
    --out /scratch/mstryja/gamma_beamlet/pairs.npz

# 2. Time the rungs. The GPU rungs and the baseline over the full draw:
uv run python scripts/gamma_beamlet_benchmark.py sweep \
    --pairs /scratch/mstryja/gamma_beamlet/pairs.npz \
    --device cuda:1 --rungs rung1,rung3,rung4 \
    --criteria 1/1,2/2,3/3 --interp-fraction 10 --repeats 3 \
    --paths array,tensor \
    --out /scratch/mstryja/gamma_beamlet/sweep_main.json

#    The torch-CPU rung on a smaller draw: it only has to separate the
#    implementation from the device, and eight beamlets settle that.
uv run python scripts/gamma_beamlet_benchmark.py sweep \
    --pairs /scratch/mstryja/gamma_beamlet/pairs.npz \
    --device cuda:1 --rungs rung2 --criteria 1/1,2/2,3/3 \
    --repeats 1 --paths array --limit 8 \
    --out /scratch/mstryja/gamma_beamlet/sweep_cpu.json

# 3. Reduce one or more sweeps into the parity, timing and throughput tables.
uv run python scripts/gamma_beamlet_benchmark.py report \
    --sweep /scratch/mstryja/gamma_beamlet/sweep_main.json \
    --sweep /scratch/mstryja/gamma_beamlet/sweep_cpu.json \
    --out-dir docs/gamma_beamlet
```

## The maps subcommand and provenance

`maps` computes the raw gamma map of every (beamlet, criterion, rung) and
compares each rung against rung 1 (and rung 3 against rung 2) in memory, in the
dtype the backend produced, before the pass-rate reduction zeroes the NaNs.
Maps are saved in native precision under `<out dir>/maps/`. `--provenance` on
any subcommand writes a `manifest.json` and raw system dumps beside `--out`.
Both are described in [gamma_evidence.md](gamma_evidence.md).

## What is measured, and how

* The evaluation dose comes from a **real prediction**, not from a perturbed
  copy of the reference. The cost of the gamma search depends on how far it has
  to travel before it converges, so a fabricated disagreement would give a
  timing that is not the timing of the metric in use.
* Every measurement is preceded by **one untimed call**, which absorbs CUDA
  context creation and the first kernel launch, then repeated `--repeats` times.
  The **minimum** of the repeats is reported, as the value least contaminated by
  other load on a shared machine.
* Every GPU timing is taken after an explicit `torch.cuda.synchronize`, so it
  measures completed work rather than enqueued work.
* The per-criterion statistic over beamlets is the **median** of the per-beamlet
  minima. A mean would be pulled by the few beamlets whose search converges
  late, and the question is what a typical beamlet costs.
* `gamma_index` zeroes its inputs below the cutoff in place, so the cached
  arrays are never handed to it; a fresh de-normalised copy is made per call and
  that copy is inside the timed region for the array path, exactly as it is in
  the scripts that call the metric.

## Outputs

`report` writes three files under `--out-dir`, sharing a basename:

| File | Contents |
|---|---|
| `gamma_beamlet_results.json` | every table plus the raw per-measurement rows |
| `gamma_beamlet_results.md` | the same tables, rendered |
| `gamma_beamlet_results.csv` | one row per measurement, for re-analysis |

The tables are the deviation of each rung from pymedphys (per criterion, over
beamlets), the time per beamlet with the speed-up over pymedphys, and the wall
time of one gamma pass over a validation pool of a given size -- the last being
the number the training loop is budgeted against.

## Building the technical report

`scripts/analysis/report_gamma_acceleration.py` reads these sweeps together with
the plan-level `docs/gamma_gpu/gamma_gpu_results.json` and writes the LaTeX
tables, the figure and the CSVs of
`reports/technical-reports/gamma-pass-rate/`. See that directory's `Makefile`.
