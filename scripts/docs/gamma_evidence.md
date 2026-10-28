# The gamma evidence harnesses (EXP-0008)

> **Reproducibility / portability.** Every path below is the original
> development environment's; replace it with your own. Every run writes a
> `manifest.json` beside its results when `--provenance` is given: thread
> environment, CPU affinity, library and driver versions, the GPU's UUID and the
> processes on it, the git state of both repositories (with the diff saved when
> the worktree is dirty), and SHA-256 hashes of the inputs. Nothing under
> `/scratch` is committed; the results are the JSON files, the report is built
> from them.

---

Five scripts produce the evidence behind the gamma-acceleration technical
report (`reports/technical-reports/gamma-pass-rate/`). They share the four-rung
ladder of [`gamma_beamlet_benchmark.md`](gamma_beamlet_benchmark.md) and one
timing protocol: one untimed warm-up per backend and dtype, then repeated timed
calls with the device synchronised before and after each, every repetition
kept, medians and interquartile ranges reported.

Run them under one controlled environment. The file used for EXP-0008 pinned
every thread pool to the 24 physical cores and reserved one GPU by UUID:

```bash
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMBA_NUM_THREADS=24
export CUDA_VISIBLE_DEVICES=GPU-<uuid>     # inside the process it is cuda:0
```

| Experiment | Script | Question |
|---|---|---|
| A | `gamma_beamlet_benchmark.py sweep` | How do the four rungs compare on the *same* 32 beamlets? |
| B | `gamma_beamlet_benchmark.py maps`, `gamma_plan_map_agreement.py` | Do the float64 maps agree before any cast to float32? |
| C | `gamma_pool_benchmark.py` | What does a pass over 200, 948 and 2000 beamlets actually cost? |
| D | `gamma_scaling_benchmark.py` | How does runtime scale with size when everything else is fixed? |
| E | `gamma_profile_beamlet.py` | Where does a beamlet evaluation spend its time? |

## A: matched four-rung sweep

```bash
uv run python scripts/gamma_beamlet_benchmark.py sweep \
    --pairs $RUN/inputs/pairs.npz --device cuda:0 \
    --rungs rung1,rung2,rung3,rung4 --criteria 1/1,2/2,3/3 --interp-fraction 10 \
    --repeats 5 --paths array --provenance --out $RUN/A/beamlet_matched_array.json
```

The sweep loops over beamlets outermost and rotates the rung order per beamlet
(`--no-rotate` disables it), so no backend runs systematically cold or warm; the
order used is in every row. Rows carry `seconds_all`, the quartiles, the torch
search counters, peak device memory, and the fully resolved gamma
configuration. The tensor entry point is a second sweep over the rungs where it
is meaningful (`--rungs rung1,rung3,rung4 --paths tensor`).

`src.metrics.gamma_beamlet_report.paired_speedups` reduces a sweep to per-case
ratios on **matched** cases and raises `MatchedSetError` otherwise. It reports
three conventions because they answer different questions and can differ by a
factor of two on skewed data: the median of the paired ratios, the ratio of the
medians, and the ratio of the sums.

## B: unquantised map agreement

```bash
uv run python scripts/gamma_beamlet_benchmark.py maps --pairs $RUN/inputs/pairs.npz \
    --device cuda:0 --rungs rung1,rung2,rung3,rung4 --criteria 1/1,2/2,3/3 \
    --provenance --out $RUN/B_beamlet/beamlet_maps.json
uv run python scripts/gamma_plan_map_agreement.py --device cuda:0 \
    --plans LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4 \
    --criteria 1/1/10,2/2/10,3/3/10 --provenance --out $RUN/B_plan/plan_maps.json
```

Maps are compared in memory in their native dtype, before the pass-rate
reduction zeroes the NaNs, so the evaluated mask survives. Beamlet maps are
saved as `.npy` in native precision; plan maps are not saved (24 maps of up to
772 MB each), their SHA-256 digests are. Gates are predeclared in
`src.metrics.gamma_map_agreement.GATES`; measured values are reported whether
or not they pass. `bitwise_identical` is asserted only by an actual
`array_equal` in the native dtype.

## C: pools

```bash
uv run python scripts/gamma_pool_benchmark.py pools --out-dir $RUN/C --device cuda:0
uv run python scripts/gamma_pool_benchmark.py time --pool $RUN/C/pool_test200.npz \
    --device cuda:0 --passes 3 --provenance --out $RUN/C/time_test200.json
uv run python scripts/gamma_pool_benchmark.py integrated --pool-ids $RUN/C/pool_test200_ids.json \
    --h5 <test h5> --device cuda:0 --passes 3 --out $RUN/C/integrated_test200.json
```

`pools` defines three pools by record id and caches their dose pairs: a seeded
200-record subset of the held-out test set, the complete test set (948), and a
seeded 2000-record subset of the training run's own validation split (14,029
records, reproduced exactly as `train_adota.py` splits it). `time` measures
gamma alone on cached pairs; `integrated` reads, infers and scores each record
with inference and gamma timed apart. The two are never mixed.

## D: scaling

```bash
uv run python scripts/gamma_scaling_benchmark.py \
    --plans LUNG1-195_Publication_Plan_2,Prostate-AEC-007_Publication_Plan_4 \
    --targets 144000,1000000,8000000,32000000 --repeats 5 --device cuda:0 \
    --provenance --out $RUN/D/scaling.json
```

Nested crops around the centroid of the region above half the reference
maximum, with the full grid added as the largest. The global normalisation
dose is pinned to the full plan's maximum so the dose tolerance and cutoff do
not move with the crop; the timed region excludes input copies (the entry
point's in-place masking is a no-op on non-negative dose, which is asserted).
Every crop's bounds, shape, cutoff population and hashes are recorded. Crop
content still affects convergence, which is why evaluated points are reported
beside voxels.

## E: profiling

```bash
uv run python scripts/gamma_profile_beamlet.py --pairs $RUN/inputs/pairs.npz \
    --sweep $RUN/A/beamlet_matched_array.json --iterations 5 --device cuda:0 \
    --provenance --out-dir $RUN/E
```

Picks the easy, median and hard beamlets by the reference rung's median time
and profiles the GPU rungs under `torch.profiler`. Chrome traces and a CSV of
kernel time, copy time, launch count and time, synchronisation, allocation and
the remaining host CPU time are written. Profiling adds overhead; the wall
times here are for attribution only.

## Building the report

```bash
uv run python scripts/analysis/report_gamma_acceleration.py \
    --plan-json docs/gamma_gpu/gamma_gpu_results.json \
    --exp7-json docs/gamma_beamlet/gamma_beamlet_results.json \
    --evidence-dir $RUN --out-dir reports/technical-reports/gamma-pass-rate
uv run python scripts/analysis/audit_gamma_report.py --report-dir reports/technical-reports/gamma-pass-rate
```

The first writes every table, figure and CSV, plus `tables/numbers.tex`, the
macros the prose quotes. The second recomputes every macro from the CSVs and
lists numeric literals in the abstract and conclusion that no data backs. The
report's `Makefile` wraps both (`make assets`, `make audit`).
