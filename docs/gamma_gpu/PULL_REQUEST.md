# GPU gamma index

Merges `gamma_gpu` into `refactor_software_baseline`. Implements
[`docs/gpu_gamma_plan.md`](../gpu_gamma_plan.md).

`pymedphys.gamma` dominated gamma pass rate evaluation. It is the reason
`src/training/gpr_pool.py` exists at all, computing GPR on a frozen *subset* of
the validation set. This adds `src/metrics/gamma_torch/`, a torch implementation
of the gamma shell method that keeps both dose grids on the device, behind an
opt-in backend switch.

**The reported metric does not change.** The backend defaults to pymedphys, and
the `gamma_values -> gamma_pass_rate` arithmetic is shared verbatim between the
two paths, quirky denominator included (voxels with gamma exactly 0 are excluded
from both numerator and denominator). Nothing in the training loop or the plan
pipeline moves.

## Results

All 40 (plan x criterion) pairs of the eight-plan OpenTPS corpus, one A40.
Full tables in [`docs/gamma_gpu/`](.).

| comparison | tolerance | max abs delta | verdict |
|---|---|---|---|
| rung 2 (torch-CPU float64) vs rung 1 (pymedphys) | 0.01 pp | **0.000000 pp** | PASS |
| rung 3 (GPU float64) vs rung 2 | 0.001 pp | **0.000000 pp** | PASS |
| rung 4 (GPU float32) vs rung 2 | 0.1 pp | 0.047677 pp | PASS |
| rung 1 vs rung 0 (recorded) | reported | 0.056401 pp | not gated |

The float64 backend reproduces the pymedphys pass rate exactly on every pair, on
CPU and GPU alike. Voxel-level on the two plans whose maps were persisted: zero
gamma = 1 boundary crossings over 9.3 M evaluated voxels, and no disagreement
about which voxels were evaluated.

Wall time for the whole corpus, same machine, warm-up (~1.0 s, once per process)
excluded from every timed region:

| rung | | total | vs rung 1 |
|---|---|---|---|
| 1 | pymedphys 0.41, this machine | 3016.0 s (50.3 min) | 1.0x |
| 2 | gamma_torch, torch-CPU, float64 | 1767.9 s | 1.7x |
| 3 | gamma_torch, one A40, float64 | **159.0 s (2.65 min)** | **19.0x** |
| 4 | gamma_torch, one A40, float32 | 111.3 s | 27.1x |

Peak device memory 2.1 to 2.8 GiB, on grids from 67.5 M to 100.5 M voxels.

The 4.13 h recorded in the plans' `gamma_metrics.json` is provenance from
another machine on pymedphys 0.40, not a controlled measurement, so no speedup
is quoted against it.

## A finding that is not about this work

**The pymedphys 0.40 to 0.41 interpolator change moves the already-published
gamma pass rates**, by up to 0.056 pp in absolute value, in both directions
(mean 0.0075 pp over the 40 pairs). It is concentrated in the tightest criterion
(1%/1mm) and in two plans, LUNG1-062 and Prostate-AEC-069. Same code path, same
doses, different interpolator underneath.

It may need reporting wherever those figures were published. It also makes the
CPU path about 4.9x faster on its own, which is most of the gap between the
recorded 4.13 h and the 50 min re-run.

## What is in the commits

1. **Python floor to 3.10** so pymedphys can move to 0.41. `numba` added
   (pymedphys ships it only as an optional extra and `gamma` raises without it);
   `torch` pinned `<2.9`, because the 3.10 resolution would otherwise take torch
   2.11, whose CUDA build does not load on these machines' 535.x driver.
2. **The deviation-ladder harness** plus `scripts/gamma_benchmark.py` and its
   guide, landed before any kernel existed so the baseline was established first.
3. **`src/metrics/gamma_torch/`** -- the kernel. Apache-2.0 rather than the
   repository's MIT and free of every `src.` import, so it can be copied to
   PyMedPhys unchanged; a package only because of the 500-line limit, split
   shells / interpolation / loop.
4. **The backend switch**, defaulting to pymedphys.
5. **A memory-lifetime fix**: the interpolation recursion was a nested closure,
   which refers to itself through its own cell and so forms a reference cycle;
   its temporaries survived until the cyclic collector ran. Peak device memory on
   the largest plan dropped 32.3 GiB -> 2.8 GiB with identical results.
6. **The recorded results** and the 1.5.0 CHANGELOG entry.

## Deliberately not done

* **The default is not flipped.** The torch backend is opt-in, so it can run
  alongside the CPU path on real validation data before it becomes the default.
* **`gpr_pool.py` and the frozen-subset logic are untouched.**
* **The pass-rate definition is untouched.** It is unusual, and it is what
  training runs are compared against longitudinally. It is now in one place,
  `_gamma_pass_rate`, rather than open to drifting between two backends.

## The one thing not reproduced exactly

`pymedphys` allocates its per-shell minimum relative dose difference with
`np.ones_like(flat_dose_reference)`, which is **float32** for every dose in this
pipeline, so a float64 minimum is quantised to float32 on assignment.
`gamma_torch` keeps it at the working dtype.

That is the entire residual: on float64 input the two agree to **1.2e-14**, pure
float64 rounding. On float32 doses it accounts for a systematic ~1e-7 difference
in gamma, which costs zero boundary crossings and zero change in all 40 pass
rates. Replicating it would mean rounding an intermediate to float32 inside a
float64 code path, defeating the float64 mode; if bit-parity on float32 input is
ever required it is one cast, not a redesign.

## Known gaps

* Scalar thresholds only; the sequence form `pymedphys.gamma` answers with a
  dict raises `NotImplementedError`.
* Uniform grids only, checked on entry.
* `local_gamma` is not exercised by the corpus (all 40 recorded criteria are
  global); it is covered against `pymedphys.gamma` on synthetic cases.

## Testing

* `run-tests.py unit`: **760 passed**, 7 deselected.
* `tests/test_gamma_torch.py` -- 21 synthetic parity tests against
  `pymedphys.gamma`: 1D/2D/3D, anisotropic spacing, `local_gamma` (including
  `lower_percent_dose_cutoff=0`, where it divides by zero-dose voxels),
  `skip_once_passed`, `random_subset`, unbounded `max_gamma`, out-of-bounds
  handling, and the error paths.
* `tests/test_gamma_torch_corpus.py` -- 3 plan-scale tests, marked
  `integration`, `slow`, `gpu`. Pass on this machine; skip with a reason naming
  `$ADOTA_GAMMA_CORPUS` when the plans are absent (verified).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
