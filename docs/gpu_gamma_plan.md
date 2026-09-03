# Agent brief — GPU gamma index for adota

## What you are building

A GPU implementation of the gamma index (Low et al. dose comparison metric) as a
**self-contained torch module** at `src/metrics/gamma_torch.py`, to replace the
`pymedphys.gamma` call that currently dominates gamma pass rate (GPR) evaluation
in this repository.

The machine has 3× NVIDIA A40. Torch 2.8 + CUDA 12.8 is already installed and working.

## Why, and the constraint that shapes everything

Two consumers, one file:

1. **adota, now.** GPR is a training-loop validation metric. It is so slow that
   `src/training/gpr_pool.py` exists purely to compute it on a frozen *subset* of
   the validation set. Making it fast changes what is measurable.
2. **PyMedPhys, later.** This same code is intended for upstream contribution to
   `pymedphys/pymedphys` as an optional torch backend for
   `lib/pymedphys/_gamma/implementation/shell.py`.

**Therefore `src/metrics/gamma_torch.py` must import nothing from `adota`.**
Only `numpy`, `torch`, and the standard library. No `src.*` imports, no config
objects, no adota logging. If it needs a helper, the helper goes in the same file.
Anything adota-specific (tensor unwrapping, scale dicts, the `(N,C,D,H,W)` layout)
belongs in a thin adapter in `src/metrics/gamma_pass_rate.py`, not in the kernel module.

Breaking this rule means the upstream port becomes a rewrite instead of a copy.
It is the single most important constraint in this brief.

Put an Apache-2.0 licence header and a copyright line at the top of
`gamma_torch.py` from the first commit. The rest of adota is MIT; PyMedPhys is
Apache-2.0, and this one file is destined for there.

## Hard constraint: do not change the reported metric

`src/metrics/gamma_pass_rate.py::gamma_index` computes a two-element pass rate
with a specific and slightly unusual denominator, after `np.nan_to_num`. Training
runs are compared longitudinally against that exact definition — that is the
entire reason `gpr_pool.py` freezes its sample set.

**Your change replaces the computation of the gamma array only.** The
`gamma_values → gamma_pass_rate` arithmetic downstream must be untouched, and it
must receive an array with identical NaN semantics to today's (NaN where the
reference dose is below the cutoff and gamma was not evaluated). Do not "fix" the
pass-rate definition, do not change the NaN handling, do not rename the return
tuple. If you think the metric is wrong, say so in your report and leave it alone.

## The algorithm you are reimplementing

Read `/home/mstryja/projects/pymedphys/lib/pymedphys/_gamma/implementation/shell.py`
and `/home/mstryja/projects/pymedphys/lib/pymedphys/_utilities/createshells.py`
in full before writing anything. That checkout is at pymedphys `main` (0.42.0-dev0).
Note that adota currently runs pymedphys **0.40.0** on Python 3.9, which is an
older interpolation path — see "Step 0" below.

Summary of the shell method (Wendling et al. 2007, http://dx.doi.org/10.1118/1.2721657):

For every reference voxel above the lower dose cutoff, search outward at
increasing radii `r = 0, Δ, 2Δ, …` where `Δ = min(distance_mm_threshold) / interp_fraction`.
At each radius, build a shell of offsets (2 points in 1D, a circle in 2D, a sphere
in 3D) spaced no wider than `Δ`. Interpolate the *evaluation* dose at
`reference_coord + offset` for every shell point, take the minimum absolute
relative dose difference over the shell, and fold

```
gamma_at_r = sqrt( (min_rel_dose_diff / (dose_percent_threshold/100))**2
                 + (r / distance_mm_threshold)**2 )
```

into a running per-voxel minimum. A voxel stops searching once
`current_gamma <= r / distance_mm_threshold`, because no larger radius can improve it.
The loop ends when no voxel is still searching, or when `r > max(distance_mm_threshold) * max_gamma`.

### Where the time goes, and what to exploit

The CPU implementation materialises a `(shell_points × ref_points × ndim)` float64
coordinate array, interpolates it, then reduces. For a clinical 3D case that
temporary reaches ~48 GB at r = 3 mm, which is why the CPU code slices it into
hundreds of RAM chunks.

**Your kernel must never materialise it.** Tile over shell points and reference
points, and fold each tile into a running `torch.minimum`. The reduction is what
makes this a good GPU problem — the intermediate is pure waste.

Useful properties:

- Every (reference voxel, shell point) pair is independent within a distance step.
- `min` is exact and associative for floats, so reduction order does not affect
  the result. You do not need a deterministic reduction order (unlike a sum).
- Both grids are regular and uniformly spaced, so voxel lookup is
  `floor((p - x0)/dx)` — no binary search needed. Verify uniformity on entry and
  raise a clear error if the axes are not uniform.
- The convergence loop (12–21 iterations) stays in Python on the host. Do not
  port it to the GPU; it is not the bottleneck.

### Numerical parity — the four things that will bite you

1. **Out-of-bounds interpolation fills `+inf`, not NaN.** The CPU path passes
   `extrap_fill_value=np.inf`. An infinite dose difference loses every `min`,
   which is the intended behaviour. Fill with NaN and you silently poison the
   whole reduction.
2. **Local gamma divides by the reference dose.** With `local_gamma=True` and a
   zero reference voxel, the CPU path produces `inf` or `0/0 → nan` under a
   suppressed numpy warning. Normally the dose cutoff excludes those voxels, but
   `lower_percent_dose_cutoff=0` is supported and the pymedphys tests use it.
   Match the behaviour rather than guarding it away.
3. **The upper grid edge.** `np.searchsorted` at `p == x[-1]` yields
   `x0 = n-2, w = 1`. Floor-based indexing gives `x0 = n-1`, which reads one plane
   past the end. Clamp explicitly.
4. **The forced search distances.** `gamma_loop` snaps the radius to each value in
   `distance_mm_threshold` exactly, and grows the step by
   `max(r / interp_fraction / max_gamma, Δ)`. Replicate this stepping exactly —
   a different radius schedule gives different numbers and makes parity
   testing meaningless.

## Interface

```python
def gamma_index_torch_core(
    axes_reference,          # sequence of 1-D uniform coordinate arrays (np or torch)
    dose_reference,          # array/tensor, shape matching axes_reference
    axes_evaluation,
    dose_evaluation,
    dose_percent_threshold,  # float
    distance_mm_threshold,   # float
    *,
    lower_percent_dose_cutoff=20,
    interp_fraction=10,
    max_gamma=None,
    local_gamma=False,
    global_normalisation=None,
    skip_once_passed=False,
    random_subset=None,
    device=None,             # torch.device; default: cuda if available else cpu
    dtype=None,              # torch.float32 default, torch.float64 opt-in
) -> "torch.Tensor":         # gamma array, NaN below cutoff, same shape as dose_reference
```

Keyword names and semantics mirror `pymedphys.gamma` exactly. That is deliberate:
the upstream port should be a signature match, not a translation.

**The core takes and returns device-resident torch tensors.** Numpy inputs are
converted on entry; a numpy in / numpy out convenience wrapper sits on top. This
ordering matters — `gamma_index_torch` in `gamma_pass_rate.py` currently does
`.detach().cpu().numpy()` on tensors that are already on the GPU, and that round
trip should disappear.

### Scope for the first version

- 1D, 2D and 3D. Prioritise 3D; it is what adota needs.
- **Scalar thresholds only.** `pymedphys.gamma` accepts sequences for
  `dose_percent_threshold` and `distance_mm_threshold` and returns a dict. Raise
  `NotImplementedError` with a clear message for sequence input, and note it as a
  known gap. Do not build the multi-threshold machinery yet.
- Support `local_gamma`, `max_gamma`, `skip_once_passed`, `lower_percent_dose_cutoff`,
  `global_normalisation`, `interp_fraction`. These are all cheap once the loop is right.
- `random_subset` — support it, it is used for fast approximate pass rates.

## Validation corpus — real plans with recorded results

Validation runs against eight OpenTPS plan directories under
`/scratch/mstryja/opentps_plans/`. Each already contains a `gamma_metrics.json`
holding the pass rates and wall time produced by the current CPU pipeline. These
are the numbers you must reproduce.

| Plan directory | Grid (x,y,z) | Voxels | Recorded `elapsed_s` |
|---|---|---|---|
| `LUNG1-062_Publication_Plan_1` | 501 × 501 × 303 | 76.1 M | 285 |
| `LUNG1-195_Publication_Plan_2` | 500 × 500 × 270 | 67.5 M | 139 |
| `LUNG1-250_Publication_Plan_3` | 500 × 500 × 402 | 100.5 M | 1974 |
| `LUNG1-364_Publication_Plan_5` | 500 × 500 × 402 | 100.5 M | 1751 |
| `Prostate-AEC-004_Publication_Plan_1` | 500 × 500 × 358 | 89.5 M | 2181 |
| `Prostate-AEC-069_Publication_Plan_2` | 500 × 500 × 318 | 79.5 M | 2852 |
| `Prostate-AEC-006_Publication_Plan_3` | 482 × 482 × 322 | 74.8 M | 2616 |
| `Prostate-AEC-007_Publication_Plan_4` | 500 × 500 × 386 | 96.5 M | 3076 |

Total recorded CPU time: **14 874 s ≈ 4.13 h** for 5 criteria per plan, 40
gamma evaluations in all. `elapsed_s` times `plan_gamma(...)` only — not the
figures or the MAPE/RMSE metrics — so it is a clean gamma-only baseline.

Every plan uses the same five criteria `(dose%, distance_mm, cutoff%)`:
`1%/1mm/10%`, `2%/2mm/10%`, `3%/3mm/10%`, `1%/2mm/3%`, `1%/3mm/0.1%`.
The last one evaluates nearly the whole volume and will dominate the runtime;
treat it as the stress case, not an outlier to be dropped.

### Reproducing the recipe exactly

Read `scripts/run_plan_opentps.py::_run_gamma_stage` and
`src/metrics/plan_gamma.py` and follow them literally. The details that matter:

- **Take the parameters from each plan's own `gamma_metrics.json`, not from
  adota's current defaults.** The JSON records `gamma_params_base`
  (`interp_fraction: 5`, `max_gamma: 2`, `local_gamma: false`,
  `random_subset: null`) and the exact `criteria` list. `DEFAULT_GAMMA_PARAMS` in
  `src/adota/config.py` says `interp_fraction: 10`; the recorded runs used 5,
  overridden by `DEFAULT_GAMMA_EXTRA`. Defaults drift, the JSON is the record of
  what actually ran. Assume nothing.
- **Load the doses through the existing path**, not by reading the `.raw` files
  yourself: `PlanDirectory` (`src/loaders/plan_directory.py`),
  `BeamDataLibrary.from_file(bdl.txt)`, then
  `sitk.GetArrayFromImage(load_dose_gy(path, plan, bdl))`
  (`src/beamlets/dose_scaling.py`). `load_dose_gy` applies the MU→Gy scaling; a
  raw read gives you unscaled numbers that will look plausible and be wrong.
  Spacing is `plan_directory.ct.GetSpacing()[::-1]`.
- **Watch the reference/evaluation direction.** `_run_gamma_stage` calls
  `plan_gamma(dose_adota, dose_mc, ...)`, whose signature is
  `plan_gamma(dose_eval, dose_ref, ...)` — so ADoTA is *eval*, MCsquare is *ref*.
  `plan_gamma` then calls `gamma_index(ground_truth=dose_ref, prediction=dose_eval)`,
  and `gamma_index` calls `gamma(axes, ground_truth, axes, prediction)`, making
  **MCsquare the pymedphys reference grid and ADoTA the evaluation grid**. Gamma
  is not symmetric. Swapping these produces believable numbers that are wrong,
  and it is the single easiest way to fail this task silently.
- **Reproduce the pass-rate arithmetic verbatim**, including its quirk: after
  `np.nan_to_num(gamma_values, 0)`, the reported figure is
  `pass_rate[0] = 1 - count_nonzero(g > 1) / count_nonzero(g > 0)`, times 100.
  Voxels with gamma exactly 0 are excluded from both numerator and denominator.
  Do not fix this — see "Hard constraint" above.

### The deviation ladder

Four things could move a pass rate, and a single before/after comparison cannot
tell you which one did. Measure them separately, each rung against the one above:

| Rung | What it is | Isolates |
|---|---|---|
| 0 | The recorded `gamma_metrics.json` (pymedphys 0.40.0, Python 3.9, EconForge interpolation) | — |
| 1 | CPU re-run on pymedphys 0.41.x after the step-0 bump, same machine | The interpolator change |
| 2 | `gamma_torch` on torch-CPU, float64 | Your implementation |
| 3 | `gamma_torch` on one A40, float64 | The device |
| 4 | `gamma_torch` on one A40, float32 | Precision |

Rung 1 matters more than it looks. The recorded numbers were produced by
pymedphys 0.40.0's EconForge interpolator; 0.41.x uses a different one. If the
bump alone shifts the pass rates, that shift is not yours, and you need to know
its size before you can interpret anything downstream. Establish rung 1 before
writing the kernel.

### Acceptance criteria

Across all 40 (plan × criterion) pairs, on `pass_rate_pct`:

- **Rung 2 vs rung 1: |Δ| ≤ 0.01 pp.** This is the correctness gate. A larger
  deviation means a genuine algorithm difference — find it, do not widen the
  tolerance.
- **Rung 3 vs rung 2: |Δ| ≤ 0.001 pp.** Same code, same precision; should be
  near-exact.
- **Rung 4 vs rung 2: |Δ| ≤ 0.1 pp.** If any pair exceeds this, report it and make
  float64 the default.
- **Rung 1 vs rung 0** is reported, not gated — it is pymedphys's change, not yours.

Report the full 40-row table, not just the maximum.

### Voxel-level parity

Pass rates can agree while the maps differ. For at least two plans — pick
`LUNG1-195_Publication_Plan_2` (fastest, 139 s) and one prostate plan — persist the
rung-1 CPU gamma maps to `/scratch` as float32 `.npy` during the baseline re-run
(they are not saved by the current pipeline; `plan_gamma` returns them in memory
for the figure only). Then report, per criterion:

- max |Δγ| over evaluated voxels, and the 99.9th percentile of |Δγ|
- **the fraction of evaluated voxels that cross the γ = 1 boundary** — this is the
  number that matters. A voxel moving 1.5 → 1.6 is irrelevant; one moving
  0.999 → 1.001 changes the result.
- a count of voxels where one implementation gives NaN and the other does not

Budget the disk: a 500 × 500 × 402 float32 map is 402 MB, five criteria per plan,
two plans ≈ 4 GB on `/scratch`. Fine, but do not do it for all eight.

## Performance test

### What to measure

Per (plan × criterion), single A40:

- wall time, and the per-plan total for direct comparison against `elapsed_s`
- peak GPU memory (`torch.cuda.max_memory_allocated`), reset per criterion
- throughput in interpolated samples/s, so plans of different grid sizes are comparable

Then the headline: total GPU time across all 8 plans against the 4.13 h CPU baseline.

### Measurement hygiene

- **The recorded `elapsed_s` values are provenance, not a controlled measurement.**
  They were produced on some machine under some load between June and September
  2026, on pymedphys 0.40.0 and Python 3.9. Never compute a speedup factor against
  them.
- **Re-run the CPU baseline on the target machine** for the controlled number. Doing
  all eight is ~4 h — acceptable as one overnight run, and it doubles as rung 1 of
  the deviation ladder, so it is not extra work. If you must economise, do
  `LUNG1-195` (139 s) and `Prostate-AEC-007` (3076 s) to bracket the range, and label
  the other six as JSON-provenance only.
- Exclude CUDA context creation and first-call warm-up from the timed region;
  report warm-up separately as a one-off cost.
- Pin BLAS threads for the CPU baseline the way the existing gamma scripts do
  (see the `E402` entries in `pyproject.toml` and the note in CLAUDE.md), so the
  CPU number is not accidentally sandbagged.
- One GPU only. Three A40s are present; do not shard across them in this task.

### How it is packaged

A pytest test marked `integration`, `slow` and `gpu` that reads the plan
directories, and **skips with a reason naming `/scratch/mstryja/opentps_plans` when
they are absent** — per the repo rule that a test must never fail for a missing
resource. Point it at the corpus by env var (follow the `$ADOTA_GOLDEN_DIR`
pattern the existing golden tests use) rather than hard-coding the path, since
this file is also headed upstream.

Write the results as a small JSON plus a markdown table. The tables are small
enough to live in the repo; the gamma maps are not, and belong on `/scratch`.

## Step 0, before any of the above

adota is on Python 3.9, which pins it to `pymedphys==0.40.0` — the old EconForge
interpolation path. `pymedphys 0.41.0` requires ≥3.10 and ships an in-house numba
interpolator that is 5–8× faster on exactly the interpolation gamma spends its
time in. The lockfile already resolves 0.41.0 behind a `python_full_version >= '3.10'` marker.

Bump `requires-python` to `>=3.10` in `pyproject.toml`, re-lock, and run the test
suite. Two reasons this comes first:

- Your benchmark baseline must be the current pymedphys, not a two-version-stale one.
  Reporting a GPU speedup against 0.40.0 would overstate the gain.
- If the bump breaks something, better to know now than to have it entangled with
  the GPU work.

If the bump turns out to break the suite, stop and report rather than working
around it — and in that case keep `gamma_torch.py` Python 3.9-compatible.
Either way, write the module with `from __future__ import annotations` and
`typing.Optional` / `typing.Tuple` style, matching the surrounding code; it costs
nothing and keeps the file portable.

Note also that the module docstring in `src/metrics/gamma_pass_rate.py` claims
`pymedphys.gamma` requires the EconForge `interpolation` package. That goes stale
with the bump — update it.

## Repository conventions (from adota CLAUDE.md — read it in full)

- Everything runs through `uv run`. Never a bare `python`.
- Tests: `uv run python scripts/run-tests.py unit`. Markers: `integration`, `e2e`,
  `gpu`, `slow`. **A test that needs something the machine lacks must skip with a
  reason naming what to install or where the data belongs — never fail.**
- Lint: `uv run ruff check .` and `uv run ruff format .`. Ruff selects `E`, `F`, `I`
  at line length 120. Import order enforced: stdlib, third-party, local.
- **500 lines per module, enforced in `src/`.** If the kernel plus its helpers
  outgrows that, split by role (shells / interpolation / loop), not at an arbitrary cut.
- `resolve_device` in `src/evaluation/cli.py` is the only device entry point in
  adota — the *adapter* should use it. The kernel module takes a plain
  `torch.device` and must not import it.
- Shared test helpers live in `tests/utils/`, not in fixtures.
- Large outputs go on `/scratch`, never in the repo or `$HOME`.
- `CHANGELOG.md` is maintained under semver; update it for anything that changes a
  public import path or an output format.

## Suggested staging

Land these as separate commits so each is reviewable and revertible:

1. Python 3.10 bump + re-lock + suite green + stale docstring fixed.
2. The harness: loads a plan directory, reads its `gamma_metrics.json`, re-runs the
   CPU path, and emits the comparison table. At this point it establishes rung 1 and
   reports rung 1 vs rung 0 — before any kernel exists. Persist the CPU gamma maps
   for the two voxel-level plans in this step.
3. `gamma_torch.py`: shells, uniform-grid interpolation, and the fused
   interpolate-and-reduce step. 3D global gamma only — which covers all 40
   comparisons, since every recorded criterion has `local_gamma: false`. Rungs 2
   and 3 green.
4. `local_gamma`, `skip_once_passed`, `random_subset`, 1D/2D. Not exercised by the
   plan corpus, so cover these with small synthetic cases checked against
   `pymedphys.gamma` directly.
5. The adapter: `gamma_pass_rate.py` gains a backend switch, defaulting to the
   existing pymedphys path. Tensors stay on device when the torch backend is used.
6. Full benchmark run across all 8 plans + the 40-row deviation table + CHANGELOG entry.

Do not flip the default to the torch backend in this task. Land it opt-in, run it
alongside the CPU path on real validation data for a while, and switch the default
in a later change once the two agree across a meaningful number of real cases.

## Report back with

- **The 40-row deviation table** — 8 plans × 5 criteria, `pass_rate_pct` at every
  rung of the ladder, with the rung-to-rung deltas. State plainly which acceptance
  criteria passed and which did not.
- **The rung 1 vs rung 0 result on its own.** How much, if anything, did the
  pymedphys 0.40 → 0.41 interpolator change move the published pass rates? This is
  a finding about the existing results regardless of how the GPU work turns out,
  and it may need reporting elsewhere.
- **The voxel-level table** for the two plans: max |Δγ|, 99.9th percentile |Δγ|,
  γ = 1 boundary-crossing fraction, NaN disagreement count.
- **The performance table**: per plan and criterion, GPU wall time and peak memory,
  against the re-run CPU baseline. Separate the controlled measurements from the
  JSON-provenance ones. Give the aggregate against 4.13 h.
- Anything in the pymedphys algorithm you could not reproduce exactly, and why.
- Any place you were tempted to change the reported metric, and what you did instead.

## Out of scope

- Do not modify `gpr_pool.py` or the frozen-subset logic.
- Do not write a custom CUDA kernel, use `torch.compile`, or reach for CUDA graphs.
  Plain torch ops first; those are follow-ups once the benchmark says where the
  remaining time is.
- Do not reformulate the algorithm (dense neighbourhood search, distance
  transforms). Different sampling gives different numbers and forfeits parity with
  the validated baseline.
- Do not touch the pymedphys checkout at `/home/mstryja/projects/pymedphys`.
  The upstream contribution is tracked separately.
