# Active learning: pool, validation set, loop

Three entry points that together answer one question: **given patient CTs the model
has never seen, which beamlets are worth simulating next, and does choosing them by
the input-only difficulty score reach a target accuracy with fewer Monte Carlo
seconds than choosing them at random?**

The unit of cost is **Monte Carlo seconds**, not sample count, because that is what
the clinic pays. Design rationale: [docs/active_learning_pipeline.md](../../docs/active_learning_pipeline.md).

| Script | Config | What it does |
|---|---|---|
| [`al_build_pool.py`](../al_build_pool.py) | `config_al.yaml` | Split held-out CTs into pool and validation roles; write the provenance CSV. Run once. |
| [`al_build_validation_set.py`](../al_build_validation_set.py) | `config_al.yaml` | Generate the frozen, difficulty-balanced validation set. Run once, before any cycle. |
| [`al_loop.py`](../al_loop.py) | `config_al.yaml` | Run one arm: sample, label, retrain, validate, repeat. One process per strategy. |

## Order of operations

```bash
# 1. Roles. Reads the tail of each collection, drops patients claimed by other
#    experiments, writes registry/al_pool_selection.csv. Overwriting it changes
#    the split, so it refuses unless you pass --overwrite.
uv run python scripts/al_build_pool.py --config scripts/config_al.yaml

# 2. The yardstick. --dry-run scores and selects but simulates nothing, and prints
#    the Monte Carlo estimate: always run that first.
uv run python scripts/al_build_validation_set.py --config scripts/config_al.yaml --dry-run
uv run python scripts/al_build_validation_set.py --config scripts/config_al.yaml --num-threads 48

# 3. The arms, at equal Monte Carlo budget, on separate GPUs.
uv run python scripts/al_loop.py --config scripts/config_al.yaml --strategy random --device-index 1 &
uv run python scripts/al_loop.py --config scripts/config_al.yaml --strategy score  --device-index 2 &
```

**Smoke test first**, as for any long Monte Carlo job. `config_al_smoke.yaml` runs the
identical code path with every dimension cut to the smallest value that still
exercises it (two CTs, one gantry, two energies, eight beamlets, one cycle, twenty
optimizer steps) and writes everything to `/scratch/mstryja/al_smoke`, so it cannot
touch the dataset root:

```bash
uv run python scripts/al_build_validation_set.py --config scripts/config_al_smoke.yaml --n-cts 2
uv run python scripts/al_loop.py --config scripts/config_al_smoke.yaml --strategy score
```

## What one cycle does

```
generate candidates on pool CTs -> score (input-only) -> strategy -> batch
label the batch with Monte Carlo -> add it to the training sources
retrain from the previous cycle's weights -> validate on the frozen set
write the cycle manifest
```

A candidate is `(CT, gantry, energy, theta_x, theta_y)`. Version 0 draws the gantry
uniformly at random (the CT is rotated into the canonical beam's-eye frame, so gantry
is metadata), puts the isocenter at the grid centre, takes the energy from a discrete
layer set and the steering uniformly from the generator's lattice inside ±1.5°.

**Validity is decided before Monte Carlo, from the inputs.** `score_candidates` flags
a ray that misses the CT as `roi_out_of_bounds` and an analytic Bragg peak that leaves
the 320 mm crop as `peak_outside_crop`. The second gate is the input-only range check
EXP-0006 validated at 97% agreement with the Monte Carlo truth, and it removes the
over-ranging failures before they cost anything. Expect a substantial rejection rate
on thoracic anatomy at the higher layers: low-density lung extends the range past the
crop. **Size the candidate pool against the valid count, not the raw count.**

## Strategies

| `--strategy` | Rule | Role |
|---|---|---|
| `random` | uniform over valid candidates | the control, and not a straw man |
| `score` | probability ∝ score^`score_alpha`, with per-patient and per-energy quotas | the hypothesis |
| `score_topk` | the hardest K | ablation: shows the collapse the quotas prevent |
| `stratified_score` | equal counts per score decile | ablation, and the validation-set recipe |

Difficulty is used **conditional on energy**. The score is dominated by path length,
so an unconstrained draw picks only the deepest beamlets; the budget is split equally
across the energy layers present and capped per patient within each layer.

## Cost: groups, not just beamlets

Work is grouped by `(patient, gantry, energy)`. Each group pays one CT rotation and
one MCsquare setup (~12 s) however many beamlets it holds, so a batch spread thinly
over many groups spends more on setup than on physics. Keep groups fat by sampling a
**subset** of pool CTs per cycle (`n_cts_per_cycle`) rather than a few beamlets on
every CT. `batch_cost_estimate` reports the split before anything is simulated, and
the `--dry-run` flag on both `al_loop.py` and `al_build_validation_set.py` prints it.

Measured throughput: **3.31 s per beamlet** in beamlet mode at 1e6 primaries on 48
threads. Beamlet-mode parallelism is one thread per spot, so this scales roughly with
the thread count — on a shared machine, divide accordingly.
`beamlet_block_size` bounds the transient dense dose grids MCsquare writes before
Python crops them (a 324-spot thoracic field is over 100 GB).

## Retraining: a step budget, not an epoch

A cycle buys a few thousand beamlets against a training split of 56k. Under uniform
sampling a new beamlet would be seen once every several epochs and the cycle would
measure nothing. So a cycle is a fixed number of optimizer steps
(`al_steps_per_epoch` × `num_epochs`) in which `al_oversample_fraction` of every batch
is drawn from the bought beamlets.

**This is a deliberate departure from the design document's "continue on the full
union".** Both arms use it, so they stay comparable to each other; neither is
comparable to a plain-union baseline.

A cycle hands **`last.pth`**, not `best.pth`, to the next one. `best` is chosen by the
loss on the HDF5 validation split, which is the distribution the model already fits;
a cycle that trains on newly bought beamlets can raise that loss while improving on
exactly the geometry it just bought, and selecting on it would carry the pre-cycle
weights forward and make the loop measure nothing.

## What is measured

On the frozen validation set, after every cycle and once on the starting checkpoint
(the zero-budget point of the learning curve):

- **GPR** at 3%/3mm with a 10% cutoff: the mean, and the tail that matters — the
  fraction below 95% and the 5th percentile. Computed on the **GPU** (`backend="torch"`);
  pass rates are identical to pymedphys and it is the only practical choice at
  thousands of beamlets per evaluation.
- **MAPE** and **RDE**, matching the training-time definitions so the numbers are
  comparable with any earlier run.
- **dR80**, the distal range error, which training validation does not compute today.
  `compute_range_metrics` returns NaN rather than a number when the depth-dose curve
  never falls back below the level, so a beamlet whose dose leaves the crop is
  excluded from the range statistics instead of poisoning them.

Per-decile summaries come free from the balanced validation set
(`validation_by_decile.csv`), so "did it only improve on easy beamlets?" is answerable
without another run.

## Outputs

```
<runs_dir>/al_<name>_<strategy>/
  loop_config.json            the resolved arm config
  baseline_metrics.json       the starting checkpoint on the frozen set
  baseline_samples.csv        per-beamlet metrics for it
  training_sources.csv        every (dir, stem) bought so far -- the union training set
  metrics.jsonl               one line per cycle: cumulative MC seconds and the metrics
  cycle_00/
    candidates.csv            every candidate scored this cycle, valid or not
    selection.csv             what the strategy chose
    groups.json               per-group Monte Carlo outcome
    manifest.json             the cycle record: selection fingerprint, MC seconds, metrics
    train_config.yaml         the config the cycle's training run used
    training.log              its stdout
    training/train_*/         a full training run directory, with checkpoints
    validation_samples.csv    per-beamlet metrics after the cycle
    validation_by_decile.csv  the same, per score decile
```

`metrics.jsonl` is the learning curve: read `mc_seconds_cumulative` against any metric.

## Resume

Everything is resumable and nothing is recomputed:

- A cycle whose `manifest.json` exists is skipped.
- Monte Carlo skips beamlets already on disk. Candidate ids are **content-addressed**
  (a hash of patient, gantry, energy and the two steering angles), so the same
  candidate selected in two cycles maps to the same files and is simulated once.
- The baseline evaluation is cached in `baseline_metrics.json`.

Rerun the same command with the same run directory.

## Requirements

The MCsquare install (`engine.mcsquare_install`), the reference HDF5 training set, the
warm-start checkpoint and its `hyperparams.json`, and a CUDA device. The model block
of `config_al_train.yaml` **must** match the warm-start checkpoint's hyperparameters
or the weights will not load — note `convolutional_steps: 1`, where v11 differs from
`config_train_adota.yaml`.
