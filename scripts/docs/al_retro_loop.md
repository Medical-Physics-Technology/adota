# al_retro_loop.py and al_compare.py

> **Reproducibility / portability.** The paths in the examples are the original
> development environment's locations. Point the config at your own dataset, and
> keep every output under a scratch directory of your choice.

---

The **retrospective active-learning benchmark** (experiment record EXP-0009). It runs
entirely on the existing training HDF5: the labels already exist, and the loop treats
them as hidden, revealing them only for the records a sampling strategy selects. No
Monte Carlo runs, nothing is generated, and no labelling cost is measured. The
question it answers is: **how does the sampling strategy influence the training
progress?** The prospective loop that buys labels with Monte Carlo is
[`al_loop.py`](al_loop.md); this benchmark is its dry run on known data, and a
design filter for it.

| Script | Config | What it does |
|---|---|---|
| [`al_retro_loop.py`](../al_retro_loop.py) `splits` | `config_al_retro_loop.yaml` | Apply the exclusion list; draw and write the validation set, the training split and the cycle-0 set. Run once. |
| [`al_retro_loop.py`](../al_retro_loop.py) `cycle0` | same | Train the shared baseline from random weights on the cycle-0 set. Run once per seed. |
| [`al_retro_loop.py`](../al_retro_loop.py) `run` | same | One strategy: resume from the cycle-0 checkpoint; each cycle scores the pool, selects `N` records, trains `epochs_per_cycle` epochs, validates. One process per strategy. |
| [`al_compare.py`](../al_compare.py) | `config_al_compare.yaml` | Read the strategy runs back, check they are comparable, write the four figures and the tables. |

## Order of operations

```bash
# 1. Splits. Exclusion list -> D; data_fraction of D; V = 15% of that (train_adota's split mechanism);
#    T = D \ V; cycle-0 set = 20% of T; pool = the rest. Written once to splits_dir.
uv run python scripts/al_retro_loop.py splits --config scripts/config_al_retro_loop.yaml

# 2. The shared cycle-0 baseline. No scoring happens here.
uv run python scripts/al_retro_loop.py cycle0 --config scripts/config_al_retro_loop.yaml --device-index 0

# 3. The strategies, each an independent run from the identical checkpoint.
C0=/scratch/mstryja/adota_runs/al_retro/train_<ts>_al_EXP-0009_cycle0_seed1234
uv run python scripts/al_retro_loop.py run --config scripts/config_al_retro_loop.yaml \
    --strategy random            --cycle0-run $C0 --device-index 0 &
uv run python scripts/al_retro_loop.py run --config scripts/config_al_retro_loop.yaml \
    --strategy score_topk        --cycle0-run $C0 --device-index 1 &
uv run python scripts/al_retro_loop.py run --config scripts/config_al_retro_loop.yaml \
    --strategy score_topk_mixed  --cycle0-run $C0 --device-index 2 &

# 4. The comparison, once the runs are complete.
uv run python scripts/al_compare.py --config scripts/config_al_compare.yaml \
    --run <random run> --run <score_topk run> --run <score_topk_mixed run>
```

**Unattended:** `scripts/run_al_retro.sh` runs all four steps, queuing every
(seed, strategy) pair over the GPUs in `GPUS` seed by seed and comparing
whatever finished. `SEEDS` defaults to the config's `seed`; with three GPUs and
three strategies each seed is one round, so the first round can be compared by
hand while the rest run. Every seed resumes from the one `CYCLE0_RUN`: the seed
changes the selection draws and the mini-batch order, not the starting weights.
EXP-0011 reuses the EXP-0009 splits and cycle-0 checkpoint, so it skips both:

```bash
GPUS="0 1 2" STRATEGIES="random score_topk score_topk_mixed" SEEDS="1234 1235 1236" SKIP_SPLITS=1 \
CYCLE0_RUN=/scratch/mstryja/adota_runs/al_retro/d30/train_20260910_192628_al_EXP-0009_d30_cycle0_seed1234 \
nohup bash scripts/run_al_retro.sh > /scratch/mstryja/adota_runs/al_retro/d30/exp0011/launch.out 2>&1 & echo "PID: $!"
```

**Smoke test first.** There is no smoke YAML: the smoke test is the main config plus
the repeatable `--set KEY=VALUE` option, which overrides any key (dotted for nested
blocks) before the config is validated. The list below runs the identical code path
on a 400-record `D` (two cycles of two epochs, an 8-record evaluation subsample,
compile and TF32 off so a resume is bit-exact) under
`/scratch/mstryja/adota_runs/al_retro_smoke`, in minutes. Every stage takes the same
list, so put it in a variable:

```bash
SMOKE="--set data_fraction=1.0 --set max_records=400 \
  --set splits_dir=/scratch/mstryja/adota_runs/al_retro_smoke/splits \
  --set runs_dir=/scratch/mstryja/adota_runs/al_retro_smoke \
  --set n_cycles=2 --set epochs_per_cycle=2 --set eval_every_n_epochs=1 \
  --set eval_subsample_size=8 --set checkpoint_every_n_epochs=1 --set scorer.n_workers=8 \
  --set training.compile=false --set training.allow_tf32=false"

uv run python scripts/al_retro_loop.py splits --config scripts/config_al_retro_loop.yaml $SMOKE
uv run python scripts/al_retro_loop.py cycle0 --config scripts/config_al_retro_loop.yaml $SMOKE
uv run python scripts/al_retro_loop.py run    --config scripts/config_al_retro_loop.yaml $SMOKE \
    --strategy score_topk_mixed --cycle0-run <smoke cycle-0 run>
```

## What one cycle does

```
score the remaining pool (input only, through the scorer interface)
select N records with the named strategy
add them to the training set; assert |train| == |cycle-0 set| + c * N
train epochs_per_cycle epochs from the previous cycle's state (weights, optimizer,
    balancer, RNG all restored; the scheduler too under lr_schedule: plateau, not
    under a fixed schedule, which carries no scheduler state)
validate; write the cycle manifest
```

- **Strategies** (`strategy`): `random` (uniform over the pool; the control),
  `score_topk` (the `N` highest scores), `score_topk_mixed` (a `top_fraction`
  share of the batch is the top-scoring prefix of `score_topk`, the rest drawn
  uniformly from everything else in the pool, scored or not; exploitation plus
  coverage). `stratified_score` (equal counts per score decile) was dropped
  after EXP-0009 showed it is uniform sampling over the score distribution, a
  second `random` run rather than a coverage strategy. They live in a registry
  in `src/active_learning/retrospective/sampling.py`; a fourth is one decorated
  function there and one config line. `strategy_params` forwards keyword
  parameters to the strategy (`{top_fraction: 0.5}` for `score_topk_mixed`;
  `random` and `score_topk` take none).
- **Scoring** happens on the fly at the start of every cycle from cycle 1 onward,
  never precomputed, through the scorer interface of `scoring.py`. The scorer is the
  deployed input-only difficulty score (`DifficultyScorer.load()`, `scorer_path`,
  `variant`, `arm`). It reads the CT, the flux and the energy, and locates the Bragg
  peak with the analytic surrogate; it never reads the stored dose, and
  `tests/retrospective/test_scoring.py` fails if it ever does.
- **Validation**: the loss on the full validation set every epoch; gamma pass rate
  (torch backend, 2%/2mm, 10% cutoff by default), MAPE, RDE and dR80 (with the
  plateau guard) every `eval_every_n_epochs` on a fixed subsample, and on the whole
  validation set at every cycle boundary. Every row of `metrics.jsonl` carries the
  cycle, the cumulative epoch and the training set size.

## Options

`splits`: `--config`, `--data-fraction`, `--max-records` (smoke tests only), `--splits-dir`.
`cycle0`: `--config`, `--device-index`, `--epochs-per-cycle`, `--seed`, `--runs-dir`,
`--resume-dir`. `run`: the same plus `--strategy`, `--cycle0-run` (required),
`--n-cycles`. Every stage also takes `--set KEY=VALUE`, repeatable, for any config key
(`--set training.compile=false`, `--set scorer.n_workers=8`; the value is parsed as
YAML). Precedence: per-field option > `--set` > YAML > defaults, through
`apply_set_overrides` and `merge_config`.

## Config reference (`config_al_retro_loop.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `dataset_path` | the training HDF5 | Every record the loop may reveal. |
| `data_fraction`, `data_fraction_seed` | 0.30, 20260911 | Share of D the experiment uses, drawn once before any split. The pilot runs at 0.30; scale-ups at 0.4, 0.5, 0.6, 1.0 each get their own `splits_dir` and `runs_dir`, and the run name carries `d<percent>`. |
| `exclude_indexes_path` | the `IndexesExclude_...txt` list | Mandatory; a missing file is an error. Cross-checked against `data/excluded_indexes/`. |
| `record_provenance_csv` | the study's `uuid_provenance_map.csv` | Patient and anatomy per record for the fingerprint; optional. |
| `splits_dir`, `runs_dir` | under `/scratch/mstryja/adota_runs/al_retro` | Where the splits and the runs go. |
| `val_fraction`, `initial_fraction` | 0.15, 0.20 | `V` as a share of `D`; the cycle-0 set as a share of `T`. |
| `split_seed`, `initial_seed` | 42, 20260910 | The two draws. |
| `batch_fraction`, `n_cycles`, `epochs_per_cycle` | 0.10, 5, 50 | `N` as a share of `T`; the loop length. |
| `eval_every_n_epochs`, `eval_subsample_size`, `eval_subsample_seed` | 5, 1000, 20260910 | The metric cadence and the fixed subsample. |
| `strategy`, `seed`, `device_index` | random, 1234, 0 | Overridable on the CLI. |
| `strategy_params` | `{}` | Keyword parameters forwarded to the strategy, for example `{top_fraction: 0.5}` for `score_topk_mixed`. |
| `scorer` | `{name: difficulty, ...}` | `scorer_path`, `variant`, `arm`, `n_workers`. |
| `gamma_params`, `gamma_cutoff_percent`, `gamma_backend` | 2%/2mm, 10, torch | Named in every metric filename. |
| `lr_schedule` | plateau | `plateau`, `constant` or `cosine_per_cycle`; see "Learning-rate schedule" below. |
| `lr_min` | 0.0 | The floor of the `cosine_per_cycle` schedule, an absolute learning rate. Unused otherwise. |
| `training` | the baseline model and optimizer | Any `TrainingConfig` field; the model is trained from scratch. |

## Learning-rate schedule

`lr_schedule` picks how the learning rate evolves within a cycle. `plateau`
(the default) is `ReduceLROnPlateau` on the validation loss, its `lr_factor` and
`lr_patience` from the `training` block, and its state carried across cycles;
this is what EXP-0009 ran. `constant` holds `training.learning_rate` fixed at
every epoch. `cosine_per_cycle` decays from `training.learning_rate` at epoch 0
to `lr_min` at the last epoch of every cycle, warm-restarting at the start of
the next one.

EXP-0009 found that `plateau` couples the learning rate to the strategy: a run
whose validation loss stalls gets its LR cut and trains slower thereafter, so
the scheduler amplifies whatever difference the training data made rather than
leaving that difference to speak for itself. EXP-0010 used `constant` so the
strategy is the only thing that differs between runs, and found the next
problem: under a constant 5e-4 the validation loss swings several-fold inside
a cycle, so the boundary row is one epoch of an oscillating trajectory (two
same-data replicates differed by 1 to 2 points of mean pass rate at the
boundary), the mini-batch order pins that trajectory so same-seed runs are not
independent samples of it, and one run in three diverged mid-cycle (a ten-fold
rise of the training loss in one epoch). EXP-0011 uses `cosine_per_cycle`, 5e-4
to `lr_min` 5e-5, so the model is at a low learning rate when it is measured,
and runs three seeds per strategy.

A fixed schedule (`constant` or `cosine_per_cycle`) carries no scheduler
object, so a strategy run under one may resume from a cycle-0 checkpoint that
was trained under `plateau`, as EXP-0010 does with the EXP-0009 cycle 0. The
run logs that cycle-0 run's `lr_schedule` and the distinct `lr` values seen in
its `metrics.jsonl`, and warns (without stopping) if a fixed-schedule run
resumes from a cycle 0 that saw more than one learning rate, since that cycle 0
is then not quite the constant-LR starting point the fixed schedule assumes.

## Outputs

A run directory is a standard training run directory: `manifest.json` (git hash,
GPU, dataset fingerprint, resolved config including `lr_schedule` and `lr_min`,
the splits fingerprint, the cycle-0 checkpoint path and SHA-256, and one entry
per cycle with the selected count, the selection fingerprint, the timings and
the full-set metrics), append-only `metrics.jsonl` (every row also carries
`lr_schedule` beside the existing `lr`), `checkpoints/cycle_XX/last.pth` with
full RNG state, `validation/` with the per-sample CSVs, and `cycles/cycle_XX/`
with `pool_scores.csv`, `selection.csv` (ids and scores), `selection.json` and
`training_ids.csv`. The cycle-0 rows are copied into each strategy's log so every
run reads on its own.

`al_compare.py` writes, under a timestamped directory of `output_dir`:
`F1_training_curves`, `F2_quality_vs_n_train_gamma_<criteria>`,
`F3_quality_vs_epochs_gamma_<criteria>`, `F4_fingerprint_score_decile`,
`F4_fingerprint_energy` (SVG, PDF and PNG, with a CSV of the numbers beside each),
`summary_boundaries_gamma_<criteria>.csv`, `epochs_to_quality.csv` and
`consistency.json`. The figure functions are in `src/figures/al_curves.py`.

Two readers added after EXP-0010 sit beside those: `divergences.csv` gives, per
run and cycle, the largest epoch-to-epoch rise of the training loss and flags it
above `divergence_ratio` (a flagged cycle, and everything after it, measures the
recovery rather than the data; `consistency.json` lists the flagged cycles under
`diverged` and the script warns); `summary_trajectory_last<k>_gamma_<criteria>.csv`
gives the median of the last `trajectory_last_k` subsample evaluations of every
cycle, a boundary estimate that does not depend on which epoch the cycle
happened to end on (on the subsample, so not interchangeable with the full-V
boundary numbers). When a strategy was run under several seeds, its runs are
labelled `<strategy>_seed<seed>` and the boundary and trajectory summaries, F2
and F3 are written a second time `_by_strategy`: mean across seeds with a
min-max band (`aggregate_over_seeds` in `compare.py`).

## Resume

Pass `--resume-dir <run dir>`: completed cycles are skipped (their selection and
checkpoint are reused) and an interrupted cycle restarts from its `last.pth`, which
restores the weights, the optimizer, the balancer and the RNG, and the scheduler
too under `lr_schedule: plateau`; a fixed schedule carries no scheduler state, so
there is nothing to restore there and the epoch index alone determines the
learning rate on resume. The strategy run verifies, tensor by tensor, that the
state it restores from the cycle-0 checkpoint equals the file, and records the
checkpoint's SHA-256 in its manifest.

What is and is not reproducible: restoring a checkpoint is bit-exact, and the
selection of every cycle is a function of the seed and the cycle index alone.
Training itself is not bit-reproducible run to run on CUDA even with `compile`
and `allow_tf32` off: two cycle-0 smoke runs with the same seed diverged at the
fifth digit of the loss after the first optimizer step (nondeterministic
backward kernels, the same property as `train_adota.py`). Compare strategies
across seeds, not against a re-run of the same seed.
