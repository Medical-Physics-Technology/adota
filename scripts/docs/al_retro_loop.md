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
    --strategy random           --cycle0-run $C0 --device-index 0 &
uv run python scripts/al_retro_loop.py run --config scripts/config_al_retro_loop.yaml \
    --strategy score_topk       --cycle0-run $C0 --device-index 1 &
uv run python scripts/al_retro_loop.py run --config scripts/config_al_retro_loop.yaml \
    --strategy stratified_score --cycle0-run $C0 --device-index 2 &

# 4. The comparison, once the runs are complete.
uv run python scripts/al_compare.py --config scripts/config_al_compare.yaml \
    --run <random run> --run <score_topk run> --run <stratified_score run>
```

**Unattended:** `scripts/run_al_retro.sh` runs all four steps, queuing the
strategies over the GPUs in `GPUS` and comparing whatever finished:

```bash
nohup bash scripts/run_al_retro.sh > /scratch/mstryja/adota_runs/al_retro/d30/pilot.out 2>&1 & echo "PID: $!"
```

**Smoke test first.** `config_al_retro_smoke.yaml` runs the identical code path on a
400-record `D` (two cycles of two epochs, an 8-record evaluation subsample) under
`/scratch/mstryja/adota_runs/al_retro_smoke`, in minutes:

```bash
uv run python scripts/al_retro_loop.py splits --config scripts/config_al_retro_smoke.yaml
uv run python scripts/al_retro_loop.py cycle0 --config scripts/config_al_retro_smoke.yaml
uv run python scripts/al_retro_loop.py run    --config scripts/config_al_retro_smoke.yaml \
    --strategy stratified_score --cycle0-run <smoke cycle-0 run>
```

## What one cycle does

```
score the remaining pool (input only, through the scorer interface)
select N records with the named strategy
add them to the training set; assert |train| == |cycle-0 set| + c * N
train epochs_per_cycle epochs from the previous cycle's state (weights, optimizer,
    scheduler, RNG all restored)
validate; write the cycle manifest
```

- **Strategies** (`strategy`): `random` (uniform over the pool; the control),
  `score_topk` (the `N` highest scores), `stratified_score` (equal counts per score
  decile). They live in a registry in `src/active_learning/retrospective/sampling.py`;
  a fourth is one decorated function there and one config line.
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
`--n-cycles`. CLI > YAML > defaults, through `merge_config`.

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
| `scorer` | `{name: difficulty, ...}` | `scorer_path`, `variant`, `arm`, `n_workers`. |
| `gamma_params`, `gamma_cutoff_percent`, `gamma_backend` | 2%/2mm, 10, torch | Named in every metric filename. |
| `training` | the baseline model and optimizer | Any `TrainingConfig` field; the model is trained from scratch. |

## Outputs

A run directory is a standard training run directory: `manifest.json` (git hash,
GPU, dataset fingerprint, resolved config, the splits fingerprint, the cycle-0
checkpoint path and SHA-256, and one entry per cycle with the selected count, the
selection fingerprint, the timings and the full-set metrics), append-only
`metrics.jsonl`, `checkpoints/cycle_XX/last.pth` with full RNG state,
`validation/` with the per-sample CSVs, and `cycles/cycle_XX/` with
`pool_scores.csv`, `selection.csv` (ids and scores), `selection.json` and
`training_ids.csv`. The cycle-0 rows are copied into each strategy's log so every
run reads on its own.

`al_compare.py` writes, under a timestamped directory of `output_dir`:
`F1_training_curves`, `F2_quality_vs_n_train_gamma_<criteria>`,
`F3_quality_vs_epochs_gamma_<criteria>`, `F4_fingerprint_score_decile`,
`F4_fingerprint_energy` (SVG, PDF and PNG, with a CSV of the numbers beside each),
`summary_boundaries_gamma_<criteria>.csv`, `epochs_to_quality.csv` and
`consistency.json`. The figure functions are in `src/figures/al_curves.py`.

## Resume

Pass `--resume-dir <run dir>`: completed cycles are skipped (their selection and
checkpoint are reused) and an interrupted cycle restarts from its `last.pth`, which
restores the weights, the optimizer, the scheduler and the RNG. The strategy run
verifies, tensor by tensor, that the state it restores from the cycle-0 checkpoint
equals the file, and records the checkpoint's SHA-256 in its manifest.

What is and is not reproducible: restoring a checkpoint is bit-exact, and the
selection of every cycle is a function of the seed and the cycle index alone.
Training itself is not bit-reproducible run to run on CUDA even with `compile`
and `allow_tf32` off: two cycle-0 smoke runs with the same seed diverged at the
fifth digit of the loss after the first optimizer step (nondeterministic
backward kernels, the same property as `train_adota.py`). Compare strategies
across seeds, not against a re-run of the same seed.
