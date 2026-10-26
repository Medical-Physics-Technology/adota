# `al_compare_experiments.py`: two retrospective AL experiments, run by run

> **Reproducibility / portability.** The paths in the examples are the original
> development environment's locations. Point the config at your own run
> directories, and keep every output under a scratch directory of your choice.

[`al_compare.py`](../al_compare.py) compares the sampling strategies *inside* one
retrospective active-learning experiment ([al_retro_loop.md](al_retro_loop.md)).
`scripts/al_compare_experiments.py` compares **two experiments** that differ in one
training setting and share everything else, for example EXP-0012 (warmup plus
cosine) against EXP-0013 (constant learning rate):

```bash
uv run python scripts/al_compare_experiments.py --config scripts/config_al_compare_experiments.yaml
# another output location
uv run python scripts/al_compare_experiments.py --config <yaml> --output-dir /scratch/<you>/compare
```

It reads only the run directories the loop already wrote; nothing is trained or
evaluated, and it runs on CPU in a few minutes.

## When a comparison is valid

The strategies of the retrospective loop are model-free (a seeded uniform draw, a
static input-only score, their mixture). So two experiments that share the splits,
the cycle-0 checkpoint, the evaluation subsample and the cycle sizes select the
**same records in the same order** for the same `(strategy, seed)`, and each such
pair of runs differs only in the training setting. The script:

1. checks that all runs of both experiments are comparable, with the same check as
   `al_compare.py` (`src.active_learning.retrospective.compare.check_consistency`);
2. pairs the runs on `(strategy, seed)`, and stops if a run has no partner;
3. compares the per-cycle `selection.csv` and `training_ids.csv` of every pair and
   writes the result to `selection_identity.csv`; it warns when a pair did not
   select identically, because that pair then differs in more than the setting.

## What it computes

Per run and cycle (`cycle_table.csv`):

- **boundary metrics** (`full_*`): the gamma pass rate (mean, 5th percentile, share
  below 95%), MAPE and the 95th percentile of |dR80| on all of V at the last epoch of
  the cycle;
- **trajectory metrics** (`traj_*`): for each run, the **median of the last
  `trajectory_last_k` subsample evaluations** of the cycle (3 by default). It does not
  depend on which single epoch the cycle ended on;
- **snapshot noise**: the boundary validation loss over the cycle's minimum
  (`val_last_over_min`) and the cycle's peak over its minimum (`val_peak_over_min`);
- **divergence**: whether the training loss rose by more than `divergence_ratio`
  (2 by default) from one epoch to the next, the rule of `al_compare.py`.

Per pair and cycle, B minus A (`paired_deltas.csv`); and per cycle and metric
(`paired_summary.csv`): the mean of each experiment over its runs (`mean_a`,
`mean_b`), the mean, minimum and maximum of the paired difference, and in how many
pairs B was better (`n_b_better` of `n_pairs`). "Better" means higher for pass
rates and lower for losses, MAPE, range errors and the noise ratios.

## Figures

Each figure is written as SVG, PDF and PNG with the CSV it draws, under `figures/`
(`src.figures.al_experiment_comparison`, house style of `src.figures.al_curves`):

| Figure | Shows |
|---|---|
| `G1_schedule_and_val_loss` | the learning rate and the validation loss against the cumulative epoch, median over runs with the min-max band, A and B overlaid |
| `G2_paired_quality_<gamma>` | per metric, the mean over runs per cycle (top) and the paired difference B minus A per cycle (bottom) |
| `G3_strategies_cycle<N>_<gamma>` | every run at the last cycle, grouped by strategy, A and B side by side, one marker per seed |
| `G4_stability` | diverged runs and the boundary validation loss over the cycle's minimum, per cycle |

## Config (`config_al_compare_experiments.yaml`)

| Key | Meaning |
|---|---|
| `experiments` | exactly two, A first: `name`, `legend`, `runs_dir`, `pattern` (a glob over run directories; the cycle-0 baseline run is dropped) |
| `output_dir` | parent of the timestamped `al_compare_experiments_<ts>/` directory |
| `title` | figure title prefix |
| `trajectory_last_k` | how many of a cycle's last subsample evaluations the trajectory median takes |
| `divergence_ratio` | the epoch-to-epoch training-loss rise that flags a cycle as diverged |

## Outputs

```
<output_dir>/al_compare_experiments_<ts>/
  al_compare_experiments.log  consistency.json
  selection_identity.csv  cycle_table.csv  paired_deltas.csv  paired_summary.csv
  figures/  G1..G4 as .svg / .pdf / .png, each with its .csv
```

## EXP-0012 against EXP-0013

The default config compares EXP-0012 (5-epoch warmup, cosine 5e-4 to 5e-5 per cycle)
with EXP-0013 (constant 1e-4 after the shared cycle 0). Its run of 2026-10-26
(`/scratch/mstryja/adota_runs/al_retro_v3/d30_patient/compare_exp0012_vs_exp0013/al_compare_experiments_20261026_131236`)
is the source of the numbers in the EXP-0013 record: 45 of 45 run-cycles selected
identically, 12 diverged run-cycles in EXP-0012 against 0 in EXP-0013.
