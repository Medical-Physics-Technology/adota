# `ensemble_hpo.py`: hyperparameter optimisation of the EXP-0014 members

`scripts/active_learning/ensemble_hpo.py` tunes one ADoTA configuration per
sequence mixer (`transformer`, `mlstm`, `none`) with the same budget for each,
and picks the EXP-0014 members from the result: C0 (best transformer), C1 (a
promoted transformer far from C0), X0 (best mLSTM) and U0 (best plain U-Net).
It trains and validates on R only, the HPO split of the EXP-0014 partition
(`scripts/docs/cross_fit_partition.md`); no F or V scan enters.

```bash
uv run python scripts/active_learning/ensemble_hpo.py run \
    --config scripts/active_learning/config_ensemble_hpo.yaml
# continue an interrupted run; finished segments are kept
uv run python scripts/active_learning/ensemble_hpo.py run --resume <output_dir>/hpo_<ts>
```

## How the search runs

| Step | What happens |
|---|---|
| Trials | per mixer, `n_trials` scrambled-Sobol points over the shared space plus the mixer's own; any over `param_cap_factor` x the reference model (v11) is skipped for the next point. `fixed_trials` (v11's configuration) take the first places. |
| Memory check | one full-batch training step per trial, before any training; the run stops if one does not fit |
| Successive halving | rungs at `min_epochs * eta**r` cumulative epochs (3, 9, 27). All trials train to rung 0; the best `1/eta` of each mixer continue from their checkpoint. A rung is promoted only once every trial in it finished, so the result does not depend on GPU timing. |
| Objective | the lowest validation loss so far (static 0.7 / 0.3 MSE / IDD weights), on a fixed seeded subsample (`val_subsample_size`) of the HPO validation scans |
| Finalists | every top-rung trial and C1: gamma (2%/2mm, 10% cutoff), MAPE, RDE and dR80 of `best.pth` on the same subsample |
| Selection | best of the top rung per mixer; C1 = the transformer trial promoted at least once that is farthest from C0 in the unit cube of the space |

Every trial trains from scratch with seed `training.seed`, a constant learning
rate and the EXP-0012 recipe otherwise. Why Sobol and not Bayesian
optimisation: the trials double as the pool C1 is drawn from, and BO trials
cluster around one optimum, which would make C1 a near-copy of C0.

With the defaults (24 trials, rungs 3 / 9 / 27, 24 -> 8 -> 2 trials) a mixer
costs 156 epochs. At about 5.5 min per epoch on an A40 (4.5 min of training on
35,985 records, 1 min of validation on 3,000) the three mixers take about 45
GPU-hours, about a day on two GPUs.

## Workers

The driver runs every segment, memory check and finalist evaluation as its own
process on one GPU of `devices` (`CUDA_VISIBLE_DEVICES` set per worker), so an
out-of-memory error or a non-finite loss ends one trial (it ranks last), not
the search. The driver rebuilds its state from the files every 10 s; killing it
and running `--resume` continues where it stopped. A killed worker leaves its
trial at the last finished epoch, and the next segment resumes from
`last.pth`.

## Smoke test

```bash
uv run python scripts/active_learning/ensemble_hpo.py run \
    --set output_dir=/scratch/mstryja/adota_runs/exp0014/hpo_smoke \
    --set n_trials=3 --set min_epochs=1 --set n_rungs=2 \
    --set max_train_records=300 --set val_subsample_size=100 --set training.compile=false
```

## Outputs

```
<output_dir>/hpo_<ts>/
  config.yaml  manifest.json  hpo.log  memcheck.log  reference.json
  ids/train_ids.csv  ids/val_ids.csv
  trials.json                         every trial: mixer, parameters, unit point, parameter count
  trials/<trial>/
    trial.json  config.yaml           the trial and its resolved TrainingConfig
    checkpoints/best.pth  last.pth    the trained model (full training state)
    metrics.jsonl  worker.log  memcheck.json
    segment_<epochs>.json             objective at each rung reached
    finalist_<gamma>.csv / .json      per-record metrics and summary, finalists only
  results.csv                         one row per trial: parameters, objective per rung, finalist metrics
  selection.json                      C0, C1, X0, U0: trial, parameters, config, checkpoint, metrics
  summary.json
```
