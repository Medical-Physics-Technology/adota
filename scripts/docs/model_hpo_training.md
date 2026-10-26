# `model_hpo_training.py`: HPO and training of one ADoTA model

`scripts/model_hpo_training.py` finds the best configuration of a single ADoTA
model, trains it, and evaluates it once on a held-out test set. It is general:
the record sets, the search space, the budgets and the metrics all come from the
YAML config (`scripts/config_model_hpo_training.yaml`), so the same script serves
a later dataset or model by changing the config.

```bash
# 1. search: Optuna TPE with median pruning, one worker per GPU
uv run python scripts/model_hpo_training.py search --config scripts/config_model_hpo_training.yaml
# resume an interrupted search (trials left RUNNING by a killed worker are marked FAIL)
uv run python scripts/model_hpo_training.py search --run-dir <output_dir>/model_hpo_<ts>
# 2. confirm: the top-k trials trained at full length, the winner chosen by a fixed rule
uv run python scripts/model_hpo_training.py confirm --run-dir <run>
# 3. final: the winner trained on the final training records
uv run python scripts/model_hpo_training.py final --run-dir <run> --device 0
# 4. test: the final model on the test records, once (a second call is refused without --force)
uv run python scripts/model_hpo_training.py test --run-dir <run> --device 0
```

`search` freezes the config into the run directory; `confirm`, `final` and
`test` read it back from there, so a run cannot change its inputs half way.

## Stages

| Stage | Trains on | Judged on | Output |
|---|---|---|---|
| search | `search_train` | per epoch: loss on a fixed subsample of `search_val` (pruning); at the end: the last checkpoint's loss on all of `search_val` (the objective) | `search/study.journal`, `search/trials.csv`, `search/trials/trial_<n>/` |
| confirm | `search_train`, `confirm.epochs` | last checkpoint on all of `search_val`: loss and gamma (`gamma_params`) | `confirm/results.csv`, `confirm/selection.json` |
| final | `final_train`, `final.epochs` | nothing: per-epoch loss on a subsample of its own training records, in-sample, for the curves only | `final/checkpoints/last.pth` |
| test | none | all of `test`, at every criterion of `test_gamma` | `test/test_<gamma>.csv`, `test/summary.json` |

**Winner rule (confirm), fixed before any run:** the lowest final validation
loss wins, unless another candidate's mean gamma pass rate is higher by more than
`confirm.gpr_guard_pp` points; then the candidate with the highest pass rate wins.

**Search.** Trials are proposed by Optuna's TPE sampler (multivariate, with
`constant_liar` so parallel workers do not propose the same point) and stopped
early by the median pruner: after `pruner_warmup_steps` epochs, a trial whose
loss is worse than the median of earlier trials at the same epoch is pruned.
A configuration over `search.max_params` trainable parameters (counted on the
`meta` device, nothing allocated) or with a head count that does not divide the
token size is rejected before training and does not count towards
`search.n_trials`. With several workers the budget can be exceeded by at most
the number of workers minus one.

## Record sets

Each entry of `id_sets` is a CSV with a `sample_id` column, or a filter
`{table, column, values}` on a table; relative paths are under `partition_dir`.
Every set loses the records of `exclude_indexes_path` and the beamlets above
`max_energy_mev` (`src.datasets.energy_cut.drop_above_energy`). The run stops if
two sets that must be disjoint (search train / search validation / test, final
train / test) share a scan (`scan_table`). The kept ids are written to `ids/`,
with the counts in `ids/counts.csv`.

With the default config: the EXP-0014 partition, searching on its HPO split of R
(30 / 8 scans), training the final model on every scan outside the frozen test
set F (R and V), and testing on F.

## The learning-rate schedule, and why

Every run, trial or final, trains from scratch with a linear warmup of
`warmup_epochs` epochs and a cosine decay of the learning rate over the run's own
epochs (`lr_schedule: warmup_cosine`, in
`src.active_learning.retrospective.lr_schedule.fixed_lr`), and keeps the **last**
checkpoint. The rate is set once per epoch; warmup epoch `e` trains at
`lr0 * (e + 1) / (W + 1)` and the cosine stops one step short of zero, so every
epoch trains. (The AL loop's `cosine_per_cycle` starts its warmup and ends its
cycle at exactly the minimum rate, which is right for a cycle that continues a
trained model but costs a from-scratch run two epochs at a learning rate of 0. A
run directory frozen before `lr_schedule` became a key keeps that legacy schedule
when resumed.) The reasons for the schedule are measured, not assumed:

- **A constant learning rate was unstable at the rates that win.** In the
  EXP-0014 ensemble HPO (constant rate), the training loss blew up to more than
  twice the previous epoch three times, and two trials diverged; four of these
  five failures happened at learning rates of 7.8e-4 or above, the range the
  selection favoured.
- **The validation loss of the last epochs was too noisy to rank on.** Over the
  last 9 of 27 epochs its coefficient of variation was 0.39 to 0.90 for the
  transformer and mLSTM trials, larger than the differences between the leading
  trials. A decay to zero shrinks the steps, so the last epochs settle.
- **The best single epoch rewards luck.** A best-so-far objective picks a
  favourable fluctuation (the EXP-0014 transformer member was selected one epoch
  before its blow-up); with the decay, the last checkpoint is the run's result
  and no held-out data is used to pick an epoch, which is also what lets the
  final model train without a validation set.
- **A warm restart is not used.** EXP-0011 showed that restarting a cosine at
  full rate multiplies the training loss 30 to 70-fold; here every run has one
  decay, with a linear warmup from zero (EXP-0012), so there is no jump.
- **Constant rates were tried and kept for the loop, not for selection.**
  EXP-0013 (constant rate in the AL loop) had no divergence at 5e-4 but noisier
  checkpoints, consistent with the above.
- Standard practice: cosine decay (Loshchilov and Hutter, SGDR, ICLR 2017) and
  linear warmup (Goyal et al., arXiv:1706.02677).

Because the schedule is tied to the run's length, a search trial is never
resumed into a longer run: `confirm` trains the top trials again from scratch at
full length.

## Gradient clipping and final-run deviations

`training.max_grad_norm` (default `null`, off) clips the global L2 norm of the
gradients to that value before every optimizer step
(`torch.nn.utils.clip_grad_norm_` in `src.training.loop.train_one_epoch`). Every
epoch's `metrics.jsonl` row now also records the median and maximum of the
per-step gradient norms (measured before clipping, so they describe the training
either way) and the number of clipped steps (`n_clipped`); the median and maximum
are what a threshold should be chosen from.

The final run trains on 1.7 times as many beamlets per epoch as the confirmation
runs, so it spends more steps at the peak learning rate. If it proves unstable,
it can be repeated with clipping and a warmup:

```bash
uv run python scripts/model_hpo_training.py final --run-dir <run> --device 0 \
    --restart --max-grad-norm <X> --warmup-epochs <W>
```

`--max-grad-norm` and `--warmup-epochs` are written into the run's frozen
`config.yaml` under `final:` and logged as a deviation from the confirmed
configuration (`stage_overrides`). `--restart` moves an existing `final/` to
`final_aborted_<ts>/` rather than deleting it; without it, `final` refuses to
overwrite a final training.

## What is fixed and why

| Fixed | Why |
|---|---|
| `norm_layer: batch` | without normalisation and with several conv steps the output is exactly zero at initialisation (a loss of 1.0 with no gradient); `group` would apply GroupNorm to the single output channel, standardising each sample's whole dose field (`src/adota/layers/encoder_decoder.py`, last decoder block) |
| `num_forward: 2` | v11's attention look-ahead; nearly a no-op, since the 3D convolutions already see 4 to 30 slices downstream |
| `weight_init: default` | with normalisation after every convolution, the init only rescales the effective step size, confounded with the learning rate |
| `causal: true` | the beam-direction prior of ADoTA: a slice attends only upstream |
| `transformer_residual`, `conv_residual` | ablation switches; turning them off is known to hurt |
| `zero_padding`, `last_activation` | shape plumbing and the output clamp, not capacity |
| `batch_size` | changes epoch time and memory, so trials would not be compared at equal cost |
| loss weights 0.7 / 0.3, static | they define the objective; the script refuses adaptive weights |

## Smoke test

```bash
uv run python scripts/model_hpo_training.py search --config scripts/config_model_hpo_training.yaml \
  --set output_dir=/scratch/mstryja/adota_runs/model_hpo_smoke --set max_records=400 --set epoch_val_size=64 \
  --set "devices=[1,2]" --set search.n_trials=3 --set search.epochs=2 --set search.n_startup_trials=2 \
  --set search.pruner_startup_trials=1 --set search.pruner_warmup_steps=0 --set training.compile=false \
  --set training.num_workers=4 --set confirm.top_k=2 --set confirm.epochs=2 --set final.epochs=2
# then confirm / final / test with --run-dir as above; about 10 minutes in all
```

`search.epochs` must exceed the largest `warmup_epochs` of the space.

**Fixed trials.** `search.fixed_trials` lists configurations (a `label` and a value
for every searched parameter) queued ahead of the sampler's own when a search
starts; they count towards `n_trials`. The default config queues v11's
architecture. **The parameter cap** counts all trainable parameters; most of them
are the element-wise LayerNorm weights of the conv blocks (92% of v11's
14.9M), so in practice the cap bounds the conv hidden width and the model size,
not the conv or transformer capacity.

## Outputs

```
<output_dir>/model_hpo_<ts>/
  config.yaml  manifest.json  model_hpo_training.log
  ids/  search_train.csv  search_val.csv  search_val_subsample.csv  final_train.csv
        final_train_monitor.csv  test.csv  counts.csv
  search/  study.journal  trials.csv  worker_<k>.log  trials/trial_<n>/{config.yaml, metrics.jsonl, result.json, checkpoints/}
  confirm/ trial_<n>/{..., final_val_<gamma>.csv}  results.csv  selection.json
  final/   config.yaml  metrics.jsonl  result.json  checkpoints/last.pth
  test/    test_<gamma>.csv (one per criterion)  summary.json
```

Pruned trials keep their curves (`metrics.jsonl`) but not their checkpoints.
