# `difficulty_independence.py`: is the difficulty score model-independent?

The closing analysis of EXP-0014. The input-only difficulty score is fitted to the error of
dose engines, so "this beamlet is difficult" could mean only "this model got it wrong". This
script fits the score to all members of a ladder but one, and asks how well that predicts the
left-out member compared with a score fitted to the member itself. It trains nothing and runs
no model: it reads per-record errors and analytic features, and runs on the CPU in minutes.

```bash
uv run python scripts/active_learning/difficulty_independence.py \
    --config scripts/active_learning/config_difficulty_independence.yaml
```

## Inputs

| Config key | What it is | Made by |
|---|---|---|
| `ladder` | Per-record errors of the target members (name to CSV) | `ensemble_agreement.py` with `config_independence_v3_ladder_{a,b}.yaml` |
| `gate_members` | The members that pass the quality gate, for a sensitivity run | Chosen from the members' pass rates |
| `external_f` | Errors on F of engines outside the target | `ensemble_agreement.py` with `config_independence_v3_external.yaml` |
| `pool_members_dir`, `pool_external` | Every engine's errors on the pool's test split | `ensemble_agreement.py` with `config_independence_pool.yaml` |
| `v3_features`, `pool_features` | Analytic features (CT, flux and energy only) | `acquisition_input_only_features.py --mode analytic` |
| `v3_index`, `pool_index` | Record indexes, which give each record its scan | `build_beamlet_h5.py`, `al_build_beamlet_pool.py export` |
| `v3_fit_ids`, `v3_f_ids` | The records of the fit scans and of F | The EXP-0014 partition |

The population is the beamlets with the analytic peak inside the crop. The score is fitted on
the fit scans only. The pool's test split is the primary test set and F the secondary. The
pool's train and validation splits must never be used to fit the score: they are the active
learning pool and its validation set.

## The verdict

For each left-out member, gap = Pearson(score fitted to it) minus Pearson(score fitted to the
others), both against that member's error. The endpoint is the median gap over members, with
an interval from a bootstrap over scans.

| `verdict()` | Rule |
|---|---|
| `model-independent` | Median gap at most 0.03, upper bound at most 0.05 |
| `approximately model-independent` | Median gap at most 0.10, upper bound above 0.05 |
| `model-specific` | Median gap above 0.10 |

The thresholds were fixed before any number existed. Do not move them after a run.

## Outputs

In a timestamped `difficulty_independence_<date>_<time>/` under `output_dir`:

- **`summary.json`**: the verdict, the primary gap on the pool and the secondary gap on F.
- **`lomo_summary.csv`, `lomo_members.csv`**: the gap per test set, score variant and member
  set, including the pool's anatomies.
- **`transfer.csv`**: Pearson of scorer v2, the deployed v1 and energy alone against each error.
- **`cv_gap_pool.csv`**: the gap with both scores fitted inside the pool, for every engine.
- **`spread.csv`**: how much the members disagree, and the share of their variance that is shared.
- **`pool_scores.csv`**: scorer v2 and the shared target per pool beamlet.
- **`scorer_v2_fit.json`**: fitted on the fit scans only. This is the evaluated scorer, and the one
  vendored as `src/acquisition/data/analytic_scorer_v2_exp0014.json`.
- **`scorer_v2_unseen.json`**: fitted on the fit scans plus F. It is not evaluated anywhere.
- **`config.yaml`, `manifest.json`**: the resolved config, the commit and the inputs. Run from a
  checkout with no untracked files, or the manifest records `dirty: true`.

## On other inputs

There is no `--dry-run`, and this script has so far run on one data family only, so check the
first run's `difficulty_independence.log` carefully.

1. Evaluate every engine on the same records with `ensemble_agreement.py`, once for the fit
   records and once for each test set. A member must not have trained on any of them.
2. Compute analytic features for the same records.
3. Point the config at those files, and name your members under `ladder`.
4. Keep the score's fit records and every test set in separate scans. This script does not check
   whether a member trained on them: that check is yours.

## Limits

- **All ladder members trained on the same 30 scans,** so the gap measures independence from the
  architecture and hyperparameters, not from the training data. Cross-fitting was not run.
- **The reference is shared.** The pool's Monte Carlo at 1e6 primaries is in every engine's error;
  [difficulty_noise_floor.md](difficulty_noise_floor.md) tests what that does to the verdict.
- **RDE carries a normalisation.** Every RDE contains the ratio of dose volume to maximum, which is
  also strongly correlated with the score. The gap may partly reflect that ratio rather than a
  shared notion of difficulty; see the open question in EXP-0014.
- **The gap is not accuracy.** A small gap says the choice of target model matters little. It does
  not say the score predicts gamma or range errors, and it says nothing about whether selecting
  by the score helps a model.
