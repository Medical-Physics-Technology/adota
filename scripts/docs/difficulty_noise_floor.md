# `difficulty_noise_floor.py`: the Monte Carlo noise floor under EXP-0014

The closing analysis (`difficulty_independence.py`) measures every engine against one
Monte Carlo run at 1e6 primaries. That run's noise is in every error and is shared by
every engine, so it can inflate their agreement. This script reads the paired noise set,
979 beamlets of 12 test-split patients simulated at both 1e6 and 1e7
([beamlet_pool.md](beamlet_pool.md)), and tests whether the verdict survives. It loads no
model; the GPU only runs the gamma of the 1e6 dose against the 1e7 one.

```bash
uv run python scripts/active_learning/difficulty_noise_floor.py \
    --config scripts/active_learning/config_difficulty_noise_floor.yaml --device-index 0
```

## Inputs

| Config key | What it is | Made by |
|---|---|---|
| `independence_config` | The fit scans, ladder and gate members | `config_difficulty_independence.yaml` |
| `scorer_v2` | The evaluated scorer | `difficulty_independence.py` (`scorer_v2_fit.json`) |
| `members_1e6`, `members_1e7` | Every engine's errors against each reference | `ensemble_agreement.py` with `config_independence_noise_{1e6,1e7}.yaml` |
| `h5_1e6`, `h5_1e7`, `index` | The exported noise set | `al_build_beamlet_pool.py export` |
| `features` | Analytic features (CT, flux and energy only, so one pass covers both) | `acquisition_input_only_features.py --mode analytic` |

The population is the beamlets with the analytic peak inside the crop, as in the closing
analysis. The bootstrap resamples patients (`pool_patient_id`), not `patient_key`, which
identifies one field.

## The four checks

The thresholds were fixed before the per-beamlet results existed. `outcome` is
`model-independent` when all four pass, otherwise `review`: no automatic label, and the
thresholds are not moved afterwards.

| Check (`checks.csv`) | Question | Passes if |
|---|---|---|
| `gap_against_1e7` | Does the leave-one-member-out gap hold against the cleaner reference? | `verdict()` says `model-independent`: median at most 0.03, upper bound at most 0.05 |
| `target_1e6_vs_1e7` | Does noise change which beamlets count as difficult? | Pearson of the shared target at least 0.95 |
| `agreement_drop` | Does a shared reference inflate agreement? | Mean pairwise Pearson of the ladder, against 1e6 minus across references, at most 0.02 |
| `score_vs_noise_partial` | Does scorer v2 track noise rather than difficulty? | Absolute partial Pearson with the floor, 1e7 target fixed, at most 0.20 |

With 12 patients the gap's interval can be wide, so a small median with a large upper bound
reads `approximately model-independent` under `verdict()`. The pool remains the primary
evidence for the gap.

## The hottest voxel

RDE and gamma are normalised by the reference's maximum, which one deposit in a low-density
voxel can set at 1e6. `summary.json` reports the 1e6/1e7 maximum ratio, how often the 1e6
maximum sits more than `spike_depth_mm` from the 1e7 one in depth, and how often it lies below
-500 HU. The test set `noise_1e6_robust_max` repeats the gap with `robust_percentile` of the
reference as the maximum, computed by rescaling the stored RDE without inference. This is a
sensitivity check and not one of the four checks; gamma is not recomputed.

## The exploratory outputs

`exploratory_correlations.csv`, `exploratory_deciles.csv` and `exploratory_target_overlap.csv`
were added after the first run failed `score_vs_noise_partial` (0.54 against 0.20) and are
not part of the rule; they cannot overturn a check. They separate the two things the floor
RDE mixes: noise, and the ratio of dose volume to maximum that every RDE carries. Scoring
them on beamlets that did not inform them would take new Monte Carlo.

## Outputs

`checks.csv`, `summary.json`, `per_beamlet.csv`, `engines.csv`, `lomo_members.csv`,
`lomo_summary.csv`, `agreement_pairs.csv`, `agreement_summary.csv`, `score_noise.csv`,
`transfer.csv`, `selection_overlap.csv` (top-k overlap of the deployed score and scorer v2,
on this set and on the pool's test split), `noise_floor_rde.{svg,pdf,png}`
(`src.figures.ensemble_agreement.noise_floor_figure`), `config.yaml`, `manifest.json`.

## Limits

- 1e7 still has about a third of the per-voxel noise of 1e6, so the checks somewhat
  understate its effect.
- The pass rates use the engines' criteria, 2%/2 mm with a 10% cutoff, not the 3%/3 mm
  headline, because the stored engine errors hold that criterion only.
