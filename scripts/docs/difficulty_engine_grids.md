# `difficulty_engine_grids.py`: the difficulty score against dose engines

Does a score computed from a beamlet's CT, flux and energy alone predict how badly dose
engines handle that beamlet? This script answers it on saved beamlet grids, and draws one
slide that follows a beamlet from its input to each engine's dose and error.

It tests the score as an **estimator**. It does not show that selecting by the score
improves a model per Monte Carlo second: only an active-learning loop against random
selection can show that.

## Run

```bash
export TMPDIR=/scratch/mstryja/tmp
uv run python scripts/active_learning/difficulty_engine_grids.py \
    --config scripts/active_learning/config_difficulty_engine_grids.yaml --dry-run   # the selection only
uv run python scripts/active_learning/difficulty_engine_grids.py \
    --config scripts/active_learning/config_difficulty_engine_grids.yaml
```

`--limit-blocks` and `--limit-beamlets` give a smoke run. Point a smoke run at a copy made of
symlinks, because the `adota` stage adds files to its sources.

## Stages

| Stage | What it does | Where it writes |
|---|---|---|
| `adota` | Adds each ADoTA engine's prediction where a beamlet has none | Source directory, as `{id}{suffix}` plus a provenance file |
| `score` | Gamma per engine against the saved MC, then MAPE on the 1 mm crop and the 2 mm grid | `<output_root>/<block>/metrics.json`, `mape.csv` |
| `difficulty` | Score from the input alone, its percentile in a reference population, the example beamlets | `features_analytic.csv`, `difficulty.csv`, `examples.csv` |
| `analyse` | Within-field and pooled Spearman, per-decile tables, per-engine summary | `analysis/` |
| `figure` | Story figure, one figure per trend (gamma, MAPE), the numbers behind them | `figures/`, in `figure.formats` |

Files are only ever added to a source directory, each created exclusively, so a rerun never
changes an existing dose.

## Use it on your own data

1. List your beamlet directories under `sources`. Each needs `{id}_ct.npy`, `{id}_flux.npy`,
   `{id}_ds.npy` and `{id}_sim_res.json`, with stems `aXX_YY`. Leave out any scan an engine
   trained on: nothing here checks that for you.
2. Set `selection.grid_n` and `selection.min_beamlets`. `min_beamlets` equal to `grid_n`
   squared keeps only complete grids.
3. List `engines`. A deployed model takes `model_dir` and `fname`; a training run takes
   `checkpoint` and `model_config`, whose `scale` is then used. An engine whose doses already
   exist needs only `name`, `kind` and `suffix`. `label` is its name in the figures.
4. Remove `seed_metrics_from` unless you have gamma results from this same code and criteria.
5. Set `generation_config` to the config that produced the beamlets: it names the BDL used to
   convert dose to Gy per MU.

## How to read the outputs

- **`d_score`** runs from 0 to 1: the share of the reference beamlets
  (`difficulty.percentile_reference_features`) that score lower, so d = 0.9 is harder than 90% of
  them. `d_score_raw` is what the scorer returns, a linear fit to a standardised error, which is
  unbounded and can be negative. The two give the same order.
- **`analysis/spearman_summary.csv`** is the primary table. `median_within_field` compares
  beamlets of one field, the comparison selection relies on, because it rescales the score
  within an energy layer. A pass rate should fall with difficulty (negative), a MAPE rise
  (positive). `pooled_spearman` also mixes patients and energies.
- **`analysis/deciles.csv`** holds each metric per score decile within a field.
- **`figures/examples`**: the example beamlets come from the scores alone (`figure.examples`),
  before any error is read, and are written to `examples.csv` by the `difficulty` stage. A
  single example may not follow the trend; the trend panels carry the claim.

## Limits

- **Few patients.** The interval is a bootstrap over patients, so a handful of patients gives a
  wide one and supports no claim about an anatomy.
- **Beamlets without a score.** The score is defined only when the analytic peak lies inside the
  crop. Others have no score and leave the analysis; `analysis/population.json` gives the count.
- **Monte Carlo noise.** At 1e6 primaries the reference is noisy, and the score correlates with
  that noise (EXP-0014), so part of a correlation may come from the reference.
- **MAPE grid.** `mape_1mm` matches the gamma grid and is the headline. `mape_2mm` is on the
  model's own grid and is lower because averaging reduces noise.
