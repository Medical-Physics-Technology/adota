# `beamlet_hardness.py`: what makes a beamlet hard, per dose engine?

`scripts/active_learning/beamlet_hardness.py` asks which input-only physics
properties of a beamlet are associated with its error, which of them every dose
engine shares, and which belong to one architecture or to the pencil beam. It is
the C1 (model-independence) test of the difficulty score in small, part of
EXP-0014.

```bash
uv run python scripts/active_learning/beamlet_hardness.py \
    --config scripts/active_learning/config_beamlet_hardness.yaml --device-index 0
```

On one GPU the full run takes about 2.5 h: about 1 h 50 min to score the 11,531
table B beamlets with eight engines, where gamma (about 60 ms per record and
engine) is the cost, 5 min of metrics on CPU, about 30 min of analysis. Set
`device_indices: [0, 1, 2]` to shard table B over several GPUs. `--smoke` runs
every stage on subsets (400 + 900 beamlets, few bootstrap draws) in about 15 min.
`--tables-dir <run>/tables --stages analyse,examples,report` reruns the analysis
on tables already built.

## Inputs: the two tables

| Table | Beamlets | Metrics | Errors |
|---|---|---|---|
| A | the 3,000 HPO validation records (`record_ids`) of the v3 HDF5 | `record_features(..., "analytic")`: crop at the analytic peak, every energy kept | the engines' finalist CSVs (or inferred from `checkpoint` + `model_config`) |
| B | the 11,531 beamlets of the engine comparison (`comparison_csv`) | the saved `_ct.npy`, `_flux.npy` through `model_grid_inputs` + `analytic_features`, the path a candidate takes | every engine on the 2 mm grid against the saved Monte Carlo dose: neural engines run on the saved inputs, `saved_dose` engines (`_ds_pba`, `_ds_pred`) resampled like the Monte Carlo dose |

Every error is computed by `record_metrics`, the function the HPO finalists were
scored with: gamma failure (1 - pass rate; 2%/2 mm, 10% cutoff, global), MAPE,
relative dose error, |dR80|. Table B keeps the comparison file's own 1 mm errors
as `file_<engine>_<column>` beside the recomputed ones.

Checks before anything is analysed: no engine of table A trained on a validation
scan (`table_a_held_out.csv`); no table B CT shares its grid (x size, z size, z
origin) with a scan of the training HDF5 (`table_b_geometry_leakage.csv`).

## Engines are config entries

```yaml
engines:
  - {label: C0, trial: transformer_021, family: transformer, consensus: true}
  - {label: PBA, saved_dose: _ds_pba, family: pencil_beam}
residuals: [U0, PBA, ADoTA-v11]
```

A `trial` resolves inside `hpo_dir` to its `best.pth`, `config.yaml` and finalist
CSV. For retrained members, give `checkpoint` + `model_config` (and `errors_csv`
if it exists) and rerun unchanged. `consensus: true` marks the engines whose mean
within-cell error rank is "hard for everyone"; `residuals` lists the engines whose
rank minus that consensus is fitted ("hard specifically for this engine").
`tables` restricts an engine to `[A]` or `[B]`.

## What it computes

Everything happens inside **cells**: one scan times one 20 MeV energy bin, at
least `min_cell_size` beamlets, the cells of `ensemble_agreement.py`. Intervals
are 95% bootstraps over scans with the same draws as that script. The population
is the deployed score's: the analytic peak inside the 320 mm crop and at most
`max_energy_mev` (250 MeV, MCsquare's nuclear-data limit).

| Step | Output (`results/`) |
|---|---|
| sanity: C0 against X0, within-cell gamma Spearman (agreement analysis: 0.82) | `summary.json` |
| 1. median within-cell Spearman of every metric and cluster with every target | `univariate_<table>_<error>.csv`, heatmaps |
| 2. clusters: average linkage on 1 - \|within-cell Spearman\|, cut at `cluster_threshold` | `clusters.csv` (table A), `clusters_table_B_own.csv`, adjusted Rand in the summary |
| 3. the score refit per target, leave one scan out: percentile metrics (grids from the training scans), centred per cell, standardised; ridge (alpha 1) on the centred within-cell rank | `fit_evaluation_*`, `coefficients_*` (with bootstrap intervals), `importance_*` (permuting one cluster inside cells), `scorers_*.json` (full fits, readable by `DifficultyScorer.load(arm=<target>)`) |
| 4. consensus and residual targets go through step 3 like any engine | same files |
| 5. cross transfer: each engine-fitted score against every engine's error | `transfer_*`, `transfer_summary_*` (diagonal against off-diagonal) |
| 6. strata (energy terciles, sites; cells cut at the boundary) for the primary error | `strata_{univariate,evaluation,importance,populations}_*` |
| sensitivity: the consensus and residual fits on every beamlet | `sensitivity_all_*` |
| 7. examples: three high and three low beamlets of one cell for the two most important consensus clusters, through `finalist_beamlet_figure` | `examples/` |

`figures/slide_cluster_importance_<error>` is the slide figure: the top clusters
by importance, one bar per entry of `slide.bars`, in the HPO figures' colours.
`what_makes_a_beamlet_hard.md` opens with the three sentences generated from the
results, then one table per step and the limitations.

Associations inside cells are not causes; the report words them as such.
