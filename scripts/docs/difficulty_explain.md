# `difficulty_explain.py`: why a beamlet gets its difficulty score

The deployed difficulty score (EXP-0014 v2) is linear in percentile-transformed features,
`score = b + sum_k w_k p_k(x)`. With a reference population R as background it splits exactly:

    score(x) - mean_R(score) = sum_k w_k (p_k(x) - mean_R p_k)

so every feature's contribution is exact, and equal to its SHAP value with R as background.
This script places named beamlets against R, decomposes their scores, sums the contributions
within groups of correlated features, contrasts pairs and checks that the models agree.
It trains nothing and changes no model. CPU only, about 2 minutes.

## Run

```bash
export TMPDIR=/scratch/mstryja/tmp
uv run python scripts/active_learning/difficulty_explain.py \
    --config scripts/active_learning/config_difficulty_explain.yaml

# any beamlets, any pairs (labels are yours):
uv run python scripts/active_learning/difficulty_explain.py \
    --example a=pool:c09e362a8d63f --example b=grid:<block>/<stem> --pair a:b \
    --member-fits <earlier run>/member_fits.json      # skips the 80 s member refit
```

A grid beamlet is `grid:<block>/<stem>`, a saved EXP-0021 beamlet under `sources.grid.root`.
A pool beamlet is `pool:<sample_id>`, a record of the pool test export. The analytic peak
must lie inside the crop, because the score is not defined otherwise.

## What it computes

| Step | Code | Note |
|---|---|---|
| Placement | `src.acquisition.explain.placement` | Mid-rank percentile and robust z, `(x - median) / (IQR/1.349)`, of each feature within R |
| Member fits | `src.acquisition.explain.members` | Score v2 is one ridge on the mean of five members' standardised `log(1 + RDE)`. A ridge is linear in its target, so the five members' own ridges on the same rows average to it. They are refitted from `config_difficulty_independence.yaml` and rejected unless the mean matches the deployed weights, intercept and grids to 1e-8. |
| Contributions | `src.acquisition.explain.contributions` | Exact per feature; every decomposition asserts `sum = score - baseline` to 1e-9, and the ensemble's contributions must equal the member mean |
| Groups | `src.acquisition.explain.groups` | Average linkage on `1 - |Spearman|` of the features' percentiles on R, cut at `group_threshold` (default 0.7). No within-cell centring, so the energy-to-depth link stays. |
| Contrasts | `pair_contrast` | Per pair, groups and features ranked by `harder - easier` contribution |
| Agreement | `analysis.agreement` | Per member and the sparse variant: top groups, overlap with the ensemble's, Spearman of the group differences |
| Profiles | `src.acquisition.explain.profiles` | Model-grid CT, flux and analytic dose; HU, flux-weighted HU, cumulative WEPL and IDD on the beam axis. Reproduces the saved features to 1e-12. |

## Outputs

One run directory `difficulty_explain_<timestamp>/` under `output_dir`:

- `summary.md`: one generated sentence per pair (numbers only) and the stored gamma pass rates.
  The physical reading is written by hand below it.
- `tables/`: `placement.csv`, `contributions_features.csv` and `contributions_groups.csv` (all
  scorers), `contrast_groups.csv`, `contrast_features.csv`, `agreement.csv`, `sum_check.csv`,
  `member_fit_check.csv`, `member_weights.csv`, `groups.csv`, `correlation_spearman.csv`,
  `example_errors.csv`, `features_by_example.{csv,tex}`.
- `figures/` (SVG, PDF, PNG, each with a `*_numbers.csv` beside it): `contributions_by_group`,
  `contrast_by_group`, `agreement_<pair>`, `feature_correlation`, `beam_path_<label>`.
- `member_fits.json`: the member ridges, reusable with `--member-fits`.

## Reading the results

- Read groups, not single features. Correlated features share credit and often carry large
  weights of opposite sign (`sobel_dw_edge_energy` +1.6 against `sobel_dw_mean` -1.2).
- "depth and energy" mixes range with geometric depth. Geometric depth includes the air gap
  in front of the patient and grows with low-density tissue on the path. Compare
  `wepl_mean` with `bp_range_max_mm` to tell these apart.
- `interface_bp_distance` returns the zone length when the peak zone has no interface, so a
  homogeneous zone looks like "interface far away". Its weight is positive, so this term
  favours homogeneous beamlets slightly. It is a definitional artefact.
- The beam-path marks are the region boundaries of `analyse_density_regions` and the location
  of `max_hu_gradient`, so they show where the features' values come from.
