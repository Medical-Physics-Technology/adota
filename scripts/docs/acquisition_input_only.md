# Input-only difficulty features: validation against the reference study

Three scripts that establish whether the difficulty score can be computed for a
beamlet that has not been simulated. Background: the score of
`research/acquisition_function_final_summary.md` locates the Bragg peak with the
Monte Carlo dose; `src/acquisition/surrogate.py` replaces that dose with an
analytic one built from the CT, the flux and the energy. These scripts measure
what that substitution costs, on the study's own 69,290 reference beamlets, and
refit the score on the result. Experiment record: EXP-0006.

None of them loads a model or touches a GPU. The cost is HDF5 reads plus the
metric computation, so they fan out over a process pool with BLAS pinned to one
thread per process.

## 1. `acquisition_input_only_features.py`: compute both arms

```bash
uv run python scripts/analysis/acquisition_input_only_features.py \
    --out /scratch/<user>/adota_runs/acquisition_input_only/full --workers 8 --chunk 100
uv run python scripts/analysis/acquisition_input_only_features.py \
    --out .../subset --stride 230 --workers 4            # a quick 1-in-230 look
```

For every record it computes the thirty metrics twice, sharing the metric code
and differing only in the dose handed to it:

| arm | dose used to locate the peak and weight the edges | purpose |
|---|---|---|
| `gt` | the stored Monte Carlo dose, de-normalised as the study did | must reproduce the study; the current-physics baseline for the refit |
| `analytic` | `analytic_dose(ct, flux, energy)` | what a candidate gets at selection time |

In both arms the lateral 30x30 crop is centred where that arm's dose peaks, as
the training loader does. Each row also carries `peak_inside_crop` (does the
peak stop inside the 320 mm crop, by that arm's dose), the analytic-versus-gt
peak depth error in mm, and the crop-centre offset in voxels.

| option | default | meaning |
|---|---|---|
| `--out` | required | output directory, on `/scratch` |
| `--h5` | the reference set | HDF5 file of stored records |
| `--results` | the study's `results.csv` | defines the record ids and their order |
| `--stride`, `--limit` | 1, none | thin the record list for a quick run |
| `--workers`, `--chunk` | 1, 200 | process-pool size and records per task |
| `--mode` | `gt analytic` | which arms to compute |

Outputs: `features_gt.csv`, `features_analytic.csv`, `manifest.json`.

## 2. `acquisition_input_only_compare.py`: the gate and the agreement

```bash
uv run python scripts/analysis/acquisition_input_only_compare.py --features-dir .../full
```

Two questions, in order. **Does `gt` reproduce `results.csv`?** Every metric
that does not depend on stopping power must agree to 1e-3 absolute or 1e-4
relative; the script exits 1 otherwise, and nothing downstream should be
trusted. The seven RSP-based metrics (`wepl_*`, `pflugfelder_hi`, `isi_*`) are
reported separately: `results.csv` predates the unified stopping-power model of
1.3.0, so they legitimately differ by a few percent. **How far is `analytic`
from `gt`?** Per metric: Spearman rank agreement (what the percentile-normalised
score consumes), median and 95th percentile absolute difference. Reported for
all records and for the beamlets whose peak both arms place inside the crop,
with the confusion matrix of the two arms' `peak_inside_crop` verdicts.

Outputs: `reproduction.csv`, `agreement.csv`, `agreement_inside.csv`.

## 3. `acquisition_input_only_refit.py`: the study's fit, on the new features

```bash
uv run --with scikit-learn python scripts/analysis/acquisition_input_only_refit.py --features-dir .../full
```

scikit-learn is not in the project environment, hence `--with`. The protocol is
`acquisition_frozen_test.py`'s, unchanged: `log(1 + RDE)` target, percentile
features with grids from the training rows, Lasso (alpha 0.0008) for the sparse
score, ridge (alpha 1) for the full one, a gradient-boosted ceiling, five-fold
patient-grouped cross-validation on the development patients and one evaluation
on the same frozen seven test patients (`frozen_test_ids.csv`). Run on both
arms and two populations (all records; both-inside-crop), plus
leave-one-anatomy-out for the two linear variants, which is the criterion for
choosing the deployed variant.

Outputs: `refit_results.csv`, `anatomy_transfer.csv`, and `analytic_scorer.json`,
which `DifficultyScorer.load(path, variant, arm="analytic/both_inside_crop")`
reads.

## Requirements

The reference HDF5 set and the study's run directory
(`/scratch/mstryja/adota_runs/20260707_124010`, for `results.csv`,
`uuid_provenance_map.csv` and `frozen_test_ids.csv`). About two hours for the
full set with eight processes.
