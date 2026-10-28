# `hpo_figures.py`: figures of a finished HPO run

`scripts/active_learning/hpo_figures.py` draws the results of an
`ensemble_hpo.py` run (EXP-0014). It trains nothing: the learning curves come
from the trials' `metrics.jsonl`, and the beamlet figures run each finalist's
`best.pth` forward on a handful of validation beamlets.

```bash
uv run python scripts/active_learning/hpo_figures.py \
    --config scripts/active_learning/config_hpo_figures.yaml
# named beamlets instead of the picks; GPU 1
uv run python scripts/active_learning/hpo_figures.py --config <yaml> \
    --sample-id 81efc122-60fa-464a-bf4d-79dbdd60938f --device-index 1
```

The finalists' error agreement reuses `ensemble_agreement.py`, with the
finalists' per-record CSVs as members (no inference, CPU, about a minute):

```bash
uv run python scripts/active_learning/ensemble_agreement.py \
    --config scripts/active_learning/config_ensemble_agreement_hpo.yaml
```

## Figures

| Figure | What it shows | Code |
|---|---|---|
| `learning_curves` | Training (top) and validation loss (bottom) per epoch, log scale, one column per mixer. Trials cut after the first rung are grey, those cut after the second are thin in the mixer's color, top-rung trials are bold and named. A star marks the checkpoint each member (C0, C1, X0, U0) was selected on; a cross marks a non-finite loss. | `src.figures.hpo.hpo_learning_curves_figure`, rules from `src.figures.al_curves.draw_cycle_rules` |
| `finalists_beamlet_<pick>_<id>_<gamma>` | One validation beamlet: the ground truth and every finalist in the axial and coronal planes through the ground-truth Bragg peak, over the CT, in percent of the ground-truth maximum on one scale; then the integrated depth doses and the lateral profiles at the ground-truth peak depth, all overlaid. Each row states that model's pass rate, MAPE and dR80 on the beamlet, read from its finalist CSV. | `src.figures.hpo.finalist_beamlet_figure` |
| `agreement_matrix_<metric>` (from `ensemble_agreement.py`) | Finalists by finalists: the median within-cell Spearman of the per-record errors (scan x 20 MeV cells). | `src.figures.ensemble_agreement.agreement_matrix_figure` |

The planes are the ones `src.figures.single_beam.publication_figure` calls
axial and sagittal; for the lateral beams of the dataset the second is the
coronal plane, which is the name used here.

## Choosing the beamlets

`sample_ids` (or repeated `--sample-id`) wins. Otherwise one beamlet per rule in
`picks`, from the finalists' per-record pass rates on the HPO validation
subsample:

- `typical`: the mean pass rate over the finalists closest to its median;
- `disagreement`: the largest spread of pass rate across the finalists;
- `hardest`: the lowest mean pass rate over the finalists.

## Guards

Each finalist is rebuilt from its trial configuration and its `best.pth`. On
every drawn beamlet its loss must match the `loss_mse` of its finalist CSV
within `loss_check_rtol` (1%), or the script stops: the panel then shows the
model the metrics describe.

## Outputs

```
<output_dir, default <hpo_dir>/analysis>/hpo_figures_<ts>/
  config.yaml  manifest.json  hpo_figures.log
  learning_curves.{svg,pdf,png}  learning_curves.csv  learning_curves_trials.csv
  finalists_beamlet_<pick>_<id8>_<gamma>.{svg,pdf,png}
  finalists_beamlet_<pick>_<id8>_<gamma>_idd.csv       depth-dose curves as plotted
  finalists_beamlet_<pick>_<id8>_<gamma>_profile.csv   lateral profiles as plotted
  finalists_beamlets_<gamma>.csv                       per beamlet and finalist: the metrics shown
```
