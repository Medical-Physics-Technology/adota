# Active-learning pipeline: design

Status: design, 2026-09-05. Decisions taken with the supervisor on 2026-09-04;
the slots marked **to measure** are filled after the first runs, not guessed.
Companion documents: `research/sampling_architecture.md` (the earlier framing,
superseded where this disagrees), `research/publication_plan.md` (the results
blueprint), and the difficulty score that this loop selects with
(`research/acquisition_function_final_summary.md`, rebuilt input-only in
`src/acquisition/`, record EXP-0006).

## 1. What the loop is for

ADoTA is trained on Monte Carlo beamlets that cost roughly 20 s each to
simulate. The loop asks: given a pool of patient CTs the model has never seen,
which beamlets are worth simulating next, and does choosing them by an
input-only difficulty score reach a target accuracy with fewer simulations than
choosing them at random? The unit of cost is **Monte Carlo seconds**, not sample
count, because that is what the clinic pays.

Two starting points, both required, because they answer different questions:

- **`init: scratch`**: train from random weights on the designated training set
  until the validation trigger fires, then start sampling. Shows what the loop
  does for a model that is still learning the physics.
- **`init: warm`**: start from the deployed checkpoint
  (`models/DoTA_v3_grid_search_v11`). Shows the added value of active learning
  for a model that is already good, which is the deployment scenario.

Each starting point runs with two sampling strategies at **equal Monte Carlo
budget**: random and score-based. Random is not a straw man; it is the control
that most published active-learning results fail to beat.

## 2. Data

### 2.1 The CT pool

Four anatomies, at least 20 patients each, none used for training:

| anatomy | collection | location | status |
|---|---|---|---|
| thoracic | Lung-PET-CT-Dx | `/data/lung_cancer_dataset/manifest-1608669183333` (356 patients) | available |
| pelvic | StageII-Colorectal-CT | `/scratch/mstryja/manifest-1646429317311` (231 patients) | available |
| head and neck | to be delivered | | pending |
| brain | to be delivered | | pending |

Leakage rule for the two available collections: the first roughly fifty
patients of each sorted collection were used for training, the last thirty are
safe. Ten of those sixty are already claimed as robustness test patients
(`registry/robustness_clean_selection.csv`). From the remaining fifty: **five
per anatomy become validation CTs, the rest the pool**, both recorded in the
registry so they never mix. The pool is dataset-agnostic: a CT enters with an
anatomy label and a role, and the loop stratifies over whatever anatomies are
present, so the two pending collections plug in without code changes.

### 2.2 The validation set

New, generated once before cycle 1, frozen, at least **20,000** Monte Carlo
beamlets, **balanced in difficulty**: candidates are generated on the validation
CTs, scored with the input-only score, and drawn with equal counts per score
decile, stratified by anatomy and energy layer. A model that only improves on
easy beamlets cannot hide in such a set, and per-decile learning curves come for
free. The score decides what the yardstick contains, so any bias in the score is
a bias in the yardstick; the existing held-out sets (`testset_pelvis`, 11,580
records; `testset_Lung-PET-CT-Dx_v3`, 2,008) remain as an independent second
reading, and are also the regression guard against forgetting the original
distribution.

Cost: roughly 35 to 40 hours of Monte Carlo in beamlet mode on one node, or a
DelftBlue array. Versioned per anatomy, so head-and-neck and brain extend it
rather than replace it.

### 2.3 The training set

The designated training set is the reference HDF5 file. Newly labelled
beamlets are Monte Carlo output directories at 1 mm, and they join training
through a `DirBeamletDataset` that reuses `get_single_record` (the same
trilinear resample to the model's 2 mm grid that inference uses) inside a
`ConcatDataset` with the HDF5 set. No HDF5 rewriting; the training set is a
list of sources in the cycle manifest.

## 3. Candidates and their validity

A candidate is `(CT, gantry, energy, theta_x, theta_y)`. Generation, version 0:

- **gantry**: uniform random per candidate (rotate-to-canonical handles it; the
  model consumes a canonical beam's-eye frame and gantry is metadata);
- **isocenter**: the grid centre, as in every generated dataset so far;
- **energy**: a discrete layer set spanning the training range;
- **steering**: uniform on the generator's lattice within the screened
  plus or minus 1.5 degrees.

Version 1 moves to plan-like placement (body-mask, then segmentation-based
isocenters, clinical gantry arcs). Both realism modes are kept as an ablation.

**Validity is decided before Monte Carlo, from the inputs**, by
`score_candidates`: a ray that misses the CT is `roi_out_of_bounds`; an
analytic peak that leaves the 320 mm crop is `peak_outside_crop`. The second
gate is the input-only range check EXP-0006 validated at 97 percent agreement
with the Monte Carlo truth, and it removes the thin-thorax over-ranging failures
of the robustness run before they cost anything. After labelling, the
generator's own QA (deposition ratio) still applies.

## 4. Sampling strategies

All strategies see the same valid candidate table and the same budget.

| strategy | rule | role |
|---|---|---|
| `random` | uniform over valid candidates | the control |
| `score` | score-then-sample: draw with probability proportional to a power of the difficulty score, with per-patient and per-energy-layer quotas | the hypothesis |
| `score_topk` | the hardest K | ablation: shows the collapse onto deep beamlets that quotas prevent |
| `stratified_score` | equal counts per score decile | ablation, and the validation-set recipe |

Which score: both linear variants are logged for every candidate; selection
uses the **30-metric score**, which in EXP-0006's refit transfers across anatomy
better than the sparse one (leave-one-anatomy-out Pearson 0.755 against 0.730)
and reaches 0.879 / 0.906 on the frozen test patients within the valid
population. It is `DifficultyScorer.load()`'s default.
Difficulty is used **conditional on energy** (quotas per energy layer) rather
than raw, because the score is dominated by path length and unconstrained
selection would pick only deep beamlets. Later strategies (model uncertainty by
Monte Carlo dropout, structure-aware placement) slot in as new rows.

## 5. The cycle

```
state: model weights, training sources, cycle index, budget spent
repeat:
  train B_c epochs (continue from the previous cycle's weights, full union)
  validate: GPR, MAPE, dR80 on the frozen validation set, plus the regression guard
  if trigger not yet fired: continue (scratch arm only)
  sample: generate candidates on pool CTs -> score_candidates -> strategy -> batch
  label: MC in beamlet mode via mc_generation, resumable, provenance per beamlet
  extend: add the new directory to the training sources
  record: cycle manifest (selected candidates, scores, MC seconds, metrics)
until budget exhausted or cycles done
```

- **`B_c`**, **beamlets per cycle**, **number of cycles**, and the **Monte Carlo
  budget** are configuration variables; the budget is the primary axis of every
  plot.
- **Trigger** (scratch arm): GPR, MAPE and dR80 thresholds on the validation
  set, **to measure**: the deployed checkpoint is evaluated on the new
  validation set first, and thresholds are proposed from those values.
- **dR80 is not computed in training validation today** and must be added from
  `src.metrics.range_metrics` (with the plateau guard: undefined when the IDD
  never falls below the level).
- **Retraining**: continue from the previous cycle's weights on the full union.
  Forgetting is measured on the regression guard, not assumed away.
- **Resumability**: every cycle writes a manifest; Monte Carlo is resumable per
  beamlet already; a crashed loop restarts from the last complete cycle.

## 6. What is measured

Primary: learning curves against Monte Carlo seconds, for GPR (mean and the
tail: fraction below 95 percent, 5th percentile), MAPE, and dR80 (median,
95th percentile of the absolute value), on the frozen validation set, per
strategy and per arm. Derived: labels-to-target and MC-seconds-to-target for
the trigger thresholds; the area under the learning curve. Secondary: the same
on the regression guard (forgetting), the selection fingerprint (energy,
anatomy, score-decile distribution of what each strategy chose), and, once the
loop is stable, plan-level gamma through `run_plan_opentps`.

At least three seeds per arm and strategy before any claim; paired comparison
at equal budget.

## 7. Where the code goes

```
src/active_learning/
  pool.py         the CT pool: registry roles, per-anatomy stratification
  candidates.py   candidate generation v0/v1 on one CT -> BeamletCandidate list
  sampling.py     the strategies of Section 4 over a scored candidate table
  oracle.py       MC labelling of a batch through mc_generation, resumable
  dataset.py      DirBeamletDataset and the union with the HDF5 set
  validation.py   the difficulty-balanced validation set recipe; dR80 in validation
  loop.py         the cycle of Section 5, its manifest and resume
scripts/al_loop.py, scripts/al_build_validation_set.py, configs alongside
```

Reused, not rebuilt: `src.acquisition.score_candidates` (scoring and validity),
`src.mc_generation` (geometry and the MCsquare boundary), `src.training`
(loop, checkpoints, validation), `src.loaders.dir_based` (1 mm to 2 mm),
`scripts/train_adota.py`'s warm start and resume.

## 8. Order of work

1. Registry roles for the pool and validation CTs (extends
   `docs/patient_registry_design.md`).
2. Candidate generation v0 plus the validation-set recipe; measure the deployed
   checkpoint on the existing held-out sets; launch the 20k validation MC.
3. `DirBeamletDataset`, dR80 in validation, the cycle loop, dry run with a tiny
   budget on one CT.
4. Trigger thresholds from the measurement in step 2; the four arm-by-strategy
   runs; then seeds.

## 9. Open decisions

- Beamlets per cycle and cycles per run, once the epoch time on the union set
  is measured.
- Whether the score is refit between cycles on the labels the loop buys
  (a learned acquisition function) or stays frozen. Frozen is the main arm;
  refit is an ablation.
- Percentile grids for scoring pool candidates: the frozen reference-pool grids
  (comparable to the study) or grids over the candidate pool (better-calibrated
  ranking). Frozen for the main arm.
