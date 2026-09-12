# Active-learning pipeline: design

Status: implemented 2026-09-08 (`src/active_learning/`, branch
`feat/active-learning-loop`); the slots marked **to measure** are still filled
after the first runs, not guessed. Design decisions taken with the supervisor on
2026-09-04. **Section 10 records where the implementation departs from this
document and why** -- read it before comparing a run against this text.
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

## 10. Where the implementation departs from this design

Three departures, all deliberate, all made during implementation and verified by
the end-to-end smoke run. They are recorded here rather than folded silently into
the text above, because a reader comparing a run against this document needs to
know which parts of it the code does not do.

### 10.1 A cycle is a step budget with oversampling, not a pass over the union

Section 5 says "train B_c epochs (continue from the previous cycle's weights, full
union)". The code instead trains a fixed number of optimizer steps
(`al_steps_per_epoch` x `num_epochs`) in which `al_oversample_fraction` of every
batch is drawn from the beamlets the loop bought.

The reason is arithmetic. A cycle buys a few thousand beamlets against a training
split of 56,114. Under uniform sampling a new beamlet is seen once every several
epochs, so a cycle at any budget that fits a night would move the validation
metrics by nothing measurable, and the experiment would report a null result
caused by the sampler rather than by the acquisition function.

The cost of the departure: both arms use the same regime, so they stay comparable
to **each other**, which is what the paired comparison needs. Neither is
comparable to a plain-union baseline, so the absolute learning curve is not the
curve this document originally described. A plain-union arm remains available by
setting `al_oversample_fraction` to the natural share.

### 10.2 A cycle hands on `last.pth`, not `best.pth`

`best` is selected by the loss on the HDF5 validation split -- the distribution
the model already fits. A cycle trained on newly bought beamlets can raise that
loss while improving on exactly the geometry it just bought, so selecting on it
would carry the pre-cycle weights forward and the loop would measure nothing. A
cycle is a fixed budget, so what it produced is what it hands on
(`LoopConfig.checkpoint_selection`, default `last`).

### 10.3 Monte Carlo cost is grouped, and the grouping is not neutral

Section 6 makes Monte Carlo seconds the budget axis. Measured on this machine, a
beamlet costs **3.31 s** in beamlet mode at 1e6 primaries on 48 threads, and each
`(patient, gantry, energy)` group costs a further **~12 s** of MCsquare setup
whatever it holds. Beamlet-mode parallelism is one thread per spot, so throughput
scales roughly with the thread count; on a shared machine, divide.

**This makes the cost axis strategy-dependent, which the design did not
anticipate.** In the smoke run the same eight beamlets cost 203 s under `score`
(2 groups, 4 per group) and 292 s under `random` (4 groups, 2 per group): a 44%
overhead from packing alone. `random` spreads over more patients by construction,
so at equal beamlet count it spends more seconds, and a learning curve plotted
against seconds credits `score` for packing rather than for choosing well.

Mitigations, none of them yet chosen: report both axes (beamlets and seconds) and
say so; equalise the budget in seconds rather than beamlets; or drive
`n_cts_per_cycle` low enough that both strategies pack similarly. **The first
results must state which was used.**

## 12. The retrospective benchmark (EXP-0009)

Before the prospective loop above buys a single Monte Carlo label, the same cycle
runs retrospectively on the reference HDF5 set: the labels exist, the loop hides
them and reveals them only for the records a strategy selects, and the model is
trained from scratch on a fixed schedule ``0.2 |T| + c N``. It answers a narrower
question, how the sampling strategy shapes the training progress at equal record
count, and it is a lower bound and a design filter rather than a result: the pool
was itself drawn uniformly, so a retrospective run can only reweight what is
present and understates what a prospective loop finds in the far tail. Code:
``src/active_learning/retrospective/``; entry points ``scripts/al_retro_loop.py``
and ``scripts/al_compare.py``; guide ``scripts/docs/al_retro_loop.md``.

## 11. What the first runs measured

From the end-to-end smoke run of 2026-09-08 (`scripts/config_al_smoke.yaml`),
which exercised both arms with real Monte Carlo and real training:

- **Candidate validity is strongly energy-dependent.** At 80 and 105 MeV, 80 of
  80 thoracic candidates were valid. At 155-180 MeV a large fraction fail
  `peak_outside_crop`: low-density lung extends the range past the 320 mm crop.
  Section 3's validity gate does its job before any Monte Carlo is paid for, but
  **the candidate pool must be sized against the valid count, not the raw count**,
  and the energy layer set of `config_al.yaml` needs a `--dry-run` measurement per
  anatomy before the first real launch.
- **Out-of-body dose is real physics here, not a geometry fault.** Thoracic
  beamlets at random gantry deposited 47-61% of their dose inside the body mask,
  confirmed against an independent HU threshold on the same crop. The patient
  starts 20-29 mm into the crop (the configured standoff), only 9-15% of dose
  lands before entry, and the remainder is beyond and lateral to a thorax that is
  far shorter than the 320 mm crop. Deposition ratio stayed at 0.995, so nothing
  escapes the crop.
- **The validation recipe does not balance patients.** `select_balanced`
  stratifies by anatomy, energy layer and score decile as specified; with two CTs
  and eight beamlets it drew all eight from one patient. Immaterial at 4,000
  beamlets over ten CTs, but it is not a guarantee, and a per-patient quota is the
  obvious fix if the drawn set turns out lopsided.
- **Arms share the beamlet cache, not their training sets.** Candidate ids are
  content-addressed, so a candidate selected by both arms is simulated once and
  reused; each arm still trains only on its own `training_sources.csv`. The reused
  beamlet is counted in the selecting arm's Monte Carlo seconds, which is the cost
  of what it chose.

