# Paper 2: active learning implementation plan

Status: PROPOSAL, 2026-11-20, awaiting approval. No code written, nothing launched.
Source: supervisor meeting of 20 Nov 2026 (Zoltan Perkó, Danny Lathouwers) and its brief.
Supersedes the experiment design half of [al_pool_warmstart_design.md](al_pool_warmstart_design.md);
the readiness findings there still stand and are referenced, not repeated.

## 1. Where the brief and the code disagree

The brief says to trust the code. These are the places it differs.

| Brief says | Code / data says | Consequence |
|---|---|---|
| Data covers head and neck, lung, prostate, abdominal, pelvic | Pool v1 has thoracic (27,338 beamlets), abdominal (13,132), pelvic (10,153). v3 is thorax + pelvis. No head and neck anywhere. | Report three sites. Head and neck is new MC generation, out of scope. |
| Final score = mean of per-model fits | One ridge (alpha 1) fitted to the mean of per-member standardised `log(1+RDE)` ([independence.py:86](../src/active_learning/ensemble/independence.py#L86)). Ridge is linear in its target, so this equals the mean of five per-member ridge fits on the same rows. | Same thing. Phase 0 asserts the identity numerically and saves the per-member fits it implies. |
| Easy/medium/hard GPR 99+ / 98.1 / 95 shows the score works | Those are three example beamlets (EXP-0021 story figure). Over deciles the best models lose about 1 GPR point from easiest to hardest (EXP-0021, within-field Spearman v11 -0.38, EXP-0019 -0.48, xLSTM -0.52, pencil beam -0.81, U-Net +0.01). | Phase 0 shows the tier table next to the examples so the supervisors see the real effect size. |
| U-Net is not used further | U0 is one of the five members whose errors define the score target. | Keep as is (the score is frozen); say so when presenting. |
| Coordinates are normalised across all DICOMs | They are not. DICOM z origin spans 2.4 m within one collection. Pool records store the CT rotated to the beam's-eye frame with the isocenter at the grid centre, plus `bixel_shift_xy_mm`, gantry, energy, steering angles, patient and collection. | Oracle option A cannot use absolute coordinates. A body-relative coordinate (body-mask extents or TotalSegmentator vertebral level) is needed, see Phase 5. |
| Every training beamlet stores body part, source CT, isocenter X/Y | True for the pool (`pool_anatomy`, `pool_patient_id`, `bixel_shift_xy_mm`, ...). v3 stores `patient_key` and `source_dataset` only; anatomy follows from `source_dataset`. | Fine for logging; no extra work. |
| MC runs as SLURM jobs on DelftBlue | The only sbatch file is pool generation, and its config is a template with `<...>` placeholders. Every MC and training run so far ran on Apollo (3 x A40, 48 cores) with nohup. MC on demand already exists in-process: `label_batch` in [oracle.py](../src/active_learning/oracle.py) calls `generate_beamlets` (12 s setup per patient/gantry/energy group + 3.3 s per beamlet at 1e6 primaries). | Default: Apollo, in-process MC. DelftBlue needs MCsquare, the env and about 65 GB of data moved first (decision D2). |
| "Oracle" = the true-error strategy | `src/active_learning/oracle.py` is the MC labeller of the prospective loop, not a strategy. | New strategy code uses the names `true_error_*` and `error_neighbours` to avoid the clash. |
| Models support MC dropout | The model has `dropout_rate` (attention dropout), but the EXP-0019 model, the chosen one, was trained with 0.0. | BALD's posterior source is decision D5. |
| The difficulty score applies to every beamlet | v2 applies only where the analytic Bragg peak is inside the crop. 34% of v3 (23,444 of 69,290) is unscored; the pool fraction is not measured yet. | Unscored beamlets stay selectable by random and by the random part of mixed strategies; the ledger counts them. |
| Retrain the model each round | The retrospective loop reads one HDF5 file; only the pool test split is exported. | Phase 1 adds the train/validation exports and multi-source reading. |
| (implicit) The baseline is trained on "the ADoTA dataset" | The EXP-0019 model was trained on 61,355 v3 records with a patient-level test set of 7,938 held out (`provenance.json`). | L0 is exactly those 61,355 ids; the 7,938 become a seen-collection test set. |
| (implicit) Pool patients are new | Checked: the pool registry has no training-scan match in train (433 patients), validation (98) or test (130); the 48 matches are all `excluded`. The pending `al_pool_selection.csv` fix concerns the old prospective loop, not this pool. | Clean; nothing to do. |

The score itself is clean for this experiment: v2 was fitted on 19 v3 scans only, with the pool held
out (`analytic_scorer_v2_exp0014.json`, `n_fit` 17,288), so it has seen no pool validation or test
label.

## 2. Answers to the brief's questions

- **Score features, per-model fits, ensemble.** Features: `src/acquisition/features.py`.
  Scorer: `DifficultyScorer` in [scorer.py](../src/acquisition/scorer.py), score = intercept +
  sum of weight x percentile(feature), with `contributions()` already implemented. Fit and LOMO
  check: [independence.py](../src/active_learning/ensemble/independence.py). The weights are
  interpretable only with care: the 30 features are strongly collinear, so single-feature
  contributions can trade off. Phase 0 reports feature groups and checks against the sparse variant.
- **MC dropout / ensembles.** Dropout exists but is 0 in the chosen model. Seven checkpoints exist
  from EXP-0014 (five ladder members, EXP-0019, v11); only their errors were saved, not their
  predictions. The confirm #41 checkpoint was trained with dropout. They are different
  architectures, frozen, so they are not a posterior of the model being trained.
- **MC.** MCsquare, external install at `/home/mstryja/tools/mcsquare`, beamlet mode, 1e6
  primaries for the pool (v3 is 1e7). Apollo, 48 threads: 3.3 s per beamlet + 12 s per group.
- **Stored per beamlet.** See the index CSV of the pool export: ids, patient, collection, anatomy,
  split, gantry, energy, steering angles, bixel shift, grid origin/size/spacing, primaries,
  flux model and widths.
- **Plan reconstruction.** [run_plan_opentps.py](../scripts/run_plan_opentps.py) (`stream` stage)
  takes any checkpoint. The 8 ADoTA paper plans are those in
  [run_publication_plans.sh](../scripts/run_publication_plans.sh): LUNG1-062, -195, -250, -364,
  Prostate-AEC-004, -069, -006, -007. All 8 patients are in the pool TEST split.

## 3. Experiment core

**Data roles.** L0 = the 61,355 v3 records M0 was trained on (1e7). U = pool train (35,519, 1e6,
labels already exist and are revealed only when selected). V = pool validation (7,552), from which
two disjoint fixed subsamples are drawn: V_stop (2k) drives early stopping, V_eval (2k) gives the
primary endpoint, so the stopping rule never selects on the number it is judged by. The full V is
evaluated at each round end. T = pool test (7,552), the 7,938 held-out v3 records, and the 8
publication plans. T is touched only at the end of an arm, never for a decision.

Because U is already simulated, every strategy except on-demand generation runs retrospectively
at zero new MC cost. The ledger charges each acquired beamlet the MC seconds it actually cost when
the pool was generated, so the cost axis is real.

**Arms (wave 1, one seed each).**

| Arm | New data | Sampler | Role |
|---|---|---|---|
| A0 | none | v3 only | separates "more training" from "more data" |
| A1 | all of U at once | uniform | the "simulate everything" reference; sets the target |
| R-U | random | uniform union | standard training |
| K-U | plain top-k by the score | uniform union | the strategy in the brief and the diagram; expected to over-pick thorax (87% of the top 10%) |
| S-U | top-k by the score inside collection x energy quota cells | uniform union | score selection without the site bias |
| R-M | random | managed | sample management alone |
| S-M | score with quotas | managed | both |
| E-U | true error of the current model (top-k under the same quotas) | uniform union | upper bound for any error-driven strategy; free here because U is labelled |

Managed sampler, defined here in full:

- **New-data share.** Each training batch draws a fixed share of acquired pool beamlets, equal to
  min(0.5, 10 x |acquired| / samples per round). Each acquired beamlet is then seen about 10
  times per round, and never more than half of a batch is new data.
- **Replay.** The remaining share replays L0 with probability proportional to the beamlet's
  score percentile. L0 beamlets without a score (peak outside the crop) get the median weight.
- **Uniform union**, by contrast, concatenates L0 and the acquired beamlets and shuffles.

**Budget.** 10% of U per round (3,552 beamlets), 5 rounds, 17,760 beamlets in total (half of U).

**Fairness rule (the supervisors' condition).** Every round trains until the loss on a fixed 2k
subsample of V has not improved for 3 evaluations (one evaluation every 0.5 v3-epoch equivalents),
up to 20 v3-epoch equivalents, then restores the best checkpoint before acquiring. Every arm uses
the same optimiser, learning rate (constant 1e-4, from EXP-0013), stopping rule and hyperparameters.
Epochs therefore become an outcome, not a setting, which is what the paper wants to measure. This
replaces the fixed `epochs_per_cycle: 50` of the retrospective loop.

**Endpoints.**

- Primary: fail95 on the V subsample (share of beamlets with GPR 2%/2mm < 95%, 10% dose cutoff),
  plotted against cumulative samples seen (in v3-epoch equivalents), labelled beamlets, MC seconds
  and GPU hours. Time to target = first of 3 consecutive evaluations below the target, where
  target = halfway between M0 and A1 on fail95. 2%/2mm is primary because 3%/3mm sits near 99% and
  barely moves; 3%/3mm is reported beside it.
- Converged quality at equal label budget (V at each round's best checkpoint).
- At the end of each arm (final checkpoint and target-crossing checkpoint): T beamlets by score
  tercile and by collection/anatomy (GPR 2/2 and 3/3, MAPE, RDE, dR80), and the 8 plans (plan GPR,
  target Dmean and D95, OAR D2%) against MCsquare.
- d_ij calculation time is a property of the architecture, so it is identical across strategies.
  It is reported once. AL can only improve it indirectly, if a smaller model reaches the same
  accuracy with better data. Worth saying explicitly to the supervisors.

**Cost.** EXP-0013 trained at about 69 records/s per run with three runs sharing Apollo. One v3
epoch is about 17 min, about 21 min with the full acquired budget. With 5 to 15 epochs per round,
a 6-round arm takes roughly 15 to 35 GPU hours of training. That excludes evaluation: at an EXP-0013
cycle boundary the epoch took 526 s against 170 s of training, mostly gamma on the CPU. A V_stop
loss every 0.5 EE is cheap, but gamma on V_eval across 8 concurrent runs on 48 shared cores may
dominate. So "wave 1 is about 4 days" is provisional until the Phase 1 smoke test measures the
evaluation cost; if it is too high, gamma on V_eval runs every 1 EE instead of 0.5.

## 4. Phases

Every phase: branch off `dev`, unit tests that run on CPU, a smoke test that prints dose
statistics (min, max, mean, NaN count) before any long job, the exact nohup command posted for
approval, and an EXP record via `scripts/report.py`. New modules stay under 500 lines;
`retrospective/loop.py` is already at 451, so new behaviour goes into new modules.

### Phase 0. Difficulty score diagnostics (CPU only, 1 to 2 days, for the next supervisor meeting)

Goal: answer Zoltan's two requests with existing artifacts, no training.

- New `src/acquisition/explain.py`:
  - `feature_table(scorer, features, reference) -> DataFrame` gives, per feature: value,
    percentile and robust z within the reference population (pool test), weight and contribution.
  - `contrast(scorer, a, b, reference)` returns the contribution difference between two
    beamlets, sorted.
  - `group_contributions(table, groups)` sums contributions over named groups (WEPL, HU change
    along path, Bragg peak heterogeneity, edges, energy), because single features trade off under
    collinearity.
- New `scripts/active_learning/difficulty_diagnostics.py` (typer) with
  `config_difficulty_diagnostics.yaml` and a guide in `scripts/docs/`. Stages:
  - `explain`: the EXP-0021 easy, medium and hard examples (from `difficulty.csv` in
    `exp0014/difficulty_engine_grids/energy_gantry_full`): full table plus the medium-versus-easy
    contrast.
  - `correlate`: per-member ridge fits (same rows, alpha 1) and the ensemble, each against every
    model's GPR, MAPE and RDE on pool test, using the EXP-0014 member error CSVs. Spearman with
    scan-bootstrap CI. Pencil beam comes from the EXP-0021 grids. Each model is marked "in the
    score target" (C0, C1, X0, U0, T012) or "external" (v11, EXP-0019, pencil beam).
  - `tiers`: GPR, MAPE and RDE by score tercile and decile, overall and per anatomy, per model.
- Figure: add `contribution_bars_figure` to
  [src/figures/difficulty_engine_grids.py](../src/figures/difficulty_engine_grids.py), beside
  `difficulty_story_figure`, saved with `save_figure_as_publication_formats`, CSV beside it.
  Outputs to `/scratch/mstryja/adota_runs/paper2/diagnostics/`.
- Acceptance:
  - ensemble weights equal the mean of the per-member weights (to 1e-8);
  - the top 2 or 3 groups explaining medium over easy agree between the full and sparse variants
    (if they disagree, report that instead of a story);
  - correlations come with CIs and the in-target/external labels.

### Phase 1. Warm-start loop infrastructure (3 to 4 days of development)

- **Exports** (CPU, existing command, overnight; about 55 GB with 383 GB free):
  `al_build_beamlet_pool.py export --split train` and `--split validation`.
- **Scores for U and V:** the v2 features via the EXP-0014 feature stage pointed at the new
  exports. Record the peak-inside-crop fraction.
- **New `src/active_learning/retrospective/sources.py`:** `SourceSpec(name, h5_path, index_csv)`
  and a multi-source index mapping `v3:<id>` and `pool:<id>` to their file. The trainer and the
  validation code read through it.
  - Acceptance: the pool and v3 records go through the same dose normalisation. v3 is 1e7 and the
    pool is 1e6, and units must match (Gy per MU, see the dose-units memo). Check with the same
    model's error on a few records of each.
- **New `retrospective/stopping.py`:** `RoundStopper(patience, eval_every_ee, max_ee)`, the
  fairness rule above, with restore-best.
- **New `retrospective/ledger.py`:**
  - `acquisitions.csv`: round, id, source, collection, anatomy, patient, energy, score, rank,
    strategy, MC seconds;
  - `rounds.csv`: epochs, samples seen, wall time, GPU hours, cumulative MC seconds, V metrics.
  - Per-beamlet MC seconds come from the pool state if recorded, else 3.3 s + group setup; this is
    checked in Phase 1.
- **New `retrospective/sampler.py`:** the managed sampler. Reuses `union_sampler` from
  [src/active_learning/dataset.py](../src/active_learning/dataset.py) for the share cap and adds
  score-weighted replay.
- **Strategies** in [retrospective/sampling.py](../src/active_learning/retrospective/sampling.py)
  (the registry):
  - `score_quota`: top-k inside collection x energy-bin cells sized like U;
  - `true_error_topk`: one inference pass of the current model over the remaining U per round,
    about 35k beamlets, minutes on an A40.
- **Config:** `scripts/config_al_warmstart.yaml`, with CLI > YAML > defaults through
  `merge_config`.
- **Tests:** source mapping, stopper, quota cell sizes, sampler shares, ledger schema (all CPU).
- **Smoke:** 200 records per source, 2 rounds, 1 evaluation each, dose stats printed.

### Phase 2. M0 and wave 1 (GPU, about 4 days running, launched end of week 1)

- M0 (decision D4): default is the EXP-0019 final model (61,355 v3 records, dropout 0). That is
  our best model, so "AL improves the best model" is the claim. The fallback is a fresh M0 under
  the new stopping rule, which costs about 1 GPU-day.
- Launch the eight arms of section 3, seed 1234, on GPUs 0 to 2, through
  `scripts/run_al_warmstart.sh` (new, in the style of `scripts/run_al_retro.sh`):
  `nohup bash scripts/run_al_warmstart.sh > /scratch/mstryja/adota_runs/paper2/wave1/launch.out 2>&1 &`.
  The exact command goes up for approval first. After launch: command, PIDs and log paths, then
  stop.
- Outputs: `/scratch/mstryja/adota_runs/paper2/wave1/<arm>_seed1234/` with `manifest.json`,
  `metrics.jsonl`, `rounds.csv`, `acquisitions.csv` and per-round best checkpoints.
- Acceptance:
  - every arm starts from the same M0 sha256;
  - A0 at round 0 reproduces M0's V_eval fail95 within noise;
  - each round's ledger has exactly 3,552 new ids, none repeated and none from V or T;
  - no NaN in losses;
  - the stopper fires before `max_ee` in most rounds. If it never fires, the patience is wrong.

### Phase 3. Evaluation benchmark (built while wave 1 runs)

- New `scripts/active_learning/al_benchmark.py` (typer). Stages:
  - `beamlet`: T through the shared `evaluate` engine with `H5Source`; tiers and sites.
  - `plans`: the 8 publication plans through `run_plan_opentps.py --stages stream,gamma` per
    checkpoint, using the `gamma_criteria` list of its config.
  - `efficiency`: curves and time to target from `rounds.csv`, with seed CIs once seeds exist.
- Figures reuse [al_curves.py](../src/figures/al_curves.py),
  [al_experiment_comparison.py](../src/figures/al_experiment_comparison.py)
  (`strategy_endpoint_figure`, `stability_figure`), `plan_comparison.py` and
  `dvh_comparison.py`. One results table per strategy, numbers beside every figure.
- Plan-level is end-of-arm only (D3). The "feedback" arrow in the pipeline diagram becomes an
  end-of-run evaluation.
- Outputs: `/scratch/mstryja/adota_runs/paper2/benchmark/<wave>/`.
- Acceptance:
  - M0 through the `beamlet` stage reproduces its own EXP-0014 pool-test numbers (GPR 2/2 mean
    and fail95, to the third decimal);
  - M0 through `plans` reproduces the June plan GPRs of the 8 plans within 0.5 points (the
    checkpoint differs from v11, so only an order-of-magnitude check there; an exact check with
    v11);
  - every figure has its CSV.

### Phase 4. BALD: understand first, then integrate

- **Design note** `docs/bald_regression_note.md`, covering:
  - What BALD computes. It is the mutual information between a prediction and the weights. For a
    Gaussian likelihood with noise variance s^2 it is 0.5 log(1 + Var_theta[mu(x)] / s^2), so it
    ranks by epistemic variance and ignores irreducible noise.
  - How a dose volume reduces to a scalar: sum over voxels, the high-dose region, or variance of
    R80.
  - Why BatchBALD in regression becomes a log-determinant over the batch, which rewards diverse
    high-variance picks. This connects it to score plus distance.
  - How it differs from top-k on the score. The score is fixed, model-free and mixes aleatoric
    with epistemic difficulty; BALD is model-dependent and changes every round.
  - The posterior source options:
    - (a) MC dropout: needs dropout in every arm, which changes M0;
    - (b) a checkpoint or snapshot ensemble from the round's own trajectory: free;
    - (c) a last-layer ensemble or Laplace approximation: cheap;
    - (d) a deep ensemble of 3 to 5: K times the compute, which must then be charged to the arm.
- **Toy problem** (CPU, an hour): a 1D depth-dose with a random slab interface plus a region of pure
  label noise and a data gap. Success: BALD samples the gap, while a hardness score samples the
  noise.
- **Check on real data** before integrating (about 1 to 2 GPU hours, command posted first):
  - re-run inference with the 7 checkpoints on pool test;
  - committee or snapshot variance against the current model's error, as a partial Spearman beyond
    the score.
  - If it adds nothing, BALD does not get a wave.
- **Integration:**
  - `src/active_learning/posterior.py` (an adapter over the chosen source);
  - strategies `bald` and `bald_batch` (greedy log-det) in the registry.

### Phase 5. Oracle: check similarity before generating anything

- **5a. Option A tested retrospectively, no MC.** U is labelled, so Zoltan's concern can be
  measured directly.
  - Take M0's worst 200 beamlets on a probe set (a slice of U not used for acquisition).
  - For each, collect candidates from U by three rules:
    - (i) same collection and anatomy, near the same body-relative position, gantry and energy;
    - (ii) k nearest neighbours in the standardised 30-feature space;
    - (iii) random within the same anatomy.
  - Report the feature distance and M0's actual error on each set (inference only).
  - Option A is worth building only if (i) or (ii) beat (iii) on error.
  - If (ii) wins, it becomes strategy `error_neighbours` in the retrospective loop at zero MC cost.
- **5b. Option B pilot (CT perturbation).** New `src/acquisition/perturb.py` shifts one interface
  (bone/air or tissue boundary along the beam) by plus or minus 1 or 2 mm in the beam's-eye CT
  before cropping.
  - Pilot: 20 source beamlets x 5 perturbations = 100 MC beamlets, about 10 minutes through
    `label_batch`.
  - Measure feature shift and M0 error on the perturbed versus the source beamlets.
- **Body-relative coordinate** (needed for A(i)): first body-mask extents per slice from
  `body_mask`; a TotalSegmentator vertebral-level pilot on 5 CTs only if that is too coarse.
- **5c. On-demand integration:** the prospective loop ([loop.py](../src/active_learning/loop.py)
  and `label_batch`) already labels by MC in-process. It is extended only for whichever option
  passes 5a or 5b. SLURM submission only if D2 picks DelftBlue.

### Phase 6. Waves 2 and 3, write-up

- Seeds 1235 and 1236 for the arms that matter after wave 1.
- `bald` and `error_neighbours` arms if they passed their checks.
- Benchmark on T and plans.
- EXP records and the change record for the code PRs.
- Outputs: `/scratch/mstryja/adota_runs/paper2/wave2/` (same layout as wave 1).
- Acceptance:
  - any claimed difference between two arms exceeds the seed spread (paired over seeds, at the
    same samples-seen points);
  - the results table carries all three seeds;
  - the EXP record's Results negative section lists every arm that did not beat R-U.

## 5. Timeline

| Week | Work |
|---|---|
| 23 to 27 Nov | Phase 0 (Mon to Tue). Phase 1 build, exports overnight, smoke. BALD note and toy on CPU. Wave 1 launched by Friday after approval. |
| 30 Nov to 4 Dec | Wave 1 running. Phase 3 benchmark. Phase 5a (inference only) and 5b pilot. BALD real-data check. |
| 7 to 11 Dec | Wave 2 (seeds + new arms). First results table. |

## 6. Decisions needed before Phase 1

| | Question | Default if you agree |
|---|---|---|
| D0 | May I start the two pool exports now (CPU, no new code)? | Yes; commands in the reply |
| D1 | Are the 8 paper plans those in `run_publication_plans.sh`? | Yes, LUNG1-062/195/250/364 + Prostate-AEC-004/069/006/007, all pool test |
| D2 | Apollo or DelftBlue? | Apollo for waves 1 and 2; DelftBlue only once MCsquare and data are there |
| D3 | Plan-level only at the end of each arm, not as per-round feedback? | Yes; per-round plan feedback would need new plans on pool validation patients |
| D4 | M0 = EXP-0019 final, or a fresh M0? | EXP-0019 final, with A0 as the control |
| D5 | BALD posterior source (decides whether all arms train with dropout) | Snapshot or last-layer ensemble, dropout stays 0; settled by the Phase 4 note |
| D6 | Fairness rule as in section 3 | Patience 3 x 0.5 EE, max 20 EE per round, constant 1e-4 |
| D7 | Budget and target | 10% of U x 5 rounds; target = half the M0 to A1 gap on V fail95 at 2%/2mm |

## 7. Risks

- The biggest failures come from unseen collections (LUNG1 25 to 29% fail versus Lung_Dx 5 to 7%),
  and random sampling reaches those for free. The quotas in `score_quota` exist for this. If S-U
  still loses to R-U, the honest result is that the score helps only inside a site.
- The Prostate-AEC low-energy failures on homogeneous paths are invisible to the score and to
  committee disagreement (shared short-range bias). E-U will show how much is left on the table.
- Earlier pilots (EXP-0009 to EXP-0013) showed the learning-rate schedule moving results more than
  the strategy. The constant rate and the per-round stopping rule remove that confound, but one
  seed per arm is not evidence; wave 2 seeds are mandatory before any claim.
- If the pool and v3 normalisations differ, every arm is affected. That is the first acceptance
  check of Phase 1.
