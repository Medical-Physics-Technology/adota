# Warm-start active learning from v3 onto the new-patient beamlet pool: readiness and design

Status: **proposal, 2026-11-20.** The experiment design (Sections 2 to 9) is superseded by
[paper2_al_implementation_plan.md](paper2_al_implementation_plan.md); Section 1 (readiness) still stands. Nothing here is built or launched. Section 1 is the
readiness verdict for difficulty score v2 as a selector; Sections 2 to 9 design the
experiment; Section 10 lists the decisions that are still open. Exploratory numbers are
in `/scratch/mstryja/adota_runs/al_pool_warmstart/readiness_20261120/` (scripts, CSVs,
and the MAPE rerun of the EXP-0014 gap).

## 1. Is difficulty score v2 ready for the loop?

The score is two things, and they have different answers: an **estimator** of how hard
a beamlet is, and a **selector** that is supposed to make training better. EXP-0014 and
EXP-0021 tested the estimator. Neither tested the selector, and the retrospective
benchmarks EXP-0009 to EXP-0013 (score against random inside v3) never separated the two
strategies beyond seed noise.

### 1.1 Engineering: ready

- `src/acquisition/data/analytic_scorer_v2_exp0014.json` is vendored and deployed in
  `scripts/config_al.yaml`. It is input-only, so it can score unlabelled candidates.
- Coverage on the pool test split is 7,449 of 7,552 (98.6%). The rest have the analytic
  peak outside the crop, so they have no score and fall back to random fill.
- Leakage: v2 was fitted on 19 v3 scans (EXP-0014 fit set). The pool validation and test
  splits are new patients, so v2 is clean for any validation drawn from the pool. It is
  **not** clean against a validation set drawn from v3 scans: this is the reason
  `config_al_retro_loop.yaml` keeps the EXP-0012 scorer.
- Still missing: scores for the pool train and validation splits. Only the test split
  was exported (`/scratch/mstryja/adota_pool/exports/v1_apollo_1e6_test.h5`).

### 1.2 Estimator: ready on RDE, not on MAPE

| check (pool test, 130 patients) | result |
|---|---|
| LOMO gap, RDE (EXP-0014 primary, deployed v2) | +0.009 [0.005, 0.011]: model-independent |
| LOMO gap, score **refitted on MAPE** (deferred rerun, today) | **+0.108 [0.090, 0.129]: model-specific** (rule: <= 0.10 approximate) |
| same, quality-gate members only | +0.031 [0.025, 0.039]: approximately model-independent |
| same, per anatomy, ladder | thoracic 0.070, abdominal 0.208, pelvic 0.204 |
| deployed v2 (RDE-fitted) against MAPE, Spearman | v11 0.34, EXP-0019 0.46 (within anatomy x energy 0.32 / 0.48) |

The rerun refits the score on the MAPE target (`difficulty_independence.py` fits on the
configured `error_metric`). Its verdict is therefore about **difficulty defined by
MAPE**, not about the deployed score:

- MAPE-defined difficulty is model-specific. In the abdominal and pelvic subsets, a
  score fitted on the other models predicts a model's MAPE only at r -0.03 to 0.27.
- The deployed v2 tracks MAPE moderately (Spearman 0.34 to 0.46).
- The model-independence claim holds for RDE. For MAPE it holds only for thorax and
  only for the better models.

This belongs in EXP-0014 Results negative before the paper repeats the claim.

### 1.3 Selector: partly ready, and only with quotas

Per beamlet, against the two models trained on v3 (v11, EXP-0019 `final_attndrop0`).
GPR is at 2%/2mm with a 10% cutoff.

| selector, top 10% of pool test | v11: share GPR < 95% | EXP-0019: share GPR < 95% | overlap with the 10% worst GPR |
|---|---|---|---|
| random | 13.8% | 14.3% | 10% |
| score v2 | 25.5% | 32.9% | 21% / 26% |
| score v2 within anatomy x 20 MeV cell | 24.2% | 28.2% | 20% / 24% |
| energy only (highest) | 2.2% | 3.4% | 2% |
| oracle (true error) | 100% | 100% | 100% |

- The score is not an energy or anatomy proxy. Ranking within cells keeps almost all of
  its lift, and energy alone selects *easier* beamlets than random.
- The score's top decile fails 6.6x as often as its bottom decile for the EXP-0019 model
  (32.9% against 5.0%), and 2.6x for v11.
- **Composition:** its top 10% is 87% thoracic, against 51% of the pool. Patients spread
  well: 115 of 130 appear, with a maximum share of 2.7%.
- **Blind spot:** Prostate-AEC (pelvic, a collection no model has seen) fails 10 to 16%
  of the time. The failures are mostly at 70 to 100 MeV (33% for v11) and concentrated in
  a few patients (one patient at 92%). The score ranks low-energy beamlets as easy, so
  inside Prostate-AEC it is useless for EXP-0019 (9.2% against 10.4% base) and slightly
  anti-informative for v11 (6.0% against 15.7%).
- **The largest failure source is a new collection, not geometric difficulty.** LUNG1
  (NSCLC-Radiomics) fails 25 to 29% of the time, Lung_Dx (the training collection) 5.5
  to 7%. StageII-Colorectal (the pelvic training collection, labelled abdominal in the
  pool) is at 99.4%. A random sampler gets this domain-shift gain for free, which makes
  random a strong control here.

### 1.4 Verdict

**Ready to enter the experiment as the selector of one arm, under per-collection and
per-energy quotas. Not ready to be claimed as a better selector:** that claim is what
Section 2 tests.

Conditions before launch:

1. Pool train and validation are scored with v2 once, and the scores are cached (the
   score is model-free).
2. The quota scheme is fixed before any run, as proportional to the pool's composition
   (Section 4.2). It is not tuned on the pool test split, which this check has already
   looked at.
3. The Prostate-AEC blind spot is reported per collection rather than averaged away.
4. An oracle pilot shows that selection *can* matter on this endpoint before seeds are
   paid for (Section 7).

Caveat: these numbers are on the pool **test** split, because it is the only split with
inference already run. The models are fixed and nothing was fitted, but the test split
has now been used twice for score diagnostics (EXP-0014 and this check). It stays clean
for the models trained in this experiment.

## 2. Question and hypotheses

Start from a model trained on the dataset ADoTA was trained on (v3). Add beamlets of new
patients from the labelled pool. Does choosing **which** beamlets to add (difficulty
score) and **how** they are fed to training (sample management) reach a target on an
independent validation set in fewer training epochs, while using less data?

- **H1 (primary):** score acquisition with managed sampling reaches the target in fewer
  samples-seen than random acquisition with uniform sampling.
- **H2 (acquisition main effect):** score acquisition beats random acquisition at equal
  sampling.
- **H3 (management main effect):** managed sampling beats uniform sampling at equal
  acquisition.
- **H4 (data):** the winning arm reaches the target with a fraction of the pool, where
  the all-in arm uses all of it.

## 3. Data roles

| role | source | size | patients | labels |
|---|---|---|---|---|
| L0, initial labelled set | v3 `trainset_pelvis_initial_test_one_ct_downsampled_v3_all_SingleGaussian.h5` | 70,146 | 56 scans (40 thorax Lung_Dx, 16 StageII) | MC 1e7 |
| U, acquisition pool | beamlet pool v1, train split | 35,519 | 433 (LUNG1 162, Lung_Dx 79, Prostate-AEC 82, StageII 110) | MC 1e6 |
| V, target set | beamlet pool v1, validation split | 7,552 | 98 | MC 1e6 |
| T, final test | beamlet pool v1, test split | 7,552 | 130 | MC 1e6 |

- **Leakage:** the pool registry excludes all 56 v3 scans (match_training, 40/40 + 16/16).
  The pool is split by patient. Scorer v2 never saw a pool patient.
- **Retrospective:** the U labels already exist. The loop reveals them only for what a
  strategy selects. Cost is still reported as the per-beamlet `mc_seconds` of the pool
  index: 131 process-hours for the whole of U, so what each arm would have paid is
  known.
- **Mixed fidelity:** L0 labels are 1e7 and U labels are 1e6. 1e6 does not limit the
  measured GPR: the models score the same against 1e6 and 1e7 on the noise set (97.53
  against 97.56). But the score correlates with MC noise (EXP-0014 partial r 0.54), so a
  score arm buys noisier labels. Every batch logs a noise proxy (`volume_over_max`,
  r 0.96 with the measured noise floor).

## 4. Arms

All arms start from the same checkpoint M0 (Section 5), at the same learning rate, with
the same step budget and paired seeds.

| arm | acquisition from U | training sampler | role |
|---|---|---|---|
| A0 | none | uniform over L0 | does more training on v3 alone reach the target? |
| A1 | all of U at once | uniform over L0 + U | data ceiling; sets the target |
| A1-M | all of U at once | managed | the "use everything" comparator for the epochs claim |
| R-U | random, N per cycle | uniform over L0 + acquired | conventional baseline |
| S-U | score, N per cycle | uniform over L0 + acquired | H2 |
| R-M | random, N per cycle | managed | H3 |
| S-M | score, N per cycle | managed | H1 |
| O-M | oracle (true loss of the current model), N per cycle | managed | pilot only, ceiling of any selector |

R-U, S-U, R-M and S-M form a 2 x 2 factorial, so the acquisition effect and the
management effect are separable.

**Which baseline the epochs claim is measured against.** A1 and A1-M hold all of U from
step 0, and the target is half of A1's own gain. They will probably cross first in
samples seen, and "active learning is slower than using everything" is a possible honest
result. The claims are therefore stated per comparator:

- against R-U, the conventional loop: fewer samples seen to the target (H1);
- against A1-M: the samples seen **and** the beamlets used at the crossing, as a
  trade-off. S-M wins outright only if it crosses no later than A1-M while using at most
  50% of U.

### 4.1 Acquisition

- N = 10% of U = 3,552 beamlets per cycle, for 5 cycles. At most 50% of U is acquired.
- **Score arm (`score_stratified`, new strategy):** quotas over collection x 20 MeV
  energy bin, proportional to what is left in U. Inside each cell it takes the top-k
  scores. Unscored beamlets in a cell are drawn at random only when the cell has fewer
  scored beamlets than its quota. This is the design doc's `score` row (Section 4 of
  `docs/active_learning_pipeline.md`), which the retrospective registry never
  implemented.
- **Random arm:** uniform over U, with no quotas. A random draw already matches the
  pool's composition in expectation.
- **Oracle:** the same quotas, ranked by the current model's per-beamlet loss on U. That
  needs inference over the remaining U each cycle, with no gamma (about 35k forward
  passes).

### 4.2 Sample management (default proposal, see Section 10)

The prospective loop already has the mechanism (`union_sampler`, `al_oversample_fraction`
in `src/active_learning/dataset.py`). Managed means:

1. **New-data share, capped:**
   - the share of each batch from acquired beamlets is
     `min(0.5, 10 x |acquired| / samples per cycle)`, so an acquired beamlet is drawn
     about 10 times per cycle at most;
   - a fixed 50% would draw the 3,552 cycle-1 beamlets about 30 times each in cycle 1,
     against 3 times for a v3 record;
   - under the uniform sampler the acquired share is only 4 to 20%: 3.5k to 17.8k among
     74k to 88k records;
   - the train-to-validation loss gap on acquired records is logged as an overfitting
     check.
2. **Difficulty-weighted replay:**
   - the rest of the batch replays L0, with probability rising linearly with the
     record's v2 score rank, so the hardest v3 records are drawn about twice as often as
     the median;
   - v2 was fitted on v3 scans, which is allowed here because L0 is training data;
   - about 34% of L0 has no score (23,444 of the 69,290 records EXP-0006 featured have
     the analytic peak outside the crop). They get the median weight, and their share of
     replay is logged.

If S-M wins, an ablation separates components 1 and 2. That is not in the main budget.

## 5. Starting model M0 and training recipe

Recommended: train M0 once on all of L0 with the recipe the continuation uses.

- Architecture of `config_al_retro_loop.yaml` (v11-like), **dropout 0**. EXP-0019 showed
  the deterministic-eval gap that dropout causes.
- Constant learning rate 1e-4 after a 2-epoch warmup, and gradient clipping at norm 10.
- 60 passes over L0, about 20 GPU-hours to measure.

The continuation keeps the same constant learning rate and carries the optimizer state.
So there is no warm restart, which was the EXP-0011 and EXP-0012 failure mode, and it
matches EXP-0013's finding: constant 1e-4, zero divergences.

Alternatives are in Section 10: start from EXP-0019 `final_attndrop0` or from v11. Both
save a day, but both ended on a schedule the continuation would have to restart from.

## 6. Endpoint and what is recorded

**Effort axis.** An epoch is a different number of steps in each arm (L0 alone is 70k
records, L0 + U is 106k). The primary axis is therefore **samples seen since M0**,
reported in **v3-epoch equivalents** (1 EE = 70,146 samples). Each arm's own epoch count
is reported beside it. Every arm gets the same budget: 5 cycles x 3 EE = 15 EE.

**Evaluation.** Every 0.5 EE, on a fixed 2,000-beamlet subsample of V stratified by
collection, the same subsample in every arm. The full V is evaluated at every cycle end.
T is evaluated once per run, at the crossing checkpoint and at the end.

**Primary metric:** fail95, the share of V beamlets with GPR (2%/2mm, 10% cutoff) below
95%. M0-class models sit at about 14% on the pool. Mean GPR has only about 1 point of
headroom, so the tail is where a difference can show. Secondary metrics:

- mean and p05 GPR, MAPE, and |dR80| median and p95;
- the same at 3%/3mm, the repository's headline criterion;
- fail95 per collection.

**Target, frozen before any AL arm runs:**

- target = M0 level - 0.5 x (M0 level - A1 plateau);
- the A1 plateau is the median of A1's last three full-V evaluations, averaged over A1's
  seeds;
- a crossing is the first evaluation that starts a run of **3 consecutive** evaluations
  at or below the target;
- a run that never crosses is censored at 15 EE and reported as "> 15 EE".

**Data used, per evaluation:**

- beamlets acquired, as a count and as a percentage of U;
- patients touched;
- MC seconds of the acquired beamlets;
- unique records seen;
- samples seen;
- the selection fingerprint: collection, energy, score decile, and the noise proxy.

## 7. Order and go/no-go

1. **Infrastructure and smoke test** (Section 8), including one 0.1 EE cycle per arm
   type.
2. **M0 training:** about one day, on one GPU.
3. **Pilot, 1 cycle-pair (2 cycles), all under the managed sampler:** R-M with 2 seeds,
   S-M with 1 seed, O-M with 1 seed, and A1 with 1 seed, on 3 GPUs.
   - The pilot does not use the uniform sampler, because there acquired beamlets are
     only 4 to 10% of the samples. Section 10.1 of `docs/active_learning_pipeline.md`
     shows that regime hides any acquisition effect and would give a false no-go.
   - **Go** if O-M is ahead of both R-M seeds by more than their spread on fail95 at the
     end of cycle 2.
   - **No-go** otherwise: no acquisition function can win on this endpoint at this N,
     so N or the endpoint gets revisited before money is spent on seeds.
4. **Main runs:** the target is frozen from A1. Then A0, A1 and A1-M with 3 seeds each,
   and the factorial with 3 seeds per cell.
5. **Analysis and record.**

**Statistics:**

- Crossings are reported per seed.
- Paired differences use the same M0 and the same data-order seed.
- A claim needs all three pairs to have the same sign **and** a median difference
  larger than the random arm's seed range.
- If the pilot shows effects near the seed range, two more seeds go to R-U and S-M only.

## 8. What has to be built

None of this exists yet. Each item needs its own approval where it is long.

1. **Export** the pool train and validation splits to v3 layout with the existing
   `export` stage. That is about 43k beamlets at about 0.22 s each (about 2.6 h) and
   about 55 GB on `/scratch` (383 GB free). It is a long command.
2. **Score** U, V and L0 with v2 once, using the analytic features on the exported files,
   and cache the results.
3. **Extend the retrospective loop** (`src/active_learning/retrospective/`):
   - a multi-file source (L0 plus pool files by `sample_id`);
   - `val_split: external`;
   - warm start from an external checkpoint instead of cycle 0;
   - a samples-seen step budget per cycle;
   - evaluation every 0.5 EE;
   - the `score_stratified` and `oracle` strategies;
   - the managed sampler, ported from `src/active_learning/dataset.py`;
   - the data ledger.
   
   The 500-line ratchet applies: `loop.py` is at 451 lines.
4. **Extend `al_compare`:** samples-seen and EE axes, the sustained-crossing rule (the
   current `crossing` table takes the first crossing only), and per-collection panels
   through `src/figures/`.
5. Records: EXP for the experiment, CHG for the code PR.

## 9. Cost (estimates, to be measured in the smoke test)

- **Training throughput:** about 58 samples/s for the v11-like model on one A40, from
  EXP-0012. That makes 1 EE about 20 min and 15 EE about 5 h of training per run.
- **Evaluation:** 30 subsample evaluations plus 5 full-V evaluations add about 1.5 to
  2.5 h. A run is about 7 h.
- **Totals:**
  - pilot: 5 runs at 2 cycles each, under 1 day on 3 GPUs;
  - main: A0, A1 and A1-M at 3 seeds each, plus 4 cells at 3 seeds each, so 21 runs,
    about 147 GPU-hours, about 2 days on 3 A40s;
  - M0: about 1 day.

## 10. Open decisions

1. **Does "full dataset" mean v3 + the new-patient pool?** That is the assumption here:
   L0 = v3, U = pool train. The alternative is a retrospective benchmark inside v3, which
   is what EXP-0009 to EXP-0013 already did.
2. **What does "improved samples management" mean?** Section 4.2 proposes a new-data
   share plus difficulty-weighted replay. Other options are curriculum ordering, pruning
   easy v3 records, or loss-based reweighting.
3. **M0:** train fresh on v3 (recommended), EXP-0019 `final_attndrop0`, or v11.
4. **Primary metric:**
   - fail95 at 2%/2mm (recommended: headroom, and continuity with EXP-0009 to EXP-0013);
   - or 3%/3mm, the repository headline, where there is little headroom.
5. **Target rule:** 50% of the gap from M0 to A1 (recommended), or an absolute level.
6. **Scale:** N = 10% of U per cycle, 5 cycles, 3 EE per cycle.
7. **A model-based uncertainty arm** (MC-dropout or ensemble disagreement). It is the
   natural detector of the domain-shift failures the score misses, but it is a separate
   experiment.
