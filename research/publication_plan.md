# Publication Plan: Active Learning for Clinic-Specific Adaptation of Proton Dose Surrogates

Status: draft v1, 2026-07-02. Decisions locked with M. Stryja; see section 8.
Companions: [active_learning_literature_review.md](active_learning_literature_review.md),
[references.md](references.md).

---

## 1. Core narrative (the spine)

A general, pretrained proton dose surrogate (ADoTA) is adapted to a new clinic,
treatment site, or anatomy using a **minimal number of new Monte-Carlo
simulations**, selected by a **physics-informed, deployment-aware active
learning** loop, and validated **at the level of the clinic's actual treatment
plans** (DVH, plan gamma), not only per-beamlet test error.

One claim, three ingredients (not three separate claims):

| Ingredient | Role in the framework | Asset already in repo |
|---|---|---|
| Physics-informed acquisition | *How* candidates are scored: a difficulty predictor g(heterogeneity metrics) trained to predict per-beamlet error (GPR/RDE) | Metric suite in `training_set_analysis_advanced_metrics.py` (Pflugfelder HI, ISI, Sobel/structure tensor, WEPL) |
| Deployment-aware candidate pool | *Where* candidates come from: the clinic's own plan spot distribution (energies, positions, MU/influence weights) | Plan parser + spot expansion in `run_plan_opentps.py` |
| Plan-level validation | *What success means*: reconstructed plan quality on clinic plans | Full plan pipeline: accumulate, DVH, plan gamma (`run_plan_opentps.py`) |

Why this framing is robust: even if informed selection ties with random, the
paper still delivers (a) the first quantification of the adaptation cost of a
dose surrogate to a new anatomy, (b) the plan-level evaluation methodology, and
(c) an honest benchmark. The negative result is publishable by design.

## 2. Two-paper sequence

**Paper 1 (primary, Phys. Med. Biol. or Medical Physics):**
"Clinic-specific adaptation of a transformer dose engine with physics-informed
active learning." Clinical framing, plan-level endpoints, the full framework.
Statistical bar: 3 seeds with variance bands, fixed held-out test sets.

**Paper 2 (ML venue: MIDL / MELBA / NeurIPS workshop):**
The acquisition-function benchmark extracted and hardened: more seeds, more
baselines (SBAL, BatchBALD-style batching, coreset), ablations, the
difficulty-predictor transfer study. Reuses Paper 1's infrastructure and pool.

Experiments below are designed so their outputs serve both papers.

## 3. Hypotheses

- **H0 (prerequisite):** a meaningful zero-shot gap exists: the pelvis-trained
  ADoTA degrades on the target anatomy (expected: lung, driven by
  heterogeneity). *If H0 fails, the adaptation narrative dies; see kill-switches.*
- **H1:** AL fine-tuning reaches the accuracy of full-data fine-tuning with a
  small fraction of the MC budget (measured in MC-seconds, not sample count).
- **H2:** physics-informed and hybrid acquisition beat random and
  pure-uncertainty selection on **tail** endpoints (fraction of beamlets below
  clinical GPR threshold; worst-case RDE).
- **H3:** the difficulty predictor g(metrics) → error **transfers across
  anatomies** (train on pelvis, rank lung beamlets usefully).
- **H4:** deployment-aware candidate weighting (clinic spot distribution,
  MU/influence weighted) improves **plan-level** endpoints beyond
  input-space-only selection.
- **H5:** adaptation does not destroy general performance (no catastrophic
  forgetting; replay mixture controls it).

## 4. Experiments

Ordered; each has a purpose, cost class, and (where relevant) a kill-switch.

**E0. Zero-shot gap measurement.** Run the existing trained model over the
lung/colorectal beamlet sets already on /scratch. No training. Produces the
gap table that motivates everything.
*Kill-switch:* no gap on any anatomy → stop, rethink (options: harder beam
model, higher resolution, different site).

**E1. Acquisition-signal audit (mostly done once, cheap).**
(a) Re-establish metric-vs-GPR correlations on the current model (the old
correlation CSVs are no longer in `runs/`).
(b) Train the difficulty predictor g on pelvis; test ranking transfer to lung
(H3, Spearman of predicted vs actual error).
(c) MC label-noise check in the tail: re-simulate a small sample of
high-heterogeneity beamlets at higher statistics; quantify how much of "hard"
is actually "noisy label".
*Kill-switch:* if g does not transfer at all AND raw metrics rank poorly on the
new anatomy, drop H2/H3 to secondary and lead with H1/H4.

**E2. Pool generation (work package, on the critical path).**
Purpose-built multi-CT pool. Spec to be frozen separately, but the decision
points are: anatomies (proposal: pelvis + lung + one more; lung = adaptation
target), number of CTs per anatomy, beamlets per CT, energy sampling strategy,
MC histories per beamlet (noise-matched; informed by E1c), beam model
(consistent `SingleGaussian` unless upgraded deliberately), and stratification
by heterogeneity at generation time. Also record per-beamlet MC wall-time: this
is the cost axis of every learning curve and the motivation-section number.

**E3. Retrospective AL adaptation benchmark (the core).**
Pretrained model + pool from E2 with labels hidden. Strategies: random
(mandatory), uncertainty (MC-dropout variance), diversity (k-center in feature
space), physics (g from E1), hybrid, each with SBAL-style stochastic batching.
Fine-tuning with replay mixture (H5); from-scratch confirmation runs for the
2-3 survivors only. 3 seeds, learning curves vs MC-seconds.

**E4. Plan-level evaluation vs budget (signature experiment).**
For each strategy and budget round, reconstruct the clinic plans
(`run_plan_opentps.py`, 7 Prostate-AEC plans + new-anatomy plans if available)
and report plan gamma and DVH deltas vs MCsquare. This figure is the paper.

**E5. Deployment-aware variant (H4).**
Candidate pool restricted/weighted by the clinic's plan spot distribution
(energy histogram, spatial distribution, MU weights). Compare against
input-space-only selection at equal budget.

**E6. Prospective capstone (optional; Paper 2 or revision ammunition).**
opentps/MCsquare in the loop generating genuinely new beamlets proposed by the
winning strategy, beyond the pre-generated pool.

## 5. Results-section blueprint (what the paper measures)

The paper is built around one comparison: **how much labeling budget each
strategy needs to reach a target performance**, reported as a time-to-target and
a speedup ratio, and confirmed at the clinical (plan) level. Everything below is
defined so the Results section can be drafted before the numbers exist.

### 5.0 Shared definitions (fixed once, apply everywhere)

- **Strategies compared (the columns of every table):**
  - `baseline` = no-AL / random sampling (the mandatory reference; "vanilla").
  - `uncertainty` = ADoTA MC-dropout predictive variance.
  - `diversity` = k-center / coreset in the parsimonious feature basis.
  - `physics` = CT-feature difficulty head.
  - `hybrid` = uncertainty + diversity + feature (the proposed method).
  - `full` = model trained on the entire pool (the upper-bound ceiling).
- **Budget axis (x-axis of curves; unit of "time"):** primary =
  **MC-seconds** (the honest cost, since a low- and high-energy beamlet do not
  cost the same to simulate); secondary = number of labeled beamlets. Report both.
- **Quality metrics (held-out beamlets, model level):**
  - central tendency: mean GPR, RDE, RMSE;
  - **tail** (primary): % beamlets with GPR < 95 %, P5 GPR, P95 |ΔR100|,
    max |ΔR100| (ΔR100 is the robust range metric after the R80 fix).
- **Clinical metrics (held-out patients, plan level, via `run_plan_opentps.py`):**
  plan gamma pass rate (2 %/2 mm and 3 %/3 mm), target D95/D98/D2 error vs MC,
  OAR Dmean/Dmax error vs MC.
- **Performance TARGET (defines "t"):** report two, one relative and one
  absolute/clinical, e.g. (rel) reach 99 % of the `full` model's tail metric;
  (abs) reach plan gamma (2 %/2 mm) >= 95 %.
- **Statistical protocol:** fixed stratified held-out beamlet test set + held-out
  patient plans; >= 3 seeds with variance bands; paired comparisons across
  seeds/patients; `baseline` and `full` always shown.

### 5.1 Headline table T1 -- Time-to-target and speedup

The core result the reader remembers. One row per strategy.

| Strategy | MC-sec to target | # beamlets to target | Speedup vs baseline | AULC |
|---|---|---|---|---|
| baseline (no AL) | t_baseline | n_baseline | 1.0x | ... |
| uncertainty | t_uncertainty | ... | t_baseline / t_uncertainty | ... |
| diversity | ... | ... | ... | ... |
| physics | ... | ... | ... | ... |
| **hybrid (ours)** | **t_hybrid** | ... | **t_baseline / t_hybrid** | ... |

Produced for each target (relative and absolute) and for at least one model-level
and one plan-level target (so we can state "AL reaches clinical plan-gamma 95 %
in t_hybrid vs t_baseline, a Nx reduction in MC budget").

### 5.2 Figure F1 -- Learning curves (model level)

Quality vs budget (MC-seconds), one curve per strategy, 3-seed bands. Three
panels: (a) mean GPR, (b) tail % beamlets < 95 %, (c) P95 |ΔR100|. Horizontal
line at each target makes t_* readable off the x-axis; AULC in the legend.

### 5.3 Figure F2 -- Plan-level learning curves (signature figure)

Same x-axis (budget), but y = **plan gamma pass rate** and **target D95 error**
on held-out patient plans, per strategy. This is the clinical translation of F1
and the figure that makes it a medical-physics paper: "at a fixed adaptation
budget, the AL-trained model reconstructs clinical plans within gamma X % vs
baseline Y %."

### 5.4 Table T2 -- Fixed-budget comparison

At one clinically realistic budget, all model- and plan-level metrics per
strategy, with significance vs baseline. The "who wins at equal cost" snapshot.

### 5.5 Worst-case analysis (T3 + F3)

The tail is the clinical risk, so it gets dedicated treatment.

- **T3 (worst-case table):** per strategy at matched budget: max |ΔR100|,
  min GPR, count of catastrophic beamlets (|ΔR100| > threshold), and the worst
  plan (lowest plan gamma, largest OAR overdose). Tests whether AL specifically
  shrinks the dangerous tail, not just the mean.
- **F3 (worst-case distribution + gallery):** CDF/violin of |ΔR100| and GPR per
  strategy (tail emphasis), plus the worst-beamlet diagnostic panels (reuse
  `scripts/range_failure_diagnostics.py`) for hybrid vs baseline at matched
  budget.

### 5.6 Supporting figures

- **F4 selection fingerprints:** energy / heterogeneity / beamlet-angle coverage
  of the selected batches per round vs the pool (distributional-bias guard, cf.
  the quantum-water failure).
- **F5 acquisition-term ablation:** uncertainty vs diversity vs physics vs hybrid
  on the tail metric (which ingredient carries the gain).
- **F6 forgetting curve:** general (source-anatomy) performance during adaptation
  (H5), showing adaptation does not degrade the base model.
- **F0 zero-shot gap (motivation):** per-anatomy gap of the un-adapted model (E0).

### 5.7 Cost accounting (T4)

MC-seconds per selected beamlet, fine-tune GPU-hours per round, total wall-clock
per strategy. Grounds the "time" claims in real resource use and yields the
clinic-facing "adaptation recipe" (the framework deliverable).

## 6. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| H0 fails (no zero-shot gap) | Fatal to narrative | E0 first, before any other spend |
| Random ties informed selection | Medium | Framing already negative-result-proof; plan-level + cost quantification carry the paper |
| Physics metrics rank poorly on new anatomy | Medium | E1b/E1c early; hybrid strategy hedges |
| MC label noise masquerades as difficulty | Medium | E1c noise audit; noise-matched histories in E2 spec |
| Catastrophic forgetting | Medium | Replay mixture; H5 tracked every round |
| Scooped on "AL for dose engines" | Medium | Systematic novelty search still TODO before manuscript claims |
| Pool generation slips (calendar) | Medium | E0/E1 run on existing data in parallel with E2 |
| Single beam model (`SingleGaussian`) questioned | Low | Acknowledge as limitation; framework is beam-model-agnostic |

## 7. Open items (blocking, assigned to discussion)

1. Actual correlation strengths from the earlier advanced-metrics runs
   (CSVs missing from `runs/`; re-run = E1a).
2. True MC cost per beamlet / per training set (needed for the motivation
   paragraph and the cost axis).
3. E2 pool spec numbers: anatomies, #CTs, #beamlets/CT, histories, energies.
4. Systematic novelty search (Scholar/arXiv/PMB/MedPhys 2024-2026) before any
   "first" claim is written.

## 8. Decision log

- 2026-07-02: All three wedges unified under the clinic-adaptation narrative
  (not three parallel claims). Venue: Med Phys/PMB first, ML venue second.
  Negative result will be published as the finding. A new purpose-built
  multi-CT pool will be generated (E2). Adaptation experiments are fine-tuning
  runs from the existing pretrained ADoTA, not from-scratch retrains; the
  ~100-run budget therefore covers the full design with margin.
