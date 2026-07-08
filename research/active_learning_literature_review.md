# Active Learning for Surrogate Dose Models: Literature Review

**Scope.** This review analyses the delivered corpus (15 local PDFs in
[../publications/](../publications/)) and situates it in the wider literature on
(a) active-learning / adaptive-sampling *methodology* and (b) *AI in
radiotherapy* (dose prediction, dose calculation, autosegmentation, QA). The
goal is to ground the ADoTA active-learning project: how to select training
beamlets so a proton-dose surrogate reaches a target accuracy, especially in the
clinical **tail**, with the fewest expensive Monte-Carlo labels.

Full linked bibliography: [references.md](references.md). Every work cited by
number (e.g. **[7]**) resolves there, with a clickable local PDF or DOI.

---

## 1. Executive summary

- The delivered corpus is strong on **adaptive sampling for engineering/physics
  surrogates** and on **proton-therapy heterogeneity metrics**, but light on the
  **deep-AL methodology backbone** (coreset, batch-mode Bayesian AL, deep
  ensembles, regression-specific acquisition) and on **AL evaluation hygiene**.
  Those gaps are filled in [references.md, section B](references.md#b-recommended-additions-external-not-yet-in-the-corpus).
- A recurring, load-bearing finding across the corpus: **active learning does not
  automatically beat random sampling.** The quantum-liquid-water study **[9]**
  shows random *winning*, and the reproducibility literature (Munjal 2022; Lüth
  2023) shows AL gains are often an artefact of weak baselines or few seeds. This
  makes the *evaluation protocol* as important as the acquisition function.
- **Novelty wedge for ADoTA:** the RT-AI literature builds ever-better dose
  models but treats the **training set as given**. No one is using
  physics-informed acquisition to *choose which beamlets to simulate*. The
  heterogeneity metrics in this project (Pflugfelder HI **[11]**, Bueno tissue
  quantification **[12]**, the ISI/Sobel/WEPL family already implemented) are a
  ready-made, empirically-validated acquisition signal. That combination, namely
  physics-informed active learning for a proton-dose surrogate, is unoccupied.

---

## 2. The delivered corpus, analysed

### 2.1 Active learning and adaptive sampling (methodology cluster)

**Query by Committee. Seung, Opper & Sompolinsky, 1992 [1].**
The foundational committee method: train several hypotheses on the same labelled
data, query the point of **maximal disagreement**. Proves committee queries yield
*asymptotically finite information gain* and exponentially-decaying generalization
error, versus a slow inverse power law for random inputs. *Relevance:* this is the
theoretical parent of the uncertainty/committee baseline; a deep-ensemble variance
or MC-dropout disagreement is the modern instantiation. Note the contrast with **[9]**,
which shows the promise does not always survive in practice.

**A survey of adaptive sampling for global metamodeling. Liu, Ong & Cai, 2018 [2].**
The best *map* in the corpus. Categorises adaptive sampling into
**exploration** (space-filling / variance-driven) versus **exploitation**
(error-driven, gradient/nonlinearity-driven) and their balance. Also flags the
practical failure modes: over-exploitation clustering, cost of the infill
criterion, batch versus sequential. *Relevance:* gives us the vocabulary to
classify each acquisition function; our physics metrics are an *exploitation*
signal and must be balanced with a diversity/exploration term.

**Adaptive sampling for the Reduced Basis Method. Chellappa, Feng & Benner, 2020 [3].**
A greedy scheme driven by a **sharp a-posteriori error estimator** interpolated
(via RBFs) over a fine candidate set; points are added *and removed* to keep the
training set small with guaranteed accuracy on a test set. *Relevance:* the
add-and-prune loop and the "cheap error surrogate guides expensive sampling"
pattern map directly onto our setting, where a heterogeneity metric is the cheap
proxy for where MC labels are worth spending.

**Deep Adaptive Sampling for Surrogate Modeling Without Labeled Data. Wang et al., 2024 [4].**
Uses the **PDE residual as an unnormalized density** and resamples collocation
points where residual is high (a physics-informed, label-free loop). *Relevance:*
conceptually elegant but assumes a differentiable physics residual we do **not**
have; our "label" is a Monte-Carlo dose that is expensive and non-differentiable.
Useful as a contrast that clarifies why our problem is *pool/label-based* AL, not
residual-driven collocation.

**Physics-and-data co-driven surrogate modeling. Xian & Wang, 2024 [5].**
Fuses a cheap **low-fidelity physical model** with a data-driven error correction,
trained by active learning targeting high correlation and low bias for
**rare-event** simulation. *Relevance:* strong analogue. A pencil-beam (analytic)
dose is our low-fidelity physics, MC is the truth; AL could target beamlets where
the analytic model and MC disagree most. Their rare-event framing also motivates
our **tail** endpoint (the clinically dangerous minority of beamlets).

**Multi-Resolution Active Learning of Fourier Neural Operators. Li et al., 2023 [6].**
AL that jointly selects **inputs and fidelity/resolution** via a
**utility-to-cost ratio** acquisition, with a probabilistic (ensemble MC) operator
for posterior inference and a cost-annealing trick. *Relevance:* the utility/cost
acquisition is exactly right for us, where MC cost varies by beamlet (energy,
depth). Directly motivates budgeting AL in **MC-seconds, not sample count**.

**Stochastic Batch Acquisition (SBAL). Kirsch et al., 2023 [7].**
Shows a simple **score-then-stochastically-sample** rule turns any single-point
acquisition into a batch method that matches BatchBALD/BADGE at a fraction of the
compute, because it accounts for scores shifting as the batch fills. *Relevance:*
our single-sample retrain is infeasible (~12 h/run) so we *must* batch; SBAL is the
cheap, strong default and a must-have baseline. Pairs with BatchBALD in
[references.md, section B.1](references.md#b1-active-learning-methodology-backbone).

**Adaptive sampling with batch selection for surrogate models (geotechnical). 2025 [8].**
An applied batch-AL surrogate pipeline emphasising **batch diversity** to avoid
redundant, informative-but-similar samples. *Relevance:* a template for the applied
"framework" paper we want to write; confirms batch diversity is the central
engineering problem once you commit to batch AL.

**Random versus Active Learning for ML potentials of quantum liquid water. Stolte et al., 2025 [9].**
The corpus's essential cautionary tale. With query-by-committee AL, **random
sampling gave lower test error** for a given set size; the AL set was *biased*
(small systematic energy offsets from preferentially-added structures). Switching
to a shift-invariant error measure recovered parity. *Relevance:* three warnings we
must design around. (i) Always include a strong random baseline. (ii) AL can
introduce **distributional bias**. (iii) The *metric* you optimise or evaluate
changes the verdict. This is why our protocol pins tail metrics and multi-seed
bands.

### 2.2 Radiotherapy and proton-therapy domain cluster

**ADoTA base model. Stryja, Lathouwers & Perkó [10].** The angle-dependent dose
transformer this project trains and improves. Defines the surrogate whose *training
process* the AL work aims to make more data-efficient. This is the project's own
base publication.

**Pflugfelder heterogeneity number. 2007 [11].** Introduces `H_i`, a per-spot
**lateral heterogeneity** number, and shows it **correlates with pencil-beam-versus-MC
dose error** and with setup-error sensitivity. *Relevance:* this is essentially a
validated *acquisition score*, a cheap input-space quantity predictive of where a
fast dose model fails. It is already implemented in the project's metric suite and
is a prime candidate acquisition function.

**Bueno et al. MC-need algorithm. 2013 [12].** Quantifies tissue heterogeneity
to **decide when MC is required** over analytic dose for small proton fields.
*Relevance:* the same decision our AL loop makes at training time, namely "is this
beamlet hard enough to be worth an expensive MC label?" A ready-made,
physics-grounded difficulty score and a direct citation for the framework's logic.

**Albertini et al. DAPT clinical workflow. 2024 [13].** Documents a real daily
online-adaptive proton workflow where **speed is clinical**. *Relevance:* the
motivation layer. Fast, reliable surrogates matter clinically, and a training
framework that reaches accuracy with less compute has real deployment value.
Supplies the clinical framing for the paper's introduction.

**Outeiral et al. Network-score QA metric for auto-segmentation. 2023 [14].**
Derives a metric from the network output that **correlates with clinical contour
accuracy** for QA. *Relevance:* methodological sibling. Using model-internal
signals to flag low-quality outputs is the same idea as uncertainty-based
acquisition; a good cross-task precedent for "confidence estimation in RT."

**Jungo, Balsiger & Reyes. Uncertainty QA for brain-tumor segmentation. 2020 [15].**
A critical look at whether **uncertainty estimates are trustworthy** (aggregation
can hide miscalibration). *Relevance:* a direct caution for our uncertainty-based
acquisition baseline. If the surrogate's uncertainty is poorly calibrated, an
uncertainty-driven query will misfire. Motivates checking calibration before
trusting uncertainty sampling, and strengthens the case for physics-informed
acquisition as a robust alternative or complement.

---

## 3. Gaps in the corpus, and how the wider literature fills them

Full entries with links in
[references.md, section B](references.md#b-recommended-additions-external-not-yet-in-the-corpus).

### 3.1 Deep-AL methodology backbone (missing)
- **Diversity/coreset:** Sener & Savarese (core-set / k-center greedy), the
  canonical diversity baseline; also shows many classification AL heuristics fail
  for CNNs in batch mode.
- **Batch-mode Bayesian AL:** BatchBALD (Kirsch et al.), principled joint-batch
  mutual information; the redundancy-aware counterpart to SBAL **[7]**.
- **Uncertainty backbone:** Deep Ensembles (Lakshminarayanan et al.), practical,
  well-calibrated uncertainty; the engine for the uncertainty-AL baseline (with
  MC-dropout as the cheaper single-model alternative).
- **Regression-specific acquisition:** Expected Model Change for regression (Cai
  et al.; robust variant Park & Kim) and **Black-Box Batch AL for Regression**
  (Holzmüller et al.). Important because dose prediction is *regression*, while
  most AL theory is classification.
- **Taxonomy:** Settles survey; Ren et al. deep-AL survey, for shared vocabulary.

### 3.2 AL evaluation hygiene (missing, and critical)
- Munjal et al. (2022) and Lüth et al. (2023): AL advantages are frequently
  **brittle**; they vanish under proper regularization, tuning, or more seeds.
  Reading these *before* running experiments makes the protocol referee-proof.
  Together with **[9]** they justify: a strong random baseline, at least 3 seeds
  with variance bands, a fixed held-out test set, and honest budget accounting.

### 3.3 AI in radiotherapy: wider context (partially covered)
- **Dose prediction (photon KBP):** Nguyen et al. hierarchically-dense U-Net, the
  reference architecture family; our surrogate is the proton/transformer analogue.
- **Proton dose prediction:** deep-learning spot-scanning field-dose prediction
  (2023), the closest task neighbours.
- **MC dose acceleration:** Deep Dose Plugin (real-time MC via DL denoising);
  Neph et al. (DL proton dose engine). These frame *why* a cheap surrogate is
  wanted and what a "gold-standard MC label" costs.
- **DoTA lineage:** Wu et al. LSTM proton dose in heterogeneous tissue, the direct
  ancestor of ADoTA, which already centres **heterogeneity** and reinforces our
  acquisition signal.
- **Label-efficient RT learning:** AIDE (annotation-efficient segmentation); the
  autosegmentation review; patient-specific daily-updated models. These establish
  that *data efficiency* is an active theme in RT; our contribution extends it from
  segmentation labels to **simulation labels**.

---

## 4. Synthesis: implications for the ADoTA active-learning project

1. **Frame it as pool-based, batch-mode, regression AL with a physics-informed
   acquisition function.** Not residual-driven (**[4]**), because MC labels are
   expensive and non-differentiable; not single-sample, because retraining is
   ~12 h (**[7]** batching is mandatory).

2. **Candidate acquisition functions**, spanning the corpus's exploration/
   exploitation axis (**[2]**):
   - *Physics/exploitation:* Pflugfelder `H_i` **[11]**, Bueno heterogeneity
     **[12]**, and the project's ISI/Sobel/WEPL metrics, already shown to
     correlate with GPR.
   - *Uncertainty:* deep-ensemble / MC-dropout variance (heir to QBC **[1]**),
     with a **calibration check** first (**[15]**).
   - *Diversity:* coreset / k-center to prevent the bias failure of **[9]** and
     the redundancy problem of batch AL (**[8]**).
   - *Hybrid and cost-awareness:* utility-to-cost ratio in **MC-seconds** (**[6]**);
     an add-and-prune outer loop (**[3]**).

3. **Evaluate on the clinical tail, with hygiene.** Primary endpoint: fraction of
   beamlets below a clinical GPR threshold and worst-case RDE, as learning curves
   versus both sample count and MC-seconds, compressed to AULC and
   budget-to-target. Mandatory: a strong random baseline, at least 3 seeds, and a
   fixed stratified held-out test set (**[9]**, Munjal, Lüth). A tail/rare-event
   framing is also the natural read of the co-driven and rare-event work (**[5]**).

4. **The novelty claim is defensible.** RT-AI optimises the *model* given data
   (**[10]**, Nguyen, Neph); adaptive-sampling optimises *data* for generic
   physics surrogates (**[2]** to **[8]**). ADoTA sits at the intersection:
   **adaptive sampling, driven by validated proton-physics heterogeneity metrics,
   for a clinical dose surrogate, judged on the clinical tail.** That intersection
   is empty in the surveyed literature.

---

## 5. Open questions and suggested further reading

- **Does physics-informed acquisition beat generic uncertainty/diversity?** This
  is the empirical crux; it requires the ablation in the protocol.
- **Is the surrogate's uncertainty calibrated enough to trust?** (**[15]**) Check
  before committing to uncertainty sampling.
- **Distributional bias of AL sets** (**[9]**). Monitor selected-batch coverage
  across energy and heterogeneity bins each round.
- **Suggested reads not yet in the corpus:** BatchBALD, Sener & Savarese, Deep
  Ensembles, Holzmüller (batch AL for regression), Lüth (evaluation pitfalls), all
  in [references.md, section B.1](references.md#b1-active-learning-methodology-backbone).
  I can pull any of these into [../publications/](../publications/) and add a full
  analysis section on request.

---

_Prepared 2026-07-02. Companion: [references.md](references.md). Next planned
deliverable: a frozen experimental protocol (test-set construction, acquisition
definitions, metrics, statistical plan)._
