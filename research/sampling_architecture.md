# Active-Learning Sampling Architecture (ADoTA, Framing B)

Status: draft v1, 2026-07-07. Companion to [publication_plan.md](publication_plan.md)
and [active_learning_literature_review.md](active_learning_literature_review.md).
This is the design for the prospective, simulator-in-the-loop sampling stage: how
we choose which beamlets to Monte-Carlo-simulate from full patient CTs.

## 1. Premise

A beamlet is fully determined by six parameters:

    (patient CT, isocenter, gantry_angle, energy, spot_x, spot_y)

where `spot_x/y` map to the steering angles `ba0/ba1` via `spot_position_to_angles`.

**Scoring a candidate beamlet needs no MC.** The existing extraction
(`src/beamlets/extraction.py::run_extraction`: rotate-around-isocenter, crop the
VoI, project flux) turns those six numbers into exactly the CT+flux VoI that
ADoTA consumes, cheaply. MC (MCsquare) is spent only on the *selected* beamlets.
That asymmetry (cheap scoring, expensive labeling) is the whole economic premise.

## 2. The two pools

- **Patient Pool (outer, static):** full CTs, 252 NSCLC-Radiomics (lung) +
  231 StageII-Colorectal-CT, each with a DICOM segmentation. Patient selection is
  diversity-driven coverage of anatomy/heterogeneity.
  - Paths: `/scratch/mstryja/manifest-1603198545583/NSCLC-Radiomics`,
    `/scratch/mstryja/manifest-1646429317311/StageII-Colorectal-CT`.
- **Beamlet Pool (inner, on-the-fly):** for a chosen CT, a *generated* set of
  candidate `(gantry, iso, energy, spot)` tuples, scored by the acquisition
  function; the top-K are MC-simulated and added to the training set. This is
  the not-yet-implemented "pool 2".

The acquisition function therefore acts at two granularities: which patient
(coverage) and which beamlet geometry within it (uncertainty/difficulty).

## 3. Acquisition function (modular; uncertainty + diversity, features as prior)

Decision (2026-07-07): lead with model uncertainty and diversity; the CT-feature
difficulty head is a secondary, cheap prior, because the correlation research
shows CT features are only weakly predictive of true error (GPR-tail AUC ~0.70,
range weaker and depth-confounded). Build the terms modular so the benchmark can
reweight them.

For each candidate `(CT, gantry, iso, energy, spot)`:

1. **Geometric trace, no MC:** extract the would-be VoI + flux (reuse
   `run_extraction` via a synthetic-plan builder).
2. **Uncertainty term (primary):** ADoTA MC-dropout / small-ensemble predictive
   variance on the VoI. No MC.
3. **Diversity term (primary):** k-center / coreset distance in the parsimonious
   feature basis, plus stratification across patient / energy / angle bins. Guards
   against the collapse-onto-redundant-hard-cases failure mode (cf. the
   quantum-water cautionary result).
4. **Feature difficulty prior (secondary):** the ~5 de-collinearised metrics from
   the clustering (depth/path-length, WEPL-variance, one edge term, ...) mapped to
   predicted error via the correlation research.
5. **Score = w_u * uncertainty + w_d * diversity + w_f * feature_prior**; select a
   batch (SBAL-style score-then-sample) under the diversity constraint.

Steps 1-4 are cheap and parallelizable; only the selected batch hits MCsquare.

## 4. Candidate generation (both realism modes; body-mask then segmentation)

Decision (2026-07-07): support two placement scopes and two realism modes, and
sequence them.

- **Placement scope, sequenced:**
  1. *Body-mask first:* isocenters anywhere inside the thresholded body contour.
     Simpler, no SEG parsing, broad coverage to stand up the pipeline.
  2. *Segmentation next:* narrow isocenters to the DICOM-segmented target
     (GTV/tumor). Clinically meaningful; needs a SEG loader + target masks.
- **Realism mode, as an ablation:**
  - *Plan-like:* target-aimed isocenter, clinical gantry arcs, range-matched
    energy (Bragg peak lands in tissue). Trains a deployable model
    (deployment-aware wedge).
  - *Broad coverage:* wide gantry/energy/position sampling incl. unusual angles,
    to stress the model and surface failure modes (robust general model).
- **Physical validity prior (always):** reject candidates whose beam misses the
  body or whose energy stops in air; range-match energy to the traced path so the
  Bragg peak is in tissue. Generate a few thousand valid candidates per CT,
  score, keep top-K.

## 5. Hard problems / risks

1. **Weak acquisition signal.** CT features only modestly predict true error;
   hence uncertainty+diversity lead. Design must survive "features barely beat
   random."
2. **Circularity / transfer.** The difficulty head is trained on ADoTA's errors on
   the *current* distribution; using it to propose *new* geometries assumes that
   transfers (publication-plan H3). Load-bearing and untested.
3. **MC oracle throughput** sets the loop cadence (batch size, rounds,
   fine-tune vs retrain). Needs the real MC-seconds-per-beamlet number.
4. **Distribution matching.** If the target use is clinical plans, the sampled
   training distribution should match the clinical spot distribution or we
   data-efficiently optimize the wrong region.
5. **Body/target masking correctness** across two DICOM datasets (HU thresholds,
   SEG-to-CT registration, couch/immobilization removal).

## 6. Integration points (reuse, not rebuild)

- `src/beamlets/extraction.py::run_extraction` + a NEW **synthetic-plan builder**
  (assemble `Field`/`ControlPoint`/`Spot` from sampled parameters instead of
  parsing `PlanPencil.txt`): the candidate-VoI generator. Main new piece.
- `training_set_analysis_advanced_metrics` metric extraction: the feature vector
  for scoring (already computes the basis on a VoI).
- `run_plan_opentps.py` MC comparison stage: the pattern for the MC oracle wrapper
  (pool-2 labeling).
- Metric clustering (`figures/metric_clustering/parsimonious_basis.csv`): defines
  the feature space for both the difficulty head and the diversity term.

## 7. Phased build

- **P0** Synthetic beamlet generator + no-MC VoI + feature scorer, end to end on
  one NSCLC CT. No MC, no loop. De-risks everything.
- **P1** Candidate generation with the physical validity prior, body-mask scope
  first, then segmentation scope; both realism modes behind a flag.
- **P2** Selection: ADoTA-uncertainty + k-center diversity + feature prior ->
  ranked batch.
- **P3** MC oracle wrapper (opentps/MCsquare) labeling the selected batch.
- **P4** Full AL loop + evaluation vs random/uncertainty/coreset baselines.

P0-P2 are MC-free and buildable now, in parallel with the research run; P3-P4 wait
on MC integration and the finalized acquisition head.

## 8. Decision log

- 2026-07-07: Realism = both plan-like and broad-coverage, as an ablation.
  Acquisition core = uncertainty + diversity primary, CT-feature difficulty head
  secondary. Beam placement = body-mask first, then narrow to DICOM
  segmentations. Patient Pool = NSCLC-Radiomics + StageII-Colorectal-CT.
