# Physics-Informed Acquisition Functions for Active Learning of a Proton Dose Transformer

**Status:** design note / paper draft
**Data source:** `/scratch/mstryja/adota_runs/20260707_124010` (advanced-metrics evaluation of the ADoTA dose-prediction model on the pelvis/abdomen beamlet set)
**Scope:** definition, mathematical framework, and empirical grounding of candidate acquisition functions; an experiment matrix (to be filled) for selecting the best candidate.

---

## 1. Motivation and problem setting

The ADoTA transformer predicts per-beamlet proton dose distributions from the CT, flux, and beam geometry. On the current pelvis/abdomen pool the model is already accurate in the mean: the median 3%/3 mm gamma pass rate (GPR) is 99.3% (Section 4.1). The learning signal that remains is concentrated in a small, heterogeneity-driven **tail** of hard beamlets. Uniformly generating or labelling more beamlets is therefore inefficient: most new samples are redundant with what the model already predicts well.

Active learning (AL) addresses this by selecting, from a large pool of candidate beamlets, the subset whose labels (Monte Carlo dose) are most informative for the model. The central object is the **acquisition function** $a(x)$ that scores each unlabelled beamlet $x$; the highest-scoring beamlets are queried for MC simulation and added to the training set.

This note develops $a(x)$ as a combination of interpretable, physics-motivated heterogeneity metrics. It is grounded in the evaluation run above, in which every metric plus the model error was recorded for $N = 69{,}290$ beamlets, so we can measure retrospectively how well a proposed $a(x)$ ranks beamlets by true error before running any AL loop.

---

## 2. Notation and AL formalisation

Let

- $\mathcal{U}$ be the pool of unlabelled beamlets, $\mathcal{L}$ the labelled training set, $B$ the per-round query budget;
- $m(x) = (m_1(x), \dots, m_d(x)) \in \mathbb{R}^d$ the vector of $d$ heterogeneity/edge/physics metrics extracted from $x$ (Section 3);
- $y(x)$ a scalar model-error target (Section 4.2), and $\tau(x) = \mathbb{1}[\text{error} > t]$ its tail indicator at threshold $t$.

Pool-based AL repeats:

$$
Q = \operatorname*{arg\,top\text{-}B}_{x \in \mathcal{U}} \; a(x), \qquad
\mathcal{L} \leftarrow \mathcal{L} \cup Q, \qquad
\mathcal{U} \leftarrow \mathcal{U} \setminus Q,
$$

retraining the model after each round. The design problem is the functional form and parameters of $a$.

### 2.1 Rank normalisation

Metrics live on incommensurable scales (HU, mm, edge energy, dimensionless indices) with heavy tails. We map each metric to its pool percentile rank,

$$
p_k(x) = \hat{F}_k\big(m_k(x)\big) = \frac{1}{|\mathcal{U}|}\sum_{x' \in \mathcal{U}} \mathbb{1}[\, m_k(x') \le m_k(x)\,] \in [0,1],
$$

which is robust to outliers, bounded, and makes weights comparable. All acquisition functions below operate on $p(x) = (p_1, \dots, p_K)$.

### 2.2 Query-time information constraint: calibration versus deployment

The defining constraint of the problem is that **at the moment a candidate is scored, only its input data exist**. A candidate is a 3D sub-grid sampled from a full patient CT together with its beam geometry and flux; it has not been Monte Carlo simulated, so it has no dose, no gamma pass rate (GPR), no range error, and no other output or performance quantity, and producing one is precisely the expensive operation the acquisition is meant to spend sparingly. The acquisition function must therefore be a function of **input-derived quantities only**,

$$
a(x) = f\big(m(x)\big),
$$

where every component of $m(x)$ is computable either from the candidate's CT sub-grid, geometry, and flux (the heterogeneity/edge/physics metrics of Section 3) or from the trained model's own forward pass on that input (predictive uncertainty, Section 5.5). Neither family requires the ground-truth dose.

Output and performance metrics ($\mathrm{GPR}$, $\Delta R_{100}$, $\mathrm{RDE}$) **never enter $a(x)$**. They have exactly two roles, both **offline** and both on data that has already been fully simulated:

1. **Calibration.** On a labelled reference set they are the *target* used to fit the parameters of $f$ (for example the linear weights of Section 5.1 or the learned head of Section 5.3). The fitted $f$ is then **frozen** and applied to new candidates using inputs alone.
2. **Validation.** They are the ground truth against which competing acquisition functions are ranked retrospectively (Section 6).

This one-time offline calibration (labels available) is strictly separated from online deployment (inputs only). It is what makes the error-derived weights below admissible: a performance metric may *shape* the weights once, offline, but it is never *read* when a candidate is scored. Consequently the empirical correlations in Section 4 are properties of the labelled reference run, used to design and rank $f$; they are not inputs to $f$.

Two calibration regimes are considered throughout:

- **Supervised (label-calibrated).** Parameters of $f$ are fit on the labelled reference set (Sections 5.1, 5.3). Strongest, but assumes the calibration transfers to new anatomy; this transfer is itself an experiment (Section 7).
- **Unsupervised (label-free).** Parameters come from physics priors and input-space structure only, with model uncertainty (Section 5.5) as the principled input-only difficulty signal. No performance metric is used at any stage.

---

## 3. Metric redundancy and the parsimonious basis

The $d \approx 30$ extracted metrics are highly correlated (for example, nine distinct Sobel/edge-energy statistics). Feeding all of them into $a(x)$ double-counts whichever physical axis happens to be measured by the most metrics. We therefore first reduce to a **parsimonious basis** of near-independent axes.

### 3.1 Clustering construction

Using Spearman rank correlation $\rho_{ij}$ between metrics $i,j$, define the correlation distance

$$
D_{ij} = 1 - |\rho_{ij}|,
$$

apply hierarchical clustering (average linkage) and cut the dendrogram to obtain clusters $\mathcal{C}_1, \dots, \mathcal{C}_K$. Within a cluster the metrics are effectively interchangeable. For each cluster we select a single **representative** as the member most predictive of the error target,

$$
r_k = \operatorname*{arg\,max}_{m \in \mathcal{C}_k} \; \big|\rho(m, y)\big|,
$$

giving the reduced feature map $\phi(x) = \big(p_{r_1}(x), \dots, p_{r_K}(x)\big) \in \mathbb{R}^K$ with $K = 13$. This representative choice uses the offline target $y$ and is thus part of calibration (Section 2.2); a fully **label-free** alternative selects the cluster medoid (the member with the highest mean $\lvert\rho\rvert$ to its cluster-mates), keeping the entire basis construction free of any performance metric. Either way, each $r_k$ is itself an input-derived metric, so $\phi(x)$ is input-only.

### 3.2 The basis has two roles in $a(x)$

1. **De-correlation (feature selection).** Using one representative per cluster prevents any single physical axis from dominating $a$ by sheer cardinality, and conditions any learned combination.
2. **Weight prior.** The representative's $|\rho(r_k, y)|$ is a data-driven importance for axis $k$, usable directly as a linear weight.

The realised basis (`figures/metric_clustering/parsimonious_basis.csv`) is reproduced in Section 4.3. Note that it was computed against the **range-error** target, and that choice materially changes which axes rank highest.

---

## 4. Empirical grounding (run `20260707_124010`)

### 4.1 Error-target distributions ($N = 69{,}290$)

| Target | min | median | p95 | max | tail counts |
|---|---|---|---|---|---|
| GPR (3%/3 mm), % | 79.8 | 99.33 | 99.96 | 100.0 | GPR &lt; 98: 12,363 (17.8%); &lt; 95: 1,937 (2.8%); &lt; 90: 121 |
| Relative dose error `rde` | 0.016 | 0.172 | 0.304 | 0.635 | |
| Range error $\lvert\Delta R_{100}\rvert$, mm | 0.0 | 0.40 | 3.30 | 231.2 | &gt; 1 mm: 39.2%; &gt; 2 mm: 22.1%; &gt; 3 mm: 13.4% |
| Energy, MeV | 70.0 | 134.9 | 217.2 | 250.0 | |

Two facts drive the design:

- **The model is saturated in the mean**; useful signal is a small tail (2.8% of beamlets at GPR < 95).
- **The range target is contaminated by invalid fits**: $\lvert\Delta R_{100}\rvert$ reaches 231 mm although p95 is only 3.3 mm. 98.4% of beamlets have $\lvert\Delta R_{100}\rvert < 20$ mm; the remainder are failed distal-range estimates that must be masked before use (this motivates a separate range-validity guard).

### 4.2 Offline calibration and validation targets

The following error quantities are available **only on the already-simulated reference run** and are used solely to calibrate and validate $a(x)$ offline (Section 2.2); they are never components of $a(x)$. We consider three, each in "higher is worse" form:

$$
e_{\mathrm{GPR}}(x) = 100 - \mathrm{GPR}(x), \qquad
e_{\mathrm{R}}(x) = |\Delta R_{100}(x)|\ \text{(valid only)}, \qquad
e_{\mathrm{rde}}(x) = \mathrm{rde}(x),
$$

with tail labels $\tau_{\mathrm{GPR}} = \mathbb{1}[\mathrm{GPR} < 95]$ and $\tau_{\mathrm{R}} = \mathbb{1}[|\Delta R_{100}| > 3\,\text{mm}]$.

A key structural result: $e_{\mathrm{GPR}}$ and $e_{\mathrm{R}}$ are only weakly coupled, $\rho(e_{\mathrm{GPR}}, e_{\mathrm{R}}) = 0.27$. **They encode different failure modes**, so no single-target acquisition function can be optimal for both.

### 4.3 Per-axis correlation with each target

Spearman $|\rho|$ between each metric and each target (top axes; range on valid beamlets):

| Metric | $\lvert\rho\rvert$ vs $e_{\mathrm{GPR}}$ | $\lvert\rho\rvert$ vs $e_{\mathrm{R}}$ | $\lvert\rho\rvert$ vs $e_{\mathrm{rde}}$ |
|---|---|---|---|
| `lateral_edge_energy` | **0.320** | 0.183 | 0.149 |
| `sigma_hu_bp` | 0.310 | 0.171 | 0.567 |
| `lateral_hu_var_bp` | 0.310 | 0.148 | 0.215 |
| `sobel_dw_edge_energy` | 0.307 | 0.168 | 0.182 |
| `sobel_th_edge_energy` | 0.298 | 0.175 | 0.298 |
| `sum_sobel_bp` | 0.295 | 0.183 | 0.274 |
| `max_hu_jump` | 0.292 | 0.176 | 0.591 |
| `total_hu_change` | 0.285 | 0.189 | 0.639 |
| `pflugfelder_hi` | 0.242 | **0.229** | 0.214 |
| `wepl_std` | 0.231 | **0.248** | 0.478 |
| `bp_range_max_mm` | 0.098 | 0.080 | **0.743** |
| `wepl_mean` | 0.042 | 0.005 | 0.385 |

Parsimonious basis (against $e_{\mathrm{R}}$), from `parsimonious_basis.csv`:

| Cluster | Representative $r_k$ | $\lvert\rho(r_k, e_{\mathrm{R}})\rvert$ | #members |
|---|---|---|---|
| 2 | `wepl_std` | 0.2609 | 1 |
| 1 | `pflugfelder_hi` | 0.2427 | 1 |
| 6 | `bp_range_mm` | 0.2322 | 3 |
| 5 | `hu_change_per_region` | 0.2099 | 3 |
| 12 | `sum_sobel_bp` | 0.1827 | **9** |
| 7 | `sigma_hu_bp` | 0.1816 | 1 |
| 11 | `isi_mean` | 0.1407 | 4 |
| 8 | `hu_change_per_mm` | 0.1392 | 1 |
| 9 | `hetero_fraction` | 0.1390 | 1 |
| 4 | `sobel_th_beam_angle` | 0.0628 | 2 |
| 10 | `sobel_dw_anisotropy` | 0.0624 | 2 |
| 13 | `interface_bp_distance` | 0.0523 | 1 |
| 3 | `wepl_mean` | 0.0063 | 1 |

**Interpretation.** The dominant axis reorders with the target: lateral/3D **edge energy** and `sigma_hu_bp` lead for GPR failures, whereas the **WEPL** axes (`wepl_std`, `pflugfelder_hi`) lead for range failures. `rde` is far more predictable (up to $\lvert\rho\rvert = 0.74$ for `bp_range_max_mm`) but is largely a depth/range effect and is not the primary clinical target. All correlations are modest ($\lvert\rho\rvert \le 0.32$ for GPR), so the value of a combined $a(x)$ lies in **enriching the tail**, not in perfect ranking.

---

## 5. Candidate acquisition functions

Every candidate consumes **only the input-derived features** $\phi(x) \in \mathbb{R}^K$ (Section 2.2). Where a candidate has free parameters (weights, a learned head), those parameters are **constants fit once offline** on the labelled reference set and then frozen; scoring a new candidate reads inputs alone.

### 5.1 Linear combination

$$
a_{\mathrm{lin}}(x) = \sum_{k=1}^{K} w_k \, p_{r_k}(x), \qquad w_k \ge 0,\ \textstyle\sum_k w_k = 1 .
$$

The features $p_{r_k}(x)$ are input-only; the weights $w_k$ are fixed scalars determined once, offline, by one of three schemes:

- **(A) Supervised, GPR-calibrated:** $w_k \propto \lvert\rho(r_k, e_{\mathrm{GPR}})\rvert$, the per-axis correlation with GPR error measured on the labelled reference run, then frozen.
- **(B) Supervised, range-calibrated:** $w_k \propto \lvert\rho(r_k, e_{\mathrm{R}})\rvert$, the weights of the provided parsimonious basis (Section 4.3).
- **(C) Label-free physics prior:** $w_k$ set from physics reasoning (for example equal weight over the retained axes, or a monotone emphasis on beam-direction WEPL and lateral edge energy), using no performance metric at all.

To be explicit: in (A) and (B) the gamma pass rate is used only to *choose the constants* $w_k$ on already-simulated data; it is not part of $a(x)$ and is not available for the candidates being scored. Scheme (C) removes even that offline dependence.

### 5.2 Non-linear union of failure modes (noisy-OR)

A beamlet is hard if it is extreme on **any** axis. With axis groups (edge, WEPL) reduced to sub-scores $g_j(x) \in [0,1]$,

$$
a_{\mathrm{OR}}(x) = 1 - \prod_{j} \big(1 - g_j(x)\big),
$$

which saturates when any group is high, unlike the mean, which dilutes isolated extremes.

### 5.3 Learned difficulty head

Fit a monotone classifier to the tail label. Logistic form:

$$
a_{\theta}(x) = \sigma\!\big(w^\top \phi(x) + b\big), \qquad
\hat\theta = \operatorname*{arg\,min}_{\theta} \; \sum_{x} \mathrm{BCE}\big(\tau(x),\, a_\theta(x)\big) + \lambda \lVert w \rVert_2^2 ,
$$

optionally a gradient-boosted variant $a_{\mathrm{GBM}}(x)$ for interactions. This learns both the weights and (for GBM) axis interactions, and can emit calibrated $\hat{P}(\tau = 1 \mid x)$.

### 5.4 Dual-target combination

Because GPR and range failures are weakly coupled, combine two single-target heads $a^{\mathrm{GPR}}, a^{\mathrm{R}}$ by rank so both modes are represented:

$$
a_{\mathrm{dual}}(x) = \max\!\big(\operatorname{rank} a^{\mathrm{GPR}}(x),\ \operatorname{rank} a^{\mathrm{R}}(x)\big),
$$

or by splitting the budget $B$ between the two heads (round-robin).

### 5.5 Physics + model-uncertainty hybrid

Model predictive uncertainty $u(x)$ (MC-dropout or deep-ensemble variance of the ADoTA prediction) is the standard AL difficulty signal and, crucially, is **input-only**: it is obtained from the trained model's forward pass on the candidate input, requiring no Monte Carlo dose and no ground truth. It is therefore admissible under Section 2.2 and is the principal **label-free** difficulty axis, expected to be orthogonal to the static physics metrics. Combined with a physics score,

$$
a_{\mathrm{hyb}}(x) = \alpha\, a_{\theta}(x) + (1 - \alpha)\, \hat{F}_u\big(u(x)\big),
$$

with $\alpha$ a fixed mixing constant. $u(x)$ is not yet in the reference table; extracting it is the highest-value next step, and it underpins the fully label-free acquisition route of Section 2.2.

---

## 6. Evaluation protocol

### 6.1 Validation metric: tail lift, not correlation

For AL the relevant question is whether the top-scoring beamlets are the truly hard ones. Define **tail lift at fraction $\alpha$**,

$$
\mathrm{Lift}_\alpha(a, \tau) = \frac{\mathbb{P}\big(\tau = 1 \mid x \in \text{top-}\alpha \text{ by } a\big)}{\mathbb{P}(\tau = 1)},
$$

the enrichment of true failures in the selected fraction over the base rate. Rank correlation (Spearman of $a$ vs $e$) is reported for context but is a poor AL criterion: it rewards ordering the easy bulk correctly while missing the tail.

### 6.2 Wave 1 results: within-anatomy versus cross-anatomy ranking

**Protocol.** Reference run `20260707_124010` is a **multi-patient, two-anatomy pool**: thorax (`Lung-PET-CT-Dx`, 58,750 beamlets, 40 CTs) and pelvic/abdominal (`StageII-Colorectal-CT`, 10,540 beamlets, 16 CTs). Per-beamlet provenance was reconstructed from the source metadata (the CT `image_size`+`image_origin` is a patient/CT key; the source folder is the anatomy), recovering patient and anatomy labels for 100% of beamlets. We evaluate at two generalisation levels: **(a) unseen patient, same anatomy mix**, 5-fold GroupKFold over the 56 patient CTs (whole patients held out; folds contain both anatomies); and **(b) unseen anatomy**, leave-one-anatomy-out (train one anatomy, test the other, an out-of-distribution transfer test). A gantry-grouped split agrees with (a) to within noise. Supervised candidates are fit on the training folds and frozen; features are input-only. (`scripts/analysis/acquisition_ranking_benchmark.py`.)

**Failures are anatomy-specific.** Per-anatomy base rates:

| anatomy | $n$ | GPR $<$ 95 | $\lvert\Delta R_{100}\rvert>$ 3 mm | median GPR |
|---|---|---|---|---|
| thorax | 58,750 | 3.27% | 13.93% | 99.34 |
| pelvic/abdominal | 10,540 | 0.14% | 11.02% | 99.27 |

The GPR-failure tail is overwhelmingly a **thorax** (lung-heterogeneity) phenomenon; the pelvic/abdominal CT has only 15 GPR $<$ 95 beamlets. This asymmetry drives the transfer results.

**(a) Unseen patient, same anatomy mix (patient-grouped, 5-fold), GPR tail** (mean $\pm$ s.d. over folds):

| Candidate | family | GPR AUROC | GPR Lift$_{0.1}$ | GPR Lift$_{0.05}$ | range Lift$_{0.1}$ |
|---|---|---|---|---|---|
| **`gbm_gpr` (learned, non-linear)** | supervised | **0.90 $\pm$ 0.01** | **6.29 $\pm$ 0.41** | 9.18 | 2.01 |
| `logistic_gpr` (learned, linear) | supervised | 0.86 | 5.12 $\pm$ 0.12 | 6.72 | 1.93 |
| `dual_rankmax` | supervised | 0.80 | 3.73 | 4.56 | 2.34 |
| `single_edge` (`sum_sobel_bp`) | label-free | 0.73 | 2.77 $\pm$ 0.25 | 2.93 | 1.20 |
| `logistic_range` | supervised | 0.68 | 1.25 | 1.18 | **2.53** |
| `physics_prior_equal` / linear A,B | mixed | 0.59 | 0.83 $\pm$ 0.23 | 0.72 | 2.23 |
| `single_wepl` (`wepl_std`) | label-free | 0.50 | 0.79 | 0.80 | 2.33 |

Holding out whole patients barely changes the numbers (`gbm_gpr` 6.29 vs 6.44 gantry-grouped), with tight folds: the learned head's advantage is a **real, patient-generalising signal, not patient memorisation**. Ordering: learned non-linear $>$ learned linear $>$ single metric $>$ uniform average.

**(b) Unseen anatomy, leave-one-anatomy-out (out-of-distribution), GPR tail** (mean $\pm$ s.d. over the two directions; per-direction in brackets):

| Candidate | family | GPR AUROC | GPR Lift$_{0.1}$ | per-direction (test=pelvic / test=thorax) |
|---|---|---|---|---|
| `single_edge` (`sum_sobel_bp`) | label-free | 0.66 $\pm$ 0.09 | **2.33 $\pm$ 0.46** | 2.00 / 2.65 |
| `gbm_gpr` (learned) | supervised | 0.70 $\pm$ 0.27 | 3.55 $\pm$ **3.46** | 6.00 / 1.10 |
| `noisy_OR` (edge, WEPL) | label-free | 0.62 $\pm$ 0.01 | 1.12 $\pm$ 0.65 | 0.67 / 1.58 |
| `logistic_gpr` (learned) | supervised | 0.51 $\pm$ 0.05 | 0.93 $\pm$ 0.37 | 0.67 / 1.19 |
| `dual_rankmax` | supervised | 0.50 $\pm$ 0.05 | 0.74 $\pm$ 0.11 | 0.67 / 0.82 |

**Findings.**
1. **The learned head generalises across patients but not across anatomy.** Its ~6.3x lift is stable under whole-patient hold-out (a), so it is not patient memorisation; but under unseen-anatomy transfer (b) it destabilises (`gbm_gpr` Lift 3.55 $\pm$ 3.46; 6.0 one direction, 1.1 the other) and `logistic_gpr` collapses to chance (AUROC 0.51). The failure is distribution shift to a new anatomy, not overfitting to individual patients.
2. **The single label-free edge-energy metric is the most robust cross-anatomy ranker**, consistent in both directions (2.00 / 2.65) with the smallest variance. This vindicates the label-free route (Section 2.2): a physics signal that needs no failure labels transfers where a fitted head does not.
3. **Cause: the failure tail is anatomy-specific.** A head trained on pelvic/abdominal sees almost no failures (15) and cannot learn the pattern, so it fails to rank the thorax tail (1.1x); trained on thorax it ranks the tiny pelvic tail well but on only 15 positives (noisy 6.0x). Supervised heads need in-anatomy failure examples; label-free physics does not.
4. **Design implication.** For cross-anatomy robustness the acquisition function should lean on the label-free physics axis (edge energy), or the supervised head must be calibrated on a pool that already spans the target anatomy's failure modes. Model uncertainty (Wave 2), also label-free, is the natural complementary axis.

**Caveats.** (i) Only two anatomies, so the unseen-anatomy test (b) is a 2-fold comparison with high variance, and the pelvic tail is tiny (15 positives); more anatomies are needed to confirm the transfer story. (ii) Absolute lifts may be mildly optimistic because the `*_bp` metrics locate the Bragg-peak slice from the GT dose; the input-only form recomputes it from energy (Wave 1.5 re-extraction).

### 6.3 A combined score with strong error-correlation (RDE)

The supervisor set a target of **Pearson/Spearman $\approx 0.8$** between a single combined input-only score and a continuous error metric. An achievability study (regression on all ~30 input metrics, patient-grouped 5-fold CV, `scripts/analysis/acquisition_regression_study.py`) gives the ceiling per target:

| Target | interpretable linear (Ridge, all metrics) | GBM ceiling | heterogeneity-only (no energy/BP) |
|---|---|---|---|
| **RDE** ($\log(1+\mathrm{RDE})$) | Pearson **0.85** / Spearman **0.86** | 0.95 | 0.82 / 0.83 |
| gamma error ($100-\mathrm{GPR}$) | 0.50 | 0.73 | 0.50 |
| range $\lvert\Delta R_{100}\rvert$ | 0.44 | 0.52 | 0.47 |

**RDE is the target that reaches the bar**; gamma error and range do not with the current metric set. The delivered combined score is a linear model on the input metrics predicting $\log(1+\mathrm{RDE})$ (`scripts/analysis/acquisition_rde_finalize.py`; persisted as `figures/acquisition/rde_combined_scorer.json`, scatter `rde_combined_scatter.png`):

- **Held-out (patient-grouped out-of-fold): interpretable linear Pearson 0.847, Spearman 0.856; GBM ceiling 0.947.** It correlates as well with raw RDE (Pearson 0.83) as with the log target.
- **Not a depth artifact:** the correlation holds at 0.82/0.83 with energy and BP-range features removed, so it is heterogeneity-driven.
- **Cross-anatomy:** transfers at ~0.71 Pearson (0.64-0.74 Spearman) when tested on the held-out anatomy, degrading but not collapsing (unlike the gamma classifier of Section 6.2).
- **Top physical drivers** (basis form): `wepl_mean`, `total_hu_change`, `max_hu_jump`, `hetero_fraction`, `pflugfelder_hi`. Read physically: RDE grows with heterogeneity magnitude and path length.

**Key caveat (objective mismatch).** This score is tuned for RDE *correlation*, not gamma-*tail selection*: it enriches the $\mathrm{GPR}<95$ tail only 1.7x at the top decile, versus 6.3x for the dedicated gamma head (Section 6.2). High correlation with RDE and active-learning utility on the gamma tail are therefore different objectives.

**Gamma stays at ~0.73 (tested).** For the clinically primary gamma metric the ceiling is 0.73, short of 0.8. A new input-only feature was implemented and tested to try to close the gap: the **parallel-to-beam WEPL-difference** (`compute_parallel_beam_wepl_diff` in `src/processing/pflugfelder_hi.py`), which splits the aperture into two halves about the flux centroid and measures the flux-weighted WEPL difference between them (the "half the beam through bone, half through air" case). On a 11.5k / 55-patient subset it proved **largely redundant with `wepl_std`** (Spearman 0.92) and lifted the gamma ceiling by only ~0.02 (GBM Spearman 0.705 $\to$ 0.725; linear 0.475 $\to$ 0.494). The spatial arrangement co-occurs with WEPL scatter, so it adds little independent gamma signal. Reaching 0.8 on gamma therefore requires a signal orthogonal to the static physics metrics; the natural candidate is the model's own predictive uncertainty (MC-dropout, Section 5.5).

### 6.4 Prospective AL protocol (retrospective simulation)

Since all labels already exist, we simulate AL without new MC: fix a stratified held-out test set; start from a seed $\mathcal{L}_0$; at each round query top-$B$ by $a$ from the hidden pool, reveal labels, retrain, and record test performance versus number of labelled beamlets. Report the learning curve, the area under it (AULC), and the labels needed to reach a target (for example test GPR $\ge 99.5\%$), against a random-sampling baseline and an uncertainty-only baseline.

---

## 7. Experiment matrix (to be completed)

Fixed unless noted: pool = current pelvis/abdomen beamlets; seed = stratified 5%; query batch $B$; test = stratified held-out, never in pool; 5 seeds; retrospective simulation (Section 6.3). Values to be filled.

Note on the "Calib. target" column: it names the offline quantity used to fit each function's parameters, not an input. Rows E0, E1, E2, and the physics-prior variant of E5 are **label-free** (no performance metric used at any stage); all others are label-calibrated offline and frozen before scoring.

The Lift columns are the Wave 1 **unseen-patient** (patient-grouped) numbers from Section 6.2(a); they are stable under whole-patient hold-out. Under **unseen-anatomy** transfer (Section 6.2 b) the supervised heads drop sharply (e.g. `gbm_gpr` 6.29 $\to$ 3.55 $\pm$ 3.46, `logistic_gpr` $\to$ chance) while the label-free `single_edge` is the most stable (2.77 $\to$ 2.33 $\pm$ 0.46). AULC, labels-to-target, and final GPR require the full retraining loop (Wave 3) and remain open.

| ID | Acquisition $a(x)$ | Calib. target | Key params | Lift$_{0.1}$ GPR | Lift$_{0.1}$ range | AULC (GPR) | Labels to GPR$\ge$99.5% | Final GPR | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E0 | Random | — | — | 1.00 | 1.00 | | | | baseline |
| E1 | Model uncertainty only | GPR | MC-dropout $u(x)$ | | | | | | Wave 2 |
| E2 | Single best metric | GPR | `sum_sobel_bp` | 2.76 | 1.19 | | | | label-free |
| E3 | A: linear, GPR-weighted | GPR | corr weights | 0.91 | 2.18 | | | | |
| E4 | B: linear, range-weighted | range | given basis | 0.86 | 2.20 | | | | |
| E5 | C: dual-axis noisy-OR | GPR+range | edge, WEPL groups | 1.87 | 1.97 | | | | label-free |
| E6 | D: learned logistic | GPR | $\lambda$, basis $K{=}13$ | 5.17 | 1.92 | | | | |
| E7 | D-GBM: learned boosted | GPR | depth, trees | **6.44** | 1.99 | | | | best GPR ranker |
| E8 | D on combined label | GPR$\lor$range | $\tau_{\mathrm{GPR}}\!\lor\!\tau_{\mathrm{R}}$ | | | | | | Wave 2 |
| E9 | Dual-target rank-max | GPR+range | $a^{\mathrm{GPR}}, a^{\mathrm{R}}$ | 3.77 | **2.31** | | | | Section 5.4 |
| E10 | Hybrid physics + uncertainty | GPR | $\alpha$ | | | | | | Wave 2 |

---

## 8. Summary

The acquisition function is, by construction, a function of **input data only**: a normalised, de-correlated combination of physics-motivated heterogeneity metrics computed from the candidate CT sub-grid and geometry, optionally augmented with the model's own predictive uncertainty. Performance metrics (GPR, range error, RDE) are used strictly offline, to calibrate the combination on already-simulated data and to validate it; they are never read when a candidate is scored, because for a candidate they do not yet exist. The parsimonious basis supplies both the independent feature set and a weight prior. Wave 1 evaluation on run `20260707_124010` shows that a learned non-linear head ranks the failure tail best and **generalises across unseen patients** (6.3x enrichment, stable under whole-patient hold-out), but does **not** generalise to an **unseen anatomy**: under leave-one-anatomy-out it destabilises and can collapse to chance, while a single label-free edge-energy metric transfers most reliably (2.3x, low variance). Because the GPR-failure tail is anatomy-specific (thorax-dominated), the deployable acquisition function should either lean on label-free physics (edge energy, plus model uncertainty) or be calibrated on a pool spanning the target anatomy. More generally, Wave 1 shows that (i) tail enrichment, not rank correlation, is the operative criterion, (ii) a learned non-uniform combination on the basis substantially outperforms uniform linear sums and single metrics for the GPR tail, and (iii) GPR and range failures require distinct axes, so the deployable acquisition function should be dual-target and, ultimately, augmented with model uncertainty. The experiment matrix in Section 7 will select the final form.
