# Annotated Bibliography: Active Learning for Surrogate Dose Models

Every entry links to the source for deeper reading. Local PDFs live in
[../publications/](../publications/) and are linked with relative paths (click to
open in the IDE). External works link to the canonical URL/DOI.

Companion document: [active_learning_literature_review.md](active_learning_literature_review.md).

---

## A. Delivered corpus (local PDFs)

### A.1 Active learning and adaptive sampling: methodology

| # | Work | Local PDF | Source |
|---|------|-----------|--------|
| 1 | Seung, Opper & Sompolinsky (1992), *Query by Committee* | [query_ny_committee.pdf](../publications/query_ny_committee.pdf) | [ACM](https://dl.acm.org/doi/10.1145/130385.130417) |
| 2 | Liu, Ong & Cai (2018), *A survey of adaptive sampling for global metamodeling* | [adaptive_survey_engineering.pdf](../publications/adaptive_survey_engineering.pdf) | [doi:10.1007/s00158-017-1739-8](https://doi.org/10.1007/s00158-017-1739-8) |
| 3 | Chellappa, Feng & Benner (2020), *An Adaptive Sampling Approach for the Reduced Basis Method* | [an_adaptive_Sampling_approach_for_the_RBM.pdf](../publications/an_adaptive_Sampling_approach_for_the_RBM.pdf) | [arXiv:1910.00298](https://arxiv.org/abs/1910.00298) |
| 4 | Wang, Tang, Zhai, Wan & Yang (2024), *Deep Adaptive Sampling for Surrogate Modeling Without Labeled Data* | [deep_adaptive_Sampling.pdf](../publications/deep_adaptive_Sampling.pdf) | [doi:10.1007/s10915-024-02711-1](https://doi.org/10.1007/s10915-024-02711-1) |
| 5 | Xian & Wang (2024), *A physics and data co-driven surrogate modeling method for high-dimensional rare event simulation* | [physics_and_data_driven_surrogate_modeling.pdf](../publications/physics_and_data_driven_surrogate_modeling.pdf) | [doi:10.1016/j.jcp.2024.113069](https://doi.org/10.1016/j.jcp.2024.113069) |
| 6 | Li, Yu, Xing, Kirby, Narayan & Zhe (2023), *Multi-Resolution Active Learning of Fourier Neural Operators* | [multi_resulution_active_learning.pdf](../publications/multi_resulution_active_learning.pdf) | [arXiv:2309.16971](https://arxiv.org/abs/2309.16971) |
| 7 | Kirsch, Farquhar, Atighehchian, Jesson, Branchaud-Charron & Gal (2023), *Stochastic Batch Acquisition: A Simple Baseline for Deep Active Learning* (SBAL) | [SBAL.pdf](../publications/SBAL.pdf) | [OpenReview](https://openreview.net/forum?id=vcHwQyNBjW) |
| 8 | *A novel adaptive sampling approach with batch selection for surrogate models in geotechnical engineering* (2025) | [geotechnical batch-selection paper](../publications/a-novel-adaptive-sampling-approach-with-batch-selection-for-the-automatic-generation-of-surrogate-models-in-geotechnical-engineering.pdf) | [doi:10.1017/dce.2025.10036](https://doi.org/10.1017/dce.2025.10036) |
| 9 | Stolte, Daru, Forbert, Marx & Behler (2025), *Random Sampling Versus Active Learning Algorithms for Machine Learning Potentials of Quantum Liquid Water* | [quantum-liquid-water paper](../publications/random-sampling-versus-active-learning-algorithms-for-machine-learning-potentials-of-quantum-liquid-water.pdf) | [doi:10.1021/acs.jctc.4c01382](https://doi.org/10.1021/acs.jctc.4c01382) |

### A.2 Radiotherapy and proton-therapy domain

| # | Work | Local PDF | Source |
|---|------|-----------|--------|
| 10 | Stryja, Lathouwers & Perkó, *Angle Dependent Dose Transformer Algorithm for fast proton dose calculations* (ADoTA base model) | [AngleDependentDoseTransformerAlgorithm.pdf](../publications/AngleDependentDoseTransformerAlgorithm.pdf) | project's own base paper |
| 11 | Pflugfelder, Wilkens, Szymanowski & Oelfke (2007), *Quantifying lateral tissue heterogeneities in hadron therapy* | [Pflugfelder 2007](../publications/Medical%20Physics%20-%202007%20-%20Pflugfelder%20-%20Quantifying%20lateral%20tissue%20heterogeneities%20in%20hadron%20therapy.pdf) | [doi:10.1118/1.2710329](https://doi.org/10.1118/1.2710329) |
| 12 | Bueno, Paganetti, Duch & Schuemann (2013), *An algorithm to assess the need for clinical Monte Carlo dose calculation for small proton fields* | [Bueno 2013](../publications/Medical%20Physics%20-%202013%20-%20Bueno%20-%20An%20algorithm%20to%20assess%20the%20need%20for%20clinical%20Monte%20Carlo%20dose%20calculation%20for%20small%20proton.pdf) | [doi:10.1118/1.4812682](https://doi.org/10.1118/1.4812682) |
| 13 | Albertini et al. (2024), *First clinical implementation of a highly efficient daily online adapted proton therapy (DAPT) workflow* | [Albertini 2024](../publications/Albertini_2024_Phys._Med._Biol._69_215030.pdf) | [doi:10.1088/1361-6560/ad7cbd](https://doi.org/10.1088/1361-6560/ad7cbd) |
| 14 | Rodríguez Outeiral et al. (2023), *A network score-based metric to optimize QA of automatic radiotherapy target segmentations* | [Outeiral_etal.pdf](../publications/Outeiral_etal.pdf) | [doi:10.1016/j.phro.2023.100500](https://doi.org/10.1016/j.phro.2023.100500) |
| 15 | Jungo, Balsiger & Reyes (2020), *Analyzing the Quality and Challenges of Uncertainty Estimations for Brain Tumor Segmentation* | [fnins-14-00282.pdf](../publications/fnins-14-00282.pdf) | [doi:10.3389/fnins.2020.00282](https://doi.org/10.3389/fnins.2020.00282) |

---

## B. Recommended additions (external, not yet in the corpus)

### B.1 Active-learning methodology backbone

| Work | Why it matters | Source |
|------|----------------|--------|
| Settles (2009), *Active Learning Literature Survey* | The canonical taxonomy (uncertainty, QBC, EMC, EER, density). Baseline vocabulary. | [tech report](https://minds.wisconsin.edu/handle/1793/60660) |
| Ren et al. (2021), *A Survey of Deep Active Learning* | Modern deep-AL taxonomy; locates batch and representation methods. | [doi:10.1145/3472291](https://doi.org/10.1145/3472291) |
| Sener & Savarese (2018), *Active Learning for CNNs: A Core-Set Approach* | The canonical diversity/coreset baseline (k-center greedy). | [arXiv:1708.00489](https://arxiv.org/abs/1708.00489) |
| Kirsch, van Amersfoort & Gal (2019), *BatchBALD* | Principled batch-mode Bayesian AL; solves intra-batch redundancy. | [arXiv:1906.08158](https://arxiv.org/abs/1906.08158) |
| Lakshminarayanan, Pritzel & Blundell (2017), *Deep Ensembles* | Practical uncertainty backbone for the uncertainty-AL baseline. | [arXiv:1612.01474](https://arxiv.org/abs/1612.01474) |
| Cai, Zhang & Zhou (2013), *Maximizing Expected Model Change for AL in Regression* | Regression-specific acquisition (most AL theory is classification). | [IEEE](https://ieeexplore.ieee.org/document/6729489/) |
| Park & Kim (2020), *Robust Expected Model Change for AL in Regression* | Outlier-robust EMC for regression. | [doi:10.1007/s10489-019-01519-z](https://doi.org/10.1007/s10489-019-01519-z) |
| Holzmüller et al. (2023), *Black-Box Batch Active Learning for Regression* | Batch AL designed for regression with deep nets. | [arXiv:2302.08981](https://arxiv.org/abs/2302.08981) |
| Munjal et al. (2022), *Towards Robust and Reproducible Active Learning Using Neural Networks* | Shows AL gains vanish under proper tuning/seeds; evaluation hygiene. | [arXiv:2002.09564](https://arxiv.org/abs/2002.09564) |
| Lüth et al. (2023), *Navigating the Pitfalls of Active Learning Evaluation* | A referee-proof AL evaluation protocol. | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/1ed4723f12853cbd02aecb8160f5e0c9-Paper-Conference.pdf) |

### B.2 AI in radiotherapy: wider context

| Work | Task | Source |
|------|------|--------|
| Nguyen et al. (2019), *3D radiotherapy dose prediction, hierarchically dense U-Net* | Dose prediction (photon KBP) | [arXiv:1805.10397](https://arxiv.org/abs/1805.10397) |
| Deep-learning field dose prediction for spot-scanning proton therapy (2023) | Proton dose prediction | [PMC10312803](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10312803/) |
| Wu et al. (2021), *Deep Dose Plugin: real-time MC dose via DL denoising* | MC dose acceleration | [doi:10.1088/2632-2153/abdbfe](https://iopscience.iop.org/article/10.1088/2632-2153/abdbfe) |
| Neph et al. (2020), *Improving Proton Dose Calculation Accuracy Using Deep Learning* | DL proton dose engine | [arXiv:2004.02924](https://arxiv.org/abs/2004.02924) |
| Wu et al. (2020), *LSTM networks for proton dose calculation in highly heterogeneous tissues* | Direct DoTA lineage; heterogeneity focus | [arXiv:2006.06085](https://arxiv.org/abs/2006.06085) |
| Wang et al. (2021), *AIDE: Annotation-efficient deep learning for medical image segmentation* | Label-efficient learning | [Nature Comms](https://www.nature.com/articles/s41467-021-26216-9) |
| *Deep learning for autosegmentation in RT: state of the art* (2024) | Autosegmentation review | [doi:10.1007/s00066-024-02262-2](https://link.springer.com/article/10.1007/s00066-024-02262-2) |

_Last updated: 2026-07-02._
