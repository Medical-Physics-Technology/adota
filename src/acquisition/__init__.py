"""Input-only difficulty scoring of candidate beamlets for active learning.

- :mod:`.bragg_curve`: Bortfeld's analytic Bragg curve with the beam's own
  energy spread.
- :mod:`.surrogate`: the analytic pencil-beam dose that locates the peak.
- :mod:`.features`: the thirty metrics of the difficulty score.
- :mod:`.scorer`: the frozen weighted-percentile score.
- :mod:`.candidates`: the single call that scores candidates on a full CT.
- :mod:`.reference`: the study's record path, for validating all of the above.
"""
from src.acquisition.candidates import BeamletCandidate, prepare_ct, score_candidates
from src.acquisition.features import FEATURE_NAMES, FeatureConfig, compute_features
from src.acquisition.scorer import DifficultyScorer
from src.acquisition.surrogate import analytic_dose, peak_inside_crop

__all__ = ["BeamletCandidate", "prepare_ct", "score_candidates", "FEATURE_NAMES", "FeatureConfig",
           "compute_features", "DifficultyScorer", "analytic_dose", "peak_inside_crop"]
