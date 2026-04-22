"""
Shared dataclass schemas for ADoTA scripts.

Submodules
----------
configs
    Configuration dataclasses that control script behaviour.
results
    Per-sample / per-beamlet result containers produced by inference scripts.
analysis
    Dataclasses for pure analysis pipelines (no model inference).
"""

from src.schemas.analysis import BeamletRecord, BeamletResult, SweepPoint
from src.schemas.configs import (
    AdvancedAnalysisConfig,
    AnalysisConfig,
    EvaluationConfig,
    MetricsConfig,
    VLMConfig,
)
from src.schemas.results import (
    BPRecord,
    EvaluationResult,
    H5EvaluationResult,
    SampleRecord,
    SampleResult,
    VLMResult,
)

__all__ = [
    # configs
    "AnalysisConfig",
    "AdvancedAnalysisConfig",
    "EvaluationConfig",
    "MetricsConfig",
    "VLMConfig",
    # results
    "BPRecord",
    "EvaluationResult",
    "H5EvaluationResult",
    "SampleRecord",
    "SampleResult",
    "VLMResult",
    # analysis
    "BeamletRecord",
    "BeamletResult",
    "SweepPoint",
]
