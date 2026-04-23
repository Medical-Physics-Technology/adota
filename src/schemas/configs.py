"""Configuration dataclasses for ADoTA scripts."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.adota.config import DEFAULT_GAMMA_PARAMS, DEFAULT_SCALE

# ── Evaluation (model inference) ────────────────────────────────────────────


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Used by ``run_model.py`` and ``analysis_texture_with_inference.py``.
    """

    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())
    gamma_params: dict = field(default_factory=lambda: DEFAULT_GAMMA_PARAMS.copy())
    normalize_flux: bool = True
    resolution: tuple = (2.0, 2.0, 2.0)


# ── Texture metrics toggles ────────────────────────────────────────────────


@dataclass
class MetricsConfig:
    """Toggle flags for the three texture-metric families.

    Each flag can be set via CLI (``--enable-heterogeneity`` etc.) or in the
    YAML config under the ``metrics`` key.
    """

    heterogeneity: bool = True
    glcm: bool = True
    intensity: bool = True


# ── Analysis (training set) ─────────────────────────────────────────────────


@dataclass
class AnalysisConfig:
    """Configuration for the tissue-interface analysis.

    Used by ``training_set_analysis.py``.
    """

    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())
    gamma_params: dict = field(default_factory=lambda: DEFAULT_GAMMA_PARAMS.copy())
    resolution: tuple = (2.0, 2.0, 2.0)
    bp_radius_mm: float = 5.0  # radius of spherical neighbourhood [mm]
    sigma_hu_threshold: float = 150.0  # σ_HU threshold for "interface"
    max_energy_mev: float = 250.0  # skip beamlets above this energy [MeV]


@dataclass
class AdvancedAnalysisConfig:
    """Configuration for the advanced-metrics analysis.

    Used by ``training_set_analysis_advanced_metrics.py``.
    Extends :class:`AnalysisConfig` with smoothing, energy-binning,
    Bragg-peak estimation, and Sobel parameters.
    """

    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())
    gamma_params: dict = field(default_factory=lambda: DEFAULT_GAMMA_PARAMS.copy())
    resolution: tuple = (2.0, 2.0, 2.0)
    max_energy_mev: float = 250.0
    smoothing_sigma: float = 1.0
    smoothing_method: str = "gaussian"  # "gaussian" or "median"
    energy_bins: list = field(
        default_factory=lambda: [70, 100, 130, 160, 190, 220, 250]
    )
    # Bragg-peak range estimation
    proximal_fraction: float = 0.50
    fall_fraction: float = 0.10
    # Flux weighting
    flux_threshold_frac: float = 0.10
    # Sobel percentile
    sobel_percentile: float = 95.0
    # Region method: "bp_range" (z_min-z_max along beam) or "sphere"
    region_method: str = "bp_range"
    # Sphere radius [mm] for sphere-based Sobel analysis
    sphere_radius_mm: float = 10.0
    # Whether to compute Sobel on raw CT (True) or smoothed CT (False)
    sobel_use_raw: bool = False
    # Interface Severity Index — severity weighting mode
    # ("rsp_sq", "rsp_abs", or "density_sq")
    isi_severity_mode: str = "rsp_sq"


# ── VLM-based analysis ──────────────────────────────────────────────────────


@dataclass
class VLMConfig:
    """Configuration for the VLM-based analysis.

    Used by ``training_set_vlm_based_quantification.py``.

    .. note::
       The ``vlm_prompt`` default is intentionally empty.  The script
       should set it from its own ``DEFAULT_VLM_PROMPT`` constant.
    """

    # Data / model
    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())
    gamma_params: dict = field(default_factory=lambda: DEFAULT_GAMMA_PARAMS.copy())
    resolution: tuple = (2.0, 2.0, 2.0)
    max_energy_mev: float = 250.0
    smoothing_sigma: float = 1.0
    smoothing_method: str = "gaussian"

    # BP estimation (using GT for now)
    proximal_fraction: float = 0.50
    fall_fraction: float = 0.10
    flux_threshold_frac: float = 0.10

    # VLM settings
    vlm_model: str = "llava-hf/llava-1.5-7b-hf"
    vlm_device_index: int = 0  # GPU index for VLM (can differ from ADoTA)
    vlm_temperature: float = 0.0
    vlm_n_votes: int = 3
    vlm_prompt: str = ""
    vlm_max_new_tokens: int = 256

    # Output
    save_panels: bool = True

    # Energy bins for stratified analysis
    energy_bins: list = field(
        default_factory=lambda: [70, 100, 130, 160, 190, 220, 250]
    )
