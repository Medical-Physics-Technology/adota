"""Result / record dataclasses produced by ADoTA inference scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


# ── Model evaluation results ────────────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Container for a single sample's evaluation results.

    Used by ``run_model.py`` and ``analysis_texture_with_inference.py``.
    """

    sample_id: str
    energy_mev: float
    beamlet_angles: tuple
    gpr: float
    rmse: float
    mape: float
    rde: float
    calc_time: float
    prediction: Optional[torch.Tensor] = field(default=None, repr=False)
    ground_truth: Optional[torch.Tensor] = field(default=None, repr=False)
    input_data: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass
class H5EvaluationResult:
    """Container for a single sample's evaluation results (HDF5 pipeline).

    Used by ``run_model_h5py.py``.  Differs from :class:`EvaluationResult`
    in that it carries ``tv`` / ``cv`` instead of ``beamlet_angles`` / ``gpr``.
    """

    sample_id: str
    energy_mev: float
    rmse: float
    mape: float
    rde: float
    tv: float
    cv: float
    calc_time: float
    prediction: Optional[torch.Tensor] = field(default=None, repr=False)
    ground_truth: Optional[torch.Tensor] = field(default=None, repr=False)
    input_data: Optional[torch.Tensor] = field(default=None, repr=False)


# ── Texture-with-inference results ──────────────────────────────────────────


@dataclass
class SampleResult:
    """Container for one sample's combined inference + texture results.

    Used by ``analysis_texture_with_inference.py``.
    """

    sample_id: str
    energy_mev: float
    beamlet_angles: tuple

    # Model performance metrics
    gpr: float
    rmse: float
    mape: float
    rde: float
    calc_time: float

    # Beam-aligned heterogeneity metrics (optional)
    g_phi: Optional[float] = None
    r_roughness: Optional[float] = None
    h_phi: Optional[float] = None

    # GLCM homogeneity — volume-level mean (optional)
    glcm_homogeneity_mean: Optional[float] = None
    glcm_homogeneity_std: Optional[float] = None

    # Global intensity heterogeneity (optional)
    intensity_mean: Optional[float] = None
    intensity_std: Optional[float] = None
    intensity_iqr: Optional[float] = None
    intensity_mad: Optional[float] = None
    intensity_entropy: Optional[float] = None
    intensity_uniformity: Optional[float] = None
    intensity_skewness: Optional[float] = None
    intensity_kurtosis: Optional[float] = None

    # Cached tensors for publication figures
    prediction: Optional[torch.Tensor] = field(default=None, repr=False)
    ground_truth: Optional[torch.Tensor] = field(default=None, repr=False)
    input_data: Optional[torch.Tensor] = field(default=None, repr=False)


# ── Advanced-metrics sample record ──────────────────────────────────────────


@dataclass
class SampleRecord:
    """Container for a single beamlet's extracted data summary.

    Used by ``training_set_analysis_advanced_metrics.py``.
    """

    sample_id: str
    energy_mev: float
    ct_min_hu: float
    ct_max_hu: float
    flux_max: float
    gt_dose_min: float
    gt_dose_max: float
    bp_range_min_mm: float
    bp_range_max_mm: float
    max_grad_depth_mm: float
    n_density_regions: int
    total_hu_change: float
    max_hu_jump: float
    sigma_hu_bp: float
    max_hu_gradient: float
    lateral_hu_var_bp: float
    hetero_fraction: float
    interface_bp_distance: float
    mean_sobel_axial: float
    p95_sobel_bp: float
    sum_sobel_bp: float
    # Method DW: continuous dose-weighted structure-tensor metrics
    sobel_dw_mean: float = 0.0
    sobel_dw_anisotropy: float = 0.0
    sobel_dw_beam_angle: float = 0.0
    sobel_dw_edge_energy: float = 0.0
    # Method TH: binary-threshold structure-tensor metrics (5 % global dose)
    sobel_th_mean: float = 0.0
    sobel_th_anisotropy: float = 0.0
    sobel_th_beam_angle: float = 0.0
    sobel_th_edge_energy: float = 0.0
    pflugfelder_hi: float = 0.0
    wepl_mean: float = 0.0
    wepl_std: float = 0.0
    isi_sum: float = 0.0
    isi_max: float = 0.0
    isi_mean: float = 0.0
    isi_axial_sum: float = 0.0
    gpr: float = 0.0
    rde: float = 0.0
    extract_time: float = 0.0


# ── VLM result ──────────────────────────────────────────────────────────────


@dataclass
class VLMResult:
    """Per-beamlet result from VLM-based evaluation.

    Used by ``training_set_vlm_based_quantification.py``.
    """

    sample_id: str
    energy_mev: float
    bp_range_min_mm: float
    bp_range_max_mm: float
    n_density_regions: int
    vlm_difficulty: float  # median of K votes
    vlm_confidence: float  # median of K votes
    vlm_reason: str  # from the median vote
    vlm_votes: str  # JSON list of all votes
    gpr: float
    panel_path: str
    query_time_s: float


# ── Bragg-peak estimation record ────────────────────────────────────────────


@dataclass
class BPRecord:
    """One row per beamlet: BP depth [mm] from each method.

    Used by ``bragg_peak_estimation.py``.
    """

    sample_id: str
    energy_mev: float
