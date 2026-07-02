"""Configuration dataclasses for ADoTA scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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


# ── Training (DoTA model fitting) ───────────────────────────────────────────


@dataclass
class TrainingConfig:
    """Configuration for an ADoTA training run.

    Used by ``scripts/train_adota.py``. A YAML config under
    ``scripts/config_train_adota.yaml`` populates this dataclass; CLI
    flags override individual fields.
    """

    # ── Run identification ──────────────────────────────────────────────
    config_name: str = "default"

    # Base directory under which each run directory
    # (``train_<timestamp>_<config_name>/``) is created. All training
    # artifacts for a run live inside it.
    runs_dir: str = "/scratch/mstryja/adota_runs"

    # ── Data ────────────────────────────────────────────────────────────
    dataset_path: str = ""
    excluded_indexes_file: str = ""
    train_test_split: float = 0.2
    split_random_state: int = 42
    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())

    # ── DataLoader ──────────────────────────────────────────────────────
    batch_size: int = 56
    num_workers: int = 4
    augmentation: bool = True
    normalize_flux_only: bool = True

    # ── Optimization ────────────────────────────────────────────────────
    num_epochs: int = 400
    learning_rate: float = 5e-4
    weight_decay: float = 1e-3
    patience: int = 100  # early-stopping patience (epochs)
    seed: int = 1234
    device_index: int = 0

    # ReduceLROnPlateau scheduler
    lr_factor: float = 0.9
    lr_patience: int = 20

    # ── Acceleration (opt-in; defaults keep the FP32 eager path untouched) ──
    # compile: wrap the model with torch.compile for fused forward/backward.
    # compile_mode: one of "default" / "reduce-overhead" / "max-autotune".
    # allow_tf32: enable TF32 matmul/conv kernels (small numeric change, large
    #   throughput gain on Ampere+). Both relax cuDNN determinism when enabled.
    compile: bool = False
    compile_mode: str = "default"
    allow_tf32: bool = False

    # ── Loss balancing ──────────────────────────────────────────────────
    # Initial static weights for the (LMSE, LPS) pair. After
    # ``adaptive_after_epoch`` the TwoObjectiveBalancer takes over.
    #
    # loss_mode controls which objectives are active:
    #   "mse_idd"  -- LMSE + LPS combined via TwoObjectiveBalancer (default)
    #   "mse_only" -- LMSE only; LPS and the balancer are skipped
    loss_mode: str = "mse_idd"

    initial_weight_mse: float = 0.7
    initial_weight_ps: float = 0.3
    balancer_smoothing: float = 0.9
    adaptive_after_epoch: int = 10_000  # disable adaptive balancing by default

    # LPS pixel spacing (used to weight the IDD integral)
    lps_dx_mm: float = 2.0
    lps_dy_mm: float = 2.0

    # ── Validation cadence ──────────────────────────────────────────────
    # RMSE/MAPE/RDE are computed every epoch on the full val set.
    # GPR is computed every ``gpr_every_n_epochs`` on a fixed, persisted pool.
    gpr_every_n_epochs: int = 10
    # gpr_subset_size is the (stable) pool size; gpr_comparable_size is the
    # nested subset reported for apples-to-apples comparison with prior runs.
    # The pool is frozen before the loop and written to run_dir/gpr_samples.json.
    gpr_subset_size: int = 50
    gpr_comparable_size: int = 20
    # Optional path to a gpr_samples.json to pin the exact gamma set (overrides
    # the deterministic draw). Empty => draw fresh (and persist).
    gpr_samples_file: str = ""
    gamma_params: dict = field(default_factory=lambda: DEFAULT_GAMMA_PARAMS.copy())
    # Voxel spacing used by the gamma index (mm). Matches the downsampled
    # (160, 30, 30) grid produced by the H5 loader.
    gpr_resolution_mm: Tuple[float, float, float] = (2.0, 2.0, 2.0)

    # Per-energy validation breakdown
    energy_bins_fixed: List[float] = field(
        default_factory=lambda: [
            70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0,
            140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
            210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0,
        ]
    )
    energy_bins_quantile_n: int = 10

    # ── Worst-case sample tracking ─────────────────────────────────────
    worst_k_samples: int = 10

    # ── Checkpoint retention ───────────────────────────────────────────
    # "best.pth" and "last.pth" are always saved; in addition every
    # ``checkpoint_every_n_epochs`` an ``epoch_NNNN.pth`` snapshot is kept.
    checkpoint_every_n_epochs: int = 25

    # ── Attention canary ────────────────────────────────────────────────
    # Only used when ``num_transformers > 0``. Saves the attention maps
    # for a fixed validation sample every K epochs.
    attention_every_n_epochs: int = 10

    # ── Run-control flags ──────────────────────────────────────────────
    max_hours: Optional[float] = None  # wall-time budget
    smoke_test: bool = False  # 2 epochs, 4 batches/epoch, exit cleanly
    resume_from: Optional[str] = None  # path to a checkpoint .pth
    # Warm-start: load only the model weights from resume_from (fresh
    # optimizer / scheduler / epoch counter). Used to fine-tune a prior best
    # checkpoint with a new LR schedule.
    weights_only: bool = False

    # ── Model hyperparameters (passed verbatim to DoTA3D_v3) ───────────
    input_shape: Tuple[int, int, int, int] = (2, 160, 30, 30)
    num_transformers: int = 1
    num_heads: int = 4
    # Feed-forward hidden dim of the transformer layers. None -> defaults to the
    # model's token_size (original behavior).
    dim_feedforward: Optional[int] = None
    num_levels: int = 4
    enc_features: int = 32
    kernel_size: int = 3
    convolutional_steps: int = 2
    conv_hidden_channels: int = 128
    dropout_rate: float = 0.1
    causal: bool = True
    zero_padding: bool = True
    last_activation: bool = False
    num_forward: int = 2

    # ── Optional free-form notes saved next to the run ─────────────────
    comment: str = ""

    # ── Ablation controls ───────────────────────────────────────────────
    # flux_mode selects what the second input channel carries:
    #   "analytical"      -- the raw H5 flux projection (default)
    #   "angle_broadcast" -- a constant volume whose value is
    #                        sqrt(theta_x^2 + theta_y^2), removing all
    #                        spatial structure from the flux channel
    flux_mode: str = "analytical"

    # Residual-connection ablations (passed verbatim to DoTA3D_v3). Both
    # default to True, reproducing the original architecture.
    #   transformer_residual -- additive residuals inside the transformer layers
    #   conv_residual        -- encoder-decoder skip connections in the decoder
    transformer_residual: bool = True
    conv_residual: bool = True

    # Convolutional regularization (passed verbatim to DoTA3D_v3). Defaults
    # reproduce the original architecture.
    #   weight_standardization -- per-filter weight standardization in convs
    #   norm_layer             -- "batch" | "group" | "none" (group pairs with WS)
    #   weight_init            -- "default" | "kaiming" | "xavier"
    weight_standardization: bool = False
    norm_layer: str = "batch"
    weight_init: str = "default"


# ── Cross-run validation experiment (validation_adota.py) ───────────────────


@dataclass
class RunRef:
    """One trained run to evaluate in the comparison.

    ``run_dir`` is a directory under ``runs_dir`` containing ``hyperparams.json``
    and a ``checkpoints/`` folder. ``checkpoint_fname`` (when set) overrides the
    experiment-level default for this run.
    """

    name: str
    run_dir: str
    checkpoint_fname: Optional[str] = None


@dataclass
class ValidationExperimentConfig:
    """Configuration for the cross-run validation experiment.

    Evaluates several trained ADoTA runs on one shared validation set (defined
    by the data + split fields below) and reports a Run x Metric table. No
    training is performed.
    """

    # ── Shared validation set (defines the identical val set for all runs) ──
    dataset_path: str = ""
    excluded_indexes_file: str = ""
    train_test_split: float = 0.2
    split_random_state: int = 42
    normalize_flux_only: bool = True
    flux_mode: str = "analytical"
    scale: dict = field(default_factory=lambda: DEFAULT_SCALE.copy())

    # ── Runs to compare ──────────────────────────────────────────────────
    runs: List[RunRef] = field(default_factory=list)
    checkpoint_fname: str = "best.pth"  # default checkpoint per run (inference mode)

    # ── Reporting methodology ────────────────────────────────────────────
    # report_mode:
    #   "logs"      -- reproduce the published ablation: read each run's
    #                  metrics.jsonl, select the epoch minimising val MAPE
    #                  (select_by), and report that epoch's LOGGED metrics.
    #                  No model/GPU needed. Range errors are unavailable.
    #   "inference" -- fresh batched inference on each run's checkpoint
    #                  (checkpoint_fname); computes range errors too. Use for
    #                  runs that saved a best_mape.pth at the min-MAPE epoch.
    # select_by: metric to minimise when choosing the reporting epoch
    #            ("val_mape" => mape_pct_mean, "val_loss" => loss_combined_mean).
    report_mode: str = "logs"
    select_by: str = "val_mape"

    # ── Evaluation control ───────────────────────────────────────────────
    device_index: int = 0
    batch_size: int = 32  # batched inference (eval mode => results batch-invariant)
    num_workers: int = 4  # DataLoader workers for parallel H5 reads
    n_samples: Optional[int] = None  # cap the val set (first-N) for quick runs
    mape_mask_frac: float = 0.1  # MAPE mask: y_pred > frac * max(y_pred)
    strict_split_check: bool = True  # error (vs warn) on split/scale mismatch

    # ── Range-metric extraction (range_metrics.compute_range_metrics) ────
    range_dz_mm: float = 2.0
    range_oversample: int = 20
    range_min_peak_dose: float = 0.0

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "/scratch/mstryja/adota_runs/validation_experiments"
    experiment_name: str = "validation_experiment"
