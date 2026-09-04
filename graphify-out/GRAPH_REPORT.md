# Graph Report - /home/mstryja/projects/adota  (2026-07-08)

## Corpus Check
- Large corpus: 299 files · ~1,611,004 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 2492 nodes · 5946 edges · 120 communities (113 shown, 7 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 273 edges (avg confidence: 0.68)
- Token cost: 191,318 input · 0 output

## Community Hubs (Navigation)
- Texture Analysis & Validation
- Interface Severity & Tissue
- Model Inference & Evaluation
- Beam Data Library
- Threshold Sweep & Diagnostics
- Training Config & Factory
- Advanced Metrics Analysis
- Range-Fidelity Analysis
- Beamlet Geometry & Cropping
- GPR Pool & Training
- VLM-Based Quantification
- CT Rotation
- Streaming & Dose Accumulation
- Isocenter & Structures
- Training Run Orchestration
- Beamlet Timing Benchmark
- Dose Deposit & Extraction Tests
- BEV Rotation Timing
- Plan Geometry Check & Gamma
- Rotation Performance Analysis
- Training Losses
- Plan Parsing & Directory
- Checkpoint Management
- Texture-with-Inference Correlations
- Multi-Radius Analysis
- Beamlet Extraction Geometry
- Plan Directory & Contours
- Training-Set Analysis
- Perf Refactor Tests
- Evaluation Sample Sources
- Data Loading & Collate
- Model Layers & Architecture
- Directory Loader & Worker
- Plan Spots & ROI Cropping
- PlanPencil Parser
- Pflugfelder Heterogeneity Index
- Test Model Hyperparams
- Dvh Comparison
- Test Inference
- Bragg Peak Estimation
- Run Plan Opentps
- Rotation
- Range Metrics
- Texture Analysis
- Bragg Peak Estimation
- Test Conv Regularization
- Dvh
- Plan Comparison
- Active Learning Literature Review
- Validation Adota
- Layers
- Test Evaluation Engine
- Test Flux 2Mm
- Beamlet Extraction Integration Plan
- Intensity Heterogeneity
- Validation Adota
- Test Validation Adota
- Gamma Comparison
- Flux
- Readme
- Test Evaluation Outputs
- Edge Detection
- Run
- Rsp
- Ct Texture Analysis
- Layers
-  Goldenlib
- Heterogeneity
- Test Plan Metrics
- Test Bev Beamlet Rotation
- Aggregate Results
- Conftest
- Test Beamlet Input Figure
- Sobel
- Test Validation Adota
- Ct Texture Analysis
- Utils
- Test Flux Gpu
- Scripts Refactor Phase1 Plan
- Plot Beamlets
- Losses
- Test Timing Report
- Config A Analytical Mse Idd
- Compare Plan Dose
- Config Multi Radius Analysis
- Run Model
- Bragg Peak Estimation
- Config Bp Estimation
- Config Range Analysis
- Test Train Pipeline Smoke
- Run Plan Opentps
- Bragg Peak Estimation
- Add Excluded Index
- Remove H5 Record
- Config Analysis Texture With Inference
- Config Beamlet Bev Rotation Timing
- Ideas
- Single Beam
- Config Validation Adota
- Run Ab
- Run Ablation
- Run All Plans
- Run Grid Factor Ab
-   Init  
- Conftest

## God Nodes (most connected - your core abstractions)
1. `DoTA3D_v3` - 77 edges
2. `H5PYGenerator` - 61 edges
3. `BeamDataLibrary` - 48 edges
4. `main()` - 42 edges
5. `load_model()` - 41 edges
6. `PlanDirectory` - 41 edges
7. `ExtractionConfig` - 40 edges
8. `inverse_minmax()` - 39 edges
9. `Plan` - 38 edges
10. `run_streaming_pipeline()` - 37 edges

## Surprising Connections (you probably didn't know these)
- `Release 1.0.0 — DoTA3D_v3 model refactor` --references--> `DoTA3D_v3 model`  [INFERRED]
  CHANGELOG.md → README.md
- `ADoTA base model (Stryja, Lathouwers & Perko) [10]` --references--> `DoTA3D_v3 model`  [INFERRED]
  research/active_learning_literature_review.md → README.md
- `evaluate_single_sample()` --calls--> `model()`  [INFERRED]
  scripts/analysis_texture_with_inference.py → tests/test_training_losses.py
- `generate_publication_figures()` --calls--> `model()`  [INFERRED]
  scripts/analysis_texture_with_inference.py → tests/test_training_losses.py
- `_infer_bev_dose()` --calls--> `model()`  [INFERRED]
  scripts/beamlet_bev_rotation_timing.py → tests/test_training_losses.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Beamlet geometric error suspects (S1-S7)** — docs_beamlet_extraction_integration_plan_s1_wrong_bdl, docs_beamlet_extraction_integration_plan_s2_pivot_frame, docs_beamlet_extraction_integration_plan_s3_x_flip, docs_beamlet_extraction_integration_plan_s4_half_voxel_shift, docs_beamlet_extraction_integration_plan_s5_sign_chain, docs_beamlet_extraction_integration_plan_s6_entrance_frame, docs_beamlet_extraction_integration_plan_s7_dose_normalization [EXTRACTED 1.00]
- **Physics-informed active learning framework** — research_active_learning_literature_review_novelty_wedge, research_publication_plan_core_narrative, research_sampling_architecture_acquisition_function, research_publication_plan_difficulty_predictor [INFERRED 0.85]
- **Plan-level dose pipeline (extract-infer-accumulate-gamma)** — readme_plan_pipeline, readme_beamlets_package, scripts_docs_run_plan_opentps_stream_mode, scripts_docs_run_plan_opentps_gamma_stage [EXTRACTED 1.00]
- **2x2 Flux-mode x Loss-mode Ablation Study** — scripts_ablation_config_a_analytical_mse_idd_config, scripts_ablation_config_b_angle_broadcast_mse_idd_config, scripts_ablation_config_c_analytical_mse_only_config, scripts_ablation_config_d_angle_broadcast_mse_only_config [EXTRACTED 1.00]
- **Residual-connection Ablation Set (vs reference A)** — scripts_config_ablation_no_conv_residual_config, scripts_config_ablation_no_transformer_residual_config, scripts_ablation_config_a_analytical_mse_idd_config [INFERRED 0.85]
- **train_adota.py Config Family** — scripts_config_train_adota_config, scripts_ablation_config_a_analytical_mse_idd_config, scripts_ablation_config_b_angle_broadcast_mse_idd_config, scripts_ablation_config_c_analytical_mse_only_config, scripts_ablation_config_d_angle_broadcast_mse_only_config, scripts_config_ablation_no_conv_residual_config, scripts_config_ablation_no_transformer_residual_config [EXTRACTED 1.00]

## Communities (120 total, 7 thin omitted)

### Community 0 - "Texture Analysis & Validation"
Cohesion: 0.06
Nodes (78): compute_heterogeneity_metrics(), evaluate_single_sample(), generate_publication_figures(), SampleResult, CT Texture & Model Inference Analysis  Combines DoTA model evaluation with CT he, Compute beam-aligned heterogeneity metrics on the original CT grid.      Paramet, Evaluate a single sample: run inference, compute model metrics and     the enabl, Generate publication figures for best, worst, and mean cases. (+70 more)

### Community 1 - "Interface Severity & Tissue"
Cohesion: 0.05
Nodes (51): plot_hu_material_density(), help, ndarray, Option, Path, HU → Material Composition & Stopping Power Conversion  Converts a CT volume (in, Generate a 3×2 figure: HU, material assignment, and density map.      Row 1: CT, Generate a water-box phantom (HU=0) and compute decomposition. (+43 more)

### Community 2 - "Model Inference & Evaluation"
Cohesion: 0.04
Nodes (70): advanced_metrics_and_figures(), density_variability_vs_gpr(), discover_sample_ids(), evaluate_samples(), generate_gpr_plot(), evaluate_samples(), generate_correlation_figures(), main() (+62 more)

### Community 3 - "Beam Data Library"
Cohesion: 0.05
Nodes (56): angles_to_spot_position(), BeamDataLibrary, _find_index(), ndarray, Path, Beam data library (BDL) parsing and spot/angle conversions.  The BDL holds the g, Protons per monitor unit, per nominal energy., Protons per MU at ``energy`` (OpenTPS ``computeMU2Protons``).          Linear in (+48 more)

### Community 4 - "Threshold Sweep & Diagnostics"
Cohesion: 0.06
Nodes (53): _build_config(), main(), DataFrame, help, Option, Path, Regenerate the established CT+Sobel and 3D-dose figures for the worst range-erro, Rebuild SampleRecord objects from a results.csv slice (typed by field). (+45 more)

### Community 5 - "Training Config & Factory"
Cohesion: 0.07
Nodes (53): Configuration for an ADoTA training run.      Used by ``scripts/train_adota.py``, TrainingConfig, build_adota_model(), build_config_from_yaml(), build_optimizer_scheduler(), maybe_compile_model(), Any, device (+45 more)

### Community 6 - "Advanced Metrics Analysis"
Cohesion: 0.07
Nodes (58): _add_derived_columns(), analyse_density_regions(), compute_advanced_metrics(), _correlate_target(), denorm_ctx(), extract_all_samples(), generate_angle_performance_analysis(), generate_correlation_analysis() (+50 more)

### Community 7 - "Range-Fidelity Analysis"
Cohesion: 0.06
Nodes (49): discover_sample_ids(), main(), _make_per_sample_fn(), normalize_test_data_config(), plot_delta_histogram(), plot_energy_stratified(), plot_r80_scatter(), plot_worst_idd_overlays() (+41 more)

### Community 8 - "Beamlet Geometry & Cropping"
Cohesion: 0.06
Nodes (51): crop_around_spatial_point(), extract_beamlet_roi(), Image, ndarray, Extract a spot's BEV CT crop and its beamlet entrance point.      Port of ``get_, Crop an ``(H, W, D)`` ROI around a physical point, air-padding OOB regions., beamlet_ray(), check_if_point_in_cube() (+43 more)

### Community 9 - "GPR Pool & Training"
Cohesion: 0.07
Nodes (49): RandomState, main(), help, Option, Path, ADoTA training entry point.  Usage:     uv run python scripts/train_adota.py --c, build_dataloaders(), Build train / val dataloaders from the resolved training config. (+41 more)

### Community 10 - "VLM-Based Quantification"
Cohesion: 0.08
Nodes (47): aggregate_votes(), analyse_density_regions(), estimate_bp_range(), extract_sample(), generate_correlation_analysis(), _load_vlm(), main(), _parse_vlm_response() (+39 more)

### Community 11 - "CT Rotation"
Cohesion: 0.09
Nodes (45): expanded_reference_grid(), Image, Per-field CT rotation around the isocenter (SimpleITK).  The gantry rotation is, Rotate a CT image in the axial plane around the physical isocenter.      Args:, Build an empty grid that fully contains ``image`` rotated about the isocenter., rotate_ct_around_isocenter(), _argmax_index_xyz(), _corner_phantom() (+37 more)

### Community 12 - "Streaming & Dose Accumulation"
Cohesion: 0.10
Nodes (42): main(), Verify the streaming pipeline produces the same dose as the staged pipeline.  Fo, accumulate_dose(), AccumulationConfig, _load_beamlet_array(), _load_records_by_field(), Image, ndarray (+34 more)

### Community 13 - "Isocenter & Structures"
Cohesion: 0.10
Nodes (40): Flip, isocenter_index_zyx(), isocenter_physical(), plan_isocenter_index_ct(), Image, Isocenter coordinate handling: plan frame -> CT frame (x-flip convention).  The, Convert a plan isocenter ``(x, y, z)`` index to the CT frame (x flipped).      A, Physical point of the plan isocenter in the CT frame (x flipped).      Args: (+32 more)

### Community 14 - "Training Run Orchestration"
Cohesion: 0.08
Nodes (35): FrameType, _config_to_dict(), dump_nan_context(), _file_fingerprint(), _git_info(), _gpu_info(), GracefulShutdown, MetricsLog (+27 more)

### Community 15 - "Beamlet Timing Benchmark"
Cohesion: 0.09
Nodes (41): discover_sample_ids(), _float_or_nan(), has_required_files(), load_raw_volume_dhw(), main(), parse_test_dataset(), device, ndarray (+33 more)

### Community 16 - "Dose Deposit & Extraction Tests"
Cohesion: 0.10
Nodes (40): deposit_crop(), Add ``weight * crop`` into ``grid`` at the crop's original window.      The wind, ExtractionConfig, Extract per-spot ADoTA inputs for a whole plan (sequential reference).      This, Configuration for :func:`run_extraction`.      Attributes:         roi_size: ``(, run_extraction(), Field, Represents a treatment field. (+32 more)

### Community 17 - "BEV Rotation Timing"
Cohesion: 0.08
Nodes (40): _build_timing_report(), _coerce_beams(), estimate_end_to_end(), _flux_overlay(), _format_end_to_end_table(), _format_timing_table(), generate_validation_figures(), _infer_bev_dose() (+32 more)

### Community 18 - "Plan Geometry Check & Gamma"
Cohesion: 0.08
Nodes (35): check_plan_geometry(), _compare(), GridGeometry, _overlap_fraction(), PlanGeometryError, Image, ndarray, Path (+27 more)

### Community 19 - "Rotation Performance Analysis"
Cohesion: 0.12
Nodes (35): build_pivot_rotation_matrix_3d(), build_results_table(), correctness_vs_scipy(), _cupy_rotate_one(), _draw_subplot(), FrameworkResult, load_plan_data(), main() (+27 more)

### Community 20 - "Training Losses"
Cohesion: 0.07
Nodes (32): CaptureFixture, LMSE, LossLPD, LPS, Loss functions and adaptive weight balancers for ADoTA training.  This module is, Lateral Profile Difference loss.      For each depth slice ``z`` and each sample, Initialize the loss.          Args:             epsilon: Denominator stabilizer, Initialize the balancer.          Args:             smoothing: Exponential smoot (+24 more)

### Community 21 - "Plan Parsing & Directory"
Cohesion: 0.08
Nodes (34): _bdl_preview(), _indent(), _plan_preview(), Loader for an OpenTPS plan directory.  An OpenTPS plan directory bundles every i, Return a human-readable preview of the loaded plan directory.          Args:, Return the first *n_lines* non-empty lines of the BDL for a preview., Build a truncated, readable tree of the parsed plan., _collect_field_ids() (+26 more)

### Community 22 - "Checkpoint Management"
Cohesion: 0.12
Nodes (30): CheckpointManager, device, Module, Optimizer, Return the underlying module behind a ``torch.compile`` wrapper.      ``torch.co, Save and load full training snapshots.      Each snapshot contains everything ne, Persist a training snapshot. Returns the ``last.pth`` path., Restore a snapshot in-place; returns the bookkeeping fields. (+22 more)

### Community 23 - "Texture-with-Inference Correlations"
Cohesion: 0.09
Nodes (34): compute_correlations(), evaluate_all_samples(), generate_correlation_plots(), generate_gpr_plot(), generate_metrics_description(), load_model(), main(), print_combined_results_table() (+26 more)

### Community 24 - "Multi-Radius Analysis"
Cohesion: 0.10
Nodes (31): compute_multi_radius_metrics(), load_input_csv(), main(), process_beamlets(), Argument, help, ndarray, Option (+23 more)

### Community 25 - "Beamlet Extraction Geometry"
Cohesion: 0.11
Nodes (31): Convert a bixelgrid shift to beamlet angles (degrees).      Args:         y_spot, spot_position_to_angles(), _build_manifest(), _build_sim_res(), _extract_impl(), _FieldTiming, _prepare_output_dir(), _process_spot() (+23 more)

### Community 26 - "Plan Directory & Contours"
Cohesion: 0.10
Nodes (30): _image_summary(), _load_contours(), load_plan_directory(), parse_opentps_config(), Image, Path, Load every ADoTA input from an OpenTPS plan directory.      Args:         plan_d, Load every structure-mask ``.mhd`` (all but CT and the reference dose). (+22 more)

### Community 27 - "Training-Set Analysis"
Cohesion: 0.09
Nodes (29): evaluate_samples(), generate_scatter_plot(), generate_sigma_hu_histogram(), generate_tv_vs_gpr_scatter(), generate_tv_vs_rde_scatter(), generate_violin_plots(), main(), print_summary() (+21 more)

### Community 28 - "Perf Refactor Tests"
Cohesion: 0.14
Nodes (28): Restore original handlers (call at end of training)., bench(), build_layer(), build_model(), install_original_mask_path(), _original_causal_mask(), pick_device(), device (+20 more)

### Community 29 - "Evaluation Sample Sources"
Cohesion: 0.11
Nodes (19): Dataset, hdf5_samples(), Argument, Load random samples from training HDF5 and compute decomposition., DirSource, H5Source, Path, HDF5-backed source wrapping an existing :class:`H5PYGenerator`.      Iterates th (+11 more)

### Community 30 - "Data Loading & Collate"
Cohesion: 0.11
Nodes (24): collate_h5(), limited_loader(), LimitedLoader, load_record_ids(), DataLoader, Path, Data-pipeline helpers for ADoTA training.  Everything needed to go from an on-di, Wrap a DataLoader and stop after ``max_batches`` iterations. (+16 more)

### Community 31 - "Model Layers & Architecture"
Cohesion: 0.10
Nodes (13): ConvEncoder3D, CroppingLayer, LinearProj, PositionalEmbedding, Calculates the token size at the given depth., _summary_      Args:         nn (_type_): _description_      Raises:         Val, Project scalars to token vectors., Class ConvEncoder3D.     General-purpose DoTA Encoder, responsible for convert t (+5 more)

### Community 32 - "Directory Loader & Worker"
Cohesion: 0.13
Nodes (24): main(), ADoTA batch inference worker.  Runs in the ADoTA venv (Python 3.9 + PyTorch). Ca, get_single_record_no_gt(), postprocess_prediction(), prepare_input_from_arrays(), ndarray, Tensor, Build the 2-channel ADoTA model input from in-memory CT/flux crops.      This is (+16 more)

### Community 33 - "Plan Spots & ROI Cropping"
Cohesion: 0.14
Nodes (23): clip_axis_window(), Sub-volume (ROI) extraction for beamlet inputs.  Port of datagenerator's ``cropp, Clip a ``[start, start+length)`` window to ``[0, size)``.      Shared by croppin, adjusted_gantry_angle(), expand_plan_to_spots(), group_by_field(), Expand a parsed plan into per-spot extraction records.  This is the single sourc, Group spot records by their ``beam`` (field) index, preserving order.      Rotat (+15 more)

### Community 34 - "PlanPencil Parser"
Cohesion: 0.16
Nodes (26): parse_plan(), Parse a PlanPencil text dump into a Plan object.      Args:         path: Path t, _minimal_plan(), Path, Unit tests for :mod:`src.loaders.plan_parser`.  Covers the happy path, the optio, ``#NumberOfFractions`` is metadata; the loop follows the actual blocks.      Thi, Optional field-level range-shifter block is captured., Optional control-point-level range-shifter block is captured. (+18 more)

### Community 35 - "Pflugfelder Heterogeneity Index"
Cohesion: 0.12
Nodes (19): compute_pflugfelder_hi(), compute_wepl_map(), pflugfelder_hi(), ndarray, Pflugfelder (2007) lateral tissue heterogeneity index.  Implements the Water-Equ, Compute the per-ray WEPL map up to the Bragg-peak depth.      Parameters     ---, Pflugfelder heterogeneity index from a WEPL map.      Parameters     ----------, Convenience wrapper: CT + dose → Pflugfelder HI.      Uses the GT IDD argmax as (+11 more)

### Community 36 - "Test Model Hyperparams"
Cohesion: 0.12
Nodes (26): _ff_linears(), Tests for model-config fixes (backlog items #1, #2, #3).  * #1 -- ``zero_padding, Legacy hyperparams (no dim_feedforward) build identical FF shapes., With num_transformers=0, the eval attn placeholder is (1, D+1, D+1)., self.mask / generate_subsequent_mask are gone, causal forward still works., Trained depth 160 reproduces the previous hardcoded 161., forward always returns (dose, attention); attention is None in training., With zero_padding=False the forward pass must run (previously crashed). (+18 more)

### Community 37 - "Dvh Comparison"
Cohesion: 0.15
Nodes (24): _generate_comparison_figures(), Generate the ADoTA vs MCsquare dose-comparison + DVH figures/metrics.      Reads, compute_structure_dvhs(), dvh_comparison_figure(), dvh_metrics(), _ordered_names(), ndarray, Path (+16 more)

### Community 38 - "Test Inference"
Cohesion: 0.15
Nodes (23): discover_spot_ids(), InferenceConfig, device, Module, Path, Stage 2: in-process batched ADoTA inference over extracted beamlets.  Reuses the, Run batched ADoTA inference over all complete beamlets.      Writes ``{id}_ds_pr, Configuration for :func:`run_inference`.      Attributes:         batch_size: Nu (+15 more)

### Community 39 - "Bragg Peak Estimation"
Cohesion: 0.14
Nodes (24): build_estimator(), denormalize_energy(), load_schneider_calibration(), load_yaml_config(), main(), plot_energy_stratified(), plot_error_histogram(), plot_scatter() (+16 more)

### Community 40 - "Run Plan Opentps"
Cohesion: 0.11
Nodes (23): _build_timing_report(), _format_timing_report(), load_ct_from_dicom_dir(), main(), _merge_timing_report(), help, Logger, Option (+15 more)

### Community 41 - "Rotation"
Cohesion: 0.15
Nodes (22): Size, build_lateral_axis_rotation_matrix_3d(), center_pivot_dhw(), compose_affine_matrices(), _make_torch_grid(), device, ndarray, Tensor (+14 more)

### Community 42 - "Range Metrics"
Cohesion: 0.14
Nodes (22): interp1d, compute_range_metrics(), _distal_crossing(), _fine_grid(), integrated_depth_dose(), _nan_metrics(), ndarray, Range-fidelity metrics for proton beamlet depth-dose (IDD) curves.  These functi (+14 more)

### Community 43 - "Texture Analysis"
Cohesion: 0.13
Nodes (22): Enum, Argument, help, Option, Load CT images and run the texture analysis pipeline., run(), discover_images(), ImageFormat (+14 more)

### Community 44 - "Bragg Peak Estimation"
Cohesion: 0.11
Nodes (15): CSDACorrectedEstimator, CSDAWaterEstimator, CTDensityGradientEstimator, GTIDDEstimator, hu_to_rsp(), load_pstar_table(), ndarray, Load PSTAR CSV → interpolator  energy_MeV → CSDA_range_cm. (+7 more)

### Community 45 - "Test Conv Regularization"
Cohesion: 0.17
Nodes (17): Conv3D, DoTA3D_v3, Tensor, Dose Transformer (DoTA) 3D model, version 3.      Encoder-decoder architecture f, _conv_modules(), Tests for the convolutional regularization options (backlog item #5 + init).  Co, _record(), test_defaults_backward_compatible_state_dict() (+9 more)

### Community 46 - "Dvh"
Cohesion: 0.13
Nodes (15): DVH, ndarray, Dose-volume histogram (DVH) computation, adapted from OpenTPS.  Faithful port of, Volume receiving at least ``dose_gy`` Gy (in % or cm^3)., Return the common DVH metrics as a dict (Gy)., Cumulative dose-volume histogram of a ROI for one dose grid.      Attributes:, Compute the DVH.          Args:             mask: Boolean ROI mask ``(z, y, x)``, The ``(dose_bins_gy, cumulative_volume_pct)`` arrays for plotting. (+7 more)

### Community 47 - "Plan Comparison"
Cohesion: 0.16
Nodes (20): dose_comparison_metrics(), _overlay(), _overlay_contour(), plan_dose_comparison(), ndarray, Path, Plan-level dose comparison figures (ADoTA vs MCsquare).  Renders two full-grid d, Compare two plan dose maps on the CT (axial/coronal/sagittal + profile).      Ar (+12 more)

### Community 48 - "Active Learning Literature Review"
Cohesion: 0.13
Nodes (20): ADoTA base model (Stryja, Lathouwers & Perko) [10], AL does not automatically beat random sampling, Bueno et al. MC-need algorithm [12], Physics-informed AL novelty wedge, Active Learning for Surrogate Dose Models: Literature Review, Pflugfelder heterogeneity number [11], Query by Committee (Seung et al. 1992) [1], Stochastic Batch Acquisition (SBAL, Kirsch 2023) [7] (+12 more)

### Community 49 - "Validation Adota"
Cohesion: 0.17
Nodes (17): aggregate_run(), _log_cell(), Any, Cross-run validation experiment for ADoTA (inference only).  Evaluates several t, Nan-aware mean/std per metric over a run's per-sample results., Format ``mean +/- std`` (or ``mean``) for a logged metric, or ``n/a``., Render the table to the log and save comparison.md + comparison.csv., Per-sample metrics for one validation record under one run's model. (+9 more)

### Community 50 - "Layers"
Cohesion: 0.15
Nodes (10): ConvBlock3D_v2, ConvDecoder3D, Calculates the token size at the given depth., Class repreenting a ConvBlock3D layer. ConvBlock is responsible for processing a, Constructs the convblock as described in the https://arxiv.org/abs/1505.04597 pa, Builds the per-conv normalization layer per ``norm_layer``., _layernorm(), Unit tests for ConvBlock3D_v2 token_size handling (backlog items #9, #10).  #9 - (+2 more)

### Community 51 - "Test Evaluation Engine"
Cohesion: 0.16
Nodes (14): denorm_pair(), Tensor, De-normalize ground truth and prediction to physical units (NumPy).      Mirrors, Convenience wrapper for :func:`denorm_pair` on this context., _FakeModel, _FakeSource, _make_sample(), Unit tests for src/evaluation/engine.py.  Uses a fake model and a fake source on (+6 more)

### Community 52 - "Test Flux 2Mm"
Cohesion: 0.16
Nodes (19): _flux_pair(), _lateral_centroid(), _norm(), ndarray, Flux equivalence on the 2x2x2 grid (field-resampling P2).  The flux at 2mm must, The cell-center 2mm flux is centroid-aligned with downsample(1mm); grid-point is, Flux for the same physical beamlet at 1mm and 2mm (entrance scaled by spacing)., Lateral 1-sigma in mm from the 2nd moment of a fixed-depth slice along ``axis``. (+11 more)

### Community 53 - "Beamlet Extraction Integration Plan"
Cohesion: 0.14
Nodes (18): Dose calibration factor, ADoTA Changelog, Parallel extraction (run_extraction_pooled), Release 1.0.0 — DoTA3D_v3 model refactor, Release 1.2.0 — Plan-level dose pipeline, Datagenerator source-to-target port map, Geometric ground-truth test policy, Plan-level ADoTA pipeline integration plan (+10 more)

### Community 54 - "Intensity Heterogeneity"
Cohesion: 0.16
Nodes (17): compute_glcm_metrics(), compute_intensity_metrics(), ndarray, Compute GLCM homogeneity on sampled axial slices.      Returns     -------     d, Compute global intensity heterogeneity on the original CT grid.      Parameters, _central_moments(), global_intensity_heterogeneity(), GlobalIntensityHeterogeneity (+9 more)

### Community 55 - "Validation Adota"
Cohesion: 0.15
Nodes (18): _build_config(), _build_val_loader(), _cell(), evaluate_run(), load_run_model(), main(), DataLoader, device (+10 more)

### Community 56 - "Test Validation Adota"
Cohesion: 0.22
Nodes (18): _check_split_consistency(), Verify a run's saved config shares the experiment's split + scale.      Guards a, Pick the epoch minimising ``select_by`` from a run's metrics.jsonl.      Returns, Paper-faithful reporting: logged metrics at the min-val-MAPE epoch., _run_logs_mode(), select_epoch_from_logs(), One trained run to evaluate in the comparison.      ``run_dir`` is a directory u, Configuration for the cross-run validation experiment.      Evaluates several tr (+10 more)

### Community 57 - "Gamma Comparison"
Cohesion: 0.20
Nodes (16): _draw_gamma(), plan_gamma_figure(), ndarray, Path, Plan-level gamma-map figure (one column per gamma criterion).  Renders the gamma, Hide ticks on an image panel (gamma maps carry no spatial ticks)., CT grayscale + alpha-masked gamma overlay; return the overlay image., Render the gamma maps (3 views x N criteria) around the isocenter.      Args: (+8 more)

### Community 58 - "Flux"
Cohesion: 0.17
Nodes (15): flux_projection(), flux_projection_gpu(), ndarray, Proton-flux projection for the ADoTA input channel.  Faithful port of datagenera, GPU/torch twin of :func:`flux_projection` (identical math, on ``device``)., Generate a proton-flux projection along an angled beamlet.      Faithful port; t, The GPU flux at 2mm is float32-identical to the NumPy flux at 2mm., test_gpu_2mm_matches_numpy_2mm() (+7 more)

### Community 59 - "Readme"
Cohesion: 0.19
Nodes (16): Data directory (example inputs), Models directory (checkpoints + hyperparams), 2x2 factorial ablation study, angle_broadcast flux mode, DoTA3D_v3 model, HDF5 beamlet dataset (160x30x30 grid), Training losses (LMSE + LPS, adaptive balance), Beam-aligned heterogeneity metrics (G_phi, R, H_phi) (+8 more)

### Community 60 - "Test Evaluation Outputs"
Cohesion: 0.25
Nodes (14): _save_per_sample_csv(), CsvColumn, Any, Path, Shared output writers for the evaluation scripts.  The Tier-1 scripts each emitt, One CSV column.      Attributes:         name: Header / field name.         row:, Write per-sample results to CSV, with an optional summary block.      The summar, save_results_csv() (+6 more)

### Community 61 - "Edge Detection"
Cohesion: 0.28
Nodes (14): dog_response(), _kill_roll_borders(), log_edges(), log_edges_significant(), log_response(), _neighbor_offsets(), ndarray, Integrated, parameterized LoG edge detector that returns only "important" edges. (+6 more)

### Community 62 - "Run"
Cohesion: 0.12
Nodes (16): Any, DataLoader, device, Module, Optimizer, Path, Tensor, Run one training epoch; returns aggregate stats. (+8 more)

### Community 63 - "Rsp"
Cohesion: 0.18
Nodes (12): hu_to_density(), hu_to_rsp(), hu_to_rsp_density(), ndarray, Path, Shared HU → density / RSP conversion utilities.  References ---------- - Schneid, Convert HU → mass density [g/cm³] via piecewise-linear table., Approximate RSP ≈ ρ(HU) / ρ_water  (energy-independent). (+4 more)

### Community 64 - "Ct Texture Analysis"
Cohesion: 0.19
Nodes (11): compute_texture_metrics(), ndarray, CT Texture Analysis Pipeline  A command-line tool for loading CT images and comp, Compute texture analysis metrics for a CT image array.      Computes GLCM homoge, Create a 3×2 figure (3 axial slices × [CT, edges]) and save as PNG.      Slice i, save_edge_detection_figure(), LogEdgeParams, Parameters controlling how "important" edges are selected.      - scale_normaliz (+3 more)

### Community 65 - "Layers"
Cohesion: 0.19
Nodes (6): Permute, device, Return the causal attention mask, building and caching it lazily.          The m, Construct the causal mask (0 where attention is allowed, -inf else).          Bu, Constructor of ConvBlock3D class.          Args:             in_channels (int):, TransformerEncoderLayerDoTA

### Community 66 - " Goldenlib"
Cohesion: 0.19
Nodes (13): check_against_golden(), compare_csv(), golden_dir(), _is_timing(), Path, Shared helpers for the characterization (golden) tests.  These tests pin the cur, Capture a baseline or compare against it.      Returns ``(status, messages)`` wh, Return (creating if needed) the directory holding reference CSVs. (+5 more)

### Community 67 - "Heterogeneity"
Cohesion: 0.26
Nodes (12): beam_aligned_global_heterogeneity(), beam_axis_roughness(), beam_weighted_gradient(), gradient_magnitude_3d(), _normalize_weights(), ndarray, Beam-aligned CT heterogeneity metrics.  Provides three complementary scores that, Beam-axis roughness **R**.      .. math::          R = \\frac{1}{K-1} \\sum_{k} (+4 more)

### Community 68 - "Test Plan Metrics"
Cohesion: 0.29
Nodes (12): high_dose_mask(), plan_dose_metrics(), ndarray, Boolean mask of voxels above ``frac`` of the reference's percentile dose.      T, Plan-level error metrics comparing ``dose_eval`` to reference ``dose_ref``., Tests for :mod:`src.metrics.plan_metrics`., _ref(), test_high_dose_mask_threshold_is_frac_of_percentile() (+4 more)

### Community 69 - "Test Bev Beamlet Rotation"
Cohesion: 0.21
Nodes (12): _centered_angled_flux(), _depth_centroid_spread(), ndarray, Correctness tests for the per-beamlet BEV reinterpolation rotation.  These pin t, Std over depth of the per-slice lateral (z, y) centroid, in voxels.      A beam, Angled flux entering at the lateral centre of the x=0 face., Rotating by (0, 0) returns the crop unchanged., The angled flux ridge becomes axis-aligned after the forward rotation.      This (+4 more)

### Community 70 - "Aggregate Results"
Cohesion: 0.21
Nodes (9): LogRecord, _fmt(), _load_run(), main(), _print_table(), Path, Aggregate ablation study results from multiple training runs.  Reads ``manifest., Aggregate ablation study results and print a comparison table. (+1 more)

### Community 71 - "Conftest"
Cohesion: 0.25
Nodes (10): ModuleType, datagenerator_geometry(), datagenerator_utils(), _import_datagenerator_module(), make_phantom(), Shared fixtures for the beamlet tests.  Provides access to the external ``datage, Import a datagenerator submodule, or return None if unavailable., The datagenerator geometry module, or skip if it cannot be imported. (+2 more)

### Community 72 - "Test Beamlet Input Figure"
Cohesion: 0.33
Nodes (10): beamlet_input_figure(), Path, Plot a constructed beamlet input (CT crop + flux) for correctness checks.      R, save_figure_as_publication_formats(), _ct_and_flux(), Path, Smoke tests for :func:`src.figures.single_beam.beamlet_input_figure`., test_non_3d_raises() (+2 more)

### Community 73 - "Sobel"
Cohesion: 0.25
Nodes (10): compute_sobel_metrics(), compute_sobel_metrics_sphere(), compute_structure_tensor_metrics_sphere(), ndarray, Sobel-based edge metrics on CT volumes around the Bragg peak., Compute 3-D Sobel metrics inside a sphere around the Bragg peak.      The Bragg, Compute dose-weighted (DW) and threshold-masked (TH) structure-tensor     Sobel, Compute 3-D Sobel-based edge metrics in the Bragg-peak zone.      The CT volume (+2 more)

### Community 74 - "Test Validation Adota"
Cohesion: 0.27
Nodes (11): _ctx(), _gaussian_profile(), _metrics(), ndarray, SampleResult, Tensor, Make a (1, D, H, W) dose whose lateral sum equals ``profile`` (D,)., Build a minimal InferenceContext-like object for the metric fn. (+3 more)

### Community 75 - "Ct Texture Analysis"
Cohesion: 0.27
Nodes (10): load_dcm(), load_image(), load_mhd(), load_npy(), Image, Path, Dispatch to the appropriate loader based on *fmt*.      Args:         path: Path, Load an .mhd / .raw image pair using SimpleITK.      Args:         path: Path to (+2 more)

### Community 76 - "Utils"
Cohesion: 0.22
Nodes (9): get_all_lrs(), get_lr(), Optimizer, Tensor, Training-loop utilities for ADoTA.  Helpers in this module operate on objects th, Return ``True`` if every tensor lies within ``[low, high]``.      Used as a guar, Return the learning rate of an optimizer's first parameter group.      Args:, Return the learning rate of every parameter group.      Useful when the optimize (+1 more)

### Community 77 - "Test Flux Gpu"
Cohesion: 0.33
Nodes (9): _cpu(), _gpu(), Equivalence tests: ``flux_projection`` (NumPy) vs ``flux_projection_gpu`` (Torch, Math identity: Torch (CPU, float64) reproduces NumPy to round-off., The stored/consumed artifact (float32) is bit-identical to the NumPy path., CUDA path matches NumPy to float64 round-off and is float32-bit-identical., test_cuda_matches_numpy(), test_torch_cpu_bit_identical_after_float32_cast() (+1 more)

### Community 78 - "Scripts Refactor Phase1 Plan"
Cohesion: 0.31
Nodes (9): Release 1.1.0 — Scripts refactor part 1, Canonical thresholded MAPE fix, src/evaluation package (cli/sources/engine/outputs), Characterization (golden) tests, Scripts refactor Part 1 action plan, resolve_device unified device handling, Duplicated inference-evaluation pipeline, Scripts refactor plan (draft) (+1 more)

### Community 79 - "Plot Beamlets"
Cohesion: 0.28
Nodes (8): _discover_spot_ids(), main(), help, Option, Path, Plot constructed beamlet inputs (CT crop + flux) for correctness checks.  Reads, Return the spot ids present in a beamlets directory, sorted., Render a CT/flux mosaic per spot.

### Community 80 - "Losses"
Cohesion: 0.22
Nodes (5): Tensor, Compute the loss.          Args:             y_pred: Predicted dose, shape ``(B,, Compute the loss.          Args:             y_pred: Predicted dose, shape ``(B,, Compute the next pair of objective weights.          Args:             loss1: Fi, Compute the loss.          Args:             y_pred: Predicted dose, shape ``(B,

### Community 81 - "Test Timing Report"
Cohesion: 0.50
Nodes (8): _accumulation(), _extraction(), _inference(), Tests for the pipeline timing report (build / format / JSON)., test_build_report_aggregates_and_is_json_serializable(), test_format_report_table(), test_merge_preserves_prior_stages(), test_partial_stages_omitted()

### Community 82 - "Config A Analytical Mse Idd"
Cohesion: 0.50
Nodes (8): Ablation A Config (analytical flux + MSE+IDD), Ablation B Config (angle-broadcast flux + MSE+IDD), Ablation C Config (analytical flux + MSE-only), Ablation D Config (angle-broadcast flux + MSE-only), Ablation Config: conv_residual OFF, Ablation Config: transformer_residual OFF, ADoTA Training Config (baseline anneal), train_adota.py (ADoTA training)

### Community 83 - "Compare Plan Dose"
Cohesion: 0.29
Nodes (6): main(), help, Option, Path, Visual comparison of two completed plan dose maps: ADoTA vs MCsquare.  Loads the, Render the ADoTA vs MCsquare dose comparison figure (doses in Gy).

### Community 84 - "Config Multi Radius Analysis"
Cohesion: 0.33
Nodes (7): Multi-Radius Heterogeneity Analysis Config, Threshold Sensitivity Sweep Config, Training Set Analysis Config (interface prevalence), multi_radius_analysis.py, threshold_sweep.py, training_set_analysis.py, Slides: CT Density Variability & Model Performance

### Community 85 - "Run Model"
Cohesion: 0.29
Nodes (7): anatomical_site_summary_rows(), format_mean_std(), print_anatomical_site_summary(), Logger, Format mean ± std with the current population-std convention., Build rows for the anatomical-site publication summary., Print the requested publication table by anatomical site.

### Community 86 - "Bragg Peak Estimation"
Cohesion: 0.33
Nodes (4): energy_to_r80_mm(), R80DensityCorrectedEstimator, Grevillot et al. (2011) analytical fit: energy [MeV] → r80 [mm].      Returns th, Grevillot r80 in water, walked through density-based RSP field.      Approach (m

### Community 87 - "Config Bp Estimation"
Cohesion: 0.33
Nodes (6): bragg_peak_estimation.py, Training Set Advanced Metrics Config, Bragg Peak Estimation Config, VLM-Based Difficulty Quantification Config, training_set_analysis_advanced_metrics.py, training_set_vlm_based_quantification.py

### Community 88 - "Config Range Analysis"
Cohesion: 0.33
Nodes (6): Beamlet Range-Fidelity Analysis Config, ADoTA Model Evaluation Config (directory-based), ADoTA Model Evaluation Config (HDF5-based), range_analysis.py, run_model_h5py.py, run_model.py

### Community 89 - "Test Train Pipeline Smoke"
Cohesion: 0.40
Nodes (5): _pick_device(), device, End-to-end training smoke test on a tiny slice of the real dataset.  Trains DoTA, Freest visible CUDA device (avoids GPUs busy with training), else CPU., test_training_smoke_on_real_data_slice()

### Community 90 - "Run Plan Opentps"
Cohesion: 0.40
Nodes (5): GPU flux projection (flux_projection_gpu), Plan gamma stage (GPR per criterion vs MC), Field-level 2mm resampling (grid_factor), run_plan_opentps.py end-to-end plan pipeline guide, Stream (fused, disk-free) execution mode

### Community 91 - "Bragg Peak Estimation"
Cohesion: 0.40
Nodes (4): Protocol, BPEstimator, Protocol every BP estimation method must satisfy., Return estimated BP depth in mm (along axis 0).

### Community 92 - "Add Excluded Index"
Cohesion: 0.40
Nodes (4): main(), Path, Add one or more record IDs to an exclusion index file., Append the given ID(s) to the exclusion file if not already present.      Specif

### Community 93 - "Remove H5 Record"
Cohesion: 0.40
Nodes (4): main(), Path, Remove a single record from an HDF5 file by its ID., Delete the record with the given ID from the HDF5 file.

### Community 94 - "Config Analysis Texture With Inference"
Cohesion: 0.50
Nodes (4): analysis_texture_with_inference.py, CT Texture + Inference Analysis Config, CT Texture Analysis Config, ct_texture_analysis.py

### Community 95 - "Config Beamlet Bev Rotation Timing"
Cohesion: 0.50
Nodes (4): beamlet_bev_rotation_timing.py, Beamlet BEV Rotation Timing Config, ADoTA End-to-End Plan Pipeline Config, run_plan_opentps.py

### Community 96 - "Ideas"
Cohesion: 0.67
Nodes (3): Material-interface numerical metric, Complex meta-metric for plan evaluation, Research ideas

### Community 97 - "Single Beam"
Cohesion: 0.67
Nodes (3): identify_axes(), Axes, Helper to identify the Axes in the examples below.      Draws the label in a lar

## Knowledge Gaps
- **32 isolated node(s):** `run_ab.sh script`, `run_ablation.sh script`, `run_all_plans.sh script`, `run_grid_factor_ab.sh script`, `Models directory (checkpoints + hyperparams)` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DoTA3D_v3` connect `Test Conv Regularization` to `Texture Analysis & Validation`, `Layers`, `Model Inference & Evaluation`, `Threshold Sweep & Diagnostics`, `Training Config & Factory`, `Advanced Metrics Analysis`, `Test Model Hyperparams`, `VLM-Based Quantification`, `Validation Adota`, `Layers`, `Training Losses`, `Validation Adota`, `Texture-with-Inference Correlations`, `Test Train Pipeline Smoke`, `Training-Set Analysis`, `Perf Refactor Tests`, `Model Layers & Architecture`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `run_streaming_pipeline()` connect `Streaming & Dose Accumulation` to `Directory Loader & Worker`, `Plan Spots & ROI Cropping`, `Beam Data Library`, `Test Model Hyperparams`, `Run Plan Opentps`, `Beamlet Geometry & Cropping`, `CT Rotation`, `Isocenter & Structures`, `Dose Deposit & Extraction Tests`, `BEV Rotation Timing`, `Beamlet Extraction Geometry`, `Flux`, `Edge Detection`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `model()` connect `Test Model Hyperparams` to `Texture Analysis & Validation`, `Model Inference & Evaluation`, `Threshold Sweep & Diagnostics`, `Training Config & Factory`, `Advanced Metrics Analysis`, `Range-Fidelity Analysis`, `GPR Pool & Training`, `VLM-Based Quantification`, `Streaming & Dose Accumulation`, `BEV Rotation Timing`, `Training Losses`, `Checkpoint Management`, `Perf Refactor Tests`, `Directory Loader & Worker`, `Test Inference`, `Test Conv Regularization`, `Validation Adota`, `Run`, `Test Train Pipeline Smoke`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `DoTA3D_v3` (e.g. with `TestDataset` and `SampleResult`) actually correct?**
  _`DoTA3D_v3` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 63 inferred relationships involving `ValueError` (e.g. with `_select_spot_ids()` and `load_raw_volume_dhw()`) actually correct?**
  _`ValueError` has 63 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `H5PYGenerator` (e.g. with `BPEstimator` and `CSDACorrectedEstimator`) actually correct?**
  _`H5PYGenerator` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `BeamDataLibrary` (e.g. with `ExtractionConfig` and `_FieldTiming`) actually correct?**
  _`BeamDataLibrary` has 3 INFERRED edges - model-reasoned connections that need verification._