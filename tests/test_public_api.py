"""Pin the importable surface of the modules that the refactor splits.

Flow:
1. ``PUBLIC_API`` maps a module path to every name defined in it today.
2. Each name is imported from that path and must resolve.

The inventory was generated from the pre-refactor code, so it records the
surface that call sites already depend on. When an oversized module is split,
its entry moves to the new location(s) and the names must still resolve --
nothing may quietly disappear. Private (underscore) names are included because
tests and sibling modules import several of them.
"""

from __future__ import annotations

import importlib

import pytest

PUBLIC_API: dict[str, tuple[str, ...]] = {
    "src.adota.layers": (
        "Conv3D",
        "ConvBlock3D_v2",
        "ConvDecoder3D",
        "ConvEncoder3D",
        "CroppingLayer",
        "LinearProj",
        "Permute",
        "PositionalEmbedding",
        "ReshapeLayer",
        "TransformerEncoderLayerDoTA",
    ),
    # src/training/run.py was split into four role-named modules; the same
    # symbols must still resolve, now from their new homes.
    "src.training.logging_utils": (
        "BANNER_WIDTH",
        "PHASE_FIELD_WIDTH",
        "RelativeTimeFormatter",
        "_DEFAULT_PHASE",
        "format_duration",
        "log_banner",
        "log_phase",
        "log_section",
        "setup_training_logging",
        "silence_pymedphys",
    ),
    "src.training.run_dir": (
        "MetricsLog",
        "RUN_SUBDIRS",
        "_config_to_dict",
        "_file_fingerprint",
        "_git_info",
        "_gpu_info",
        "save_resolved_config",
        "setup_training_run_directory",
        "write_manifest",
    ),
    "src.training.checkpoints": (
        "CheckpointManager",
        "_restore_rng_state",
        "_rng_state",
        "_unwrap_compiled",
    ),
    "src.training.diagnostics": (
        "GracefulShutdown",
        "compute_grad_norm",
        "compute_param_norm",
        "dump_nan_context",
    ),
    "src.beamlets.extraction": (
        "ExtractionConfig",
        "ROI_SIZE",
        "_FieldTiming",
        "_build_manifest",
        "_build_sim_res",
        "_extract_impl",
        "_prepare_output_dir",
        "_process_spot",
        "_save_field_overlay",
        "_save_spot",
        "_union_seconds",
        "run_extraction",
        "run_extraction_pooled",
    ),
    # src/figures/single_beam.py kept publication_figure and shed the rest.
    "src.figures.single_beam": (
        "publication_figure",
    ),
    "src.figures.axes_utils": (
        "aligned_colorbar",
        "identify_axes",
        "save_figure_as_publication_formats",
    ),
    "src.figures.input_comparison": (
        "compare_two_inputs",
    ),
    "src.figures.beamlet_input": (
        "beamlet_input_figure",
    ),
    # The Bragg-peak diagnostic moved to its own module.
    "src.figures.ct_visualizations": (
        "HU_LUT",
        "N_CLASSES",
        "_COLORS",
        "plot_ct_with_segmentation",
        "segment_hu",
        "smooth_ct",
    ),
    "src.figures.bp_diagnostic": (
        "_DEFAULT_METHOD_COLORS",
        "_FALLBACK_COLORS",
        "plot_bp_estimation_diagnostic",
    ),
    # src/training/validation.py shed its binning and attention helpers.
    "src.training.validation": (
        "_gamma_pass_rate",
        "evaluate_validation",
        "pick_canary",
        "pick_gpr_subset",
    ),
    "src.training.binning": (
        "_SampleMetrics",
        "_bin_by_fixed_edges",
        "_bin_by_quantile",
        "_worst_k_records",
    ),
    "src.training.attention": (
        "save_attention_snapshot",
    ),
}

API_CASES = [
    (module_name, symbol)
    for module_name, symbols in PUBLIC_API.items()
    for symbol in symbols
]


@pytest.mark.parametrize(("module_name", "symbol"), API_CASES)
def test_symbol_is_importable(module_name: str, symbol: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name} no longer exposes {symbol}"
