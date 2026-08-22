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
    "src.training.run": (
        "BANNER_WIDTH",
        "CheckpointManager",
        "GracefulShutdown",
        "MetricsLog",
        "PHASE_FIELD_WIDTH",
        "RUN_SUBDIRS",
        "RelativeTimeFormatter",
        "_DEFAULT_PHASE",
        "_config_to_dict",
        "_file_fingerprint",
        "_git_info",
        "_gpu_info",
        "_restore_rng_state",
        "_rng_state",
        "_unwrap_compiled",
        "compute_grad_norm",
        "compute_param_norm",
        "dump_nan_context",
        "format_duration",
        "log_banner",
        "log_phase",
        "log_section",
        "save_resolved_config",
        "setup_training_logging",
        "setup_training_run_directory",
        "silence_pymedphys",
        "write_manifest",
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
    "src.figures.single_beam": (
        "aligned_colorbar",
        "beamlet_input_figure",
        "compare_two_inputs",
        "identify_axes",
        "publication_figure",
        "save_figure_as_publication_formats",
    ),
    "src.figures.ct_visualizations": (
        "HU_LUT",
        "N_CLASSES",
        "_COLORS",
        "_DEFAULT_METHOD_COLORS",
        "_FALLBACK_COLORS",
        "plot_bp_estimation_diagnostic",
        "plot_ct_with_segmentation",
        "segment_hu",
        "smooth_ct",
    ),
    "src.training.validation": (
        "_SampleMetrics",
        "_bin_by_fixed_edges",
        "_bin_by_quantile",
        "_gamma_pass_rate",
        "_worst_k_records",
        "evaluate_validation",
        "pick_canary",
        "pick_gpr_subset",
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
