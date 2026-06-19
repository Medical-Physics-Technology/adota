"""End-to-end ADoTA plan pipeline CLI.

Given an OpenTPS plan directory this script will extract per-spot
ADoTA inputs, run inference, accumulate the predicted dose, and compare it
against the MCsquare reference.

This first iteration wires up the foundation only:

1. load the ADoTA model into memory,
2. Load .mhd file with CT and MCsquare dose,
3. Parse PlanPencil.txt file and extract beams and beamlets.
4. Perform the extraction stage (per-spot CT crop + flux projection),
5. Run inference on the extracted beamlets (ADoTA forward pass),
6. Accumulate the predicted dose into a single 3D dose grid (Dose_ADoTA.mhd),
7. Generate comparison figures (plan dose comparison, DVH comparison, gamma map) and metrics. 

Different running modes supported: stream, extract+infer+accumulate, or any subset of the stages. The streaming mode fuses all three stages into a single pass, avoiding disk I/O and saving time.

Usage:
    uv run python scripts/run_plan_opentps.py \\
        --config scripts/config_run_plan_opentps.yaml
        
For detailed usage, see the ./scripts/README.md and the --help output.
"""

import json
import logging
import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import pydicom
import SimpleITK as sitk
import typer

# Add the project root to the path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import load_yaml_config, setup_logging, setup_run_directory
from src.adota.utils import load_model
from src.beamlets.accumulation import AccumulationConfig, run_accumulation
from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.dose_scaling import load_dose_gy
from src.beamlets.extraction import (
    ExtractionConfig,
    run_extraction,
    run_extraction_pooled,
)
from src.beamlets.inference import InferenceConfig, run_inference
from src.beamlets.isocenter import isocenter_index_zyx
from src.beamlets.streaming import StreamingConfig, run_streaming_pipeline
from src.beamlets.structures import load_oriented_structures
from src.dcm.load_data import list_all_files
from src.evaluation.cli import resolve_device
from src.figures.dvh_comparison import dvh_comparison_figure, write_dvh_metrics_json
from src.figures.gamma_comparison import plan_gamma_figure
from src.figures.plan_comparison import plan_dose_comparison
from src.loaders.plan_directory import load_plan_directory
from src.metrics.plan_gamma import parse_criteria, plan_gamma
from src.metrics.plan_metrics import plan_dose_metrics

# Pipeline stages, in execution order. "extract", "infer", "accumulate", "gamma"
# and "stream" are implemented; "compare" is reserved for a later task. "stream"
# is the fused, disk-free alternative to extract+infer+accumulate.
ALL_STAGES = ("extract", "infer", "accumulate", "stream", "gamma", "compare")
BEAMLET_SUBDIR = "adota_beamlets"
ADOTA_DOSE_NAME = "Dose_ADoTA.mhd"

# Default gamma criteria as [dose%, distance_mm, dose_cutoff%]; extra pymedphys
# params merge {"interp_fraction": 5} over DEFAULT_GAMMA_PARAMS (faster on full grids).
DEFAULT_GAMMA_CRITERIA = [[1, 1, 10], [2, 2, 10], [3, 3, 10], [1, 2, 3], [1, 3, 0.1]]
DEFAULT_GAMMA_EXTRA = {"interp_fraction": 5}

logger = logging.getLogger(__name__)

app = typer.Typer(help="End-to-end ADoTA plan pipeline (OpenTPS plan directory).")

# Decision: all heavy outputs live on /scratch, never under /home (see the
# integration plan, section 2).
DEFAULT_RUNS_DIR = Path("/scratch/mstryja/tmp_adota/runs")


def load_ct_from_dicom_dir(dicom_dir: Path, logger: logging.Logger) -> list:
    """Load CT images from a directory of DICOM files (DORMANT).

    Kept from the original HPTC script for the DICOM-input path. The plan
    pipeline reads ``CT.mhd`` (the grid MCsquare simulated on) instead, so this
    is currently unused; retained for plans delivered as DICOM CT.

    Args:
        dicom_dir: Directory containing DICOM CT files.
        logger: Logger instance.

    Returns:
        CT ``pydicom`` datasets sorted by instance number.
    """
    file_lists = list_all_files(str(dicom_dir), maxDepth=0)
    dicom_files = file_lists.get("Dicom", [])

    if not dicom_files:
        logger.error(f"No DICOM files found in {dicom_dir}")
        return []

    ct_slices = []
    for file_path in dicom_files:
        try:
            dcm = pydicom.dcmread(file_path)
            if getattr(dcm, "Modality", "Unknown") == "CT":
                ct_slices.append(dcm)
        except Exception as exc:
            logger.warning(f"Could not read {file_path}: {exc}")

    ct_slices.sort(key=lambda x: getattr(x, "InstanceNumber", 0))
    logger.info(f"Loaded {len(ct_slices)} CT slices from {dicom_dir}")
    return ct_slices


def _row(label: str, seconds: float, per_beamlet_ms: Optional[float] = None) -> str:
    """Format one timing row, optionally with a per-beamlet figure."""
    row = f"  {label:<24}: {seconds:8.2f} s"
    if per_beamlet_ms is not None:
        row += f"   ({per_beamlet_ms:8.3f} ms/beamlet)"
    return row


def _step(total_s: float, per_beamlet_ms: Optional[float] = None) -> dict:
    """One timing entry: total seconds (and optional per-beamlet ms)."""
    out = {"total_s": round(float(total_s), 3)}
    if per_beamlet_ms is not None:
        out["ms_per_beamlet"] = round(float(per_beamlet_ms), 3)
    return out


def _build_timing_report(
    *,
    total_s: float,
    model_load_s: float,
    plan_load_s: float,
    extraction: Optional[dict],
    inference: Optional[dict],
    accumulation: Optional[dict],
    figure_s: float,
    streaming: Optional[dict] = None,
) -> dict:
    """Assemble the structured per-stage timing report (single source of truth)."""
    report: dict = {"total_s": round(float(total_s), 3), "stages": {}}
    stages = report["stages"]

    if model_load_s > 0:
        stages["model_load"] = _step(model_load_s)
    stages["plan_load"] = _step(plan_load_s)

    if streaming is not None:
        n = max(int(streaming["n_spots"]), 1)
        report["n_spots"] = int(streaming["n_spots"])
        t = streaming["timing"]
        stages["streaming"] = {
            **_step(streaming["elapsed_s"], streaming["elapsed_s"] / n * 1000),
            "n_spots": n,
            "n_fields": streaming["n_fields"],
            "grid_factor": streaming.get("grid_factor", 1),
            "grid_mode": streaming.get("grid_mode", "1mm"),
            "steps": {k: _step(v) for k, v in t.items()},
        }

    if extraction is not None:
        n = max(int(extraction["n_spots"]), 1)
        report["n_spots"] = int(extraction["n_spots"])
        tpf = extraction.get("timing_per_field", {}).values()
        rotation = sum(t["rotation_s"] for t in tpf)
        crop = sum(t["crop_s_total"] for t in tpf)
        flux = sum(t["flux_s_total"] for t in tpf)
        save = sum(t["save_s_total"] for t in tpf)
        stages["extraction"] = {
            **_step(extraction["elapsed_s"], extraction["elapsed_s"] / n * 1000),
            "n_spots": n,
            "parallel": bool(extraction.get("parallel", False)),
            "workers": int(extraction.get("workers", 0)),
            "steps": {
                "rotation": _step(rotation),
                "ct_cropping": _step(crop, crop / n * 1000),
                "flux_projection": _step(flux, flux / n * 1000),
                "save_beamlets": _step(save, save / n * 1000),
            },
        }

    if inference is not None:
        n = max(int(inference["n_spots"]), 1)
        stages["inference"] = {
            **_step(inference["elapsed_s"], inference["elapsed_s"] / n * 1000),
            "n_spots": n,
            "n_batches": inference["n_batches"],
            "batch_size": inference["batch_size"],
            "device": inference.get("device"),
            "steps": {
                "record_load": _step(inference["read_s"], inference["ms_per_spot_read"]),
                "downsample": _step(inference["downsample_s"], inference["ms_per_spot_downsample"]),
                "forward": _step(inference["forward_s"], inference["ms_per_spot_forward"]),
                "upsample": _step(inference["upsample_s"], inference["ms_per_spot_upsample"]),
                "save_write": _step(inference["save_write_s"], inference["ms_per_spot_save_write"]),
            },
        }

    if accumulation is not None:
        n = max(int(accumulation["n_spots"]), 1)
        steps = {
            "deposit": _step(accumulation["deposit_s"]),
            "derotate": _step(accumulation["derotate_s"]),
        }
        if "write_s" in accumulation:
            steps["write"] = _step(accumulation["write_s"])
        stages["accumulation"] = {
            **_step(accumulation["elapsed_s"], accumulation["elapsed_s"] / n * 1000),
            "n_fields": accumulation["n_fields"],
            "steps": steps,
        }

    if figure_s > 0:
        stages["comparison_figures"] = _step(figure_s)

    return report


def _merge_timing_report(existing: dict, new: dict) -> dict:
    """Merge a freshly built timing report into a previously saved one.

    Stages that ran in this invocation overwrite their matching entry; stages
    recorded by earlier runs (but not re-run now) are kept, so running a single
    stage later never discards previously measured timings. ``total_s`` keeps
    its per-run meaning (the wall-clock of this invocation), while
    ``aggregate_total_s`` is the sum of the merged top-level stage totals, i.e.
    the cumulative measured work across every run that touched this plan.
    """
    merged = dict(existing)
    stages = dict(existing.get("stages", {}))
    stages.update(new.get("stages", {}))
    merged["stages"] = stages
    if "n_spots" in new:
        merged["n_spots"] = new["n_spots"]
    merged["total_s"] = new.get("total_s", existing.get("total_s"))
    merged["aggregate_total_s"] = round(
        sum(float(s.get("total_s", 0.0)) for s in stages.values()), 3
    )
    return merged


def _format_timing_report(report: dict) -> str:
    """Render the timing report as the human-readable summary table."""
    stages = report["stages"]
    lines = ["=" * 70, "TIMING SUMMARY", "=" * 70]
    if "model_load" in stages:
        lines.append(_row("model load", stages["model_load"]["total_s"]))
    lines.append(_row("plan load", stages["plan_load"]["total_s"]))

    st = stages.get("streaming")
    if st:
        s = st["steps"]
        # Distinct mode label: 1mm per-beamlet vs 2mm field-level resampling. At
        # gf=2 the crop/flux/prep/post/deposit run on the 2mm grid (the per-field
        # rotate/de-rotate carry the single resize), so those rows shrink.
        gf = st.get("grid_factor", 1)
        mode = "fused, 1mm" if gf == 1 else f"fused, {gf}mm field"
        lines.append("-" * 70)
        lines.append(
            f"Streaming ({mode})".ljust(29)
            + f"total {st['total_s']:8.2f} s   "
            f"({st['ms_per_beamlet']:8.3f} ms/beamlet)  "
            f"[{st['n_spots']} spots, {st['n_fields']} fields, no disk]"
        )
        prep_lbl = "input prep (downsample)" if gf == 1 else "input prep (no resize)"
        post_lbl = "postprocess (upsample)" if gf == 1 else "postprocess (no resize)"
        for label, key in (
            ("rotation (per field)", "rotation"), ("CT cropping", "crop"),
            ("flux projection", "flux"), (prep_lbl, "prep"),
            ("ADoTA forward", "forward"), (post_lbl, "post"),
            ("deposit", "deposit"), ("de-rotate (per field)", "derotate"),
            ("write Dose_ADoTA.mhd", "write"),
        ):
            if key in s:
                lines.append(_row(label, s[key]["total_s"]))

    ex = stages.get("extraction")
    if ex:
        s = ex["steps"]
        lines.append("-" * 70)
        mode = (
            f"pooled x{ex['workers']} threads" if ex.get("parallel") else "serial"
        )
        lines.append(
            f"Extraction               total {ex['total_s']:8.2f} s   "
            f"({ex['ms_per_beamlet']:8.3f} ms/beamlet)  [{ex['n_spots']} spots, {mode}]"
        )
        if ex.get("parallel"):
            # Sub-steps are the real wall-clock time each step was active (union of
            # the concurrent per-spot intervals). Different steps still run in
            # parallel across threads, so they can overlap each other and need not
            # sum to the stage wall total above.
            lines.append(
                "  (sub-steps are real wall-time each step was active; they run "
                "concurrently, so they can overlap)"
            )
        lines.append(_row("rotation (per field)", s["rotation"]["total_s"]))
        lines.append(_row("CT cropping", s["ct_cropping"]["total_s"], s["ct_cropping"]["ms_per_beamlet"]))
        lines.append(_row("flux projection", s["flux_projection"]["total_s"], s["flux_projection"]["ms_per_beamlet"]))
        lines.append(_row("save beamlets to disk", s["save_beamlets"]["total_s"], s["save_beamlets"]["ms_per_beamlet"]))

    inf = stages.get("inference")
    if inf:
        s = inf["steps"]
        lines.append("-" * 70)
        lines.append(
            f"ADoTA inference          total {inf['total_s']:8.2f} s   "
            f"({inf['ms_per_beamlet']:8.3f} ms/beamlet)  "
            f"[{inf['n_spots']} spots, {inf['n_batches']} batches of {inf['batch_size']}]"
        )
        lines.append(_row("record load (file read)", s["record_load"]["total_s"], s["record_load"]["ms_per_beamlet"]))
        lines.append(_row("downsample (CT -> ADoTA grid)", s["downsample"]["total_s"], s["downsample"]["ms_per_beamlet"]))
        lines.append(_row("ADoTA forward", s["forward"]["total_s"], s["forward"]["ms_per_beamlet"]))
        lines.append(_row("upsample (ADoTA -> ROI grid)", s["upsample"]["total_s"], s["upsample"]["ms_per_beamlet"]))
        lines.append(_row("save predictions to disk", s["save_write"]["total_s"], s["save_write"]["ms_per_beamlet"]))

    ac = stages.get("accumulation")
    if ac:
        s = ac["steps"]
        lines.append("-" * 70)
        lines.append(
            f"Dose accumulation        total {ac['total_s']:8.2f} s   "
            f"({ac['ms_per_beamlet']:8.3f} ms/beamlet)  [{ac['n_fields']} fields]"
        )
        lines.append(_row("deposit", s["deposit"]["total_s"]))
        lines.append(_row("de-rotate (per field)", s["derotate"]["total_s"]))
        if "write" in s:
            lines.append(_row("write Dose_ADoTA.mhd", s["write"]["total_s"]))

    if "comparison_figures" in stages:
        lines.append("-" * 70)
        lines.append(_row("comparison figure", stages["comparison_figures"]["total_s"]))

    lines.append("-" * 70)
    lines.append(f"  {'TOTAL':<24}: {report['total_s']:8.2f} s")
    lines.append("=" * 70)
    return "\n".join(lines)


@app.command()
def main(
    plan_dir: Annotated[
        Optional[Path],
        typer.Option(help="OpenTPS plan directory (CT.mhd, PlanPencil.txt, ...)."),
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(help="Path to YAML configuration file."),
    ] = None,
    model_name: Annotated[
        Optional[str], typer.Option(help="Model directory name under models/.")
    ] = None,
    model_fname: Annotated[
        Optional[str], typer.Option(help="Model weights filename.")
    ] = None,
    bdl_path: Annotated[
        Optional[Path],
        typer.Option(help="Beam data library path (default: plan-local bdl.txt)."),
    ] = None,
    device_index: Annotated[
        Optional[int], typer.Option(help="CUDA device index (-1 for CPU).")
    ] = None,
    max_fields: Annotated[
        Optional[int], typer.Option(help="Fields to expand per fraction in preview.")
    ] = None,
    max_control_points: Annotated[
        Optional[int],
        typer.Option(help="Control points to expand per field in preview."),
    ] = None,
    stages: Annotated[
        Optional[str],
        typer.Option(help="Comma-separated pipeline stages (default: extract)."),
    ] = None,
    n_spots: Annotated[
        Optional[int], typer.Option(help="Extract only the first N spots (subset).")
    ] = None,
    beams: Annotated[
        Optional[str],
        typer.Option(help="Comma-separated beam (field) indices to extract."),
    ] = None,
    overwrite: Annotated[
        Optional[bool], typer.Option(help="Allow writing into a non-empty output dir.")
    ] = None,
    no_overlays: Annotated[
        Optional[bool], typer.Option(help="Skip per-field overlay PNGs.")
    ] = None,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Enable verbose/debug logging.")
    ] = None,
    grid_factor: Annotated[
        Optional[int],
        typer.Option(help="Field-level resampling factor for the extract/infer/"
                          "accumulate and stream stages (1 = 1mm per-beamlet; "
                          "2 = 2mm field grid)."),
    ] = None,
) -> None:
    """Main CLI entry point for the ADoTA plan pipeline.

    Configurable via CLI flags, a YAML config (``--config``), or both. CLI
    flags take precedence over YAML values.
    """
    # Load YAML config if provided.
    yaml_config: dict = {}
    config_path: Optional[Path] = None
    if config is not None:
        config_path = config if config.is_absolute() else PROJECT_ROOT / config
        yaml_config = load_yaml_config(config_path)

    # Merge: CLI overrides YAML, YAML overrides defaults.
    plan_dir = plan_dir if plan_dir is not None else yaml_config.get("plan_dir")
    model_name = model_name or yaml_config.get("model_name")
    model_fname = model_fname or yaml_config.get("model_fname", "best_model.pth")
    bdl_path = bdl_path if bdl_path is not None else yaml_config.get("bdl_path")
    device_index = (
        device_index if device_index is not None else yaml_config.get("device_index", 0)
    )
    max_fields = (
        max_fields if max_fields is not None else yaml_config.get("max_fields", 2)
    )
    max_control_points = (
        max_control_points
        if max_control_points is not None
        else yaml_config.get("max_control_points", 3)
    )
    stages_raw = stages if stages is not None else yaml_config.get("stages", "extract")
    # CLI grid_factor overrides YAML; write it back so the stream stage reads it.
    yaml_config["grid_factor"] = (
        grid_factor if grid_factor is not None else int(yaml_config.get("grid_factor", 1))
    )
    n_spots = n_spots if n_spots is not None else yaml_config.get("n_spots")
    beams_raw = beams if beams is not None else yaml_config.get("beams")
    overwrite = (
        overwrite if overwrite is not None else yaml_config.get("overwrite", False)
    )
    no_overlays = (
        no_overlays if no_overlays is not None else yaml_config.get("no_overlays", False)
    )
    verbose = verbose if verbose is not None else yaml_config.get("verbose", False)

    # Parse the stage / beam lists.
    stage_list = [s.strip() for s in str(stages_raw).split(",") if s.strip()]
    unknown = [s for s in stage_list if s not in ALL_STAGES]
    if unknown:
        raise typer.BadParameter(
            f"Unknown stage(s) {unknown}; valid stages are {list(ALL_STAGES)}"
        )
    beam_list = (
        [int(b) for b in str(beams_raw).split(",") if str(b).strip() != ""]
        if beams_raw not in (None, "")
        else None
    )

    # Validate required arguments.
    if plan_dir is None:
        raise typer.BadParameter("PLAN_DIR is required (via --plan-dir or YAML config)")
    if ("infer" in stage_list or "stream" in stage_list) and model_name is None:
        raise typer.BadParameter(
            "MODEL_NAME is required for the infer/stream stage (via --model-name or YAML)"
        )

    plan_dir = Path(plan_dir)
    bdl_path = Path(bdl_path) if bdl_path is not None else None

    # Setup run directory and logging on /scratch.
    runs_dir = Path(yaml_config.get("runs_dir", DEFAULT_RUNS_DIR))
    run_dir = setup_run_directory(runs_dir, prefix="plan_", subdirs=())
    log_file = setup_logging(run_dir, verbose=verbose, log_filename="run.log")
    if config_path is not None:
        import shutil

        shutil.copy2(config_path, run_dir / config_path.name)

    logger.info("=" * 70)
    logger.info("ADoTA plan pipeline")
    logger.info("=" * 70)
    logger.info("Run directory : %s", run_dir)
    logger.info("Log file      : %s", log_file)
    logger.info("Plan directory: %s", plan_dir)
    logger.info("Stages        : %s", stage_list)

    device = resolve_device(device_index)
    logger.info("Device        : %s", device)

    pipeline_start = perf_counter()
    extraction_summary = None
    inference_summary = None
    accumulation_summary = None
    streaming_summary = None
    figure_s = 0.0

    # --- Load the model into memory (only when inference is requested) --------
    model = None
    model_load_s = 0.0
    if "infer" in stage_list or "stream" in stage_list:
        model_hub = PROJECT_ROOT / "models"
        model_path = model_hub / model_name / model_fname
        hyperparams_path = model_hub / model_name / "hyperparams.json"
        logger.info("Loading model : %s", model_path)
        load_t = perf_counter()
        model = load_model(model_path, hyperparams_path, device)
        model_load_s = perf_counter() - load_t
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model loaded  : %s (%s parameters)", type(model).__name__, f"{n_params:,}"
        )

    # --- Load + parse the plan directory -------------------------------------
    logger.info("Loading plan directory ...")
    plan_load_t = perf_counter()
    plan_directory = load_plan_directory(plan_dir, bdl_path=bdl_path)
    plan_load_s = perf_counter() - plan_load_t

    preview = plan_directory.summary(
        max_fields=max_fields, max_control_points=max_control_points
    )
    logger.info("\n%s\n%s\n%s", "-" * 70, preview, "-" * 70)

    # --- Stage: extract ------------------------------------------------------
    if "extract" in stage_list:
        output_dir = plan_dir / BEAMLET_SUBDIR
        logger.info("=" * 70)
        logger.info("Stage: extract -> %s", output_dir)
        logger.info("=" * 70)
        flux_on_gpu = bool(yaml_config.get("flux_on_gpu", False))
        grid_factor = int(yaml_config.get("grid_factor", 1))
        if flux_on_gpu:
            logger.info("Flux projection: GPU path on %s", device)
        if grid_factor != 1:
            logger.info(
                "Field-level resampling ENABLED: grid_factor=%d (%dmm crops)",
                grid_factor, grid_factor,
            )
        extraction_config = ExtractionConfig(
            n_spots=n_spots,
            beams=beam_list,
            overwrite=overwrite,
            save_overlays=not no_overlays,
            bdl_path=bdl_path,
            flux_on_gpu=flux_on_gpu,
            flux_device=str(device),
            grid_factor=grid_factor,
        )
        # Serial reference (run_extraction) vs thread-pooled twin; both
        # bit-identical. Selected by config so the serial path stays comparable.
        extraction_parallel = bool(yaml_config.get("extraction_parallel", False))
        extraction_workers = int(yaml_config.get("extraction_workers", 0))
        if extraction_parallel:
            logger.info(
                "Extraction: parallel (thread pool, workers=%s)",
                extraction_workers or "auto",
            )
            extraction_summary = run_extraction_pooled(
                plan_directory, output_dir, extraction_config, workers=extraction_workers
            )
        else:
            extraction_summary = run_extraction(
                plan_directory, output_dir, extraction_config
            )

    # --- Stage: infer --------------------------------------------------------
    if "infer" in stage_list:
        beamlets_dir = plan_dir / BEAMLET_SUBDIR
        logger.info("=" * 70)
        logger.info("Stage: infer %s", beamlets_dir)
        logger.info("=" * 70)
        inference_config = InferenceConfig(
            batch_size=yaml_config.get("batch_size", 56),
            grid_factor=int(yaml_config.get("grid_factor", 1)),
        )
        inference_summary = run_inference(beamlets_dir, model, device, inference_config)

    # --- Stage: accumulate ---------------------------------------------------
    if "accumulate" in stage_list:
        beamlets_dir = plan_dir / BEAMLET_SUBDIR
        dose_path = plan_dir / ADOTA_DOSE_NAME
        # When inference ran, the predicted dose exists; default to it.
        default_source = "prediction" if "infer" in stage_list else "flux"
        dose_source = yaml_config.get("dose_source") or default_source
        # Optional dose calibration: off by default (factor 1.0 = unchanged).
        calibration_factor = (
            float(yaml_config.get("dose_calibration_factor", 1.0))
            if yaml_config.get("dose_calibration_enabled", False)
            else 1.0
        )
        logger.info("=" * 70)
        logger.info("Stage: accumulate %s -> %s (source=%s)", beamlets_dir, dose_path, dose_source)
        if calibration_factor != 1.0:
            logger.info("Dose calibration ENABLED: scaling accumulated dose by %.4f", calibration_factor)
        logger.info("=" * 70)
        accumulation_config = AccumulationConfig(
            dose_source=dose_source, calibration_factor=calibration_factor
        )
        accumulation_summary = run_accumulation(
            plan_directory, beamlets_dir, dose_path, accumulation_config
        )
        # Auto-generate the ADoTA vs MCsquare comparison + DVH figures.
        figure_s = _generate_comparison_figures(plan_directory, plan_dir, dose_path)

    # --- Stage: stream (fused, disk-free alternative to extract+infer+accumulate) -
    if "stream" in stage_list:
        dose_path = plan_dir / ADOTA_DOSE_NAME
        calibration_factor = (
            float(yaml_config.get("dose_calibration_factor", 1.0))
            if yaml_config.get("dose_calibration_enabled", False)
            else 1.0
        )
        flux_on_gpu = bool(yaml_config.get("flux_on_gpu", False))
        grid_factor = int(yaml_config.get("grid_factor", 1))
        logger.info("=" * 70)
        logger.info("Stage: stream -> %s (fused, no per-beamlet disk I/O)", dose_path)
        if grid_factor != 1:
            logger.info(
                "Field-level resampling ENABLED: grid_factor=%d (%dmm field grid)",
                grid_factor, grid_factor,
            )
        if calibration_factor != 1.0:
            logger.info("Dose calibration ENABLED: scaling dose by %.4f", calibration_factor)
        logger.info("=" * 70)
        streaming_config = StreamingConfig(
            n_spots=n_spots,
            beams=beam_list,
            bdl_path=bdl_path,
            batch_size=yaml_config.get("batch_size", 56),
            flux_on_gpu=flux_on_gpu,
            flux_device=str(device),
            calibration_factor=calibration_factor,
            grid_factor=grid_factor,
        )
        streaming_summary = run_streaming_pipeline(
            plan_directory, model, device, dose_path, streaming_config
        )
        # Same comparison + DVH figures as the accumulate stage (quality investigation).
        figure_s = _generate_comparison_figures(plan_directory, plan_dir, dose_path)

    remaining = [
        s
        for s in stage_list
        if s not in ("extract", "accumulate", "infer", "stream", "gamma")
    ]
    if remaining:
        logger.info("Stages not yet implemented, skipped: %s", remaining)

    timing_report = _build_timing_report(
        total_s=perf_counter() - pipeline_start,
        model_load_s=model_load_s,
        plan_load_s=plan_load_s,
        extraction=extraction_summary,
        inference=inference_summary,
        accumulation=accumulation_summary,
        figure_s=figure_s,
        streaming=streaming_summary,
    )
    timing_path = plan_dir / "pipeline_timing.json"
    if timing_path.exists():
        try:
            prior = json.loads(timing_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read existing %s (%s); writing fresh report.",
                timing_path, exc,
            )
            prior = None
        if isinstance(prior, dict) and prior:
            timing_report = _merge_timing_report(prior, timing_report)
    logger.info("\n%s", _format_timing_report(timing_report))
    timing_path.write_text(json.dumps(timing_report, indent=2) + "\n")
    logger.info("Timing written to %s", timing_path)

    # --- Stage: gamma (after timing; opt-in via stages) ----------------------
    if "gamma" in stage_list:
        _run_gamma_stage(plan_directory, plan_dir, yaml_config)

    logger.info("Done.")


def _generate_comparison_figures(plan_directory, plan_dir: Path, dose_path: Path) -> float:
    """Generate the ADoTA vs MCsquare dose-comparison + DVH figures/metrics.

    Reads the just-written ``Dose_ADoTA.mhd`` and the MC ``Dose.mhd`` (both in Gy),
    writes ``dose_comparison.*``, ``dvh_comparison.*`` and ``dvh_metrics.json`` next
    to the plan, and returns the seconds spent. Shared by the ``accumulate`` and
    ``stream`` stages so both produce the same quality-investigation figures.
    """
    if plan_directory.mc_dose_path is None:
        logger.warning("No MC Dose.mhd in the plan dir; skipping comparison figure.")
        return 0.0

    logger.info("Generating ADoTA vs MCsquare comparison figure (Gy) ...")
    figure_t = perf_counter()
    bdl = BeamDataLibrary.from_file(plan_directory.bdl_path)
    ct_arr = sitk.GetArrayFromImage(plan_directory.ct)
    dose_adota = sitk.GetArrayFromImage(load_dose_gy(dose_path, plan_directory.plan, bdl))
    dose_mc = sitk.GetArrayFromImage(
        load_dose_gy(plan_directory.mc_dose_path, plan_directory.plan, bdl)
    )
    fig_paths = plan_dose_comparison(
        ct_arr,
        dose_adota,  # dose_a = ADoTA (Gy)
        dose_mc,  # dose_b = MCsquare reference (Gy)
        str(plan_dir / "dose_comparison"),
        labels=("ADoTA", "MCsquare"),
        dose_unit="Gy",
    )
    for fig_path in fig_paths:
        logger.info("  comparison figure: %s", fig_path)

    # DVH comparison (structures oriented onto the dose grid).
    try:
        structures, _flips = load_oriented_structures(plan_directory)
        spacing = plan_directory.ct.GetSpacing()
        dvh_paths = dvh_comparison_figure(
            structures, dose_adota, dose_mc, spacing,
            str(plan_dir / "dvh_comparison"), labels=("ADoTA", "MCsquare"),
        )
        write_dvh_metrics_json(
            plan_dir / "dvh_metrics.json", structures, dose_adota, dose_mc,
            spacing, labels=("ADoTA", "MCsquare"),
        )
        for dvh_path in dvh_paths:
            logger.info("  DVH figure: %s", dvh_path)
        logger.info("  DVH metrics: %s", plan_dir / "dvh_metrics.json")
    except ValueError as exc:
        logger.warning("Skipping DVH comparison: %s", exc)
    return perf_counter() - figure_t


def _run_gamma_stage(plan_directory, plan_dir: Path, yaml_config: dict) -> None:
    """Plan-level gamma pass rate + error metrics + gamma-map figure.

    Reuses the accumulated ``Dose_ADoTA.mhd`` (from this run or a prior
    ``accumulate``) and the MCsquare ``Dose.mhd``, both converted to Gy. Writes a
    ``gamma_comparison.*`` figure (3 views x N criteria, sliced at the isocenter)
    and a ``gamma_metrics.json`` next to the plan, and logs a results table.
    """
    dose_path = plan_dir / ADOTA_DOSE_NAME
    if not dose_path.exists():
        logger.warning(
            "Gamma stage: %s not found (run the accumulate stage first); skipping.",
            dose_path,
        )
        return
    if plan_directory.mc_dose_path is None:
        logger.warning("Gamma stage: no MC Dose.mhd in the plan dir; skipping.")
        return

    logger.info("=" * 70)
    logger.info("Stage: gamma (plan pass rate + metrics)")
    logger.info("=" * 70)

    bdl = BeamDataLibrary.from_file(plan_directory.bdl_path)
    ct_arr = sitk.GetArrayFromImage(plan_directory.ct)
    dose_adota = sitk.GetArrayFromImage(load_dose_gy(dose_path, plan_directory.plan, bdl))
    dose_mc = sitk.GetArrayFromImage(
        load_dose_gy(plan_directory.mc_dose_path, plan_directory.plan, bdl)
    )
    spacing_zyx = tuple(float(s) for s in plan_directory.ct.GetSpacing()[::-1])
    slice_zyx = isocenter_index_zyx(plan_directory.plan, plan_directory.ct)
    slice_zyx = tuple(int(round(c)) for c in slice_zyx)

    criteria = parse_criteria(yaml_config.get("gamma_criteria", DEFAULT_GAMMA_CRITERIA))
    extra = {**DEFAULT_GAMMA_EXTRA, **(yaml_config.get("gamma_params") or {})}

    gamma_t = perf_counter()
    results = plan_gamma(dose_adota, dose_mc, spacing_zyx, criteria, extra)
    gamma_s = perf_counter() - gamma_t

    fig_paths = plan_gamma_figure(
        ct_arr, results, slice_zyx, str(plan_dir / "gamma_comparison")
    )
    for fig_path in fig_paths:
        logger.info("  gamma figure: %s", fig_path)

    metrics = plan_dose_metrics(dose_adota, dose_mc)

    # Log a compact results table.
    lines = ["", "=" * 60, "GAMMA PASS RATE (ADoTA vs MCsquare)", "=" * 60]
    for res in results:
        lines.append(f"  {res['label']:<16}: {res['pass_rate_pct']:6.2f} %")
    lines.append("-" * 60)
    lines.append(f"  MAPE (high-dose)      : {metrics['mape_pct']:6.2f} %")
    lines.append(f"  RMSE (high-dose)      : {metrics['rmse_gy']:.4g} Gy")
    lines.append(f"  Relative dose error   : {metrics['relative_dose_error_pct']:6.3f} %")
    lines.append("=" * 60)
    logger.info("\n".join(lines))

    out = {
        "criteria": [
            {"label": r["label"], "criterion": list(r["criterion"]),
             "pass_rate_pct": r["pass_rate_pct"]}
            for r in results
        ],
        "metrics": metrics,
        "slice_zyx": list(slice_zyx),
        "gamma_params_base": extra,
        "elapsed_s": gamma_s,
    }
    gamma_json = plan_dir / "gamma_metrics.json"
    gamma_json.write_text(json.dumps(out, indent=2) + "\n")
    logger.info("Gamma metrics written to %s", gamma_json)


if __name__ == "__main__":
    app()
