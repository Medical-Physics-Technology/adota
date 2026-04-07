"""
CT Texture Analysis Pipeline

A command-line tool for loading CT images and computing texture analysis metrics.
Supports .mhd, .dcm, and .npy file formats.

Usage:
    uv run python scripts/ct_texture_analysis.py --help
    uv run python scripts/ct_texture_analysis.py /path/to/ct_images --format mhd
    uv run python scripts/ct_texture_analysis.py --config scripts/config_texture_analysis.yaml
"""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import typer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_processing.edge_detection import (
    LogEdgeParams,
    log_edges,
    log_edges_significant,
)
from src.image_processing.homogeneity_scores import glcm_homogeneity_idm
from src.utils.texture_analysis import (
    EXTENSION_MAP,
    ImageFormat,
    discover_images,
    load_config,
    save_config_copy,
    setup_logging,
    setup_run_directory,
)

logger = logging.getLogger(__name__)

app = typer.Typer(help="CT Texture Analysis Pipeline")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_mhd(path: Path) -> sitk.Image:
    """Load an .mhd / .raw image pair using SimpleITK.

    Args:
        path: Path to the .mhd header file.

    Returns:
        SimpleITK.Image object.
    """
    image = sitk.ReadImage(str(path))
    logger.info(
        "Loaded MHD image %s | size=%s spacing=%s",
        path.name,
        image.GetSize(),
        image.GetSpacing(),
    )
    return image


def load_dcm(path: Path) -> sitk.Image:
    """Load a single DICOM file and return a SimpleITK image.

    Args:
        path: Path to the .dcm file.

    Returns:
        SimpleITK.Image object.
    """
    image = sitk.ReadImage(str(path))
    logger.info(
        "Loaded DICOM image %s | size=%s",
        path.name,
        image.GetSize(),
    )
    return image


def load_npy(path: Path) -> sitk.Image:
    """Load a .npy array and wrap it as a SimpleITK image.

    Args:
        path: Path to the .npy file.

    Returns:
        SimpleITK.Image object.
    """
    array = np.load(str(path))
    logger.info(
        "Loaded NPY file %s | shape=%s dtype=%s", path.name, array.shape, array.dtype
    )
    image = sitk.GetImageFromArray(array)
    return image


LOADER_MAP = {
    ImageFormat.MHD: load_mhd,
    ImageFormat.DCM: load_dcm,
    ImageFormat.NPY: load_npy,
}


def load_image(path: Path, fmt: ImageFormat) -> sitk.Image:
    """Dispatch to the appropriate loader based on *fmt*.

    Args:
        path: Path to the image file.
        fmt: Image format identifier.

    Returns:
        SimpleITK.Image object.
    """
    loader = LOADER_MAP[fmt]
    return loader(path)


# ---------------------------------------------------------------------------
# Texture analysis placeholder
# ---------------------------------------------------------------------------


def compute_texture_metrics(
    array: np.ndarray,
    glcm_cfg: dict,
) -> dict:
    """Compute texture analysis metrics for a CT image array.

    Computes GLCM homogeneity (IDM / IDMN) on representative axial slices
    and reports per-slice scores as well as the volume-level mean.

    Args:
        array: 3-D CT image data as a numpy array (z, y, x).
        glcm_cfg: GLCM configuration dictionary from the YAML config.

    Returns:
        Dictionary of metric name → value.
    """
    n_slices = array.shape[0]

    # Basic statistics
    metrics: dict = {
        "shape": list(array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }

    # GLCM parameters
    levels = int(glcm_cfg.get("levels", 64))
    variant = str(glcm_cfg.get("variant", "idm"))
    distances = tuple(int(d) for d in glcm_cfg.get("distances", [1]))
    angles_deg = glcm_cfg.get("angles_deg", [0, 45, 90, 135])
    angles = tuple(float(a) * np.pi / 180.0 for a in angles_deg)
    symmetric = bool(glcm_cfg.get("symmetric", True))
    vr = glcm_cfg.get("value_range", None)
    value_range = tuple(float(v) for v in vr) if vr is not None else None

    # Slice selection: sample slices evenly to keep runtime manageable
    n_sample = min(int(glcm_cfg.get("n_sample_slices", 10)), n_slices)
    if n_sample >= n_slices:
        slice_indices = list(range(n_slices))
    else:
        slice_indices = np.linspace(0, n_slices - 1, n_sample, dtype=int).tolist()

    logger.info(
        "Computing GLCM homogeneity (%s) on %d slices  "
        "[levels=%d, distances=%s, variant=%s]",
        variant,
        len(slice_indices),
        levels,
        distances,
        variant,
    )

    per_slice: dict[int, float] = {}
    for idx in slice_indices:
        score = glcm_homogeneity_idm(
            array[idx],
            levels=levels,
            value_range=value_range,
            distances=distances,
            angles=angles,
            symmetric=symmetric,
            variant=variant,
        )
        per_slice[idx] = score

    scores = list(per_slice.values())
    metrics["glcm_homogeneity"] = {
        "variant": variant,
        "levels": levels,
        "distances": list(distances),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "per_slice": {str(k): v for k, v in per_slice.items()},
    }

    return metrics


# ---------------------------------------------------------------------------
# Edge detection visualisation
# ---------------------------------------------------------------------------


def save_edge_detection_figure(
    array: np.ndarray,
    output_path: Path,
    spacing_mm: Optional[tuple] = None,
    method: str = "significant",
    sigma_mm: float = 1.0,
    zero_cross_thresh: float = 0.0,
    connectivity: int = 1,
    edge_params: Optional[LogEdgeParams] = None,
    window_center: float = 0.0,
    window_width: float = 1500.0,
) -> Path:
    """Create a 3×2 figure (3 axial slices × [CT, edges]) and save as PNG.

    Slice indices: 50, middle, shape[0]-50.

    Args:
        array: 3-D CT numpy array (z, y, x).
        output_path: Destination PNG file path.
        spacing_mm: Voxel spacing (z, y, x). Passed to LoG.
        method: Edge detection method – ``"simple"`` uses basic zero-crossing
            LoG, ``"significant"`` uses the advanced detector with MAD
            thresholding, gradient gating, and optional hysteresis.
        sigma_mm: Gaussian sigma in mm for LoG.
        zero_cross_thresh: Threshold for zero-crossing (simple method only).
        connectivity: Neighbourhood connectivity (simple method only).
        edge_params: Parameters for the significant detector. If *None* and
            *method* is ``"significant"``, defaults are used.
        window_center: CT display window center (HU).
        window_width: CT display window width (HU).

    Returns:
        Path to the saved figure.
    """
    n_slices = array.shape[0]
    slice_indices = [
        min(50, n_slices - 1),
        n_slices // 2,
        max(n_slices - 50, 0),
    ]

    # Compute edge detection on the full volume
    if method == "significant":
        _, edges_3d = log_edges_significant(
            array,
            sigma_mm=sigma_mm,
            spacing_mm=spacing_mm,
            params=edge_params,
        )
        title_suffix = "LoG Significant"
    else:
        _, edges_3d = log_edges(
            array,
            sigma_mm=sigma_mm,
            spacing_mm=spacing_mm,
            zero_cross_thresh=zero_cross_thresh,
            connectivity=connectivity,
        )
        title_suffix = "LoG Simple"

    # CT windowing for display
    vmin = window_center - window_width / 2
    vmax = window_center + window_width / 2

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        f"CT Axial Slices & {title_suffix} Edge Detection (σ={sigma_mm} mm)",
        fontsize=14,
        y=0.98,
    )

    for row, idx in enumerate(slice_indices):
        ct_slice = array[idx]
        edge_slice = edges_3d[idx]

        # Column 0: original CT
        ax_ct = axes[row, 0]
        ax_ct.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax)
        ax_ct.set_title(f"CT  —  slice {idx}")
        ax_ct.axis("off")

        # Column 1: detected edges
        ax_edge = axes[row, 1]
        ax_edge.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax, alpha=0.4)
        ax_edge.imshow(edge_slice, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
        ax_edge.set_title(f"Edges  —  slice {idx}")
        ax_edge.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved edge detection figure: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def run(
    input_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to a CT image file or a directory containing CT images.",
        ),
    ] = None,
    fmt: Annotated[
        ImageFormat,
        typer.Option("--format", "-f", help="Image file format to load."),
    ] = ImageFormat.MHD,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to a YAML configuration file.",
        ),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option(
            "--runs-dir",
            help="Base directory where run results are stored.",
        ),
    ] = PROJECT_ROOT
    / "runs",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Load CT images and run the texture analysis pipeline."""
    # ---- Load config (if provided) ----
    cfg: dict = {}
    if config is not None:
        cfg = load_config(config)

    # CLI flags override config values
    if input_path is None:
        raw = cfg.get("input_path")
        if raw is None:
            typer.echo(
                "Error: input_path must be given as argument or in config.", err=True
            )
            raise typer.Exit(code=1)
        input_path = Path(raw).resolve()

    if not input_path.exists():
        typer.echo(f"Error: path {input_path} does not exist.", err=True)
        raise typer.Exit(code=1)

    fmt = ImageFormat(cfg.get("format", fmt.value))
    runs_dir = Path(cfg.get("runs_dir", str(runs_dir)))
    verbose = cfg.get("verbose", verbose)

    # ---- Set up run directory & logging ----
    run_dir = setup_run_directory(runs_dir)
    log_file = setup_logging(run_dir, verbose=verbose)

    if config is not None:
        save_config_copy(config, run_dir)

    logger.info("Texture analysis run started")
    logger.info("Run directory: %s", run_dir)
    logger.info("Input path   : %s", input_path)
    logger.info("Image format : %s", fmt.value)

    # ---- Discover images ----
    image_paths = discover_images(input_path, fmt)
    logger.info("Discovered %d image(s)", len(image_paths))

    # ---- Process each image ----
    all_results: list[dict] = []

    # GLCM homogeneity parameters (from config or defaults)
    glcm_cfg = cfg.get("glcm", {})

    # Edge detection parameters (from config or defaults)
    edge_cfg = cfg.get("edge_detection", {})
    edge_enabled = bool(edge_cfg.get("enabled", False))

    if edge_enabled:
        edge_method = str(edge_cfg.get("method", "significant"))
        sigma_mm = float(edge_cfg.get("sigma_mm", 1.0))
        window_center = float(edge_cfg.get("window_center", 0.0))
        window_width = float(edge_cfg.get("window_width", 1500.0))
        zero_cross_thresh = float(edge_cfg.get("zero_cross_thresh", 0.0))
        simple_connectivity = int(edge_cfg.get("connectivity", 1))

        sig_cfg = edge_cfg.get("significant", {})
        edge_params = LogEdgeParams(
            scale_normalized=bool(sig_cfg.get("scale_normalized", True)),
            log_k_mad=float(sig_cfg.get("log_k_mad", 6.0)),
            grad_percentile=float(sig_cfg.get("grad_percentile", 95.0)),
            hysteresis=bool(sig_cfg.get("hysteresis", True)),
            low_high_ratio=float(sig_cfg.get("low_high_ratio", 0.5)),
            connectivity=int(sig_cfg.get("connectivity", 1)),
            mode=str(sig_cfg.get("mode", "reflect")),
        )

        figures_dir = run_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        logger.info(
            "Edge detection enabled (method=%s, sigma=%.2f mm)", edge_method, sigma_mm
        )
    else:
        logger.info("Edge detection disabled")

    for path in image_paths:
        logger.info("Processing: %s", path)
        try:
            image_sitk = load_image(path, fmt)
            array = sitk.GetArrayFromImage(image_sitk)

            metrics = compute_texture_metrics(array, glcm_cfg=glcm_cfg)
            result = {
                "file": str(path),
                "metrics": metrics,
            }
            all_results.append(result)
            logger.info(
                "  -> GLCM homogeneity mean=%.6f  std=%.6f",
                metrics["glcm_homogeneity"]["mean"],
                metrics["glcm_homogeneity"]["std"],
            )

            # Generate edge detection figure (if enabled)
            if edge_enabled:
                sp = image_sitk.GetSpacing()  # (x, y, z)
                spacing_mm: Optional[tuple] = None
                if len(sp) == 3:
                    spacing_mm = (sp[2], sp[1], sp[0])  # (z, y, x)

                fig_name = f"{path.stem}_edges.png"
                save_edge_detection_figure(
                    array,
                    output_path=figures_dir / fig_name,
                    spacing_mm=spacing_mm,
                    method=edge_method,
                    sigma_mm=sigma_mm,
                    zero_cross_thresh=zero_cross_thresh,
                    connectivity=simple_connectivity,
                    edge_params=edge_params,
                    window_center=window_center,
                    window_width=window_width,
                )
        except Exception:
            logger.exception("Failed to process %s", path)

    # ---- Save results ----
    results_file = run_dir / "texture_results.json"
    with open(results_file, "w") as fh:
        json.dump(all_results, fh, indent=2)
    logger.info("Results saved to %s", results_file)

    # ---- Summary ----
    typer.echo(f"\n✅ Processed {len(all_results)}/{len(image_paths)} images.")
    typer.echo(f"   Run directory : {run_dir}")
    typer.echo(f"   Results file  : {results_file}")
    typer.echo(f"   Log file      : {log_file}")


if __name__ == "__main__":
    app()
