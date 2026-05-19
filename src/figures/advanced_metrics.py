"""Figure generation for the advanced-metrics training-set analysis.

Produces per-record CT segmentation figures and GT-vs-prediction
publication figures, and selects representative cases (best/worst GPR,
highest Pflugfelder HI, highest sum_sobel_bp) from a set of
``SampleRecord``s.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import sobel as ndimage_sobel
from tqdm import tqdm

from src.adota.models import DoTA3D_v3
from src.figures.ct_visualizations import plot_ct_with_segmentation, smooth_ct
from src.figures.single_beam import publication_figure
from src.loaders.generator import H5PYGenerator
from src.metrics.classic import calculate_pure_mape, calculate_rmse
from src.schemas.configs import AdvancedAnalysisConfig as AnalysisConfig
from src.schemas.results import SampleRecord
from src.utils.dose_grid_utils import estimate_bp_range
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy

logger = logging.getLogger(__name__)


def generate_figures_for_records(
    model: DoTA3D_v3,
    records: list[SampleRecord],
    record_ids: list,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
    device: torch.device,
    output_dir: Path,
    category_label: str,
) -> None:
    """Generate CT segmentation and publication figures for a list of records.

    Both figure types are saved into *output_dir*.

    Args:
        model: The loaded DoTA model.
        records: SampleRecord objects to plot.
        record_ids: Ordered list of record IDs matching the dataset.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.
        device: Target device for computation.
        output_dir: Directory where figures are stored.
        category_label: Human-readable label for logging (e.g. "worst_gpr").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx: dict[str, int] = {rid: i for i, rid in enumerate(record_ids)}
    scale = config.scale

    for record in tqdm(records, desc=f"Figures ({category_label})"):
        sample_idx = id_to_idx.get(record.sample_id)
        if sample_idx is None:
            logger.warning(
                f"Sample {record.sample_id} not found in dataset -- skipping"
            )
            continue

        # Reload data
        x, energy, y = dataset[sample_idx]
        x = x.to(device)
        energy = energy.to(device)
        y = y.to(device)

        # De-normalise CT and dose
        ct_norm = x[0].cpu().numpy()
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])
        flux_np = x[1].cpu().numpy()

        gt_dose_norm = y.cpu().numpy()
        gt_dose = inverse_minmax(
            gt_dose_norm if gt_dose_norm.ndim == 4 else gt_dose_norm[np.newaxis],
            scale["min_ds"],
            scale["max_ds"],
        ).squeeze()

        # Re-estimate BP range for figure annotation
        z_min, z_max = estimate_bp_range(
            ct_hu,
            gt_dose,
            proximal_fraction=config.proximal_fraction,
            fall_fraction=config.fall_fraction,
        )
        voxel_spacing_mm = config.resolution[0]

        # ── CT segmentation figure ──────────────────────────────────
        ct_hu_smooth = smooth_ct(
            ct_hu, method=config.smoothing_method, sigma=config.smoothing_sigma
        )

        g_z_raw = ndimage_sobel(ct_hu, axis=0)
        g_y_raw = ndimage_sobel(ct_hu, axis=1)
        g_x_raw = ndimage_sobel(ct_hu, axis=2)
        sobel_mag_raw = np.sqrt(g_z_raw**2 + g_y_raw**2 + g_x_raw**2)

        g_z = ndimage_sobel(ct_hu_smooth, axis=0)
        g_y = ndimage_sobel(ct_hu_smooth, axis=1)
        g_x = ndimage_sobel(ct_hu_smooth, axis=2)
        sobel_mag = np.sqrt(g_z**2 + g_y**2 + g_x**2)

        fig_path = output_dir / f"{record.sample_id}_ct_seg.png"
        plot_ct_with_segmentation(
            ct_hu_smooth,
            record.sample_id,
            fig_path,
            ct_hu_unsmoothed=ct_hu,
            gt_dose=gt_dose,
            bp_range_slices=(z_min, z_max),
            voxel_spacing_mm=voxel_spacing_mm,
            sobel_magnitude=sobel_mag,
            sobel_magnitude_raw=sobel_mag_raw,
        )

        # ── Publication figure (GT vs prediction) ───────────────────
        with torch.no_grad():
            y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]

        y_np = inverse_minmax(
            y.unsqueeze(0).detach().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )
        y_pred_np = inverse_minmax(
            y_pred.detach().cpu().numpy(),
            scale["min_ds"],
            scale["max_ds"],
        )

        gt_sq = y_np.squeeze()
        pred_sq = y_pred_np.squeeze()
        rmse = calculate_rmse(to_gy(pred_sq), to_gy(gt_sq))
        mask = gt_sq > 0.1 * np.max(gt_sq)
        mape = calculate_pure_mape(gt_sq[mask], pred_sq[mask])

        x_input = x.squeeze().cpu().numpy()  # (2, D, H, W)
        pub_path = output_dir / f"{record.sample_id}_E{record.energy_mev:.2f}MeV.svg"

        publication_figure(
            x_input,
            record.energy_mev,
            gt_sq,
            pred_sq,
            str(pub_path),
            rmse,
            mape,
            record.gpr,
            gamma_params=config.gamma_params,
            beamlet_shape=True,
        )

        logger.info(
            f"[{category_label}] Figures saved for {record.sample_id}: "
            f"GPR={record.gpr:.2f}%, pflugfelder_hi={record.pflugfelder_hi:.4f}, "
            f"sum_sobel_bp={record.sum_sobel_bp:.2f}"
        )


def generate_figures_for_selection(
    model: DoTA3D_v3,
    results: list[SampleRecord],
    record_ids: list,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
    device: torch.device,
    figures_dir: Path,
    n_samples_requested: Optional[int],
) -> None:
    """Generate CT segmentation and publication figures for selected samples.

    Generates figures for four categories of interest:
    - 3 worst performing cases (lowest GPR)
    - 3 best performing cases (highest GPR)
    - 3 cases with the highest Pflugfelder heterogeneity index
    - 3 cases with the highest sum_sobel_bp

    Each category gets its own subdirectory under ``publications/``.

    Args:
        model: The loaded DoTA model.
        results: List of SampleRecord objects (with metrics populated).
        record_ids: Ordered list of record IDs matching the dataset.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.
        device: Target device for computation.
        figures_dir: Output directory for figures.
        n_samples_requested: The n_samples value from the config/CLI.
    """
    publications_dir = figures_dir / "publications"
    publications_dir.mkdir(parents=True, exist_ok=True)

    # ── 3 worst GPR (lowest) ────────────────────────────────────────────
    sorted_by_gpr_asc = sorted(results, key=lambda r: r.gpr)
    worst_gpr = sorted_by_gpr_asc[:3]
    logger.info("3 worst GPR cases:")
    for rank, r in enumerate(worst_gpr):
        logger.info(f"  #{rank+1}: {r.sample_id}  GPR={r.gpr:.2f}%")

    # ── 3 best GPR (highest) ────────────────────────────────────────────
    best_gpr = sorted_by_gpr_asc[-3:][::-1]
    logger.info("3 best GPR cases:")
    for rank, r in enumerate(best_gpr):
        logger.info(f"  #{rank+1}: {r.sample_id}  GPR={r.gpr:.2f}%")

    # ── 3 highest Pflugfelder HI ────────────────────────────────────────
    sorted_by_phi = sorted(results, key=lambda r: r.pflugfelder_hi, reverse=True)
    top_phi = sorted_by_phi[:3]
    logger.info("3 highest pflugfelder_hi cases:")
    for rank, r in enumerate(top_phi):
        logger.info(
            f"  #{rank+1}: {r.sample_id}  pflugfelder_hi={r.pflugfelder_hi:.4f}"
        )

    # ── 3 highest sum_sobel_bp ──────────────────────────────────────────
    sorted_by_sobel = sorted(results, key=lambda r: r.sum_sobel_bp, reverse=True)
    top_sobel = sorted_by_sobel[:3]
    logger.info("3 highest sum_sobel_bp cases:")
    for rank, r in enumerate(top_sobel):
        logger.info(f"  #{rank+1}: {r.sample_id}  sum_sobel_bp={r.sum_sobel_bp:.2f}")

    # ── Generate figures for each category ──────────────────────────────
    categories = [
        ("worst_gpr", worst_gpr),
        ("best_gpr", best_gpr),
        ("highest_pflugfelder_hi", top_phi),
        ("highest_sum_sobel", top_sobel),
    ]

    for category_name, selected in categories:
        cat_dir = publications_dir / category_name
        generate_figures_for_records(
            model=model,
            records=selected,
            record_ids=record_ids,
            dataset=dataset,
            config=config,
            device=device,
            output_dir=cat_dir,
            category_label=category_name,
        )

    logger.info("Figure generation complete")


def generate_beam_angle_figures(
    model: DoTA3D_v3,
    results: list[SampleRecord],
    record_ids: list,
    dataset: H5PYGenerator,
    config: AnalysisConfig,
    device: torch.device,
    figures_dir: Path,
) -> None:
    """Generate CT segmentation and publication figures for extreme beam-angle cases.

    Selects the 3 cases with the lowest ``sobel_dw_beam_angle`` (θ ≈ 0°,
    dominant edge perpendicular to beam) and the 3 with the highest
    (θ ≈ 90°, dominant edge parallel to beam), then generates both Sobel/CT
    and dose-distribution figures for each group.

    Args:
        model: The loaded DoTA model.
        results: List of SampleRecord objects (with metrics populated).
        record_ids: Ordered list of record IDs matching the dataset.
        dataset: The H5PYGenerator dataset.
        config: Analysis configuration.
        device: Target device for computation.
        figures_dir: Output directory for figures.
    """
    angle_dir = figures_dir / "beam_angle_analysis"
    angle_dir.mkdir(parents=True, exist_ok=True)

    sorted_by_angle = sorted(results, key=lambda r: r.sobel_dw_beam_angle)
    low_angle = sorted_by_angle[:3]          # θ ≈ 0°: edge perpendicular to beam
    high_angle = sorted_by_angle[-3:][::-1]  # θ ≈ 90°: edge parallel to beam

    logger.info("3 lowest sobel_dw_beam_angle (edge perpendicular to beam):")
    for rank, r in enumerate(low_angle):
        logger.info(
            f"  #{rank+1}: {r.sample_id}  θ={r.sobel_dw_beam_angle:.1f}°  GPR={r.gpr:.2f}%"
        )

    logger.info("3 highest sobel_dw_beam_angle (edge parallel to beam):")
    for rank, r in enumerate(high_angle):
        logger.info(
            f"  #{rank+1}: {r.sample_id}  θ={r.sobel_dw_beam_angle:.1f}°  GPR={r.gpr:.2f}%"
        )

    categories = [
        ("low_angle", low_angle),
        ("high_angle", high_angle),
    ]
    for category_name, selected in categories:
        generate_figures_for_records(
            model=model,
            records=selected,
            record_ids=record_ids,
            dataset=dataset,
            config=config,
            device=device,
            output_dir=angle_dir / category_name,
            category_label=f"beam_angle_{category_name}",
        )

    logger.info("Beam-angle figure generation complete")
