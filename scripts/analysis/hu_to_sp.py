"""
HU → Material Composition & Stopping Power Conversion

Converts a CT volume (in Hounsfield Units) to material composition
vectors and computes relative stopping power (RSP) using the Schneider
decomposition scheme as implemented in the TITUS/TRACer codebase.

The decomposition assigns each voxel a 12-element weight-percent
vector over {H, C, N, O, Na, Mg, P, S, Cl, Ar, K, Ca} based on its
HU value, following the piecewise-constant tissue-class look-up table
from Schneider et al. (PMB, 2000).

Usage examples:
    # Water-box phantom (HU=0 everywhere, 160×30×30)
    uv run python scripts/analysis/hu_to_sp.py water-phantom

    # Two random samples from the training HDF5
    uv run python scripts/analysis/hu_to_sp.py hdf5-samples \\
        --h5-path /path/to/trainset.h5 --n-samples 2
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.config import DEFAULT_SCALE, denormalize_energy, setup_run_directory
from src.processing.tissue_decomposition import (
    ELEMENT_DENSITIES,
    ELEMENT_NAMES,
    MOL_WEIGHTS,
    N_ELEMENTS,
    N_TISSUE_CLASSES,
    TISSUE_LABELS as _TISSUE_LABELS_SHARED,
    TISSUE_LUT,
    Z_ARRAY,
    hu_to_density,
    hu_to_rsp_schneider,
    mat_comp,
    segment_tissue,
)
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = typer.Typer(help="HU → material composition & stopping power")


# ═══════════════════════════════════════════════════════════════════════════
#  Core conversion functions, LUT, and element catalogue are imported from
#  src.processing.tissue_decomposition (see top of this file).
# ═══════════════════════════════════════════════════════════════════════════


def summarise_composition(hu: np.ndarray, label: str = "") -> None:
    """Print a summary of the HU distribution and composition."""
    prefix = f"[{label}] " if label else ""
    logger.info(f"{prefix}Shape: {hu.shape}, dtype: {hu.dtype}")
    logger.info(
        f"{prefix}HU  — min: {hu.min():.1f}, max: {hu.max():.1f}, "
        f"mean: {hu.mean():.1f}, std: {hu.std():.1f}"
    )

    rho = hu_to_density(hu)
    logger.info(
        f"{prefix}ρ   — min: {rho.min():.4f}, max: {rho.max():.4f}, "
        f"mean: {rho.mean():.4f} g/cm³"
    )

    rsp = hu_to_rsp_schneider(hu)
    logger.info(
        f"{prefix}RSP — min: {rsp.min():.4f}, max: {rsp.max():.4f}, "
        f"mean: {rsp.mean():.4f}"
    )

    comp = mat_comp(hu)  # (..., 12)
    mean_comp = comp.reshape(-1, N_ELEMENTS).mean(axis=0)
    logger.info(f"{prefix}Mean composition (wt%):")
    for name, pct in zip(ELEMENT_NAMES, mean_comp):
        if pct > 0.05:
            logger.info(f"  {name:>2s}: {pct:6.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
#  Tissue-class figure assets (colours / colormap)
# ═══════════════════════════════════════════════════════════════════════════

_TISSUE_LABELS = _TISSUE_LABELS_SHARED

# Distinct colours for the 24 tissue classes (grouped by type)
_TISSUE_COLORS = [
    "#B3E5FC",  # Air
    "#CE93D8",  # Lung
    "#FFF9C4",  # Adipose 1
    "#FFF176",  # Adipose 2
    "#FFD54F",  # Adipose 3
    "#FFCCBC",  # Soft tissue 1
    "#FFAB91",  # Soft tissue 2
    "#FF8A65",  # Soft tissue 3
    "#EF5350",  # Muscle
    "#A5D6A7",  # Spongy bone 1
    "#81C784",  # Spongy bone 2
    "#66BB6A",  # Spongy bone 3
    "#90CAF9",  # Cortical 1
    "#64B5F6",  # Cortical 2
    "#42A5F5",  # Cortical 3
    "#2196F3",  # Cortical 4
    "#1E88E5",  # Cortical 5
    "#1976D2",  # Cortical 6
    "#1565C0",  # Cortical 7
    "#0D47A1",  # Cortical 8
    "#0D47A1",  # Cortical 9
    "#0A3D91",  # Cortical 10
    "#083480",  # Cortical 11
    "#062B6F",  # Cortical 12
]

_TISSUE_CMAP = ListedColormap(_TISSUE_COLORS)
_TISSUE_NORM = BoundaryNorm(
    boundaries=list(range(N_TISSUE_CLASSES + 1)), ncolors=N_TISSUE_CLASSES
)


def plot_hu_material_density(
    ct_hu: np.ndarray,
    sample_id: str,
    output_path: Path,
    voxel_spacing_mm: float = 2.0,
) -> None:
    """Generate a 3×2 figure: HU, material assignment, and density map.

    Row 1: CT in HU (axial + sagittal centre slices).
    Row 2: Tissue-class assignment from Schneider LUT (discrete colourmap).
    Row 3: Mass density ρ [g/cm³] from piecewise-linear conversion.

    Args:
        ct_hu: 3-D CT volume in HU ``(D, H, W)``.
        sample_id: Identifier used in the figure title / filename.
        output_path: Where to save the PNG figure.
        voxel_spacing_mm: Voxel spacing for axis labels [mm].
    """
    from src.figures.single_beam import aligned_colorbar

    D, H, W = ct_hu.shape
    axial_idx = H // 2
    sagittal_idx = W // 2

    def _slices(vol: np.ndarray):
        ax_sl = np.rot90(vol[:, axial_idx, :])
        sg_sl = np.rot90(vol[:, :, sagittal_idx])
        return ax_sl, sg_sl

    # Derived volumes
    seg = segment_tissue(ct_hu)
    rho = hu_to_density(ct_hu)

    hu_ax, hu_sg = _slices(ct_hu)
    seg_ax, seg_sg = _slices(seg)
    rho_ax, rho_sg = _slices(rho)

    # ── Figure layout ───────────────────────────────────────────────────
    fig, ax_dict = plt.subplot_mosaic(
        "AB\nCD\nEF",
        figsize=(16, 10),
        dpi=200,
        gridspec_kw={"hspace": 0.35, "wspace": 0.25},
    )

    # ── Row 1: HU ───────────────────────────────────────────────────────
    ax_dict["A"].imshow(hu_ax, cmap="gray", aspect="auto")
    ax_dict["A"].set_title("HU – centre axial slice", fontsize=11)

    im_hu = ax_dict["B"].imshow(hu_sg, cmap="gray", aspect="auto")
    ax_dict["B"].set_title("HU – centre sagittal slice", fontsize=11)
    aligned_colorbar(fig, im_hu, ax_dict["B"], "HU", label_coords=(6.0, 0.5))

    # ── Row 2: Material assignment ──────────────────────────────────────
    ax_dict["C"].imshow(seg_ax, cmap=_TISSUE_CMAP, norm=_TISSUE_NORM, aspect="auto")
    ax_dict["C"].set_title("Material class – centre axial slice", fontsize=11)

    seg_im = ax_dict["D"].imshow(
        seg_sg, cmap=_TISSUE_CMAP, norm=_TISSUE_NORM, aspect="auto"
    )
    ax_dict["D"].set_title("Material class – centre sagittal slice", fontsize=11)

    # Discrete legend: only show classes actually present
    present = np.unique(seg)
    legend_patches = [
        Patch(facecolor=_TISSUE_COLORS[i], label=_TISSUE_LABELS[i]) for i in present
    ]
    ax_dict["D"].legend(
        handles=legend_patches,
        fontsize=6,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    # ── Row 3: Density map ──────────────────────────────────────────────
    ax_dict["E"].imshow(rho_ax, cmap="inferno", aspect="auto")
    ax_dict["E"].set_title("Density ρ – centre axial slice", fontsize=11)

    im_rho = ax_dict["F"].imshow(rho_sg, cmap="inferno", aspect="auto")
    ax_dict["F"].set_title("Density ρ – centre sagittal slice", fontsize=11)
    aligned_colorbar(fig, im_rho, ax_dict["F"], "ρ [g/cm³]", label_coords=(6.0, 0.5))

    # Remove ticks from all axes
    for key in ax_dict:
        ax_dict[key].set_xticks([])
        ax_dict[key].set_yticks([])

    fig.suptitle(
        f"HU → Material & Density – {sample_id}",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI commands
# ═══════════════════════════════════════════════════════════════════════════


@app.command()
def water_phantom(
    depth: Annotated[int, typer.Option(help="Depth slices")] = 160,
    height: Annotated[int, typer.Option(help="Height voxels")] = 30,
    width: Annotated[int, typer.Option(help="Width voxels")] = 30,
) -> None:
    """Generate a water-box phantom (HU=0) and compute decomposition."""
    logger.info("=== Water-box phantom ===")
    hu = np.zeros((depth, height, width), dtype=np.float64)
    summarise_composition(hu, label="water_phantom")

    # Verify: water should be ~11.2% H, ~88.8% O
    comp = mat_comp(hu)
    h_pct = comp[0, 0, 0, 0]  # H weight-percent for first voxel
    o_pct = comp[0, 0, 0, 3]  # O weight-percent
    logger.info(
        f"Voxel (0,0,0): H={h_pct:.1f}%, O={o_pct:.1f}% "
        f"(expect ~10.8% H, ~50.9% O for soft-tissue at HU≈0)"
    )
    logger.info("Note: HU=0 maps to soft-tissue class [-23, +7], not pure water.")

    # ── Save figure ─────────────────────────────────────────────────────
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir, prefix="hu_to_sp_", subdirs=("figures",))
    fig_path = run_dir / "figures" / "water_phantom_hu_material_density.png"
    plot_hu_material_density(hu, sample_id="water_phantom", output_path=fig_path)

    logger.info(f"Results saved to: {run_dir}")
    logger.info("Done.\n")


@app.command()
def hdf5_samples(
    h5_path: Annotated[Path, typer.Argument(help="Path to HDF5 dataset")],
    n_samples: Annotated[int, typer.Option(help="Number of random samples")] = 2,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Load random samples from training HDF5 and compute decomposition."""
    import torch

    from src.loaders.generator import H5PYGenerator

    logger.info(f"=== HDF5 samples (n={n_samples}) ===")
    logger.info(f"Loading dataset: {h5_path}")

    dataset = H5PYGenerator(str(h5_path), transform=None, normalize=False)
    n_total = len(dataset)
    logger.info(f"Total samples in dataset: {n_total}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    indices.sort()

    scale = DEFAULT_SCALE

    # ── Run directory ───────────────────────────────────────────────────
    runs_dir = PROJECT_ROOT / "runs"
    run_dir = setup_run_directory(runs_dir, prefix="hu_to_sp_", subdirs=("figures",))
    figures_dir = run_dir / "figures"

    for idx in indices:
        x, energy, y = dataset[int(idx)]

        # x shape: (C, D, H, W) — channel 0 is CT (pre-normalised in H5)
        ct_norm = x[0].cpu().numpy()  # (D, H, W)
        ct_hu = inverse_minmax(ct_norm, scale["min_ct"], scale["max_ct"])

        energy_mev = denormalize_energy(energy.item(), scale)

        label = f"sample_{idx}_E{energy_mev:.0f}MeV"
        logger.info(f"\n--- Sample index {idx} (E={energy_mev:.1f} MeV) ---")
        summarise_composition(ct_hu, label=label)

        # ── Figure ──────────────────────────────────────────────────────
        fig_path = figures_dir / f"{label}_hu_material_density.png"
        plot_hu_material_density(ct_hu, sample_id=label, output_path=fig_path)

    logger.info(f"\nResults saved to: {run_dir}")
    logger.info("Done.\n")


if __name__ == "__main__":
    app()
