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
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = typer.Typer(help="HU → material composition & stopping power")

# ═══════════════════════════════════════════════════════════════════════════
#  Constants – element catalogue
# ═══════════════════════════════════════════════════════════════════════════

ELEMENT_NAMES = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "Ar", "K", "Ca"]
N_ELEMENTS = len(ELEMENT_NAMES)

# Elemental densities [g/cm³]
ELEMENT_DENSITIES = np.array(
    [
        0.00008988,
        2.267,
        0.00125,
        0.00143,
        0.97,
        1.74,
        1.82,
        2.067,
        0.003,
        0.0017837,
        0.89,
        1.54,
    ]
)

# Atomic numbers
Z_ARRAY = np.array([1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20])

# Atomic weights [g/mol]
MOL_WEIGHTS = np.array(
    [
        1.008,
        12.011,
        14.007,
        15.999,
        22.989,
        24.305,
        30.973,
        32.060,
        35.450,
        39.948,
        39.098,
        40.078,
    ]
)

# ═══════════════════════════════════════════════════════════════════════════
#  Schneider tissue decomposition look-up table
# ═══════════════════════════════════════════════════════════════════════════
#
# Each entry: (HU_low, HU_high, [H, C, N, O, Na, Mg, P, S, Cl, Ar, K, Ca])
# Values are weight-percent.

TISSUE_LUT: list[tuple[float, float, list[float]]] = [
    (
        -1000,
        -950,
        [0.0, 0.0, 75.5, 23.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3, 0.0, 0.0],
    ),  # air
    (
        -950,
        -120,
        [10.3, 10.5, 3.1, 74.9, 0.2, 0.0, 0.2, 0.3, 0.3, 0.0, 0.2, 0.0],
    ),  # lung
    (
        -120,
        -83,
        [11.6, 68.1, 0.2, 19.8, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
    ),  # adipose 1
    (
        -83,
        -53,
        [11.3, 56.7, 0.9, 30.8, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
    ),  # adipose 2
    (
        -53,
        -23,
        [11.0, 45.8, 1.5, 41.1, 0.1, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0],
    ),  # adipose 3
    (
        -23,
        7,
        [10.8, 35.6, 2.2, 50.9, 0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0],
    ),  # soft tissue 1
    (
        7,
        18,
        [10.6, 28.4, 2.6, 57.8, 0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.1, 0.0],
    ),  # soft tissue 2
    (
        18,
        80,
        [10.3, 13.4, 3.0, 72.3, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0],
    ),  # soft tissue 3
    (80, 120, [9.4, 20.7, 6.2, 62.2, 0.6, 0.0, 0.0, 0.6, 0.3, 0.0, 0.0, 0.0]),  # muscle
    (
        120,
        200,
        [9.5, 45.5, 2.5, 35.5, 0.1, 0.0, 2.1, 0.1, 0.1, 0.0, 0.1, 4.5],
    ),  # spongy bone 1
    (
        200,
        300,
        [8.9, 42.3, 2.7, 36.3, 0.1, 0.0, 3.0, 0.1, 0.1, 0.0, 0.1, 6.4],
    ),  # spongy bone 2
    (
        300,
        400,
        [8.2, 39.1, 2.9, 37.2, 0.1, 0.0, 3.9, 0.1, 0.1, 0.0, 0.1, 8.3],
    ),  # spongy bone 3
    (
        400,
        500,
        [7.6, 36.1, 3.0, 38.0, 0.1, 0.1, 4.7, 0.2, 0.1, 0.0, 0.0, 10.1],
    ),  # cortical 1
    (
        500,
        600,
        [7.1, 33.5, 3.2, 38.7, 0.1, 0.1, 5.4, 0.2, 0.0, 0.0, 0.0, 11.7],
    ),  # cortical 2
    (
        600,
        700,
        [6.6, 31.0, 3.3, 39.4, 0.1, 0.1, 6.1, 0.2, 0.0, 0.0, 0.0, 13.2],
    ),  # cortical 3
    (
        700,
        800,
        [6.1, 28.7, 3.5, 40.0, 0.1, 0.1, 6.7, 0.2, 0.0, 0.0, 0.0, 14.6],
    ),  # cortical 4
    (
        800,
        900,
        [5.6, 26.5, 3.6, 40.5, 0.1, 0.2, 7.3, 0.3, 0.0, 0.0, 0.0, 15.9],
    ),  # cortical 5
    (
        900,
        1000,
        [5.2, 24.6, 3.7, 41.1, 0.1, 0.2, 7.8, 0.3, 0.0, 0.0, 0.0, 17.0],
    ),  # cortical 6
    (
        1000,
        1100,
        [4.9, 22.7, 3.8, 41.6, 0.1, 0.2, 8.3, 0.3, 0.0, 0.0, 0.0, 18.1],
    ),  # cortical 7
    (
        1100,
        1200,
        [4.5, 21.0, 3.9, 42.0, 0.1, 0.2, 8.8, 0.3, 0.0, 0.0, 0.0, 19.2],
    ),  # cortical 8
    (
        1200,
        1300,
        [4.2, 19.4, 4.0, 42.5, 0.1, 0.2, 9.2, 0.3, 0.0, 0.0, 0.0, 20.1],
    ),  # cortical 9
    (
        1300,
        1400,
        [3.9, 17.9, 4.1, 42.9, 0.1, 0.2, 9.6, 0.3, 0.0, 0.0, 0.0, 21.0],
    ),  # cortical 10
    (
        1400,
        1500,
        [3.6, 16.5, 4.2, 43.2, 0.1, 0.2, 10.0, 0.3, 0.0, 0.0, 0.0, 21.9],
    ),  # cortical 11
    (
        1500,
        1600,
        [3.4, 15.5, 4.2, 43.5, 0.1, 0.2, 10.3, 0.3, 0.0, 0.0, 0.0, 22.5],
    ),  # cortical 12
]

# Pre-build numpy arrays for vectorised look-up
_LUT_BOUNDS = np.array([(lo, hi) for lo, hi, _ in TISSUE_LUT])  # (N, 2)
_LUT_COMP = np.array([c for _, _, c in TISSUE_LUT])  # (N, 12)


# ═══════════════════════════════════════════════════════════════════════════
#  Core functions
# ═══════════════════════════════════════════════════════════════════════════


def hu_to_density(hu: np.ndarray) -> np.ndarray:
    """Convert HU to mass density [g/cm³] using piecewise-linear model.

    Segment 1 (HU ≤ 0):  ρ = 1.0 + HU × 0.001  (air → water)
    Segment 2 (HU > 0):  ρ = 1.0 + HU × 0.0005  (water → bone)
    """
    rho = np.where(hu <= 0, 1.0 + hu * 0.001, 1.0 + hu * 0.0005)
    return np.clip(rho, 0.001, 3.0)


def mat_comp(hu: np.ndarray) -> np.ndarray:
    """Compute material composition for each voxel from HU values.

    Implements the Schneider piecewise-constant tissue decomposition
    (same look-up table as ``matComp`` in the TITUS Julia code).

    Args:
        hu: Array of HU values, any shape.

    Returns:
        Array of shape ``(*hu.shape, 12)`` with weight-percent
        composition for each of the 12 elements.  Voxels outside
        the valid range [-1000, +1600] are assigned the nearest
        boundary composition.
    """
    original_shape = hu.shape
    hu_flat = hu.ravel().astype(np.float64)
    n = len(hu_flat)
    comp = np.zeros((n, N_ELEMENTS), dtype=np.float64)

    # Clamp to valid range
    hu_clamped = np.clip(hu_flat, -1000.0, 1600.0)

    # Vectorised bin assignment
    for idx, (lo, hi, weights) in enumerate(TISSUE_LUT):
        if idx == 0:
            mask = (hu_clamped >= lo) & (hu_clamped <= hi)
        else:
            mask = (hu_clamped > lo) & (hu_clamped <= hi)
        comp[mask] = weights

    return comp.reshape(*original_shape, N_ELEMENTS)


def hu_to_rsp_schneider(hu: np.ndarray) -> np.ndarray:
    """Convert HU → relative stopping power via Schneider calibration.

    Uses the simple two-segment piecewise-linear model:
      HU ≤ 0:  RSP = 1.0 + HU × 0.001029
      HU > 0:  RSP = 1.0 + HU × 0.000487

    Args:
        hu: Array of HU values, any shape.

    Returns:
        RSP array of the same shape.
    """
    rsp = np.where(hu <= 0, 1.0 + hu * 0.001029, 1.0 + hu * 0.000487)
    return np.clip(rsp, 0.001, 2.5)


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
#  Tissue-class segmentation (from TISSUE_LUT) & figure generation
# ═══════════════════════════════════════════════════════════════════════════

# Short labels for each LUT entry (order matches TISSUE_LUT)
_TISSUE_LABELS = [
    "Air",
    "Lung",
    "Adipose 1",
    "Adipose 2",
    "Adipose 3",
    "Soft tissue 1",
    "Soft tissue 2",
    "Soft tissue 3",
    "Muscle",
    "Spongy bone 1",
    "Spongy bone 2",
    "Spongy bone 3",
    "Cortical 1",
    "Cortical 2",
    "Cortical 3",
    "Cortical 4",
    "Cortical 5",
    "Cortical 6",
    "Cortical 7",
    "Cortical 8",
    "Cortical 9",
    "Cortical 10",
    "Cortical 11",
    "Cortical 12",
]

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

N_TISSUE_CLASSES = len(TISSUE_LUT)
_TISSUE_CMAP = ListedColormap(_TISSUE_COLORS)
_TISSUE_NORM = BoundaryNorm(
    boundaries=list(range(N_TISSUE_CLASSES + 1)), ncolors=N_TISSUE_CLASSES
)


def segment_tissue(hu: np.ndarray) -> np.ndarray:
    """Assign each voxel a tissue-class index (0..23) from TISSUE_LUT."""
    seg = np.zeros(hu.shape, dtype=np.int8)
    hu_c = np.clip(hu, -1000.0, 1600.0)
    for idx, (lo, hi, _) in enumerate(TISSUE_LUT):
        if idx == 0:
            mask = (hu_c >= lo) & (hu_c <= hi)
        else:
            mask = (hu_c > lo) & (hu_c <= hi)
        seg[mask] = idx
    return seg


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
