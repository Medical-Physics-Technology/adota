"""
Schneider tissue decomposition: HU → density / RSP / material class.

Pure-numpy routines extracted from ``scripts/analysis/hu_to_sp.py`` so
they can be imported without pulling in torch, matplotlib, or typer.
The script module re-exports these names for backwards compatibility.

References
----------
- Schneider W, Bortfeld T, Schlegel W (2000). "Correlation between CT
  numbers and tissue parameters needed for Monte Carlo simulations of
  clinical dose distributions." *Phys Med Biol* 45(2):459-478.
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  Element catalogue
# ═══════════════════════════════════════════════════════════════════════════

ELEMENT_NAMES = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "Ar", "K", "Ca"]
N_ELEMENTS = len(ELEMENT_NAMES)

ELEMENT_DENSITIES = np.array(
    [
        0.00008988, 2.267, 0.00125, 0.00143, 0.97, 1.74,
        1.82, 2.067, 0.003, 0.0017837, 0.89, 1.54,
    ]
)
Z_ARRAY = np.array([1, 6, 7, 8, 11, 12, 15, 16, 17, 18, 19, 20])
MOL_WEIGHTS = np.array(
    [
        1.008, 12.011, 14.007, 15.999, 22.989, 24.305,
        30.973, 32.060, 35.450, 39.948, 39.098, 40.078,
    ]
)

# ═══════════════════════════════════════════════════════════════════════════
#  Schneider tissue decomposition LUT
# ═══════════════════════════════════════════════════════════════════════════
# Each entry: (HU_low, HU_high, [H, C, N, O, Na, Mg, P, S, Cl, Ar, K, Ca])
# Values are weight-percent.

TISSUE_LUT: list[tuple[float, float, list[float]]] = [
    (-1000, -950, [0.0, 0.0, 75.5, 23.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3, 0.0, 0.0]),  # air
    (-950, -120, [10.3, 10.5, 3.1, 74.9, 0.2, 0.0, 0.2, 0.3, 0.3, 0.0, 0.2, 0.0]),  # lung
    (-120, -83, [11.6, 68.1, 0.2, 19.8, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0]),  # adipose 1
    (-83, -53, [11.3, 56.7, 0.9, 30.8, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0]),  # adipose 2
    (-53, -23, [11.0, 45.8, 1.5, 41.1, 0.1, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0]),  # adipose 3
    (-23, 7, [10.8, 35.6, 2.2, 50.9, 0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0]),  # soft tissue 1
    (7, 18, [10.6, 28.4, 2.6, 57.8, 0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.1, 0.0]),  # soft tissue 2
    (18, 80, [10.3, 13.4, 3.0, 72.3, 0.2, 0.0, 0.2, 0.2, 0.2, 0.0, 0.2, 0.0]),  # soft tissue 3
    (80, 120, [9.4, 20.7, 6.2, 62.2, 0.6, 0.0, 0.0, 0.6, 0.3, 0.0, 0.0, 0.0]),  # muscle
    (120, 200, [9.5, 45.5, 2.5, 35.5, 0.1, 0.0, 2.1, 0.1, 0.1, 0.0, 0.1, 4.5]),  # spongy 1
    (200, 300, [8.9, 42.3, 2.7, 36.3, 0.1, 0.0, 3.0, 0.1, 0.1, 0.0, 0.1, 6.4]),  # spongy 2
    (300, 400, [8.2, 39.1, 2.9, 37.2, 0.1, 0.0, 3.9, 0.1, 0.1, 0.0, 0.1, 8.3]),  # spongy 3
    (400, 500, [7.6, 36.1, 3.0, 38.0, 0.1, 0.1, 4.7, 0.2, 0.1, 0.0, 0.0, 10.1]),  # cortical 1
    (500, 600, [7.1, 33.5, 3.2, 38.7, 0.1, 0.1, 5.4, 0.2, 0.0, 0.0, 0.0, 11.7]),  # cortical 2
    (600, 700, [6.6, 31.0, 3.3, 39.4, 0.1, 0.1, 6.1, 0.2, 0.0, 0.0, 0.0, 13.2]),  # cortical 3
    (700, 800, [6.1, 28.7, 3.5, 40.0, 0.1, 0.1, 6.7, 0.2, 0.0, 0.0, 0.0, 14.6]),  # cortical 4
    (800, 900, [5.6, 26.5, 3.6, 40.5, 0.1, 0.2, 7.3, 0.3, 0.0, 0.0, 0.0, 15.9]),  # cortical 5
    (900, 1000, [5.2, 24.6, 3.7, 41.1, 0.1, 0.2, 7.8, 0.3, 0.0, 0.0, 0.0, 17.0]),  # cortical 6
    (1000, 1100, [4.9, 22.7, 3.8, 41.6, 0.1, 0.2, 8.3, 0.3, 0.0, 0.0, 0.0, 18.1]),  # cortical 7
    (1100, 1200, [4.5, 21.0, 3.9, 42.0, 0.1, 0.2, 8.8, 0.3, 0.0, 0.0, 0.0, 19.2]),  # cortical 8
    (1200, 1300, [4.2, 19.4, 4.0, 42.5, 0.1, 0.2, 9.2, 0.3, 0.0, 0.0, 0.0, 20.1]),  # cortical 9
    (1300, 1400, [3.9, 17.9, 4.1, 42.9, 0.1, 0.2, 9.6, 0.3, 0.0, 0.0, 0.0, 21.0]),  # cortical 10
    (1400, 1500, [3.6, 16.5, 4.2, 43.2, 0.1, 0.2, 10.0, 0.3, 0.0, 0.0, 0.0, 21.9]),  # cortical 11
    (1500, 1600, [3.4, 15.5, 4.2, 43.5, 0.1, 0.2, 10.3, 0.3, 0.0, 0.0, 0.0, 22.5]),  # cortical 12
]

N_TISSUE_CLASSES = len(TISSUE_LUT)

TISSUE_LABELS = [
    "Air", "Lung",
    "Adipose 1", "Adipose 2", "Adipose 3",
    "Soft tissue 1", "Soft tissue 2", "Soft tissue 3",
    "Muscle",
    "Spongy bone 1", "Spongy bone 2", "Spongy bone 3",
    "Cortical 1", "Cortical 2", "Cortical 3", "Cortical 4",
    "Cortical 5", "Cortical 6", "Cortical 7", "Cortical 8",
    "Cortical 9", "Cortical 10", "Cortical 11", "Cortical 12",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Core conversion functions
# ═══════════════════════════════════════════════════════════════════════════


def hu_to_density(hu: np.ndarray) -> np.ndarray:
    """Convert HU to mass density [g/cm³] using piecewise-linear model.

    Segment 1 (HU ≤ 0):  ρ = 1.0 + HU × 0.001  (air → water)
    Segment 2 (HU > 0):  ρ = 1.0 + HU × 0.0005  (water → bone)
    """
    rho = np.where(hu <= 0, 1.0 + hu * 0.001, 1.0 + hu * 0.0005)
    return np.clip(rho, 0.001, 3.0)


def hu_to_rsp_schneider(hu: np.ndarray) -> np.ndarray:
    """Convert HU → relative stopping power via Schneider calibration.

    HU ≤ 0:  RSP = 1.0 + HU × 0.001029
    HU > 0:  RSP = 1.0 + HU × 0.000487
    """
    rsp = np.where(hu <= 0, 1.0 + hu * 0.001029, 1.0 + hu * 0.000487)
    return np.clip(rsp, 0.001, 2.5)


def mat_comp(hu: np.ndarray) -> np.ndarray:
    """Per-voxel weight-percent composition over the 12-element catalogue.

    Returns an array of shape ``(*hu.shape, N_ELEMENTS)`` assigning each
    voxel the composition of the TISSUE_LUT bin its HU falls into.
    Voxels outside ``[-1000, +1600]`` are clamped to the nearest
    boundary composition.
    """
    original_shape = hu.shape
    hu_flat = hu.ravel().astype(np.float64)
    n = len(hu_flat)
    comp = np.zeros((n, N_ELEMENTS), dtype=np.float64)

    hu_clamped = np.clip(hu_flat, -1000.0, 1600.0)

    for idx, (lo, hi, weights) in enumerate(TISSUE_LUT):
        if idx == 0:
            mask = (hu_clamped >= lo) & (hu_clamped <= hi)
        else:
            mask = (hu_clamped > lo) & (hu_clamped <= hi)
        comp[mask] = weights

    return comp.reshape(*original_shape, N_ELEMENTS)


def segment_tissue(hu: np.ndarray) -> np.ndarray:
    """Assign each voxel a tissue-class index (0..N_TISSUE_CLASSES-1).

    Bin membership mirrors :func:`mat_comp`: half-open intervals
    ``(lo, hi]`` for bins ≥1, closed ``[lo, hi]`` for the first (air)
    bin.  HU values outside ``[-1000, +1600]`` are clamped first.
    """
    seg = np.zeros(hu.shape, dtype=np.int8)
    hu_c = np.clip(hu, -1000.0, 1600.0)
    for idx, (lo, hi, _) in enumerate(TISSUE_LUT):
        if idx == 0:
            mask = (hu_c >= lo) & (hu_c <= hi)
        else:
            mask = (hu_c > lo) & (hu_c <= hi)
        seg[mask] = idx
    return seg
