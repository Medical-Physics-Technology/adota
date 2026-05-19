import numpy as np
from typing import Tuple


def estimate_bragg_peak(dose_grid: np.ndarray) -> Tuple[int, int, int]:
    # Return index of the bragg peak.
    # Return in the format (y, x, z)
    depth_dose_profile = dose_grid.sum(axis=(1, 2))
    depth_max_dose = np.argmax(depth_dose_profile)
    # Get the lateral profile at the depth.
    bp_slice = dose_grid[depth_max_dose, :, :]
    max_dose_index = np.unravel_index(np.argmax(bp_slice), bp_slice.shape)
    return (depth_max_dose, max_dose_index[0], max_dose_index[1])  # z, y, x


def estimate_bp_range(
    ct_grid: np.ndarray,
    dose_grid: np.ndarray,
    proximal_fraction: float = 0.50,
    fall_fraction: float = 0.10,
) -> Tuple[float, float]:
    """Estimate the depth range of the Bragg peak from a dose grid.

    The algorithm works on the Integrated Depth Dose (IDD) -- the dose
    summed over the lateral dimensions at each depth slice::

        IDD[k] = dose_grid[k, :, :].sum()

    **Proximal boundary (z_min)** -- walk backward from the Bragg-peak
    maximum towards the entrance and find the first depth slice where
    the IDD drops below ``proximal_fraction * IDD_max``.  This mirrors
    the clinical definition of the proximal range (e.g. proximal R80
    or R50) and is inherently robust against plateau noise because it
    is anchored to the peak rather than the entrance.

    **Distal boundary (z_max)** -- starting from the Bragg-peak
    maximum, walk towards deeper depths until the IDD drops below
    ``fall_fraction * IDD_max``.  This marks the distal fall-off.

    Both boundaries are returned as **depth-slice indices** (float).
    Multiply by the voxel spacing to convert to physical units.

    Args:
        ct_grid: 3-D CT volume ``(D, H, W)`` in HU (reserved for
            future density-aware refinements; currently unused).
        dose_grid: 3-D ground-truth dose grid ``(D, H, W)``.
        proximal_fraction: Fraction of peak IDD used as the proximal
            threshold when walking backward from the BP (default 50 %).
        fall_fraction: Fraction of peak IDD used as the distal
            threshold (default 10 %).

    Returns:
        ``(z_min, z_max)`` -- proximal and distal depth-slice indices
        bounding the Bragg peak.  If the dose is effectively zero
        everywhere, ``(0.0, 0.0)`` is returned.
    """
    # -- Integrated Depth Dose ------------------------------------------------
    idd = dose_grid.sum(axis=(1, 2))  # (D,)

    idd_max = idd.max()
    if idd_max < 1e-9:
        return 0.0, 0.0

    bp_idx = int(np.argmax(idd))

    # -- Proximal boundary: walk backward from BP -----------------------------
    proximal_threshold = proximal_fraction * idd_max
    z_min = 0.0  # fallback: entrance
    for k in range(bp_idx, -1, -1):
        if idd[k] < proximal_threshold:
            z_min = float(k)
            break

    # -- Distal boundary: walk from BP towards exit ---------------------------
    n_slices = len(idd)
    fall_threshold = fall_fraction * idd_max
    z_max = float(n_slices - 1)
    for k in range(bp_idx, n_slices):
        if idd[k] < fall_threshold:
            z_max = float(k)
            break

    return z_min, z_max

