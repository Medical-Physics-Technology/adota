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

