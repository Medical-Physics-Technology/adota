"""Inverse of the min-max normalisation applied to the model's inputs and dose.

The training pipeline stores CT, flux and dose min-max normalised to [0, 1];
every evaluation path has to undo that before computing physical metrics.
"""

import numpy as np


def inverse_minmax(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return (x * (x_max - x_min)) + x_min
