import numpy as np


def inverse_minmax(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return (x * (x_max - x_min)) + x_min
