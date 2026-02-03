import numpy as np


def calculate_rmse(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Calculate the Root Mean Square Error (RMSE) between two arrays."""
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference arrays must have the same shape.")
    return np.sqrt(np.mean((predicted - reference) ** 2))


# Pure mean absolute percentage error
def calculate_pure_mape(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Calculate the Mean Absolute Percentage Error (MAPE) between two arrays."""
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference arrays must have the same shape.")
    # Calculate MAPE only for non-zero reference values
    non_zero_mask = reference != 0
    abs_diff = np.abs(predicted[non_zero_mask] - reference[non_zero_mask])
    mape = np.mean(abs_diff / np.abs(reference[non_zero_mask])) * 100
    return mape
    # return np.mean(np.abs((predicted - reference)) / (reference + 1)) * 100


def calculate_relative_dose_error(
    predicted: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Calculate the Relative Dose Error (RDE) between two arrays."""
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference arrays must have the same shape.")
    number_of_voxels = np.prod(reference.shape)
    coef = 1 / number_of_voxels
    return (
        coef
        * np.linalg.norm(predicted.flatten() - reference.flatten(), ord=1)
        / np.max(reference.flatten())
        * 100
    )
