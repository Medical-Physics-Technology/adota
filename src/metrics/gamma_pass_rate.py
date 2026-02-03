import numpy as np
import torch
from pymedphys import gamma

from src.utils.unit_conversions import to_gy


def gamma_index_torch(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    scale: dict,
    gamma_params: dict,
    resolution: tuple = (1.0, 1.0, 1.0),
    cutoff: float = 0,
) -> tuple:
    """Calculate Gamma Pass Rate based on the torch tensor directly. Scale is required, due to the fact that
    orginally, tensors are in the normalized form.

    Args:
        ground_truth (torch.tensor): Torch Tensor representing Ground Truth.
        prediction (torch.tensor): Torch Tensor representing prediction.
        scale (dict): Dictionary with scale, extracted from Dataset object.
        gamma_params (dict): Parameters of the gamma index function.
        resolution (tuple, optional): Resolution of the grid. Defaults to (1., 1., 1.).
        cutoff (float, optional): Cutoff for GT. Defaults to 0.

    Returns:
        tuple: Tuple representing Gamma values map and Gamma Pass Rate.
    """
    assert (
        ground_truth.shape == prediction.shape
    ), "Shape mismatch between ground truth and prediction"
    assert (
        len(ground_truth.shape) == 5
    ), "For torch version, the shape should be 5D in the format (N, C, D, H, W). To work directly on 3D arrays, please use metrics.gamma_index() function."

    if ground_truth.shape[0] > 1:
        raise ValueError(
            "Only batch size of 1 is supported for torch tensors. Bigger batches in the future."
        )

    # Squeeze the batch dimension:
    ground_truth = ground_truth.squeeze(0)
    prediction = prediction.squeeze(0)

    # Squeeze to 3D arrays, but first validate whether there is only 1 channel on shape0:
    if ground_truth.shape[0] > 1:
        raise ValueError("Only images with one channel are supported.")

    ground_truth = ground_truth.squeeze(0)
    prediction = prediction.squeeze(0)

    grand_truth_np_arr = ground_truth.detach().cpu().numpy()
    prediction_np_arr = prediction.detach().cpu().numpy()

    # Rescaling
    grand_truth_np_arr = (
        grand_truth_np_arr * (scale["y_max"] - scale["y_min"]) + scale["y_min"]
    )
    prediction_np_arr = (
        prediction_np_arr * (scale["y_max"] - scale["y_min"]) + scale["y_min"]
    )

    gamma_results = gamma_index(
        ground_truth=grand_truth_np_arr,
        prediction=prediction_np_arr,
        scale=scale,
        gamma_params=gamma_params,
        resolution=resolution,
        cutoff=cutoff,
    )

    return gamma_results


def gamma_index(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    scale: dict,
    gamma_params: dict,
    resolution: tuple = (1.0, 1.0, 1.0),
    cutoff: float = 0,
) -> tuple:
    """Calculate gamma index between ground truth and predicted dose distributions. Dose distributions are expected 1D, 2D or 3D arrays.

    Example of gamma params dictionary:
    gamma_params = {
        'dose_percent_threshold': 1,
        'distance_mm_threshold': 1,
        'interp_fraction': 10,  # 10 or more for more accurate results
        'max_gamma': 2,
        'lower_percent_dose_cutoff': 2,
        'random_subset': None,
        'local_gamma': False,
        'quiet': True
    }

    Args:
        ground_truth (np.ndarray): 3D array representing the ground truth dose distribution.
        prediction (np.ndarray): 3D array representing the predicted dose distribution.
        scale (dict): Dictionary containing the scalling factors, must containt 'y_max' and 'y_min' keys.
        gamma_params (dict): Dictionary with gamma parameters.
        resolution (tuple, optional): Resolution of the dose arrays. Defaults to (1., 1., 1.).
        cutoff (float, optional): Value from range [0, 100], represents the percantage value of dose in ground truth since which the cutoff is applied. Defaults to 0.

    Returns:
        tuple: Tuple containing the gamma values and gamma pass rate.
    """

    assert (
        ground_truth.shape == prediction.shape
    ), "Shape mismatch between ground truth and prediction"
    if len(ground_truth.shape) == 4:
        ground_truth = np.squeeze(ground_truth, axis=0)
        prediction = np.squeeze(prediction, axis=0)

    axes = (
        np.arange(ground_truth.shape[0]) * resolution[0],
        np.arange(ground_truth.shape[1]) * resolution[1],
        np.arange(ground_truth.shape[2]) * resolution[2],
    )

    # Cut off MC noise
    ground_truth[ground_truth < (cutoff / 100) * scale["y_max"]] = 0
    prediction[prediction < (cutoff / 100) * scale["y_max"]] = 0

    gamma_values = gamma(axes, ground_truth, axes, prediction, **gamma_params)
    gamma_values = np.nan_to_num(gamma_values, 0)

    gamma_pass_rate = np.zeros(2)
    gamma_pass_rate[0] = 1 - (
        np.count_nonzero(gamma_values > 1) / np.count_nonzero(gamma_values > 0)
    )
    gamma_pass_rate[1] = np.sum(gamma_values <= 1) / np.prod(gamma_values.shape)

    return gamma_values, gamma_pass_rate


def relative_dose_error(
    pred: torch.Tensor, target: torch.Tensor, tr: float = 0.1
) -> torch.Tensor:
    """Calculate the relative dose error between target and prediction tensors.
    Relative dose error is defined by the equation:

    $$
    rde = \frac{1}{N} \cdot \frac{||\hat{D} - D||_1}{D_{max}} \cdot 100
    $$

    Args:
        target (torch.Tensor): Target tensor.
        pred (torch.Tensor): Prediction tensor.
        tr (float, optional): Threshold value. Defaults to 0.1. Threshold controls the min value of the target tensor, to prevent division by zero.

    Returns:
        torch.Tensor: Relative dose error tensor.
    """
    batch_size = target.shape[0]
    relative_dose_error = torch.zeros(batch_size)
    for i in range(batch_size):
        voxel_factor = np.prod(target[i].shape)
        target_flatten = target[i].flatten()
        target_flatten_filtered = target_flatten[
            target_flatten > tr * torch.max(target_flatten)
        ]
        pred_flatten = pred[i].flatten()
        pred_flatten_filtered = pred_flatten[
            target_flatten > tr * torch.max(target_flatten)
        ]
        l1_norm_diff = torch.linalg.norm(
            target_flatten_filtered - pred_flatten_filtered, ord=1
        )
        max_target = torch.max(target_flatten)
        relative_dose_error[i] = l1_norm_diff / (max_target * voxel_factor)
        relative_dose_error[i] *= 100

    return torch.mean(relative_dose_error)
