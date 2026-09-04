"""Gamma pass rate (GPR) between a predicted and a reference dose distribution.

Flow:
1. ``gamma_index_torch`` takes normalised (N, C, D, H, W) tensors straight from
   the model, de-normalises them with the training ``scale`` dict, and moves
   them to numpy.
2. ``gamma_index`` wraps ``pymedphys.gamma`` for plain 1D/2D/3D arrays.
3. Both return the percentage of voxels passing the criterion, after applying
   the low-dose ``cutoff``.

``relative_dose_error`` is here for convenience because it shares the same
threshold/masking conventions.

Note: ``pymedphys.gamma`` interpolates with an in-house numba kernel from
0.41 onwards (it used the econforge ``interpolation`` package up to 0.40).
``numba`` ships as an optional pymedphys extra, so adota depends on it
explicitly; without it the call fails.
"""

import numpy as np
import torch
from pymedphys import gamma


def gamma_index_torch(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    scale: dict,
    gamma_params: dict,
    resolution: tuple = (1.0, 1.0, 1.0),
    cutoff: float = 0,
    backend: str = "pymedphys",
    backend_options: dict = None,
) -> tuple:
    """Calculate Gamma Pass Rate based on the torch tensor directly. Scale is required, due to the fact that
    orginally, tensors are in the normalized form.

    With ``backend="torch"`` the tensors are de-normalised and thresholded on
    whatever device they already sit on and handed straight to the kernel, so the
    two full-volume host transfers the pymedphys path needs never happen. Only
    the resulting gamma array comes back, because the pass-rate arithmetic is
    shared verbatim with the numpy path.

    Args:
        ground_truth (torch.tensor): Torch Tensor representing Ground Truth.
        prediction (torch.tensor): Torch Tensor representing prediction.
        scale (dict): Dictionary with scale, extracted from Dataset object.
        gamma_params (dict): Parameters of the gamma index function.
        resolution (tuple, optional): Resolution of the grid. Defaults to (1., 1., 1.).
        cutoff (float, optional): Cutoff for GT. Defaults to 0.
        backend (str, optional): ``"pymedphys"`` (default) or ``"torch"``.
        backend_options (dict, optional): Forwarded to the torch backend; see
            :func:`gamma_index`.

    Returns:
        tuple: Tuple representing Gamma values map and Gamma Pass Rate.
    """
    assert (
        ground_truth.shape == prediction.shape
    ), "Shape mismatch between ground truth and prediction"
    assert (
        len(ground_truth.shape) == 5
    ), (
        "For torch version, the shape should be 5D in the format (N, C, D, H, W). "
        "To work directly on 3D arrays, please use metrics.gamma_index() function."
    )

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

    if backend == "torch":
        return _gamma_index_on_device(
            ground_truth, prediction, scale, gamma_params, resolution, cutoff,
            backend_options,
        )

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


def _gamma_index_on_device(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    scale: dict,
    gamma_params: dict,
    resolution: tuple,
    cutoff: float,
    backend_options: dict = None,
) -> tuple:
    """The torch-backend path of :func:`gamma_index_torch`, without a host copy.

    Mirrors :func:`gamma_index` step for step -- de-normalise, zero the
    sub-cutoff voxels, build the axes, compute gamma, take the pass rate -- but
    does the first two on the tensors' own device.

    Args:
        ground_truth: Normalised reference dose, 3D.
        prediction: Normalised evaluation dose, same shape.
        scale: The training scale dict (``y_min`` / ``y_max``).
        gamma_params: The gamma parameters.
        resolution: Voxel spacing.
        cutoff: Low-dose cutoff, as a percentage of ``y_max``.
        backend_options: Torch-backend extras; see :func:`gamma_index`.

    Returns:
        tuple: ``(gamma_values, gamma_pass_rate)``.
    """
    span = scale["y_max"] - scale["y_min"]
    ground_truth = ground_truth * span + scale["y_min"]
    prediction = prediction * span + scale["y_min"]

    # Cut off MC noise (the numpy path's in-place masking, on device).
    threshold = (cutoff / 100) * scale["y_max"]
    ground_truth = ground_truth.masked_fill(ground_truth < threshold, 0)
    prediction = prediction.masked_fill(prediction < threshold, 0)

    axes = tuple(
        np.arange(size) * resolution[axis]
        for axis, size in enumerate(ground_truth.shape)
    )
    gamma_values = _gamma_values(
        axes, ground_truth, prediction, gamma_params, "torch", backend_options
    )
    return _gamma_pass_rate(gamma_values)


def gamma_index(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    scale: dict,
    gamma_params: dict,
    resolution: tuple = (1.0, 1.0, 1.0),
    cutoff: float = 0,
    backend: str = "pymedphys",
    backend_options: dict = None,
) -> tuple:
    """Calculate gamma index between ground truth and predicted dose distributions.
    Dose distributions are expected 1D, 2D or 3D arrays.

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
        cutoff (float, optional): Value from range [0, 100], represents the
            percantage value of dose in ground truth since which the cutoff is
            applied. Defaults to 0.
        backend (str, optional): ``"pymedphys"`` (default, unchanged) or
            ``"torch"`` for :mod:`src.metrics.gamma_torch`. Only the computation
            of the gamma array changes; the pass-rate arithmetic below is the
            same either way.
        backend_options (dict, optional): Forwarded to the torch backend --
            ``device`` (or ``device_index`` for :func:`resolve_device`),
            ``dtype``, ``tile_elements``, ``stats``. Ignored by pymedphys.

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

    gamma_values = _gamma_values(
        axes, ground_truth, prediction, gamma_params, backend, backend_options
    )

    return _gamma_pass_rate(gamma_values)


def _gamma_values(
    axes: tuple,
    ground_truth,
    prediction,
    gamma_params: dict,
    backend: str,
    backend_options: dict = None,
):
    """Compute the raw gamma array with the selected backend.

    Args:
        axes: Coordinate axes, shared by both grids.
        ground_truth: The reference dose (numpy array or torch tensor).
        prediction: The evaluation dose, same type and shape.
        gamma_params: The pymedphys gamma parameters.
        backend: ``"pymedphys"`` or ``"torch"``.
        backend_options: Torch-backend extras; see :func:`gamma_index`.

    Returns:
        The gamma array as numpy, NaN where gamma was not evaluated.

    Raises:
        ValueError: If ``backend`` is not a known name.
    """
    if backend == "pymedphys":
        return gamma(axes, ground_truth, axes, prediction, **gamma_params)
    if backend != "torch":
        raise ValueError(
            f"Unknown gamma backend {backend!r}; expected 'pymedphys' or 'torch'."
        )

    # Imported lazily: the torch backend is opt-in, and gamma_torch is meant to
    # stay loadable on its own.
    from src.evaluation.cli import resolve_device
    from src.metrics.gamma_torch import gamma_index_torch_core

    options = dict(backend_options or {})
    device = options.pop("device", None)
    if device is None:
        device = resolve_device(options.pop("device_index", None))
    options.pop("device_index", None)
    dtype = options.pop("dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # `quiet` is a deprecated pymedphys logging flag with no torch counterpart.
    params = {key: value for key, value in gamma_params.items() if key != "quiet"}
    values = gamma_index_torch_core(
        axes, ground_truth, axes, prediction,
        device=device, dtype=dtype, **options, **params,
    )
    return values.detach().cpu().numpy()


def _gamma_pass_rate(gamma_values: np.ndarray) -> tuple:
    """The pass rate pair, extracted verbatim so both backends share it.

    Note the denominator: voxels with gamma exactly 0 -- which after
    ``nan_to_num`` includes every voxel that was not evaluated -- are excluded
    from both the numerator and the denominator of ``gamma_pass_rate[0]``.
    Training runs are compared longitudinally against this exact definition, so
    it is reproduced unchanged rather than corrected.

    Args:
        gamma_values: The raw gamma array, NaN where not evaluated.

    Returns:
        tuple: ``(gamma_values, gamma_pass_rate)`` with the NaNs zeroed.
    """
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
        tr (float, optional): Threshold value. Defaults to 0.1. Threshold controls
            the min value of the target tensor, to prevent division by zero.

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
