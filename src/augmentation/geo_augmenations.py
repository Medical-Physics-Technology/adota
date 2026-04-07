import numpy as np
from typing import Tuple
import random


def estimate_bragg_peak(dose_grid: np.ndarray) -> Tuple[int, int, int]:
    # Return index of the bragg peak.
    # Return in the format (y, x, z)
    depth_dose_profile = dose_grid.sum(axis=(1, 2))
    depth_max_dose = np.argmax(depth_dose_profile)
    # Get the lateral profile at the depth.
    bp_slice = dose_grid[depth_max_dose, :, :]
    max_dose_index = np.unravel_index(np.argmax(bp_slice), bp_slice.shape)
    return (depth_max_dose, max_dose_index[0], max_dose_index[1])  # z, y, x


def cropp_around_index(
    ct_grid: np.ndarray,
    flux_grid: np.ndarray,
    dose_grid: np.ndarray,
    initial_energy: float,
    index_to_cropp: Tuple[int, int, int] = None,
    training_roi: Tuple[int, int, int] = (30, 30, 160),
) -> Tuple:
    if index_to_cropp is None:
        index_to_cropp = estimate_bragg_peak(dose_grid)

    start_idx_0 = index_to_cropp[0] - training_roi[0] // 2
    end_idx_0 = index_to_cropp[0] + training_roi[0] // 2
    start_idx_1 = index_to_cropp[1] - training_roi[1] // 2
    end_idx_1 = index_to_cropp[1] + training_roi[1] // 2

    if start_idx_0 <= 0:
        start_idx_0 = 0
        end_idx_0 = training_roi[0]

    if end_idx_0 > ct_grid.shape[0]:
        start_idx_0 = ct_grid.shape[0] - training_roi[0]
        end_idx_0 = ct_grid.shape[0]

    if start_idx_1 <= 0:
        start_idx_1 = 0
        end_idx_1 = training_roi[1]

    if end_idx_1 > ct_grid.shape[1]:
        start_idx_1 = ct_grid.shape[1] - training_roi[1]
        end_idx_1 = ct_grid.shape[1]

    cropped_dose = dose_grid[
        start_idx_0:end_idx_0,
        start_idx_1:end_idx_1,
        : training_roi[2],
    ]
    cropped_flux = flux_grid[
        start_idx_0:end_idx_0,
        start_idx_1:end_idx_1,
        : training_roi[2],
    ]
    cropped_ct = ct_grid[
        start_idx_0:end_idx_0,
        start_idx_1:end_idx_1,
        : training_roi[2],
    ]
    return cropped_ct, cropped_flux, cropped_dose, initial_energy


def moving_window_augmentation(
    ct_grid: np.ndarray,
    flux_grid: np.ndarray,
    dose_grid: np.ndarray,
    initial_energy: float,
    max_shifts: Tuple[int, int] = (3, 3),
    max_num_attemps: int = 10,
    training_roi: Tuple[int, int, int] = (30, 30, 160),
    min_dose_ratio_threshold: float = 0.98,
) -> Tuple:
    """Moving window augmentation technique, version: v0.
    The method is responsible for cropping the input data with size bigger than provided training_roi.
    Initially, function finds the bragg peak, using the function estimate_bragg_peak.

    Args:
        ct_grid (np.ndarray): _description_
        flux_grid (np.ndarray): _description_
        dose_grid (np.ndarray): _description_
        initial_energy (float): _description_
        max_shifts (Tuple[int, int], optional): _description_. Defaults to (3, 3).
        max_num_attemps (int, optional): _description_. Defaults to 10.
        training_roi (tuple, optional): _description_. Defaults to (30, 30, 160).
        min_dose_ratio_threshold (float, optional): _description_. Defaults to 0.98.

    Returns:
        Tuple: _description_
    """
    # Determine the position of the bragg peak.
    bp_index = estimate_bragg_peak(dose_grid)
    # print("Bragg Peak Index:", bp_index)
    total_dose_deposited = dose_grid.sum()
    for _ in range(max_num_attemps):
        shift_x = random.randint(-max_shifts[0], max_shifts[0])
        shift_y = random.randint(-max_shifts[1], max_shifts[1])
        cropping_reference_point = (
            bp_index[0] + shift_y,
            bp_index[1] + shift_x,
            bp_index[2],
        )
        try:
            start_idx_0 = cropping_reference_point[0] - training_roi[0] // 2
            end_idx_0 = cropping_reference_point[0] + training_roi[0] // 2
            start_idx_1 = cropping_reference_point[1] - training_roi[1] // 2
            end_idx_1 = cropping_reference_point[1] + training_roi[1] // 2
            if start_idx_0 <= 0:
                start_idx_0 = 0
                end_idx_0 = training_roi[0]

            if end_idx_0 > ct_grid.shape[0]:
                start_idx_0 = ct_grid.shape[0] - training_roi[0]
                end_idx_0 = ct_grid.shape[0]

            if start_idx_1 <= 0:
                start_idx_1 = 0
                end_idx_1 = training_roi[1]

            if end_idx_1 > ct_grid.shape[1]:
                start_idx_1 = ct_grid.shape[1] - training_roi[1]
                end_idx_1 = ct_grid.shape[1]

            cropped_ds_grid = dose_grid[
                start_idx_0:end_idx_0,
                start_idx_1:end_idx_1,
                : training_roi[2],
            ]
            if cropped_ds_grid.shape != training_roi:
                continue
            cropped_total_dose = cropped_ds_grid.sum()
            if cropped_total_dose / total_dose_deposited > min_dose_ratio_threshold:
                cropped_ct_grid = ct_grid[
                    start_idx_0:end_idx_0,
                    start_idx_1:end_idx_1,
                    : training_roi[2],
                ]
                cropped_flux_grid = flux_grid[
                    start_idx_0:end_idx_0,
                    start_idx_1:end_idx_1,
                    : training_roi[2],
                ]
                return (
                    cropped_ct_grid,
                    cropped_flux_grid,
                    cropped_ds_grid,
                    initial_energy,
                )
        except Exception as e:
            print(e)

    # print("After {} attempts, no valid cropping found.".format(max_num_attemps), flush=True)
    start_idx_0 = bp_index[0] - training_roi[0] // 2
    end_idx_0 = bp_index[0] + training_roi[0] // 2
    start_idx_1 = bp_index[1] - training_roi[1] // 2
    end_idx_1 = bp_index[1] + training_roi[1] // 2

    if start_idx_0 <= 0:
        start_idx_0 = 0
        end_idx_0 = training_roi[0]

    if end_idx_0 > ct_grid.shape[0]:
        start_idx_0 = ct_grid.shape[0] - training_roi[0]
        end_idx_0 = ct_grid.shape[0]

    if start_idx_1 <= 0:
        start_idx_1 = 0
        end_idx_1 = training_roi[1]

    if end_idx_1 > ct_grid.shape[1]:
        start_idx_1 = ct_grid.shape[1] - training_roi[1]
        end_idx_1 = ct_grid.shape[1]

    cropped_dose = dose_grid[
        start_idx_0:end_idx_0,
        start_idx_1:end_idx_1,
        : training_roi[2],
    ]
    cropped_flux = flux_grid[
        start_idx_0:end_idx_0,
        start_idx_1:end_idx_1,
        : training_roi[2],
    ]
    cropped_ct = ct_grid[
        start_idx_0:end_idx_0,
        start_idx_1:end_idx_1,
        : training_roi[2],
    ]
    return cropped_ct, cropped_flux, cropped_dose, initial_energy
