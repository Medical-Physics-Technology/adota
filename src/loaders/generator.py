import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
import warnings

from typing import List, Dict, Tuple

from src.augmentation.geo_augmenations import (
    moving_window_augmentation,
    cropp_around_index,
)


class H5PYGenerator(Dataset):
    """The main object for loading data records from the h5py file.

    Args:
        Dataset (torch.utils.data.Dataset): The pre-defined PyTorch Dataset class.
    """

    def __init__(
        self,
        file_path: str,
        augmentation: bool = True,
        indexes: list = None,
        scale: dict = None,
        normalize: bool = True,
        **kwargs,
    ):
        """H5PYGenerator constructor.

        Args:
            file_path (str): Path to the h5py file, generated with use of the scripts/reinterpolate_ds.py script.
            augmentation (bool, optional): Augmentation flag. Defaults to True.
            indexes (list, optional): List of indexes to load from the h5py file. Defaults to None.
            scale (dict, optional): Dictionary with scale values. Defaults to None. If None, default values are used.
            normalize (bool, optional): Normalize flag. Defaults to True.

        Raises:
            ValueError: Indexes should be either list of integers or list of strings
        """
        super(H5PYGenerator, self).__init__()
        self.file_path = file_path
        self.augmentation = augmentation
        self.indexes = indexes
        self.square_slice = kwargs.get(
            "square_slice", True
        )  # By default square slice is True H == W
        self.cropp = kwargs.get("cropp", False)
        self.return_angles = kwargs.get("return_angles", False)
        self.expected_shape = kwargs.get("expected_shape", (160, 30, 30))

        # Temorary solution, empty records handled here:
        # Description: We investigated that mentioned records have an empty dose distributions. It cannot be taken into consideration into the training process.
        invalid_records = []
        self.indexes_to_exclude_list_path = kwargs.get(
            "indexes_to_exclude_list",
            "/home/mstryja/projects/dota_pytorch/auxilary_files/IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt",
        )
        with open(self.indexes_to_exclude_list_path, "r") as f:
            for line in f:
                invalid_records.append(line.strip())

        with h5py.File(file_path, "r") as dataset:
            self.record_ids = list(dataset.keys())
            # Remove invalid records from the dataset
            self.record_ids = [
                rid for rid in self.record_ids if rid not in invalid_records
            ]

        if self.indexes is not None:
            if isinstance(self.indexes[0], int):
                self.record_ids = [self.record_ids[i] for i in self.indexes]

            elif isinstance(self.indexes[0], str):
                self.record_ids = self.indexes

            else:
                raise ValueError(
                    "Indexes should be either list of integers or list of strings"
                )

        self.default_scale = {
            "min_ds": 0.0,
            "max_ds": 24732944.0,
            "min_ct": -1024,
            "max_ct": 3063,
            "min_energy": 70.00221819271046,
            "max_energy": 179.99924071411004,
        }
        self.scale = scale if scale is not None else self.default_scale
        self.normalize = normalize

        self.normalize_energy_only = kwargs.get(
            "normalize_energy_only", False
        )  # Normalize energy only is a flag which is used for tests.

        self.normalize_flux_only = kwargs.get(
            "normalize_flux_only", False
        )  # Normalize flux only is a flag which is used for tests.

        self.flux_mode = kwargs.get("flux_mode", "analytical")
        if self.flux_mode not in {"analytical", "angle_broadcast"}:
            raise ValueError(f"Unknown flux_mode: {self.flux_mode!r}")

        # Random rotation by one of angles (0, 90, 180, 270)
        self.rotk = np.arange(4) if self.square_slice else [0, 2]

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _id = self.record_ids[idx]

        with h5py.File(self.file_path, "r") as dataset:
            record_group = dataset[_id]
            ct_grid = record_group["ct"][:]
            dose_grid = record_group["dose"][:]
            flux_grid = record_group["flux"][:]

            if not self.augmentation and self.cropp:
                # If augmentation is disabled, but cropping is enabled, we cropp around the Bragg peak.
                warnings.warn(
                    "Cropping is disabled, please make sure that this is intended behavior.",
                    category=UserWarning,
                )
                ct_grid, flux_grid, dose_grid, initial_energy = cropp_around_index(
                    ct_grid, flux_grid, dose_grid, record_group.attrs["initial_energy"]
                )

            metadata = {
                "dose_deposition_ratio": record_group.attrs["dose_deposition_ratio"],
                "gantry_angle": record_group.attrs["gantry_angle"],
                "id": record_group.attrs["id"],
                "initial_energy": record_group.attrs["initial_energy"],
                "beamlet_angles": record_group.attrs["beamlet_angles"],
                "stat_uncertainty": record_group.attrs["stat_uncertainty"],
            }
            if self.augmentation:
                # print(
                #     "Before augmentation",
                #     ct_grid.shape,
                #     flux_grid.shape,
                #     dose_grid.shape,
                # )
                ct_grid, flux_grid, dose_grid, initial_energy = (
                    moving_window_augmentation(
                        ct_grid, flux_grid, dose_grid, metadata["initial_energy"]
                    )
                )
                # print(
                #     "After augmentation",
                #     ct_grid.shape,
                #     flux_grid.shape,
                #     dose_grid.shape,
                # )
                rot = np.random.choice(self.rotk)
                ct_grid = np.rot90(ct_grid, k=rot, axes=(0, 1)).copy()
                dose_grid = np.rot90(dose_grid, k=rot, axes=(0, 1)).copy()
                flux_grid = np.rot90(flux_grid, k=rot, axes=(0, 1)).copy()
                # print("After rotation", ct_grid.shape, flux_grid.shape, dose_grid.shape)

            # Convert numpy arrays to PyTorch tensors
            # Convert numpy -> torch and fix memory/layout
            ct_grid = (
                torch.as_tensor(ct_grid, dtype=torch.float32)
                .permute(2, 0, 1)
                .contiguous()
            )
            dose_grid = (
                torch.as_tensor(dose_grid, dtype=torch.float32)
                .permute(2, 0, 1)
                .contiguous()
            )
            flux_grid = (
                torch.as_tensor(flux_grid, dtype=torch.float32)
                .permute(2, 0, 1)
                .contiguous()
            )
            e = metadata["initial_energy"]

            # Guard against empty windows (any zero-sized dim)
            if 0 in ct_grid.shape or 0 in dose_grid.shape or 0 in flux_grid.shape:
                raise IndexError(
                    f"Empty sample for id={_id}: "
                    f"ct={tuple(ct_grid.shape)}, dose={tuple(dose_grid.shape)}, flux={tuple(flux_grid.shape)}"
                )

            # Handle the incorect shape of ct_grid, flux_grid and dose_grid. Each of them must have shape of (160, 30, 30)
            if ct_grid.shape != self.expected_shape:
                raise ValueError(
                    f"Incorrect shape for ct_grid: {ct_grid.shape}, expected {self.expected_shape}"
                )
            if flux_grid.shape != self.expected_shape:
                raise ValueError(
                    f"Incorrect shape for flux_grid: {flux_grid.shape}, expected {self.expected_shape}"
                )
            if dose_grid.shape != self.expected_shape:
                raise ValueError(
                    f"Incorrect shape for dose_grid: {dose_grid.shape}, expected {self.expected_shape}"
                )

            # Normalize using MinMax scaller
            # Probably not needed due to the fact, that we are doing normalization in the generation stage.
            if self.normalize:
                # We perform scale if data in h5py file is not normalized.
                ct_grid = (ct_grid - self.scale["min_ct"]) / (
                    self.scale["max_ct"] - self.scale["min_ct"]
                )
                dose_grid = (dose_grid - self.scale["min_ds"]) / (
                    self.scale["max_ds"] - self.scale["min_ds"]
                )
                e = (metadata["initial_energy"] - self.scale["min_energy"]) / (
                    self.scale["max_energy"] - self.scale["min_energy"]
                )

            if self.normalize_energy_only:
                e = (metadata["initial_energy"] - self.scale["min_energy"]) / (
                    self.scale["max_energy"] - self.scale["min_energy"]
                )

            if self.flux_mode == "angle_broadcast":
                angles = torch.tensor(
                    metadata["beamlet_angles"], dtype=torch.float32
                )
                mag = float(torch.sqrt((angles ** 2).sum()))
                flux_grid = torch.full(flux_grid.shape, mag)

            # Apply channel dimension
            ct_grid = ct_grid.unsqueeze(0)
            dose_grid = dose_grid.unsqueeze(0)
            flux_grid = flux_grid.unsqueeze(0)

            if self.normalize_flux_only and self.flux_mode == "analytical":
                flux_grid = (flux_grid - flux_grid.min()) / (
                    flux_grid.max() - flux_grid.min()
                )

            # Concatente flux and ct grid
            x = torch.cat((ct_grid, flux_grid), dim=0).contiguous()
            initial_energy = torch.tensor(e, dtype=torch.float32)
            initial_energy = initial_energy.unsqueeze(0)

            if self.return_angles:
                warnings.warn(
                    "Returning angles is enabled, please make sure that this is intended behavior.",
                    category=UserWarning,
                )
                beamlet_angles = torch.tensor(
                    metadata["beamlet_angles"], dtype=torch.float32
                )
                gantry_angle = torch.tensor(
                    metadata["gantry_angle"], dtype=torch.float32
                )
                return x, initial_energy, dose_grid, beamlet_angles, gantry_angle

            return x, initial_energy, dose_grid
