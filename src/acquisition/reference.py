"""Reading the reference HDF5 beamlets the way the difficulty study saw them.

The study's metrics were computed on records loaded through ``H5PYGenerator``
with ``augmentation=False, cropp=True, normalize=False, normalize_flux_only=True``
and de-normalised with :data:`src.adota.config.DEFAULT_SCALE`. This module
reproduces that path without the model, and adds the one thing the study could
not do: crop and compute the metrics from an analytic dose instead of the ground
truth, so the two feature vectors can be compared record by record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

from src.acquisition.features import FeatureConfig, compute_features
from src.acquisition.surrogate import analytic_dose, bragg_peak_index, crop_like_training, peak_inside_crop
from src.adota.config import DEFAULT_SCALE, denormalize_energy
from src.utils.scallers import inverse_minmax

TRAINING_ROI = (160, 30, 30)   # (D, H, W) of the model input


@dataclass(frozen=True)
class StoredRecord:
    """One H5 record in the ``(D, H, W)`` frame, physical units, uncropped."""

    sample_id: str
    ct_hu: np.ndarray
    flux: np.ndarray        # min-max normalised, as the loader hands it to the model
    dose: np.ndarray        # ground truth in physical units, as the study de-normalised it
    energy_mev: float


def read_stored_record(group: h5py.Group, sample_id: str, scale: Dict[str, float] = DEFAULT_SCALE
                       ) -> StoredRecord:
    """Load one record group. Storage is ``(y, x, z)``; the model frame is ``(z, y, x)``."""
    ct = np.asarray(group["ct"][:], dtype=np.float32).transpose(2, 0, 1)
    flux = np.asarray(group["flux"][:], dtype=np.float32).transpose(2, 0, 1)
    dose = np.asarray(group["dose"][:], dtype=np.float32).transpose(2, 0, 1)
    span = float(flux.max() - flux.min())
    if span > 0:
        flux = (flux - flux.min()) / span
    # The dose scale matters: the dose-weighted edge energies are linear in it.
    return StoredRecord(
        sample_id=sample_id,
        ct_hu=inverse_minmax(ct, scale["min_ct"], scale["max_ct"]),
        flux=flux, dose=inverse_minmax(dose, scale["min_ds"], scale["max_ds"]),
        energy_mev=float(denormalize_energy(float(group.attrs["initial_energy"]), scale)),
    )


def record_features(record: StoredRecord, mode: str, config: FeatureConfig = FeatureConfig(),
                    ) -> Optional[Dict[str, float]]:
    """The thirty metrics of a stored record, located by the ground truth
    (``mode="gt"``) or by the analytic surrogate (``mode="analytic"``).

    In both modes the 30x30 lateral crop is centred where that dose peaks, as the
    training loader does, so the two modes differ in exactly one input: which
    dose located the peak and weighted the edges. Returns ``None`` for the
    records the reference run skipped (no flux, or energy above 250 MeV).
    """
    if record.energy_mev > 250.0 or not np.any(record.flux):
        return None
    dz_mm = float(config.resolution_mm[0])
    if mode == "gt":
        dose = record.dose
    elif mode == "analytic":
        dose = analytic_dose(record.ct_hu, record.flux, record.energy_mev, dz_mm)
    else:
        raise ValueError(f"mode must be 'gt' or 'analytic', got {mode!r}")

    z_peak, y_peak, x_peak = bragg_peak_index(dose)
    crop = lambda v: crop_like_training(v, (y_peak, x_peak), TRAINING_ROI)  # noqa: E731
    out = compute_features(crop(record.ct_hu), crop(record.flux), record.energy_mev, crop(dose), config)
    out.update({"sample_id": record.sample_id, "mode": mode,
                "peak_z_full": z_peak, "peak_y_full": y_peak, "peak_x_full": x_peak,
                "peak_inside_crop": peak_inside_crop(out, TRAINING_ROI[0], dz_mm)})
    return out


def peak_agreement(record: StoredRecord, dz_mm: float = 2.0) -> Tuple[float, float]:
    """Depth error of the analytic peak against the ground-truth peak, in mm, and
    the lateral offset of the crop centres in voxels. The cheapest possible
    check of the surrogate, independent of any metric."""
    z_gt, y_gt, x_gt = bragg_peak_index(record.dose)
    z_an, y_an, x_an = bragg_peak_index(analytic_dose(record.ct_hu, record.flux, record.energy_mev, dz_mm))
    return (z_an - z_gt) * dz_mm, float(np.hypot(y_an - y_gt, x_an - x_gt))
