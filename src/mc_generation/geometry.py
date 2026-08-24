"""CT preprocessing + isocenter helpers for MC generation (adota-native).

Reuses adota's own beamlet geometry (extract_beamlet_roi, flux_projection,
angle<->spot, isocenter rotation); this module adds only the CT preprocessing the
reference pipeline applied before MC: vacuum->air clamping and isotropic resample.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk


def reduce_vacuum_to_air(image: sitk.Image, low: int = -1024, high: int = 3071) -> sitk.Image:
    """Clamp HU to ``[low, high]`` so out-of-scan vacuum reads as air (as datagenerator did)."""
    arr = sitk.GetArrayFromImage(image)
    clamped = np.clip(arr, low, high).astype(arr.dtype)
    out = sitk.GetImageFromArray(clamped)
    out.CopyInformation(image)
    return out


def resample_to_isotropic(image: sitk.Image, spacing_mm: float = 1.0,
                          interpolator: int = sitk.sitkLinear,
                          default_value: float = -1024.0) -> sitk.Image:
    """Resample to an isotropic ``spacing_mm`` grid, preserving physical extent."""
    in_size = np.asarray(image.GetSize(), dtype=float)
    in_spacing = np.asarray(image.GetSpacing(), dtype=float)
    out_spacing = np.array([spacing_mm, spacing_mm, spacing_mm], dtype=float)
    out_size = np.round(in_size * in_spacing / out_spacing).astype(int).tolist()
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(out_spacing.tolist())
    r.SetSize([int(s) for s in out_size])
    r.SetOutputOrigin(image.GetOrigin())
    r.SetOutputDirection(image.GetDirection())
    r.SetInterpolator(interpolator)
    r.SetDefaultPixelValue(default_value)
    return r.Execute(image)


def mc_isocenter(ct: sitk.Image) -> list:
    """Isocenter passed to MCsquare (image-frame center), matching datagenerator."""
    size = np.asarray(ct.GetSize())
    spacing = np.asarray(ct.GetSpacing())
    return (size * spacing // 2).tolist()


def extraction_isocenter_physical(ct: sitk.Image) -> np.ndarray:
    """World-coordinate isocenter used by extract_beamlet_roi (matches the reference)."""
    origin = np.asarray(ct.GetOrigin())
    spacing = np.asarray(ct.GetSpacing())
    center = np.asarray(mc_isocenter(ct))
    return np.round(origin + center - spacing / 2.0, 3)


def body_mask(ct: sitk.Image, body_hu_threshold: float = -500.0) -> np.ndarray:
    """Binary patient-body mask as a ``(z, y, x)`` float array (1.0 inside).

    Defined per axial slice as **everything that is NOT external air**: air voxels
    (``HU <= body_hu_threshold``) that connect to the slice border are external air;
    the patient is the complement. This robustly counts **lung** (interior air, not
    connected to the border) as *inside the patient* -- unlike hole-filling, which
    fails when lung joins external air through the airways. Dose inside this contour
    is in the patient; dose outside it is in external air.
    """
    from scipy.ndimage import label

    arr = sitk.GetArrayFromImage(ct)  # (z, y, x)
    out = np.zeros_like(arr, dtype=np.float32)
    for z in range(arr.shape[0]):
        air = arr[z] <= body_hu_threshold
        if not air.any():
            out[z] = 1.0
            continue
        lab, n = label(air)
        if n == 0:
            out[z] = 1.0
            continue
        border = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
        border = border[border != 0]
        external = np.isin(lab, border)      # air touching the slice edge = outside patient
        out[z] = (~external).astype(np.float32)
    return out


def body_center_of_mass(ct: sitk.Image, body_hu_threshold: float = -500.0) -> np.ndarray:
    """World-coordinate ``(x, y, z)`` centre of mass of the patient body mask.

    Aiming the isocenter here (instead of the grid centre) keeps the beamlet fan
    centred on the patient, so extreme steering angles stay in tissue.
    """
    mask = body_mask(ct, body_hu_threshold)                  # (z, y, x)
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        raise ValueError("empty body mask (no voxels above body_hu_threshold)")
    com_zyx = idx.mean(axis=0)
    com_xyz_index = [float(com_zyx[2]), float(com_zyx[1]), float(com_zyx[0])]
    return np.asarray(ct.TransformContinuousIndexToPhysicalPoint(com_xyz_index))


def isocenters_from_world(ct: sitk.Image, world_point: Sequence[float]):
    """Return ``(iso_mc, iso_ext)`` for an arbitrary world isocenter, matching the
    grid-centre convention: ``iso_ext = origin + iso_mc - spacing/2``."""
    origin = np.asarray(ct.GetOrigin())
    spacing = np.asarray(ct.GetSpacing())
    world = np.asarray([float(c) for c in world_point])
    iso_mc = (world - origin).tolist()
    iso_ext = np.round(world - spacing / 2.0, 3)
    return iso_mc, iso_ext


def dose_in_body_fraction(cropped_dose: np.ndarray, body_mask_crop: np.ndarray) -> float:
    """Fraction of the beamlet dose deposited inside the patient body contour.

    ``body_mask_crop`` is the body mask cropped to the same ROI as ``cropped_dose``
    (>0 = inside the patient, includes lung). ~1.0 means the beam stays in the
    patient; a low value flags a beam grazing into external air. Complements
    ``dose_deposition_ratio`` (ROI capture) with a true tissue-vs-air check.
    """
    total = float(cropped_dose.sum())
    if total <= 0:
        return 0.0
    return float(cropped_dose[body_mask_crop > 0].sum()) / total
