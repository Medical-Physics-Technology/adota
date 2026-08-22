"""CT preprocessing + isocenter helpers for MC generation (adota-native).

Reuses adota's own beamlet geometry (extract_beamlet_roi, flux_projection,
angle<->spot, isocenter rotation); this module adds only the CT preprocessing the
reference pipeline applied before MC: vacuum->air clamping and isotropic resample.
"""
from __future__ import annotations

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
