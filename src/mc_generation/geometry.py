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


def beam_entrance_index(
    ct_array: np.ndarray, lateral_window=None, tissue_hu: float = -300.0,
) -> int:
    """First index along the beam axis (x) whose slab holds tissue.

    Args:
        ct_array: ``sitk.GetArrayFromImage`` result, ``(z, y, x)``.
        lateral_window: Optional ``(z_slice, y_slice)`` restricting the search to
            the lateral region the sweep's beamlets can reach, so a couch edge or
            distant anatomy outside the beam does not define the entrance.
        tissue_hu: HU above which a voxel counts as tissue (default -300, i.e.
            anything denser than lung-ish air).

    Returns:
        The first x index containing tissue, or 0 if the volume holds none.
    """
    a = ct_array if lateral_window is None else ct_array[lateral_window[0], lateral_window[1], :]
    tissue = (a > tissue_hu).any(axis=(0, 1))
    return int(np.argmax(tissue)) if tissue.any() else 0


def trim_beam_axis(ct: sitk.Image, x_size: int, x0: int) -> sitk.Image:
    """Take ``x_size`` voxels along the beam axis (x) starting at index ``x0``.

    ``extract_beamlet_roi`` measures the ROI's depth from the grid's ``x = 0``
    face, so the beam-axis window of the grid is what decides how much of the
    patient the crop reaches. Canonicalizing a random gantry expands the grid,
    which adds air in front of the patient and pushes the distal dose out of the
    crop; trimming back with an ``x0`` chosen just before the patient surface
    restores the entrance geometry the gantry-90 data (and the trained model) has.
    ``x0`` is clamped so the window stays inside the grid.
    """
    nx = ct.GetSize()[0]
    if x_size >= nx:
        return ct
    x0 = max(0, min(int(x0), nx - x_size))
    return ct[x0:x0 + x_size, :, :]


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
