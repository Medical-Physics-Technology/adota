"""DoseCUDA pencil-beam dose on a CT grid, addressed in the MCsquare beamlet frame.

The MC generator (:mod:`src.mc_generation`) describes a beamlet as MCsquare does:
the CT written as-is, an isocenter ``iso_mc`` measured from the grid's corner
(voxel 0's lower edge), the canonical gantry 90 (beam along +x, entering at the
``x = 0`` face) and a spot ``(X, Y)`` in the isocenter plane, ``X`` diverging about
the x scanning magnet and landing on image ``-y``, ``Y`` about the y magnet onto
image ``+z`` (see ``extract_beamlet_roi`` / ``beamlet_ray`` with ``g_ang = -90``).

DoseCUDA works in DICOM patient coordinates with ``origin`` = voxel 0's centre.
Its gantry 270 puts the source at ``-x`` (it adds 180 deg internally), so the beam
runs along +x; its nozzle frame then maps spot ``(sx, sy)`` to image
``(-sx, +sy)`` in ``(y, z)``, ``sx`` diverging about VSADX and ``sy`` about VSADY.
With ``VSADX = d_smx`` and ``VSADY = d_smy`` the MC spot is therefore passed
through unchanged; :func:`mc_frame_to_dosecuda` is the single place this lives,
and the engine-agreement smoke test checks it with off-axis spots.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.pencil_beam.beam_model import PencilBeamModel, hu_to_rlsp, read_rlsp_curve
from src.pencil_beam.kernels import load_dose_kernels

__all__ = ["DOSECUDA_GANTRY_FOR_MC_90", "PencilBeamEngine", "PencilBeamResult", "PreparedField",
           "mc_frame_to_dosecuda"]

DOSECUDA_GANTRY_FOR_MC_90 = 270.0


def mc_frame_to_dosecuda(ct: sitk.Image, iso_mc: Sequence[float], mc_gantry: float = 90.0
                         ) -> Tuple[np.ndarray, float]:
    """``(isocenter_world, dosecuda_gantry)`` for an MC-frame isocenter and gantry.

    Only the canonical MC gantry (90 deg) is supported: the generator rotates the
    CT so every beamlet is simulated there (``rotate_to_canonical``).

    Raises:
        ValueError: For a non-canonical gantry or a rotated image direction.
    """
    if abs(mc_gantry - 90.0) > 1e-6:
        raise ValueError(f"only the canonical MC gantry 90 is supported, got {mc_gantry}")
    if not np.allclose(ct.GetDirection(), np.eye(3).ravel()):
        raise ValueError("CT direction must be identity (resample it first)")
    origin = np.asarray(ct.GetOrigin(), dtype=float)
    spacing = np.asarray(ct.GetSpacing(), dtype=float)
    iso_world = origin - spacing / 2.0 + np.asarray(iso_mc, dtype=float)
    return iso_world, DOSECUDA_GANTRY_FOR_MC_90


@dataclass
class PreparedField:
    """One field's RLSP volume, beam and full-grid WET (cm), from :meth:`PencilBeamEngine.prepare_field`."""

    volume: SimpleNamespace
    beam: SimpleNamespace
    wet_cm: np.ndarray
    seconds: float


@dataclass
class PencilBeamResult:
    """One pencil-beam calculation on the CT grid (arrays are ``(z, y, x)``)."""

    dose: np.ndarray
    wet_mm: np.ndarray
    seconds: float
    model_energy: float
    interpolated_from: Optional[list]


class PencilBeamEngine:
    """Commissioned DoseCUDA pencil beam for MC-frame beamlets.

    Args:
        kernels_path: ``dose_kernels.so`` built by ``provision_dosecuda.sh``.
        model_dir: A commissioned model (see :mod:`src.pencil_beam.beam_model`).
        gpu_id: CUDA device index as the process sees it.
    """

    def __init__(self, kernels_path: Path, model_dir: Path, gpu_id: int = 0):
        self.kernels = load_dose_kernels(kernels_path)
        self.model = PencilBeamModel.read(model_dir)
        self.rlsp_curve = read_rlsp_curve(model_dir)
        self.gpu_id = int(gpu_id)

    def wet(self, ct: sitk.Image, iso_mc: Sequence[float], mc_gantry: float = 90.0,
            rlsp: Optional[np.ndarray] = None) -> np.ndarray:
        """Water-equivalent depth (mm) of every voxel along its ray to the source."""
        volume, beam = self._volume_and_beam(ct, iso_mc, mc_gantry, rlsp)
        model = SimpleNamespace(VSADX=float(self.model.vsadx), VSADY=float(self.model.vsady))
        return 10.0 * self.kernels.proton_raytrace_cuda(model, volume, beam, self.gpu_id)

    def prepare_field(self, ct: sitk.Image, iso_mc: Sequence[float], mc_gantry: float = 90.0) -> "PreparedField":
        """RLSP and ray-traced WET of one field, computed once for all its spots and energies.

        The WET depends only on the CT and the field geometry (isocenter, gantry,
        point source at the mean VSAD), not on the spot or the energy, so a field's
        beamlets can share it and each needs only the cheap spot kernel.
        """
        spacing = ct.GetSpacing()
        if not np.allclose(spacing, spacing[0]):
            raise ValueError(f"DoseCUDA needs isotropic spacing, got {spacing}")
        t0 = perf_counter()
        rlsp = hu_to_rlsp(sitk.GetArrayFromImage(ct), self.rlsp_curve)
        volume, beam = self._volume_and_beam(ct, iso_mc, mc_gantry, rlsp)
        model = SimpleNamespace(VSADX=float(self.model.vsadx), VSADY=float(self.model.vsady))
        wet_cm = np.asarray(self.kernels.proton_raytrace_cuda(model, volume, beam, self.gpu_id), dtype=np.float32)
        return PreparedField(volume=volume, beam=beam, wet_cm=wet_cm, seconds=perf_counter() - t0)

    def dose_in_window(
        self,
        field: "PreparedField",
        energy: float,
        spots_xy: Sequence[Sequence[float]],
        window: Optional[Tuple[slice, slice, slice]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> PencilBeamResult:
        """Pencil-beam dose of one energy layer on a ``(z, y, x)`` window of a prepared field.

        The spot kernel is evaluated per voxel, so a window gives exactly the full
        grid's dose cut to that window (the WET still comes from the full-grid ray
        trace). ``None`` is the whole grid.
        """
        shape = field.volume.voxel_data.shape
        window = tuple(slice(0, n) for n in shape) if window is None else tuple(window)
        for sl, n in zip(window, shape):
            if not (0 <= sl.start < sl.stop <= n) or sl.step not in (None, 1):
                raise ValueError(f"window {window} outside the grid {shape}")
        weights = [1000.0] * len(spots_xy) if weights is None else list(weights)
        t0 = perf_counter()
        layer = self.model.at_energy(float(energy))
        model = layer.kernel_model()
        offset = np.array([window[2].start, window[1].start, window[0].start], dtype=np.float64)  # x, y, z
        origin = (field.volume.origin.astype(np.float64) + offset * field.volume.spacing).astype(np.float32)
        volume = SimpleNamespace(voxel_data=np.ascontiguousarray(field.volume.voxel_data[window]),
                                 origin=origin, spacing=field.volume.spacing)
        wet = SimpleNamespace(voxel_data=np.ascontiguousarray(field.wet_cm[window]), origin=origin,
                              spacing=field.volume.spacing)
        beam = SimpleNamespace(
            iso=field.beam.iso, gantry_angle=field.beam.gantry_angle, couch_angle=field.beam.couch_angle,
            spot_list=np.ascontiguousarray(
                [[float(x), float(y), float(w), 0.0] for (x, y), w in zip(spots_xy, weights)], dtype=np.float32))
        dose = self.kernels.proton_spot_cuda(model, volume, wet, beam, self.gpu_id)
        return PencilBeamResult(
            dose=np.asarray(dose, dtype=np.float32), wet_mm=10.0 * field.wet_cm[window],
            seconds=perf_counter() - t0, model_energy=float(layer.energies[0]),
            interpolated_from=layer.meta.get("interpolated_from"))

    def dose(
        self,
        ct: sitk.Image,
        energy: float,
        spots_xy: Sequence[Sequence[float]],
        iso_mc: Sequence[float],
        weights: Optional[Sequence[float]] = None,
        mc_gantry: float = 90.0,
    ) -> PencilBeamResult:
        """Pencil-beam dose of one energy layer on ``ct``'s grid.

        Args:
            ct: HU image, isotropic spacing, identity direction (the MC CT).
            energy: Nominal energy, MeV (interpolated between commissioned nodes).
            spots_xy: MC-frame spot positions in the isocenter plane, mm.
            iso_mc: MC-frame isocenter (as passed to MCsquare).
            weights: Spot weights in MCsquare plan units (default 1000 each, the
                runner's default, so the dose is on the MC dose scale).
            mc_gantry: MC gantry angle; must be the canonical 90.
        """
        field = self.prepare_field(ct, iso_mc, mc_gantry)
        result = self.dose_in_window(field, energy, spots_xy, None, weights)
        result.seconds += field.seconds
        return result

    def _volume_and_beam(self, ct, iso_mc, mc_gantry, rlsp):
        iso_world, gantry = mc_frame_to_dosecuda(ct, iso_mc, mc_gantry)
        if rlsp is None:
            rlsp = hu_to_rlsp(sitk.GetArrayFromImage(ct), self.rlsp_curve)
        volume = SimpleNamespace(
            voxel_data=np.ascontiguousarray(rlsp, dtype=np.float32),
            origin=np.asarray(ct.GetOrigin(), dtype=np.float32),
            spacing=np.asarray(ct.GetSpacing(), dtype=np.float32))
        beam = SimpleNamespace(iso=iso_world.astype(np.float32), gantry_angle=float(gantry),
                               couch_angle=0.0)
        return volume, beam
