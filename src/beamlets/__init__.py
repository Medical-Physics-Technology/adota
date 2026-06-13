"""Plan-level beamlet extraction for ADoTA.

This package turns an OpenTPS plan (CT grid + ``PlanPencil.txt`` + beam data
library) into the per-spot inputs the ADoTA model was trained on: a BEV CT crop
and a matching proton-flux projection, both of shape ``(60, 60, 320)`` in numpy
``(z, y, x)`` order, where the beam axis runs along ``x`` (depth) from the
``x = 0`` entrance face.

Coordinate conventions (fixed once, cited by every module here)
---------------------------------------------------------------
* **sitk index order is ``(x, y, z)``**; numpy array order is ``(z, y, x)``.
  ``np_index = sitk_index[::-1]``.
* **BEV / depth.** After rotating the CT so the beam axis lies along ``+x``, the
  depth crop is taken as ``x in [0, 320)`` from the ``x = 0`` face. The lateral
  ``60 x 60`` window is centred on the spot's ``(y, z)`` voxel.
* **Gantry adjustment.** Per field, the CT is rotated in the axial (``x``-``y``)
  plane by ``A = (-1) * (gantry_angle - 90)`` degrees (so a ``90`` deg field is
  not rotated and a ``-90`` deg field is rotated ``180`` deg).
* **Rotation pivot = isocenter.** The rotation is performed around the field
  isocenter, which is therefore a fixed point and is reused unchanged as the
  cropping reference. ``beamlet_ray`` is always called with ``g_ang = -90`` so
  the beam enters from the ``x = 0`` face; no x-flip is applied.
* **Isocenter convention.** The plan isocenter is a continuous voxel index
  inside the CT grid; its physical point is
  ``ct.TransformContinuousIndexToPhysicalPoint(field.isocenter)`` (handles
  non-unit spacing and any direction matrix).
* **Out-of-bounds** voxels (from rotation or cropping) are filled with air,
  ``-1024`` HU, and flagged.
* **Deterministic spot ids:** ``b{beam:02d}_l{layer:03d}_s{spot:04d}`` where
  beam = field index, layer = control-point index, spot = spot index.
"""

AIR_HU: int = -1024
ROI_SIZE: tuple[int, int, int] = (60, 60, 320)
"""Default ROI size as ``[H, W, D]`` (lateral y, lateral x/z, depth)."""
