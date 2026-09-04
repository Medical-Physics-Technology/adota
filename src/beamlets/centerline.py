"""Beam-centerline input channel: alternatives to the Gaussian flux projection.

This module builds the *centerline* of a proton beamlet, the alternative
direction encoding used to answer reviewer comment 3 (the value of the flux
projection is not established against a simpler direction encoding).

The beamlet centerline is a straight line in 3D, represented parametrically as
``r(d) = p0 + d * v`` where ``d`` is the depth-slice index (the beam propagates
along the depth axis, numpy axis 2 of a raw record), ``p0`` is the entrance
point, and ``v`` is the per-depth lateral drift. The representation is fixed by
two independent, flux-free sources:

* **Direction** from the beamlet *steering* angles only. The extraction already
  rotates the CT into the gantry-aligned beam's-eye frame, so the gantry angle
  does not enter; only the small steering angles ``(theta_x, theta_y)`` remain.
  Empirically (and exactly, since the stored flux is an analytic Gaussian about
  this line) the per-depth lateral slopes are::

      d(axis0)/d(depth) = + tan(theta_x)
      d(axis1)/d(depth) = - tan(theta_y)

* **Anchor** (entrance) from the per-beamlet source metadata JSON field
  ``rays_entrence_point = [depth, e1, e2]`` (depth is 0 at entrance). In the raw
  downsampled record frame the lateral entrance is::

      axis0 at depth 0 = e2 / downsample
      axis1 at depth 0 = e1 / downsample

  where ``downsample = roi_lateral / record_lateral`` (2 for the v2 dataset) and
  an average-pooling origin shift ``-(downsample-1)/(2*downsample)`` (``-0.25`` at
  factor 2) maps ROI voxel centers to record voxel centers. Note the axis swap
  (e2 -> axis0, e1 -> axis1); both the swap and the origin shift are verified
  against the flux ridge (residual < 0.01 vox) in ``tests`` and ``scripts``.

These conventions were established empirically against the flux ridge (per-slice
flux-weighted centroid); the analytic line matches it to well under one voxel.
Because the centerline is built in the raw record frame, it can be passed through
the same augmentation (moving-window crop + rot90) as the CT/flux/dose grids, so
it stays registered to them with no extra bookkeeping.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = ["BeamLine", "beam_line_from_metadata", "render_centerline"]


class BeamLine:
    """Parametric beamlet centerline ``r(d) = (a0 + b0 d, a1 + b1 d, d)``.

    Coordinates are in raw-record voxels: ``(axis0, axis1)`` are the two lateral
    axes and ``d`` is the depth-slice index (numpy axis 2).

    Attributes:
        a0, a1: Lateral entrance position at depth ``d = 0``.
        b0, b1: Lateral drift per depth slice (voxels/slice).
    """

    __slots__ = ("a0", "a1", "b0", "b1")

    def __init__(self, a0: float, a1: float, b0: float, b1: float):
        self.a0 = float(a0)
        self.a1 = float(a1)
        self.b0 = float(b0)
        self.b1 = float(b1)

    def lateral_center(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lateral center ``(axis0, axis1)`` at each depth-slice index."""
        d = np.asarray(depth, dtype=np.float64)
        return self.a0 + self.b0 * d, self.a1 + self.b1 * d

    def __repr__(self) -> str:
        return (
            f"BeamLine(a0={self.a0:.3f}, a1={self.a1:.3f}, "
            f"b0={self.b0:+.5f}, b1={self.b1:+.5f})"
        )


def beam_line_from_metadata(
    rays_entrence_point: Sequence[float],
    beamlet_angles_deg: Sequence[float],
    downsample: float,
) -> BeamLine:
    """Build the raw-frame centerline from source metadata.

    Args:
        rays_entrence_point: The JSON ``rays_entrence_point`` ``[depth, e1, e2]``
            in ROI voxels (depth is 0 at entrance).
        beamlet_angles_deg: Steering angles ``(theta_x, theta_y)`` in degrees.
        downsample: ``roi_lateral / record_lateral`` (e.g. 2.0 for the v2
            dataset: ROI lateral 80 -> record lateral 40).

    Returns:
        The :class:`BeamLine` in raw-record voxel coordinates.
    """
    _, e1, e2 = rays_entrence_point
    theta_x, theta_y = beamlet_angles_deg
    # average-pooling origin shift: ROI center -> record center under x{downsample}
    origin = -(downsample - 1.0) / (2.0 * downsample)
    a0 = float(e2) / downsample + origin
    a1 = float(e1) / downsample + origin
    b0 = float(np.tan(np.deg2rad(theta_x)))
    b1 = -float(np.tan(np.deg2rad(theta_y)))
    return BeamLine(a0=a0, a1=a1, b0=b0, b1=b1)


def render_centerline(
    line: BeamLine,
    shape: Sequence[int],
    mode: str = "soft",
    sigma: float = 1.5,
) -> np.ndarray:
    """Render the centerline into a raw-record volume ``(H, W, D)``.

    For each depth slice ``d`` the beam center is ``(a0 + b0 d, a1 + b1 d)``.
    ``"soft"`` places an isotropic Gaussian of fixed width ``sigma`` (a
    constant-width "tube", the reviewer's constant-width projection); ``"binary"``
    marks the single nearest lateral voxel.

    Args:
        line: The beamlet centerline in raw-record voxels.
        shape: Output ``(H, W, D)`` matching the record's ct/flux/dose grids.
        mode: ``"soft"`` (fixed-sigma Gaussian) or ``"binary"`` (nearest voxel).
        sigma: Lateral Gaussian width in voxels (``"soft"`` only).

    Returns:
        ``float32`` array of ``shape``; soft mode is normalized to a unit peak
        per non-empty slice so the channel range matches the [0, 1] flux channel.
    """
    if mode not in {"soft", "binary"}:
        raise ValueError(f"Unknown centerline mode: {mode!r}")
    H, W, D = int(shape[0]), int(shape[1]), int(shape[2])
    out = np.zeros((H, W, D), dtype=np.float32)
    d = np.arange(D)
    c0, c1 = line.lateral_center(d)  # (D,), (D,)

    if mode == "binary":
        i0 = np.rint(c0).astype(int)
        i1 = np.rint(c1).astype(int)
        ok = (i0 >= 0) & (i0 < H) & (i1 >= 0) & (i1 < W)
        out[i0[ok], i1[ok], d[ok]] = 1.0
        return out

    # soft: per-slice isotropic Gaussian about the (continuous) center
    ii = np.arange(H)[:, None]
    jj = np.arange(W)[None, :]
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    for k in range(D):
        r2 = (ii - c0[k]) ** 2 + (jj - c1[k]) ** 2
        out[:, :, k] = np.exp(-r2 * inv2s2)
    return out
