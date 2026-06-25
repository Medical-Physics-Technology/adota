"""Correctness tests for the per-beamlet BEV reinterpolation rotation.

These pin the contract of :func:`src.image_processing.rotation.rotate_beamlet_crop`
used by the DoTA-like timing pipeline:

* a zero-angle rotation is the identity;
* the analytical angled flux ridge **straightens** onto the depth centerline
  after the forward rotation (this validates the angle->axis->sign convention);
* the inverse rotation round-trips a smooth volume back to itself (validates the
  back-rotation that maps the predicted dose from the perpendicular frame to the
  crop frame).
"""

from __future__ import annotations

import numpy as np

from src.beamlets.flux import flux_projection
from src.image_processing.rotation import rotate_beamlet_crop

# gf=2 (2mm) crop shape (z, y, x); the beam/depth axis runs along x (last).
SHAPE_ZYX = (30, 30, 160)
SPACING = np.asarray([2, 2, 2], dtype=np.float32)


def _depth_centroid_spread(vol_zyx: np.ndarray) -> float:
    """Std over depth of the per-slice lateral (z, y) centroid, in voxels.

    A beam that runs straight along the depth (x) centerline has a ~constant
    lateral centroid, so a small spread means the ridge is axis-aligned.
    """
    nz, ny, nx = vol_zyx.shape
    zz, yy = np.meshgrid(np.arange(nz), np.arange(ny), indexing="ij")
    cy: list[float] = []
    cz: list[float] = []
    for ix in range(nx):
        sl = vol_zyx[:, :, ix]
        s = float(sl.sum())
        if s <= 1e-9:
            continue
        cz.append(float((zz * sl).sum() / s))
        cy.append(float((yy * sl).sum() / s))
    return float(np.hypot(np.std(cy), np.std(cz)))


def _centered_angled_flux(theta_x: float, theta_y: float, sigma: float = 2.0) -> np.ndarray:
    """Angled flux entering at the lateral centre of the x=0 face."""
    yc = (SHAPE_ZYX[1] - 1) / 2.0
    zc = (SHAPE_ZYX[0] - 1) / 2.0
    return flux_projection(
        [yc, zc, 0.0], (theta_x, theta_y), (sigma, sigma), SHAPE_ZYX, spacing=SPACING
    )


def test_zero_angle_is_identity() -> None:
    """Rotating by (0, 0) returns the crop unchanged."""
    flux = _centered_angled_flux(0.0, 0.0)
    rotated, seconds = rotate_beamlet_crop(flux, (0.0, 0.0))
    np.testing.assert_allclose(rotated, flux, atol=1e-4)
    assert seconds >= 0.0


def test_forward_rotation_straightens_angled_flux() -> None:
    """The angled flux ridge becomes axis-aligned after the forward rotation.

    This is the key geometric guarantee of the DoTA reinterpolation: a tilted
    beamlet is made perpendicular to the entrance face. The lateral centroid
    spread must collapse by a large factor versus the un-rotated ridge.
    """
    for theta_x, theta_y in [(2.0, 1.5), (1.0, 0.0), (0.0, 1.5), (-1.5, 2.0)]:
        flux = _centered_angled_flux(theta_x, theta_y)
        base = _depth_centroid_spread(flux)
        rotated, _ = rotate_beamlet_crop(flux, (theta_x, theta_y))
        straightened = _depth_centroid_spread(rotated)
        assert straightened < 0.4, (
            f"ridge not straight for ({theta_x}, {theta_y}): spread={straightened:.3f}"
        )
        if base > 0.3:  # a genuinely tilted beam must improve markedly
            assert straightened < base * 0.4


def test_inverse_round_trips_smooth_volume() -> None:
    """Forward then inverse recovers a smooth volume (validates the back-rotation)."""
    z, y, x = np.meshgrid(
        np.arange(SHAPE_ZYX[0]), np.arange(SHAPE_ZYX[1]), np.arange(SHAPE_ZYX[2]), indexing="ij"
    )
    volume = (
        np.sin(z / SHAPE_ZYX[0] * np.pi)
        * np.cos(y / SHAPE_ZYX[1] * np.pi)
        * np.sin(x / SHAPE_ZYX[2] * 2 * np.pi)
    ).astype(np.float32) + 2.0

    angles = (2.0, 1.5)
    forward, _ = rotate_beamlet_crop(volume, angles)
    restored, _ = rotate_beamlet_crop(forward, angles, inverse=True)

    interior = (slice(3, 27), slice(3, 27), slice(8, 152))
    rel_err = np.linalg.norm(restored[interior] - volume[interior]) / np.linalg.norm(
        volume[interior]
    )
    assert rel_err < 0.01, f"round-trip rel-err too high: {rel_err:.4f}"
