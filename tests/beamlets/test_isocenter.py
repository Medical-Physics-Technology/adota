"""Tests for :mod:`src.beamlets.isocenter` (plan -> CT x-flip)."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from src.beamlets.isocenter import (
    isocenter_index_zyx,
    isocenter_physical,
    plan_isocenter_index_ct,
)
from src.loaders.plan_parser import Field, Fraction, Plan


def _ct(size_xyz=(21, 13, 9), origin=(-10.0, -20.0, 5.0), spacing=(1.0, 1.0, 2.0)):
    nx, ny, nz = size_xyz
    img = sitk.GetImageFromArray(np.zeros((nz, ny, nx), dtype=np.float32))
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    return img


def test_plan_isocenter_index_ct_flips_only_x() -> None:
    assert plan_isocenter_index_ct((5.0, 7.0, 3.0), nx=21) == (15.0, 7.0, 3.0)
    # y and z untouched, x mirrored about the grid centre.
    assert plan_isocenter_index_ct((0.0, 1.0, 2.0), nx=10) == (9.0, 1.0, 2.0)


def test_isocenter_physical_matches_flipped_index() -> None:
    ct = _ct()
    plan_iso = (5.0, 7.0, 3.0)
    phys = isocenter_physical(plan_iso, ct)
    # Must equal the physical point of the x-flipped index.
    expected = ct.TransformContinuousIndexToPhysicalPoint([21 - 1 - 5.0, 7.0, 3.0])
    assert phys == pytest.approx(expected)


def test_isocenter_index_zyx() -> None:
    fld = Field(1, 0.0, 90.0, 0.0, (5.0, 7.0, 3.0), "", "", [])
    plan = Plan("t", 1.0, [Fraction(1, [1], [fld])])
    assert isocenter_index_zyx(plan, _ct()) == (3.0, 7.0, 15.0)  # (z, y, x), x flipped
