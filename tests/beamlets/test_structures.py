"""Tests for :mod:`src.beamlets.structures` (mask orientation auto-detect)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.beamlets.structures import (
    apply_flip,
    detect_target_flip,
    isocenter_index_zyx,
    load_oriented_structures,
)
from src.loaders.plan_directory import PlanDirectory
from src.loaders.plan_parser import Field, Fraction, Plan


def _blob(shape, center, r=1) -> np.ndarray:
    m = np.zeros(shape, bool)
    z, y, x = center
    m[z - r : z + r + 1, y - r : y + r + 1, x - r : x + r + 1] = True
    return m


def _img(arr) -> sitk.Image:
    return sitk.GetImageFromArray(arr.astype(np.uint8))


def test_apply_flip() -> None:
    a = np.arange(8).reshape(2, 2, 2)
    np.testing.assert_array_equal(apply_flip(a, (False, True, False)), a[:, ::-1, :])
    np.testing.assert_array_equal(apply_flip(a, (True, False, True)), a[::-1, :, ::-1])


def test_detect_target_flip_y() -> None:
    # Target at y=3; isocenter at y=7 in an 11-voxel axis -> flip-y centres it.
    target = _blob((11, 11, 11), (5, 3, 5))
    flips, dist = detect_target_flip(target, (5, 7, 5))
    assert flips == (False, True, False)
    assert dist == pytest.approx(0.0, abs=0.5)


def test_isocenter_index_zyx() -> None:
    fld = Field(1, 0.0, 90.0, 0.0, (5.0, 7.0, 9.0), "", "", [])
    plan = Plan("t", 1.0, [Fraction(1, [1], [fld])])
    assert isocenter_index_zyx(plan) == (9.0, 7.0, 5.0)  # (z, y, x) from (x, y, z)


def test_load_oriented_structures_applies_detected_flip(tmp_path: Path) -> None:
    shape = (11, 11, 11)
    target = _blob(shape, (5, 3, 5))  # needs flip-y to reach iso y=7
    oar = _blob(shape, (5, 3, 8))
    contours = {"target": _img(target), "OAR_1": _img(oar)}
    fld = Field(1, 0.0, 90.0, 0.0, (5.0, 7.0, 5.0), "", "", [])
    plan = Plan("t", 1.0, [Fraction(1, [1], [fld])])
    pd = PlanDirectory(
        plan_dir=tmp_path, plan=plan, ct=_img(np.zeros(shape)), contours=contours,
        config={}, bdl_path=None, bdl_text="", mc_dose_path=None,
    )

    oriented, flips = load_oriented_structures(pd)
    assert flips == (False, True, False)
    # Oriented target centroid lands on the isocenter (z, y, x) = (5, 7, 5).
    centroid = np.argwhere(oriented["target"]).mean(axis=0)
    np.testing.assert_allclose(centroid, [5, 7, 5], atol=0.5)
    # The same flip is applied to the OAR.
    assert "OAR_1" in oriented


def test_no_target_raises(tmp_path: Path) -> None:
    contours = {"OAR_1": _img(_blob((11, 11, 11), (5, 5, 5)))}
    fld = Field(1, 0.0, 90.0, 0.0, (5.0, 5.0, 5.0), "", "", [])
    plan = Plan("t", 1.0, [Fraction(1, [1], [fld])])
    pd = PlanDirectory(
        plan_dir=tmp_path, plan=plan, ct=_img(np.zeros((11, 11, 11))), contours=contours,
        config={}, bdl_path=None, bdl_text="", mc_dose_path=None,
    )
    with pytest.raises(ValueError, match="No target contour"):
        load_oriented_structures(pd)
