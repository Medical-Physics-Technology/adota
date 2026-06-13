"""Shared fixtures for the beamlet tests.

Provides access to the external ``datagenerator`` repository for A/B parity
tests (skip-gated when it is not present on disk) and a small synthetic SimpleITK
phantom builder with non-trivial origin/spacing for geometric ground-truth tests.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import pytest
import SimpleITK as sitk

DATAGENERATOR_ROOT = Path("/home/mstryja/projects/datagenerator")


def _import_datagenerator_module(dotted: str) -> Optional[ModuleType]:
    """Import a datagenerator submodule, or return None if unavailable."""
    if not DATAGENERATOR_ROOT.is_dir():
        return None
    if str(DATAGENERATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(DATAGENERATOR_ROOT))
    try:
        return importlib.import_module(dotted)
    except Exception:  # pragma: no cover - environment dependent
        return None


@pytest.fixture(scope="session")
def datagenerator_geometry() -> ModuleType:
    """The datagenerator geometry module, or skip if it cannot be imported."""
    module = _import_datagenerator_module(
        "datagenerator.geometry.geometry_spatial_operations"
    )
    if module is None:
        pytest.skip("datagenerator repository not available for A/B parity")
    return module


@pytest.fixture(scope="session")
def datagenerator_utils() -> ModuleType:
    """The datagenerator dataset-generation utils module, or skip."""
    module = _import_datagenerator_module(
        "datagenerator.utils.dataset_generation_utils"
    )
    if module is None:
        pytest.skip("datagenerator repository not available for A/B parity")
    return module


@pytest.fixture()
def make_phantom():
    """Return a builder for synthetic sitk phantoms with given geometry.

    The builder signature is
    ``build(size_xyz, origin, spacing, fill=-1024) -> sitk.Image`` and returns
    an image whose voxels are all ``fill`` (float32), with the requested origin
    and spacing and an identity direction.
    """

    def build(
        size_xyz: tuple[int, int, int],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        fill: float = -1024.0,
    ) -> sitk.Image:
        nx, ny, nz = size_xyz
        arr = np.full((nz, ny, nx), fill, dtype=np.float32)  # (z, y, x)
        image = sitk.GetImageFromArray(arr)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)
        return image

    return build
