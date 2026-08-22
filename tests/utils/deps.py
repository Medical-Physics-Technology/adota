"""Availability probes for optional third-party dependencies.

Flow:
1. A probe actually exercises the dependency (importing it is not enough).
2. The result is cached so the probe runs once per session.
3. Tests apply the exported ``pytest.mark.skipif`` marker.

Keeping the probes here rather than in each test module means one place states
what is missing and how to install it (TESTS.md: skip reasons must be
actionable).
"""

from __future__ import annotations

import functools

import pytest


@functools.lru_cache(maxsize=1)
def pymedphys_gamma_works() -> bool:
    """True when ``pymedphys.gamma`` can actually complete a computation.

    Importing pymedphys is not sufficient: its gamma shell needs the optional
    econforge ``interpolation`` package, and when that is absent pymedphys
    raises ``FileNotFoundError`` while building its own error message (it looks
    for ``dependency-extra.txt`` relative to a source checkout, which does not
    exist in an installed environment). So we run a tiny gamma and see.
    """
    try:
        import numpy as np
        from pymedphys import gamma
    except Exception:
        return False

    axes = (np.arange(4.0), np.arange(4.0), np.arange(4.0))
    dose = np.zeros((4, 4, 4), dtype=np.float64)
    dose[2, 2, 2] = 1.0
    try:
        gamma(
            axes, dose, axes, dose,
            dose_percent_threshold=3,
            distance_mm_threshold=3,
            quiet=True,
        )
    except Exception:
        return False
    return True


requires_pymedphys_gamma = pytest.mark.skipif(
    not pymedphys_gamma_works(),
    reason=(
        "pymedphys gamma is not usable here: it needs the optional econforge "
        "`interpolation` package. Install it (`uv pip install interpolation`) "
        "to enable these tests."
    ),
)
