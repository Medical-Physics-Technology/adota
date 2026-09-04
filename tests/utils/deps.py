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

    Importing pymedphys is not sufficient: its gamma shell interpolates through
    an optional extra -- ``numba`` from 0.41 onwards, the econforge
    ``interpolation`` package up to 0.40 -- and when that is absent the failure
    surfaces only at the first interpolation, sometimes as a ``FileNotFoundError``
    raised while pymedphys builds its own error message. So we run a tiny gamma
    and see.
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
        "pymedphys gamma is not usable here: it needs its optional interpolation "
        "extra (`numba` for pymedphys >= 0.41). Run `uv sync` to install it and "
        "enable these tests."
    ),
)
