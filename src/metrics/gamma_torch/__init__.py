# Copyright (C) 2026 Medical Physics & Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A torch implementation of the gamma index shell method.

A reimplementation of ``pymedphys._gamma.implementation.shell.gamma_shell``
(Wendling et al. 2007, http://dx.doi.org/10.1118/1.2721657) that keeps both dose
grids on a torch device and never materialises the shell-by-reference coordinate
array the numpy implementation builds. Keyword names and semantics mirror
``pymedphys.gamma`` so the two can be swapped. These files are licensed
Apache-2.0 -- the rest of adota is MIT -- because they are written for
contribution back to PyMedPhys.

**This package imports nothing but the standard library, numpy and torch.** No
project-specific helpers, config objects or logging; its internal imports are
relative rather than ``src.``-absolute, unlike the rest of the repository, so the
directory can be copied upstream unchanged. Anything adota-specific -- tensor
unwrapping, scale dicts, the ``(N, C, D, H, W)`` layout -- belongs in
:mod:`src.metrics.gamma_pass_rate`, not here.

It is a package rather than one file only because the repository enforces a
500-line limit per module. The split is by role, as three parts of one kernel:

* :mod:`~src.metrics.gamma_torch.shells` -- the offsets searched at each radius;
* :mod:`~src.metrics.gamma_torch.interpolation` -- uniform-grid description and
  the fused interpolate-and-reduce step;
* :mod:`~src.metrics.gamma_torch.loop` -- the convergence loop and public API.

Method
------
For every reference voxel at or above the lower dose cutoff, search outward at
radii ``r = 0, delta, 2 delta, ...``. At each radius build a shell of offsets (two
points in 1D, a circle in 2D, a sphere in 3D) spaced no wider than ``delta``,
interpolate the *evaluation* dose at ``reference_coord + offset``, take the
minimum absolute relative dose difference over the shell, and fold::

    gamma_at_r = sqrt( (min_rel_dose_diff / (dose_percent_threshold / 100))**2
                     + (r / distance_mm_threshold)**2 )

into a running per-voxel minimum. A voxel stops searching once its gamma is at or
below ``r / distance_mm_threshold``, since no larger radius can improve it.

Where the CPU implementation slices a ``(shell_points, ref_points, ndim)`` float64
array into RAM-sized chunks -- ~48 GB at r = 3 mm for a clinical 3D case -- this
one tiles over shell points and reference points and folds each tile into a
running ``torch.minimum``. ``min`` is exact and associative for floats, so the
reduction order does not affect the result. The convergence loop itself (12-21
iterations) stays on the host; it is not the bottleneck.

Four details decide whether the numbers match the CPU path, and all four are
deliberate:

1. Out-of-bounds interpolation fills ``+inf``, not NaN. An infinite dose
   difference loses every ``min``, which is the intended behaviour; NaN would
   poison the reduction.
2. Local gamma divides by the reference dose, so a zero reference voxel yields
   ``inf`` or ``nan``. The dose cutoff normally excludes those, but
   ``lower_percent_dose_cutoff=0`` is supported, and the behaviour is matched
   rather than guarded away.
3. Voxel lookup is ``floor((p - x0) / dx)`` on the uniform grid rather than a
   search. At ``p == x[-1]`` that indexes one plane past the end, so both corner
   indices are clamped.
4. The radius schedule snaps to ``distance_mm_threshold`` exactly and grows by
   ``max(r / interp_fraction / max_gamma, delta)``. A different schedule gives
   different numbers.

Known gaps
----------
* Scalar thresholds only. ``pymedphys.gamma`` accepts sequences for
  ``dose_percent_threshold`` / ``distance_mm_threshold`` and answers with a dict
  of gamma arrays; sequence input raises :class:`NotImplementedError` here.
* Both grids must be uniformly spaced along every axis (checked on entry).
"""

from .interpolation import DEFAULT_TILE_ELEMENTS
from .loop import gamma_index_torch_core, gamma_torch
from .shells import calculate_coordinates_shell

__all__ = [
    "gamma_index_torch_core",
    "gamma_torch",
    "calculate_coordinates_shell",
    "DEFAULT_TILE_ELEMENTS",
]
