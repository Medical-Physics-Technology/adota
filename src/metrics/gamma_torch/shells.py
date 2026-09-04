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

"""The offsets searched at each radius of the gamma shell method.

A direct port of ``pymedphys._utilities.createshells``. The offsets stay in numpy
float64 and on the host: a shell holds at most a few thousand points, building it
on the device would cost more than it saves, and computing it exactly as the CPU
path does keeps the two searching the same geometry.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["calculate_coordinates_shell"]


def calculate_coordinates_shell(
    distance: float, num_dimensions: int, distance_step_size: float
) -> Tuple[np.ndarray, ...]:
    """Offsets at ``distance`` from the origin, spaced no wider than the step.

    Args:
        distance: Shell radius, in the coordinates' units.
        num_dimensions: 1, 2 or 3.
        distance_step_size: Maximum spacing between neighbouring shell points.

    Returns:
        One 1-D float64 array of offsets per dimension. At ``distance == 0`` the
        shell is the single point at the origin.

    Raises:
        ValueError: If ``num_dimensions`` is not 1, 2 or 3.
    """
    if num_dimensions == 1:
        return _shell_1d(distance)
    if num_dimensions == 2:
        return _shell_2d(distance, distance_step_size)
    if num_dimensions == 3:
        return _shell_3d(distance, distance_step_size)
    raise ValueError("No valid dimension")


def _shell_1d(distance: float) -> Tuple[np.ndarray]:
    """The two points at the given distance in one dimension."""
    if distance == 0:
        return (np.array([0]),)
    return (np.array([distance, -distance]),)


def _shell_2d(distance: float, distance_step_size: float) -> Tuple[np.ndarray, ...]:
    """Points around a circle, no farther apart than ``distance_step_size``."""
    amount_to_check = np.ceil(2 * np.pi * distance / distance_step_size).astype(int) + 1
    theta = np.linspace(0, 2 * np.pi, amount_to_check + 1)[:-1:]
    return (distance * np.cos(theta), distance * np.sin(theta))


def _shell_3d(distance: float, distance_step_size: float) -> Tuple[np.ndarray, ...]:
    """Points over a sphere, no gap wider than ``distance_step_size``.

    Rows of constant elevation, each populated with as many azimuths as its own
    circumference needs, so the spacing holds near the poles as well as the
    equator.
    """
    number_of_rows = np.ceil(np.pi * distance / distance_step_size).astype(int) + 1
    elevation = np.linspace(0, np.pi, number_of_rows)
    # Kept as two statements, associated exactly as the CPU implementation does.
    # Folding them into `2 * pi * distance * sin(elevation)` moves the product by
    # an ulp, which is enough to shift a `ceil` and change the point count in a
    # row -- a different shell, and different gamma values.
    row_radii = distance * np.sin(elevation)
    row_circumference = 2 * np.pi * row_radii
    amount_in_row = np.ceil(row_circumference / distance_step_size).astype(int) + 1

    x_coords: List[np.ndarray] = []
    y_coords: List[np.ndarray] = []
    z_coords: List[np.ndarray] = []
    for i, phi in enumerate(elevation):
        azimuth = np.linspace(0, 2 * np.pi, amount_in_row[i] + 1)[:-1:]
        x_coords.append(distance * np.sin(phi) * np.cos(azimuth))
        y_coords.append(distance * np.sin(phi) * np.sin(azimuth))
        z_coords.append(distance * np.cos(phi) * np.ones_like(azimuth))

    return (np.hstack(x_coords), np.hstack(y_coords), np.hstack(z_coords))
