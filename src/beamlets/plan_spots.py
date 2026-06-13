"""Expand a parsed plan into per-spot extraction records.

This is the single source of truth for the spot loop that was duplicated in the
notebook and ``run_model_hptc.py``. Each spot becomes a dict carrying the
``simulation_log`` the downstream extraction/inference consumes, plus a
**deterministic id** ``b{beam:02d}_l{layer:03d}_s{spot:04d}`` (decision 7) so
reruns are idempotent and records match references by name.

The gantry angle is stored already adjusted to the extraction frame by
:func:`adjusted_gantry_angle`; ``relative_weight`` is the spot weight divided by
the plan's total meterset weight.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, List

from src.loaders.plan_parser import Plan

logger = logging.getLogger(__name__)

__all__ = ["spot_id", "adjusted_gantry_angle", "expand_plan_to_spots", "group_by_field"]


def spot_id(beam: int, layer: int, spot: int) -> str:
    """Return the deterministic spot id ``b{beam:02d}_l{layer:03d}_s{spot:04d}``."""
    return f"b{beam:02d}_l{layer:03d}_s{spot:04d}"


def adjusted_gantry_angle(gantry_angle: float) -> float:
    """Rotation (deg) that brings a field's beam into the canonical +x BEV frame.

    The extraction crop takes depth as ``x in [0, 320)`` and ``beamlet_ray`` is
    fixed at ``g_ang = -90`` (beam along +x), so each field's CT must be rotated
    to align its gantry beam with +x. The magnitude/offset ``(ga - 90)`` is tied
    to that fixed ``-90``; it is structural and unaffected by the rotation
    backend.

    The leading ``-1`` sign is the notebook's value, which was calibrated to the
    old torch rotation. Our SimpleITK rotation has its own (pinned,
    counter-clockwise-positive) sign convention, so this sign is **pending
    validation on an oblique-gantry plan** -- the Prostate-AEC-001 fields are
    ``90 -> 0`` and ``-90 -> 180``, both sign-invariant, so they cannot confirm
    it. Flip the sign here if an oblique plan's overlay shows the beam entering
    the wrong face.

    Args:
        gantry_angle: The plan gantry angle in degrees.

    Returns:
        The adjusted rotation angle in degrees.
    """
    return (-1.0) * (gantry_angle - 90.0)


def expand_plan_to_spots(plan: Plan) -> List[dict]:
    """Expand every spot in the plan into an extraction record.

    Args:
        plan: The parsed plan.

    Returns:
        A list of per-spot dicts. Each has ``id``, ``beam``/``layer``/``spot``
        indices, ``field_id``, and a ``simulation_log`` with ``gantry_angle``
        (adjusted), ``isocenter``, ``energy``, ``bixelgrid_shifts_xy``,
        ``weight`` and ``relative_weight``.
    """
    total_msw = plan.total_msw
    records: List[dict] = []

    beam = 0
    for fraction in plan.fractions:
        for field in fraction.fields:
            for layer, control_point in enumerate(field.control_points):
                for spot, spot_obj in enumerate(control_point.spots):
                    records.append(
                        {
                            "id": spot_id(beam, layer, spot),
                            "beam": beam,
                            "layer": layer,
                            "spot": spot,
                            "field_id": field.id,
                            "simulation_log": {
                                "gantry_angle": adjusted_gantry_angle(
                                    field.gantry_angle
                                ),
                                "isocenter": list(field.isocenter),
                                "energy": [control_point.energy_mev],
                                "bixelgrid_shifts_xy": [[spot_obj.x, spot_obj.y]],
                                "weight": spot_obj.weight,
                                "relative_weight": spot_obj.weight / total_msw,
                            },
                        }
                    )
            beam += 1

    logger.info("Expanded plan into %d spots across %d field(s)", len(records), beam)
    return records


def group_by_field(spots: List[dict]) -> Dict[int, List[dict]]:
    """Group spot records by their ``beam`` (field) index, preserving order.

    Rotation is performed once per field, so this is how the extraction iterates.

    Args:
        spots: Spot records from :func:`expand_plan_to_spots`.

    Returns:
        An ordered mapping ``beam_index -> [spot records]``.
    """
    grouped: "OrderedDict[int, List[dict]]" = OrderedDict()
    for record in spots:
        grouped.setdefault(record["beam"], []).append(record)
    return grouped
