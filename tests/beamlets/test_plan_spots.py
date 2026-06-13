"""Unit tests for :mod:`src.beamlets.plan_spots`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.beamlets.plan_spots import (
    adjusted_gantry_angle,
    expand_plan_to_spots,
    group_by_field,
    spot_id,
)
from src.loaders.plan_parser import (
    ControlPoint,
    Field,
    Fraction,
    Plan,
    Spot,
    parse_plan,
)

REAL_PLAN = Path(
    "/scratch/mstryja/opentps_plans/"
    "Prostate-AEC-001_Publication_Prostate-AEC-001_100MperBeam/PlanPencil.txt"
)


def _control_point(index: int, n_spots: int, energy: float) -> ControlPoint:
    return ControlPoint(
        index=index,
        spot_tuned_id=index,
        cumulative_msw=0.0,
        energy_mev=energy,
        range_shifter_setting="",
        iso_to_rs_distance=0.0,
        rs_wet=0.0,
        spots=[Spot(x=float(i), y=float(-i), weight=float(i + 1)) for i in range(n_spots)],
    )


def _synthetic_plan() -> Plan:
    field1 = Field(
        id=1,
        final_cumulative_msw=0.0,
        gantry_angle=90.0,
        couch_angle=0.0,
        isocenter=(10.0, 20.0, 30.0),
        rs_id="",
        rs_type="",
        control_points=[_control_point(0, 2, 100.0), _control_point(1, 3, 120.0)],
    )
    field2 = Field(
        id=2,
        final_cumulative_msw=0.0,
        gantry_angle=-90.0,
        couch_angle=0.0,
        isocenter=(10.0, 20.0, 30.0),
        rs_id="",
        rs_type="",
        control_points=[_control_point(0, 1, 150.0)],
    )
    fraction = Fraction(id=1, field_ids=[1, 2], fields=[field1, field2])
    return Plan(name="synth", total_msw=12.0, fractions=[fraction])


def test_adjusted_gantry_angle() -> None:
    assert adjusted_gantry_angle(90.0) == 0.0
    assert adjusted_gantry_angle(-90.0) == 180.0
    assert adjusted_gantry_angle(0.0) == 90.0


def test_spot_id_format() -> None:
    assert spot_id(0, 0, 0) == "b00_l000_s0000"
    assert spot_id(1, 12, 345) == "b01_l012_s0345"


def test_expand_counts_and_fields() -> None:
    spots = expand_plan_to_spots(_synthetic_plan())
    assert len(spots) == 2 + 3 + 1  # field1 cp0:2, cp1:3, field2 cp0:1
    # First field's spots are gantry 0, second field's are gantry 180.
    field0 = [s for s in spots if s["beam"] == 0]
    field1 = [s for s in spots if s["beam"] == 1]
    assert len(field0) == 5 and len(field1) == 1
    assert all(s["simulation_log"]["gantry_angle"] == 0.0 for s in field0)
    assert all(s["simulation_log"]["gantry_angle"] == 180.0 for s in field1)


def test_expand_ids_unique_and_stable() -> None:
    plan = _synthetic_plan()
    first = expand_plan_to_spots(plan)
    second = expand_plan_to_spots(plan)
    ids = [s["id"] for s in first]
    assert len(set(ids)) == len(ids)
    assert ids == [s["id"] for s in second]
    assert first[0]["id"] == "b00_l000_s0000"


def test_relative_weight() -> None:
    spots = expand_plan_to_spots(_synthetic_plan())
    for s in spots:
        sl = s["simulation_log"]
        assert sl["relative_weight"] == pytest.approx(sl["weight"] / 12.0)


def test_group_by_field() -> None:
    grouped = group_by_field(expand_plan_to_spots(_synthetic_plan()))
    assert list(grouped.keys()) == [0, 1]
    assert len(grouped[0]) == 5
    assert len(grouped[1]) == 1


@pytest.mark.skipif(not REAL_PLAN.is_file(), reason="real plan not available")
def test_expand_real_plan() -> None:
    spots = expand_plan_to_spots(parse_plan(str(REAL_PLAN)))
    assert len(spots) == 1924
    grouped = group_by_field(spots)
    assert len(grouped) == 2  # two fields
    assert len({s["id"] for s in spots}) == 1924
    gantries = {s["simulation_log"]["gantry_angle"] for s in spots}
    assert gantries == {0.0, 180.0}  # 90 -> 0, -90 -> 180
