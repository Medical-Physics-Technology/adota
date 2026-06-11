"""Unit tests for :mod:`src.loaders.plan_parser`.

Covers the happy path, the optional range-shifter blocks (field- and
control-point-level), both ``###FieldsID`` formats, the deliberate
treatment-fractions-vs-file-blocks divergence from the datagenerator parser,
the error paths, and a skip-gated parse of the real Prostate-AEC-001 plan.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from src.loaders.plan_parser import parse_plan

REAL_PLAN = Path(
    "/scratch/mstryja/opentps_plans/"
    "Prostate-AEC-001_Publication_Prostate-AEC-001_100MperBeam/PlanPencil.txt"
)


def _write(tmp_path: Path, text: str) -> str:
    """Write *text* (dedented) to a temp PlanPencil file and return its path."""
    path = tmp_path / "PlanPencil.txt"
    path.write_text(dedent(text).strip() + "\n")
    return str(path)


# A single control point with two spots.
_CP_BLOCK = """\
    ####ControlPointIndex
    0
    ####SpotTunnedID
    0
    ####CumulativeMetersetWeight
    0.0
    ####Energy (MeV)
    100.0
    ####NbOfScannedSpots
    2
    ####X Y Weight
    1.0 2.0 0.5
    3.0 4.0 1.5
"""


def _minimal_plan(n_fractions: int = 1, field_id: int = 1) -> str:
    """Build a minimal one-fraction, one-field plan with one control point.

    The fraction header always declares field id ``1``; ``field_id`` controls
    the id of the field block itself (used to exercise the undeclared-id path).
    """
    return f"""\
        #TREATMENT-PLAN-DESCRIPTION
        #PlanName
        TestPlan
        #NumberOfFractions
        {n_fractions}
        ##FractionID
        1
        ##NumberOfFields
        1
        ###FieldsID
        1
        #TotalMetersetWeightOfAllFields
        100.0
        #FIELD-DESCRIPTION
        ###FieldID
        {field_id}
        ###FinalCumulativeMeterSetWeight
        100.0
        ###GantryAngle
        90.0
        ###PatientSupportAngle
        0.0
        ###IsocenterPosition
        1.0 2.0 3.0
        ###NumberOfControlPoints
        1
        #SPOTS-DESCRIPTION
{_CP_BLOCK}"""


def test_minimal_plan_happy_path(tmp_path: Path) -> None:
    """A minimal plan parses into the expected nested structure."""
    plan = parse_plan(_write(tmp_path, _minimal_plan()))

    assert plan.name == "TestPlan"
    assert plan.total_msw == 100.0
    assert plan.n_treatment_fractions == 1
    assert len(plan.fractions) == 1

    fraction = plan.fractions[0]
    assert fraction.id == 1
    assert fraction.field_ids == [1]
    assert len(fraction.fields) == 1

    fld = fraction.fields[0]
    assert fld.id == 1
    assert fld.final_cumulative_msw == 100.0
    assert fld.gantry_angle == 90.0
    assert fld.couch_angle == 0.0
    assert fld.isocenter == (1.0, 2.0, 3.0)
    assert fld.rs_id == ""
    assert fld.rs_type == ""
    assert len(fld.control_points) == 1

    cp = fld.control_points[0]
    assert cp.index == 0
    assert cp.spot_tuned_id == 0
    assert cp.energy_mev == 100.0
    assert cp.range_shifter_setting == ""
    assert cp.iso_to_rs_distance == 0.0
    assert cp.rs_wet == 0.0
    assert len(cp.spots) == 2
    assert (cp.spots[0].x, cp.spots[0].y, cp.spots[0].weight) == (1.0, 2.0, 0.5)
    assert (cp.spots[1].x, cp.spots[1].y, cp.spots[1].weight) == (3.0, 4.0, 1.5)


def test_treatment_fractions_decoupled_from_file_blocks(tmp_path: Path) -> None:
    """``#NumberOfFractions`` is metadata; the loop follows the actual blocks.

    This pins the deliberate divergence from datagenerator's parser, which
    treats ``#NumberOfFractions`` as the fraction-block loop count and would
    raise here. The repo parser must read ``30`` as treatment fractions while
    finding a single fraction block.
    """
    plan = parse_plan(_write(tmp_path, _minimal_plan(n_fractions=30)))

    assert plan.n_treatment_fractions == 30
    assert len(plan.fractions) == 1
    assert plan.fractions[0].id == 1


def test_field_level_range_shifter(tmp_path: Path) -> None:
    """Optional field-level range-shifter block is captured."""
    text = """\
        #TREATMENT-PLAN-DESCRIPTION
        #PlanName
        RSPlan
        #NumberOfFractions
        1
        ##FractionID
        1
        ##NumberOfFields
        1
        ###FieldsID
        1
        #TotalMetersetWeightOfAllFields
        50.0
        #FIELD-DESCRIPTION
        ###FieldID
        1
        ###FinalCumulativeMeterSetWeight
        50.0
        ###GantryAngle
        0.0
        ###PatientSupportAngle
        0.0
        ###IsocenterPosition
        0.0 0.0 0.0
        ###RangeShifterID
        RS1
        ###RangeShifterType
        binary
        ###NumberOfControlPoints
        1
        #SPOTS-DESCRIPTION
""" + _CP_BLOCK
    plan = parse_plan(_write(tmp_path, text))

    fld = plan.fractions[0].fields[0]
    assert fld.rs_id == "RS1"
    assert fld.rs_type == "binary"


def test_control_point_level_range_shifter(tmp_path: Path) -> None:
    """Optional control-point-level range-shifter block is captured."""
    cp_with_rs = """\
        ####ControlPointIndex
        0
        ####SpotTunnedID
        0
        ####CumulativeMetersetWeight
        0.0
        ####Energy (MeV)
        120.0
        ####RangeShifterSetting
        IN
        ####IsocenterToRangeShifterDistance
        70.0
        ####RangeShifterWaterEquivalentThickness
        40.0
        ####NbOfScannedSpots
        1
        ####X Y Weight
        0.0 0.0 1.0
"""
    text = """\
        #TREATMENT-PLAN-DESCRIPTION
        #PlanName
        CPRSPlan
        #NumberOfFractions
        1
        ##FractionID
        1
        ##NumberOfFields
        1
        ###FieldsID
        1
        #TotalMetersetWeightOfAllFields
        10.0
        #FIELD-DESCRIPTION
        ###FieldID
        1
        ###FinalCumulativeMeterSetWeight
        10.0
        ###GantryAngle
        0.0
        ###PatientSupportAngle
        0.0
        ###IsocenterPosition
        0.0 0.0 0.0
        ###NumberOfControlPoints
        1
        #SPOTS-DESCRIPTION
""" + cp_with_rs
    plan = parse_plan(_write(tmp_path, text))

    cp = plan.fractions[0].fields[0].control_points[0]
    assert cp.range_shifter_setting == "IN"
    assert cp.iso_to_rs_distance == 70.0
    assert cp.rs_wet == 40.0


def test_two_fields_repeated_fieldsid(tmp_path: Path) -> None:
    """Two fields declared via repeated ``###FieldsID`` blocks both attach.

    This is the format the real OpenTPS plans use: a bare ``###FieldsID`` line
    followed by the id, repeated per field. (A pure inline ``###FieldsID N``
    header is not reachable: the fraction loop consumes a bare ``###FieldsID``
    via ``_expect_key`` before delegating to ``_collect_field_ids``.)"""
    second_field = """\
        #FIELD-DESCRIPTION
        ###FieldID
        2
        ###FinalCumulativeMeterSetWeight
        100.0
        ###GantryAngle
        270.0
        ###PatientSupportAngle
        0.0
        ###IsocenterPosition
        0.0 0.0 0.0
        ###NumberOfControlPoints
        1
        #SPOTS-DESCRIPTION
""" + _CP_BLOCK
    text = """\
        #TREATMENT-PLAN-DESCRIPTION
        #PlanName
        TwoFields
        #NumberOfFractions
        1
        ##FractionID
        1
        ##NumberOfFields
        2
        ###FieldsID
        1
        ###FieldsID
        2
        #TotalMetersetWeightOfAllFields
        200.0
        #FIELD-DESCRIPTION
        ###FieldID
        1
        ###FinalCumulativeMeterSetWeight
        100.0
        ###GantryAngle
        90.0
        ###PatientSupportAngle
        0.0
        ###IsocenterPosition
        0.0 0.0 0.0
        ###NumberOfControlPoints
        1
        #SPOTS-DESCRIPTION
""" + _CP_BLOCK + second_field
    plan = parse_plan(_write(tmp_path, text))

    assert plan.fractions[0].field_ids == [1, 2]
    field_ids = sorted(f.id for f in plan.fractions[0].fields)
    assert field_ids == [1, 2]
    gantries = {f.id: f.gantry_angle for f in plan.fractions[0].fields}
    assert gantries == {1: 90.0, 2: 270.0}


def test_bad_first_header_raises(tmp_path: Path) -> None:
    """A wrong leading header is reported as a ValueError."""
    with pytest.raises(ValueError, match="TREATMENT-PLAN-DESCRIPTION"):
        parse_plan(_write(tmp_path, "#WRONG-HEADER\n#PlanName\nX\n"))


def test_undeclared_field_id_raises(tmp_path: Path) -> None:
    """A field whose ID is not declared in any fraction is rejected."""
    text = _minimal_plan(field_id=9)
    with pytest.raises(ValueError, match="without being listed"):
        parse_plan(_write(tmp_path, text))


def test_bad_isocenter_raises(tmp_path: Path) -> None:
    """A malformed isocenter line (not three floats) raises."""
    text = _minimal_plan().replace("1.0 2.0 3.0", "1.0 2.0")
    with pytest.raises(ValueError, match="three floats"):
        parse_plan(_write(tmp_path, text))


@pytest.mark.skipif(not REAL_PLAN.is_file(), reason="real plan not available")
def test_real_prostate_plan() -> None:
    """The real Prostate-AEC-001 plan parses with the documented statistics."""
    plan = parse_plan(str(REAL_PLAN))

    assert plan.n_treatment_fractions == 1
    assert len(plan.fractions) == 1
    assert plan.fractions[0].field_ids == [1, 2]

    fields = plan.fractions[0].fields
    assert {f.id for f in fields} == {1, 2}
    assert {f.gantry_angle for f in fields} == {90.0, -90.0}

    total_spots = sum(
        len(cp.spots) for f in fields for cp in f.control_points
    )
    assert total_spots == 1924
