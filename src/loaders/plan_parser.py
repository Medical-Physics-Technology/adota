"""Parser for PlanPencil treatment plan files."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

__all__ = [
    "Spot",
    "ControlPoint",
    "Field",
    "Fraction",
    "Plan",
    "parse_plan",
]

_NUM = r"[-+]?\d+(?:\.\d+)?"
_RE_XYZ = re.compile(rf"^{_NUM}\s+{_NUM}\s+{_NUM}$")
_RE_INT = re.compile(r"\d+")


@dataclass
class Spot:
    """Represents a single spot in a control point."""

    x: float
    y: float
    weight: float


@dataclass
class ControlPoint:
    """Represents a control point in a treatment field."""

    index: int
    spot_tuned_id: int
    cumulative_msw: float
    energy_mev: float
    range_shifter_setting: str
    iso_to_rs_distance: float
    rs_wet: float
    spots: List[Spot] = field(default_factory=list)


@dataclass
class Field:
    """Represents a treatment field."""

    id: int
    final_cumulative_msw: float
    gantry_angle: float
    couch_angle: float
    isocenter: Tuple[float, float, float]
    rs_id: Optional[str]
    rs_type: Optional[str]
    control_points: List[ControlPoint] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a readable string representation of the field."""
        o = f"Field ID: {self.id}\n"
        o += f"Final cumulative MSW: {self.final_cumulative_msw}\n"
        o += f"Gantry angle: {self.gantry_angle}\n"
        o += f"Couch angle: {self.couch_angle}\n"
        o += f"Isocenter: {self.isocenter}\n"
        o += f"Range shifter ID: {self.rs_id}\n"
        o += f"Range shifter type: {self.rs_type}\n"
        o += f"Control points: {len(self.control_points)}\n"
        for cp in self.control_points:
            o += f"\tControl point index: {cp.index}\n"
            o += f"\tSpot tuned ID: {cp.spot_tuned_id}\n"
            o += f"\tCumulative MSW: {cp.cumulative_msw}\n"
            o += f"\tEnergy (MeV): {cp.energy_mev}\n"
            o += f"\tRange shifter setting: {cp.range_shifter_setting}\n"
            o += f"\tIsocenter to range shifter distance: {cp.iso_to_rs_distance}\n"
            o += f"\tRange shifter water equivalent thickness: {cp.rs_wet}\n"
            o += f"\tSpots: {len(cp.spots)}\n"
            for spot in cp.spots:
                o += f"\t\tSpot x: {spot.x}\n"
                o += f"\t\tSpot y: {spot.y}\n"
                o += f"\t\tSpot weight: {spot.weight}\n"
        return o


@dataclass
class Fraction:
    """Represents a treatment fraction."""

    id: int
    field_ids: List[int]  # IDs declared in the header
    fields: List[Field] = field(default_factory=list)


@dataclass
class Plan:
    """Represents a complete treatment plan."""

    name: str
    total_msw: float
    fractions: List[Fraction] = field(default_factory=list)
    n_treatment_fractions: int = 1  # Number of treatment fractions (not file blocks)


def parse_plan(path: str) -> Plan:
    """Parse a PlanPencil text dump into a Plan object.

    Args:
        path: Path to the PlanPencil.txt file.

    Returns:
        Plan object containing all parsed data.

    Raises:
        ValueError: If the file format is invalid.
    """
    lines = (ln.strip() for ln in pathlib.Path(path).read_text().splitlines())
    # drop blank lines early
    line_iter = (ln for ln in lines if ln)

    # 3-A. Treatment-plan header
    _expect(line_iter, "#TREATMENT-PLAN-DESCRIPTION")

    _expect_key(line_iter, "#PlanName")
    plan_name = next(line_iter)

    _expect_key(line_iter, "#NumberOfFractions")
    n_fractions = int(
        next(line_iter)
    )  # This is the number of treatment fractions, not file blocks

    # Parse fraction definitions - read until we hit #TotalMetersetWeightOfAllFields
    fractions: List[Fraction] = []

    # Peek at next lines to determine how many fraction blocks exist
    line = next(line_iter)
    while line.startswith("##FractionID"):
        # We already consumed "##FractionID", so read the ID value
        fraction_id = int(next(line_iter))

        _expect_key(line_iter, "##NumberOfFields")
        n_fields = int(next(line_iter))

        # First FieldsID key is expected
        _expect_key(line_iter, "###FieldsID")
        field_ids = _collect_field_ids(line_iter, n_fields)

        fractions.append(Fraction(id=fraction_id, field_ids=field_ids))

        # Read next line to check if there's another fraction or we hit TotalMetersetWeight
        line = next(line_iter)

    # At this point, line should be "#TotalMetersetWeightOfAllFields"
    if line != "#TotalMetersetWeightOfAllFields":
        raise ValueError(f"Expected '#TotalMetersetWeightOfAllFields', got {line!r}")
    total_msw = float(next(line_iter))

    plan = Plan(name=plan_name, total_msw=total_msw, fractions=fractions)
    # Store the number of treatment fractions as metadata
    plan.n_treatment_fractions = n_fractions

    # 3-B. Field and spot descriptions
    remaining_fields = sum(len(fr.field_ids) for fr in plan.fractions)

    while remaining_fields:
        try:
            line = next(line_iter)
        except StopIteration as exc:
            raise ValueError(
                "Unexpected end-of-file – incomplete FIELD-DESCRIPTION block."
            ) from exc

        if line != "#FIELD-DESCRIPTION":
            # Ignore any extra headers or comments we don't care about
            continue

        parsed_field = _parse_field(line_iter)

        # Attach the field to the fraction that declared its ID
        for fr in plan.fractions:
            if parsed_field.id in fr.field_ids:
                fr.fields.append(parsed_field)
                remaining_fields -= 1
                break
        else:  # no break → ID not announced in any fraction
            raise ValueError(
                f"Field ID {parsed_field.id} appears without being listed "
                "under any ##NumberOfFields section."
            )

    return plan


def _parse_field(it: Iterator[str]) -> Field:
    """Parse a single field from the iterator."""
    _expect_key(it, "###FieldID")
    fid = int(next(it))

    _expect_key(it, "###FinalCumulativeMeterSetWeight")
    final_cum_msw = float(next(it))

    _expect_key(it, "###GantryAngle")
    gantry = float(next(it))

    _expect_key(it, "###PatientSupportAngle")
    couch = float(next(it))

    _expect_key(it, "###IsocenterPosition")
    iso = _parse_xyz(next(it))

    # Optional fields: ###RangeShifterID and ###RangeShifterType
    rs_id = None
    rs_type = None

    # Peek at the next line without consuming it
    try:
        next_line = next(it)
    except StopIteration:
        raise ValueError(
            "Unexpected EOF while checking for RangeShifterID or NumberOfControlPoints"
        )

    if next_line == "###RangeShifterID":
        rs_id = next(it)  # read value
        _expect_key(it, "###RangeShifterType")
        rs_type = next(it)
        next_line = next(it)  # Now read ###NumberOfControlPoints

    if next_line != "###NumberOfControlPoints":
        raise ValueError(f"Expected '###NumberOfControlPoints', got {next_line!r}")
    n_cps = int(next(it))

    _expect(it, "#SPOTS-DESCRIPTION")

    control_points: List[ControlPoint] = []
    for _ in range(n_cps):
        control_points.append(_parse_control_point(it))

    return Field(
        id=fid,
        final_cumulative_msw=final_cum_msw,
        gantry_angle=gantry,
        couch_angle=couch,
        isocenter=iso,
        rs_id=rs_id if rs_id else "",
        rs_type=rs_type if rs_type else "",
        control_points=control_points,
    )


def _parse_control_point(it: Iterator[str]) -> ControlPoint:
    """Parse a single control point from the iterator."""
    _expect_key(it, "####ControlPointIndex")
    idx = int(next(it))

    _expect_key(it, "####SpotTunnedID")
    spot_tid = int(next(it))

    _expect_key(it, "####CumulativeMetersetWeight")
    cum_msw = float(next(it))

    _expect_key(it, "####Energy (MeV)")
    energy = float(next(it))

    # Initialize RS-related fields as optional
    rs_setting = ""
    iso_to_rs = 0.0
    rs_wet = 0.0

    # Optional: ####RangeShifterSetting
    try:
        line = next(it)
    except StopIteration:
        raise ValueError("Unexpected EOF before NbOfScannedSpots")

    if line == "####RangeShifterSetting":
        rs_setting = next(it)
        _expect_key(it, "####IsocenterToRangeShifterDistance")
        iso_to_rs = float(next(it))
        _expect_key(it, "####RangeShifterWaterEquivalentThickness")
        rs_wet = float(next(it))
        line = next(it)  # Proceed to next expected key

    if line != "####NbOfScannedSpots":
        raise ValueError(f"Expected key '####NbOfScannedSpots', got {line!r}")
    n_spots = int(next(it))

    _expect_key(it, "####X Y Weight")
    spots: List[Spot] = [Spot(*map(float, next(it).split())) for _ in range(n_spots)]

    return ControlPoint(
        index=idx,
        spot_tuned_id=spot_tid,
        cumulative_msw=cum_msw,
        energy_mev=energy,
        range_shifter_setting=rs_setting,
        iso_to_rs_distance=iso_to_rs,
        rs_wet=rs_wet,
        spots=spots,
    )


def _expect(it: Iterator[str], expected: str) -> None:
    """Expect a specific line from the iterator."""
    line = next(it)
    if line != expected:
        raise ValueError(f"Expected {expected!r}, got {line!r}")


def _expect_key(it: Iterator[str], expected_key: str) -> None:
    """Expect a specific key line from the iterator."""
    line = next(it)
    if line != expected_key:
        raise ValueError(f"Expected key {expected_key!r}, got {line!r}")


def _parse_xyz(line: str) -> Tuple[float, float, float]:
    """Parse a line containing three floats (X Y Z coordinates)."""
    if not _RE_XYZ.fullmatch(line):
        raise ValueError(f"Expected three floats, got {line!r}")
    return tuple(map(float, line.split()))  # type: ignore


def _collect_field_ids(it: Iterator[str], how_many: int) -> List[int]:
    """Collect field IDs from the iterator.

    Return exactly *how_many* integer IDs, tolerant of two formats:

        ###FieldsID
        1
        2
        3

    or

        ###FieldsID 1
        ###FieldsID 2
        ###FieldsID 3
    """
    ids: List[int] = []
    while len(ids) < how_many:
        line = next(it)

        # Strip the header keyword if present
        if line.startswith("###FieldsID"):
            # Might be "###FieldsID 7" or just "###FieldsID"
            maybe_num = line.split(maxsplit=1)[1:]  # returns [] or ["7"]
            if maybe_num:
                ids.append(int(maybe_num[0]))
            continue

        m = _RE_INT.search(line)
        if m:
            ids.append(int(m.group()))
            continue

        raise ValueError(f"Expected a field ID, got {line!r}")
    return ids
