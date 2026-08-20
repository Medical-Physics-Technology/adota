"""PlanPencil.txt and config.txt writers for MCsquare (faithful datagenerator port).

The ``_Spot`` / ``_Field`` / ``_Plan`` serialization reproduces datagenerator's
``plan.plan`` byte-for-byte, so an MCsquare run driven by this writer is identical
to the one that produced the DoTA training set. ``build_simulation_config`` /
``write_config`` mirror datagenerator's ``get_simulation_config`` /
``generate_config_file``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

PENCIL_PLAN_FILE_NAME = "PlanPencil.txt"
SIMULATION_CONFIG_FILE_NAME = "config.txt"


class _Spot:
    """One scanned spot in a control point (ported from datagenerator.plan.Spot)."""

    def __init__(self, index, spot_id, cumulative_meter_set_weight, energy,
                 scanned_spots, range_shifter=False,
                 isocenter_to_rangeshifter_dist=300.0,
                 rangeshifter_water_equivalent_thickness=0):
        self.index = index
        self.spot_id = spot_id
        self.cumulative_meter_set_weight = cumulative_meter_set_weight
        self.energy = energy
        self.range_shifter = range_shifter
        self.isocenter_to_rangeshifter_dist = isocenter_to_rangeshifter_dist
        self.rangeshifter_wet = rangeshifter_water_equivalent_thickness
        self.scanned_spots = [tuple(map(float, s)) for s in scanned_spots]
        self.header = "#" * 4
        self.keys = {
            "ControlPointIndex": self.index,
            "SpotTunnedID": self.spot_id,
            "CumulativeMetersetWeight": self.cumulative_meter_set_weight,
            "Energy (MeV)": self.energy,
            "RangeShifterSetting": "IN" if self.range_shifter else "OUT",
            "IsocenterToRangeShifterDistance": self.isocenter_to_rangeshifter_dist,
            "RangeShifterWaterEquivalentThickness": self.rangeshifter_wet,
            "NbOfScannedSpots": len(self.scanned_spots),
            "X Y Weight": self.scanned_spots,
        }

    def __repr__(self) -> str:
        o = ""
        for k, v in self.keys.items():
            o += self.header + k + "\n"
            if k == "X Y Weight" and isinstance(v, list):
                for _v in v:
                    o += f"{_v[0]} {_v[1]} {_v[2]}\n"
            else:
                o += str(v) + "\n"
        return o


class _Field:
    """One treatment field (ported from datagenerator.plan.Field)."""

    def __init__(self, field_id, final_cumulative_meter_set_weight, gantry_angle,
                 isocenter_position, spots, patient_support_angle=0,
                 range_shifter_id=None, range_shifter_type=None):
        self.field_id = int(field_id)
        self.final_cmsw = float(final_cumulative_meter_set_weight)
        self.gantry_angle = float(gantry_angle)
        self.patient_support_angle = float(patient_support_angle)
        self.isocenter_position = tuple(map(float, isocenter_position))
        self.range_shifter_id = str(range_shifter_id) if range_shifter_id else None
        self.range_shifter_type = str(range_shifter_type) if range_shifter_type else None
        self.spots = spots
        self.header = "#" * 3
        self.keys = {
            "FieldId": self.field_id,
            "FinalCumulativeMeterSetWeight": self.final_cmsw,
            "GantryAngle": self.gantry_angle,
            "PatientSupportAngle": self.patient_support_angle,
            "IsocenterPosition": self.isocenter_position,
            "RangeShifterID": self.range_shifter_id,
            "RangeShifterType": self.range_shifter_type,
            "NumberOfControlPoints": len(self.spots),
            "SPOTS-DESCRIPTION": self.spots,
        }
        if self.range_shifter_id is None:
            self.keys.pop("RangeShifterID")
        if self.range_shifter_type is None:
            self.keys.pop("RangeShifterType")

    def __repr__(self) -> str:
        o = ""
        for k, v in self.keys.items():
            if k == "SPOTS-DESCRIPTION":
                o += "\n#" + k + "\n"
                for s in self.spots:
                    o += s.__repr__()
            elif k == "IsocenterPosition" and isinstance(v, tuple):
                o += self.header + k + "\n"
                o += f"{v[0]} {v[1]} {v[2]}\n"
            else:
                o += self.header + k + "\n"
                o += str(v) + "\n"
        return o


class _Plan:
    """Full pencil plan (ported from datagenerator.plan.Plan)."""

    def __init__(self, fields, total_meterset_weight_all_fields,
                 plan_name="PlanPencil", fractions=(1,)):
        self.plan_name = plan_name
        self.fraction_ids = list(range(1, len(fractions) + 1))
        self.field_ids = list(range(1, len(fields) + 1))
        self.fields = fields
        self.total_msw = total_meterset_weight_all_fields
        self.keys = {
            "PlanName": (self.plan_name, 1, False),
            "NumberOfFractions": (len(fractions), 1, False),
            "FractionID": (self.fraction_ids, 2, True),
            "NumberOfFields": (len(fields), 2, False),
            "FieldsID": (self.field_ids, 3, True),
            "TotalMetersetWeightOfAllFields": (self.total_msw, 1, False),
            "FIELD-DESCRIPTION": (self.fields, 1, False),
        }

    def __repr__(self) -> str:
        o = "#TREATMENT-PLAN-DESCRIPTION\n"
        for k, (val, weight, repeat) in self.keys.items():
            if repeat:
                for v in val:
                    o += "#" * weight + k + "\n" + str(v) + "\n"
            elif k == "FIELD-DESCRIPTION" and isinstance(val, (list, tuple)):
                for v in val:
                    o += "#" * weight + k + "\n" + str(v) + "\n"
            else:
                o += "#" * weight + k + "\n" + str(val) + "\n"
        return o


def build_single_beamlet_plan_text(
    energy: float,
    spot_xy: Sequence[float],
    gantry_angle: float,
    isocenter: Sequence[float],
    weight: float = 1000.0,
    cumulative_meter_set_weight: float = 1000.0,
    final_cumulative_meter_set_weight: float = 1000.0,
    total_meterset_weight_all_fields: float = 1000.0,
) -> str:
    """Return the exact PlanPencil.txt text for a single-spot, single-field plan."""
    spot = _Spot(
        index=1, spot_id=1,
        cumulative_meter_set_weight=cumulative_meter_set_weight,
        energy=float(energy),
        scanned_spots=[(float(spot_xy[0]), float(spot_xy[1]), float(weight))],
    )
    field = _Field(
        field_id=1,
        final_cumulative_meter_set_weight=final_cumulative_meter_set_weight,
        gantry_angle=float(gantry_angle),
        isocenter_position=isocenter,
        spots=[spot],
    )
    plan = _Plan(fields=[field],
                 total_meterset_weight_all_fields=total_meterset_weight_all_fields)
    return repr(plan)


def write_plan_pencil(path, energy, spot_xy, gantry_angle, isocenter, **kwargs) -> Path:
    """Write PlanPencil.txt for a single beamlet; return the path."""
    text = build_single_beamlet_plan_text(energy, spot_xy, gantry_angle, isocenter, **kwargs)
    path = Path(path)
    path.write_text(text)
    return path


_DEFAULT_SIM_PARAMS = {
    "Num_Threads": 0, "RNG_Seed": 0, "Num_Primaries": 1e7,
    "E_Cut_Pro": 0.5, "D_Max": 0.2, "Epsilon_Max": 0.25, "Te_Min": 0.05,
    "Energy_MHD_Output": False,
}


def build_simulation_config(
    ct_file: str,
    pencil_plan_path: str,
    bdl_file_path: str,
    output_dir: str,
    sim_params: dict | None = None,
    scanner: str = "default",
    compute_uncertainty: bool = True,
) -> dict:
    """Build the sectioned MCsquare config dict (mirrors get_simulation_config)."""
    p = {**_DEFAULT_SIM_PARAMS, **(sim_params or {})}
    import os
    return {
        "simulation_parameters": {
            "Num_Threads": p["Num_Threads"], "RNG_Seed": p["RNG_Seed"],
            "Num_Primaries": p["Num_Primaries"], "E_Cut_Pro": p["E_Cut_Pro"],
            "D_Max": p["D_Max"], "Epsilon_Max": p["Epsilon_Max"], "Te_Min": p["Te_Min"],
        },
        "input_parameters": {
            "CT_File": ct_file,
            "HU_Density_Conversion_File": f"Scanners/{scanner}/HU_Density_Conversion.txt",
            "HU_Material_Conversion_File": f"Scanners/{scanner}/HU_Material_Conversion.txt",
            "BDL_Machine_Parameter_File": f"BDL/{os.path.basename(bdl_file_path)}",
            "BDL_Plan_File": pencil_plan_path,
        },
        "physical_params": {
            "Simulate_Nuclear_Interactions": True,
            "Simulate_Secondary_Protons": True,
            "Simulate_Secondary_Deuterons": True,
            "Simulate_Secondary_Alphas": True,
        },
        "statistical": {
            "Compute_stat_uncertainty": compute_uncertainty,
            "Stat_uncertainty": 0.0, "Ignore_low_density_voxels": False,
            "Export_batch_dose": False, "Max_Num_Primaries": 0, "Max_Simulation_time": 0,
        },
        "output_parameters": {
            "Output_Directory": output_dir,
            "Energy_MHD_Output": p["Energy_MHD_Output"],
            "Dose_MHD_Output": True,
        },
    }


def write_config(path, config_dict: dict) -> Path:
    """Serialize the sectioned config dict to config.txt (mirrors generate_config_file)."""
    o = "######################\n# Configuration file #\n######################\n\n"
    for section_name, section in config_dict.items():
        o += f"### {section_name}\n"
        for key, value in section.items():
            if isinstance(value, (int, float)):
                val = value
            elif isinstance(value, (tuple, list)):
                val = "".join(str(v) + " " for v in value)
            else:
                val = str(value)
            o += f"{key}\t{val}\n"
        o += "\n\n"
    path = Path(path)
    path.write_text(o)
    return path
