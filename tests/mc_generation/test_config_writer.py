"""Numerically meaningful tests for the MCsquare PlanPencil/config writers.

The key test is byte-parity against datagenerator's own serialization (the code
that produced the DoTA training set): identical inputs must yield an identical
PlanPencil.txt, so an MCsquare run from this port is the same run. The remaining
tests parse the written files back and assert the actual physical values
(energy, spot x/y/weight, gantry, isocenter, primaries, seed) are correct.
"""
import os
import re
import sys

import pytest

from src.mc_generation.config_writer import (
    build_simulation_config,
    build_single_beamlet_plan_text,
    write_config,
)

# Every test here compares against `datagenerator`, the external OpenTPS-side
# package that produced the DoTA training set. It is not on PyPI and lives
# outside this repository, so the whole module skips when it is absent.
DATAGENERATOR_ROOT = os.environ.get(
    "ADOTA_DATAGENERATOR_ROOT", "/home/mstryja/projects/datagenerator"
)
if DATAGENERATOR_ROOT not in sys.path:
    sys.path.insert(0, DATAGENERATOR_ROOT)

pytest.importorskip(
    "datagenerator",
    reason=(
        "the external `datagenerator` repository is required for MCsquare byte-parity; "
        f"clone it and set ADOTA_DATAGENERATOR_ROOT (currently {DATAGENERATOR_ROOT!r})"
    ),
)


def _dg_plan_text(tmp_path, energy, spot_xy, gantry, isocenter, weight=1000.0):
    """datagenerator's own PlanPencil.txt for the same beamlet (ground truth)."""
    from datagenerator.simulation.sim_config import get_single_beamlet_plan
    handle = tmp_path / "dg"
    handle.mkdir()
    get_single_beamlet_plan(
        energy=float(energy),
        bixelgrid_shifts_xy=[(float(spot_xy[0]), float(spot_xy[1]))],
        isocenter=list(map(float, isocenter)),
        gantry_angle=float(gantry),
        weights=[float(weight)],
        handle_results_path=str(handle),
        n_spots=1,
    )
    return (handle / "PlanPencil.txt").read_text()


@pytest.mark.parametrize("energy,spot_xy,gantry,iso", [
    (140.0, (0.0, 0.0), 90.0, [184.0, 184.0, 147.0]),
    (90.0, (3.5, -2.25), 47.0, [200.0, 150.0, 100.0]),
    (200.0, (-1.75, 5.18), 300.0, [10.5, 20.25, 30.75]),
])
def test_plan_text_byte_parity_with_datagenerator(tmp_path, energy, spot_xy, gantry, iso):
    ours = build_single_beamlet_plan_text(energy, spot_xy, gantry, iso, weight=1000.0)
    theirs = _dg_plan_text(tmp_path, energy, spot_xy, gantry, iso, weight=1000.0)
    assert ours == theirs


def test_plan_values_roundtrip():
    # Parse the written plan and confirm the physical numbers land correctly.
    energy, spot_xy, gantry, iso, weight = 137.2, (2.5, -3.0), 63.0, [120.0, 130.0, 140.0], 1000.0
    text = build_single_beamlet_plan_text(energy, spot_xy, gantry, iso, weight=weight)

    def after(label):
        # value on the line following a "#...label" header line
        lines = text.splitlines()
        for i, ln in enumerate(lines):
            if ln.lstrip("#") == label:
                return lines[i + 1]
        raise AssertionError(f"label {label!r} not found")

    assert float(after("Energy (MeV)")) == pytest.approx(energy)
    assert float(after("GantryAngle")) == pytest.approx(gantry)
    ix, iy, iz = map(float, after("IsocenterPosition").split())
    assert (ix, iy, iz) == pytest.approx(tuple(iso))
    x, y, w = map(float, after("X Y Weight").split())
    assert (x, y, w) == pytest.approx((spot_xy[0], spot_xy[1], weight))
    assert int(after("NbOfScannedSpots")) == 1
    assert int(after("NumberOfControlPoints")) == 1


def test_config_values_and_paths(tmp_path):
    sim = {"Num_Primaries": 5e6, "RNG_Seed": 7, "Num_Threads": 4, "Energy_MHD_Output": True}
    cfg = build_simulation_config(
        ct_file="CT.mhd", pencil_plan_path="PlanPencil.txt",
        bdl_file_path="/some/where/hptc_beam_model_rsnone.txt",
        output_dir="Outputs", sim_params=sim, scanner="default",
    )
    text = write_config(tmp_path / "config.txt", cfg).read_text()
    kv = dict(re.findall(r"^(\S+)\t(.+)$", text, flags=re.M))

    assert float(kv["Num_Primaries"]) == pytest.approx(5e6)
    assert int(kv["RNG_Seed"]) == 7
    assert int(kv["Num_Threads"]) == 4
    assert kv["CT_File"] == "CT.mhd"
    assert kv["BDL_Plan_File"] == "PlanPencil.txt"
    # BDL machine file references only the basename under BDL/ (engine-relative).
    assert kv["BDL_Machine_Parameter_File"] == "BDL/hptc_beam_model_rsnone.txt"
    assert kv["HU_Density_Conversion_File"] == "Scanners/default/HU_Density_Conversion.txt"
    assert kv["Dose_MHD_Output"] == "True"
    assert kv["Compute_stat_uncertainty"] == "True"


def test_config_serializer_parity_with_datagenerator(tmp_path):
    # The config serializer must match datagenerator's generate_config_file for the
    # same section dict (so MCsquare parses the identical file).
    from datagenerator.plan.config import generate_config_file
    cfg = build_simulation_config(
        ct_file="CT.mhd", pencil_plan_path="PlanPencil.txt",
        bdl_file_path="/x/hptc_beam_model_rsnone.txt", output_dir="Outputs",
        sim_params={"Num_Primaries": 1e7, "RNG_Seed": 0}, scanner="default",
    )
    ours = write_config(tmp_path / "ours.txt", cfg).read_text()
    theirs = generate_config_file(cfg, str(tmp_path / "theirs.txt"))
    assert ours == theirs
