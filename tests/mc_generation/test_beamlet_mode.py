"""Fast (no-MC) tests for the beamlet-mode plan and config writers."""
import pytest

from src.mc_generation.config_writer import (
    build_beamlet_field_plan_text,
    build_simulation_config,
    build_single_beamlet_plan_text,
)


def _kv(text, key):
    """Value on the line after the ``#...key`` header."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip("#") == key:
            return lines[i + 1]
    raise KeyError(key)


def test_beamlet_field_plan_holds_every_spot_in_one_control_point():
    """Spot order in the plan is the ``_Beamlet_0_0_d`` index MCsquare exports."""
    spots = [(-70.4, 90.2), (0.0, 0.0), (70.4, -90.2)]
    text = build_beamlet_field_plan_text(102.6, spots, 90.0, [10.0, 20.0, 30.0])
    assert _kv(text, "NumberOfFields") == "1"
    assert _kv(text, "NumberOfControlPoints") == "1"     # one layer -> d is the list index
    assert _kv(text, "NbOfScannedSpots") == "3"
    assert _kv(text, "Energy (MeV)") == "102.6"
    assert _kv(text, "GantryAngle") == "90.0"
    assert _kv(text, "IsocenterPosition") == "10.0 20.0 30.0"
    body = text.splitlines()
    start = body.index("####X Y Weight") + 1
    assert body[start:start + 3] == ["-70.4 90.2 1000.0", "0.0 0.0 1000.0",
                                     "70.4 -90.2 1000.0"]


def test_single_spot_field_plan_matches_the_single_beamlet_writer():
    """A one-spot field plan is the plan run_beamlet already writes -- same scale."""
    one = build_single_beamlet_plan_text(135.0, (12.0, -8.0), 90.0, [1.0, 2.0, 3.0])
    many = build_beamlet_field_plan_text(135.0, [(12.0, -8.0)], 90.0, [1.0, 2.0, 3.0])
    assert one == many


def test_beamlet_mode_section_is_opt_in():
    """The single-beamlet config is untouched; beamlet mode adds one section."""
    plain = build_simulation_config("CT.mhd", "PlanPencil.txt", "bdl.txt", "Outputs")
    assert "beamlet_simulation" not in plain

    beamlet = build_simulation_config("CT.mhd", "PlanPencil.txt", "bdl.txt", "Outputs",
                                      beamlet_mode=True)
    assert beamlet["beamlet_simulation"] == {"Beamlet_Mode": True,
                                             "Beamlet_Parallelization": True}
    # everything else is identical to the plain config
    assert {k: v for k, v in beamlet.items() if k != "beamlet_simulation"} == plain

    serial = build_simulation_config("CT.mhd", "PlanPencil.txt", "bdl.txt", "Outputs",
                                     beamlet_mode=True, beamlet_parallelization=False)
    assert serial["beamlet_simulation"]["Beamlet_Parallelization"] is False


def test_num_primaries_is_per_spot_in_the_written_config():
    """Beamlet mode runs Num_Primaries for each spot, so the value is per beamlet."""
    cfg = build_simulation_config("CT.mhd", "PlanPencil.txt", "bdl.txt", "Outputs",
                                  sim_params={"Num_Primaries": 1e7}, beamlet_mode=True)
    assert cfg["simulation_parameters"]["Num_Primaries"] == pytest.approx(1e7)


def test_comparison_figure_writes_all_formats_and_tolerates_gaps(tmp_path):
    """The three-panel comparison renders with blank cells and one shared scale."""
    import numpy as np

    from src.figures.beamlet_mode_comparison import beamlet_mode_comparison_figure

    grid = np.array([[99.0, np.nan, 97.5], [98.0, 96.0, np.nan], [95.0, 99.9, 94.0]])
    panels = {"MC beamlet vs MC sequential": grid,
              "ADoTA vs MC sequential": grid - 3.0,
              "ADoTA vs MC beamlet": grid - 2.5}
    paths = beamlet_mode_comparison_figure(panels, [-2.0, 0.0, 2.0],
                                           str(tmp_path / "cmp"), 90.0, 100.0)
    assert {p.suffix for p in paths} == {".svg", ".pdf", ".png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_comparison_figure_rejects_an_unusable_scale(tmp_path):
    import numpy as np

    from src.figures.beamlet_mode_comparison import beamlet_mode_comparison_figure

    with pytest.raises(ValueError, match="no panels"):
        beamlet_mode_comparison_figure({}, [-2.0, 2.0], str(tmp_path / "x"), 0.0, 1.0)
    with pytest.raises(ValueError, match="colour scale"):
        beamlet_mode_comparison_figure({"a": np.zeros((2, 2))}, [-2.0, 2.0],
                                       str(tmp_path / "y"), float("nan"), 1.0)
