"""Unit tests for :mod:`src.loaders.plan_directory`.

Builds a synthetic OpenTPS plan directory (small SimpleITK CT + contours,
minimal plan, fake BDL/config/reference dose) and checks loading, config
parsing, BDL path resolution, the missing-file error path, and the preview.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import SimpleITK as sitk

from src.loaders.plan_directory import (
    load_plan_directory,
    parse_opentps_config,
)

_MINIMAL_PLAN = """\
    #TREATMENT-PLAN-DESCRIPTION
    #PlanName
    SynthPlan
    #NumberOfFractions
    1
    ##FractionID
    1
    ##NumberOfFields
    1
    ###FieldsID
    1
    #TotalMetersetWeightOfAllFields
    1.0
    #FIELD-DESCRIPTION
    ###FieldID
    1
    ###FinalCumulativeMeterSetWeight
    1.0
    ###GantryAngle
    90.0
    ###PatientSupportAngle
    0.0
    ###IsocenterPosition
    0.0 0.0 0.0
    ###NumberOfControlPoints
    1
    #SPOTS-DESCRIPTION
    ####ControlPointIndex
    0
    ####SpotTunnedID
    0
    ####CumulativeMetersetWeight
    0.0
    ####Energy (MeV)
    100.0
    ####NbOfScannedSpots
    1
    ####X Y Weight
    0.0 0.0 1.0
"""


def _write_image(
    path: Path,
    size: tuple[int, int, int],
    pixel_id: int,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> None:
    """Write a small SimpleITK image with non-trivial geometry."""
    image = sitk.Image(size, pixel_id)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))


@pytest.fixture()
def plan_dir(tmp_path: Path) -> Path:
    """Create a synthetic plan directory and return its path."""
    origin = (-232.5, -91.5, -1243.5)
    spacing = (1.0, 1.0, 2.0)

    _write_image(tmp_path / "CT.mhd", (4, 4, 3), sitk.sitkFloat32, origin, spacing)
    _write_image(tmp_path / "Dose.mhd", (4, 4, 3), sitk.sitkFloat32, origin, spacing)
    _write_image(tmp_path / "target.mhd", (4, 4, 3), sitk.sitkUInt8, origin, spacing)
    _write_image(tmp_path / "OAR_1.mhd", (4, 4, 3), sitk.sitkUInt8, origin, spacing)

    (tmp_path / "PlanPencil.txt").write_text(dedent(_MINIMAL_PLAN).strip() + "\n")
    (tmp_path / "bdl.txt").write_text(
        "--UPenn beam model--\nNozzle exit to Isocenter distance\n420.0\n"
    )
    (tmp_path / "config.txt").write_text(
        "# MCsquare config\nNum_Primaries 200000000  # inline comment\n"
        "Energy_Cut 1.0\nVerbose\n"
    )
    return tmp_path


def test_load_plan_directory(plan_dir: Path) -> None:
    """All inputs load with the expected structure and geometry."""
    pd = load_plan_directory(plan_dir)

    assert pd.plan.name == "SynthPlan"
    assert pd.ct.GetSize() == (4, 4, 3)
    assert pd.ct.GetSpacing() == (1.0, 1.0, 2.0)

    # Contours are every .mhd except CT and the reference dose.
    assert set(pd.contours) == {"target", "OAR_1"}

    assert pd.bdl_path == plan_dir / "bdl.txt"
    assert "UPenn" in pd.bdl_text
    assert pd.mc_dose_path == plan_dir / "Dose.mhd"

    assert pd.config["Num_Primaries"] == "200000000"
    assert pd.config["Verbose"] is None


def test_explicit_bdl_path_overrides_plan_local(plan_dir: Path, tmp_path: Path) -> None:
    """An explicit BDL path is used instead of the plan-local bdl.txt."""
    external = tmp_path / "external_bdl.txt"
    external.write_text("--external beam model--\n")

    pd = load_plan_directory(plan_dir, bdl_path=external)

    assert pd.bdl_path == external
    assert "external" in pd.bdl_text


def test_missing_required_file_raises(plan_dir: Path) -> None:
    """Removing a required file raises a clear FileNotFoundError."""
    (plan_dir / "PlanPencil.txt").unlink()
    with pytest.raises(FileNotFoundError, match="PlanPencil.txt"):
        load_plan_directory(plan_dir)


def test_missing_directory_raises(tmp_path: Path) -> None:
    """A non-existent plan directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Plan directory not found"):
        load_plan_directory(tmp_path / "does_not_exist")


def test_summary_contains_key_sections(plan_dir: Path) -> None:
    """The preview string surfaces the main loaded sections."""
    summary = load_plan_directory(plan_dir).summary()

    assert "CT grid (SimpleITK):" in summary
    assert "Contours (2):" in summary
    assert "Beam data library:" in summary
    assert "Parsed plan:" in summary
    assert "SynthPlan" in summary


def test_parse_opentps_config_inline_comments(tmp_path: Path) -> None:
    """Config parsing strips comments and maps bare keys to None."""
    cfg = tmp_path / "config.txt"
    cfg.write_text(
        "# whole-line comment\nKey1 value1\nKey2 value2 # trailing\nBareKey\n\n"
    )
    parsed = parse_opentps_config(cfg)

    assert parsed == {"Key1": "value1", "Key2": "value2", "BareKey": None}
