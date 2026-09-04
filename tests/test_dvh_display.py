"""Unit tests for the DVH display helpers (labels, colours, robust axis)."""
import numpy as np
import pytest

from src.figures.dvh_comparison import (
    _STRUCTURE_COLORS,
    _format_structure_label,
    _robust_dose_upper,
)


@pytest.mark.parametrize("raw, expected", [
    ("Femur_Head_L", "Femur Head L"),
    ("Femur_Head_R", "Femur Head R"),
    ("Spinal-Cord", "Spinal Cord"),
    ("Lung-Right", "Lung Right"),
    ("Lungs-Total", "Lungs (Total)"),
    ("Target", "Target"),
    ("Esophagus", "Esophagus"),
])
def test_format_structure_label(raw, expected):
    assert _format_structure_label(raw) == expected


def test_formatted_labels_have_no_separators():
    for raw in ("Femur_Head_L", "Lungs-Total", "Spinal-Cord"):
        out = _format_structure_label(raw)
        assert "_" not in out and "-" not in out


def test_color_map_keyed_by_formatted_label():
    # every canonical structure resolves to a colour via its formatted label
    for raw in ("Femur_Head_L", "Lungs-Total", "Spinal-Cord", "Lung-Right", "target"):
        label = _format_structure_label(raw)
        # "target" formats to "Target"; all mapped names must be present
        if label != "Target" or "Target" in _STRUCTURE_COLORS:
            assert label in _STRUCTURE_COLORS


def test_colors_distinct_within_each_anatomy():
    for group in (
        ["Target", "Bladder", "Rectum", "Femur Head L", "Femur Head R"],
        ["Target", "Spinal Cord", "Lung Right", "Lung Left", "Lungs (Total)", "Esophagus", "Heart"],
    ):
        cols = [_STRUCTURE_COLORS[n] for n in group]
        assert len(set(cols)) == len(cols), f"colour clash within {group}"


def test_robust_dose_upper_ignores_single_outlier():
    # a realistic structure: thousands of voxels near 60 Gy, one hot outlier at 175
    dose = np.full(10000, 60.0)
    dose[0] = 175.0
    mask = np.ones(10000, dtype=bool)
    upper = _robust_dose_upper({"s": mask}, dose, dose)
    # robust upper stays near the bulk (~63 Gy), well below the 175 Gy outlier
    assert upper < 100.0


def test_robust_dose_upper_empty_structs_falls_back_to_max():
    dose = np.full((3, 3, 3), 42.0)
    assert _robust_dose_upper({}, dose, dose) == pytest.approx(1.05 * 42.0)
