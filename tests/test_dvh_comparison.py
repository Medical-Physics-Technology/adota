"""Tests for :mod:`src.figures.dvh_comparison` (figure + JSON metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.figures.dvh_comparison import (
    compute_structure_dvhs,
    dvh_comparison_figure,
    dvh_metrics,
    write_dvh_metrics_json,
)


def _scene():
    shape = (8, 12, 12)
    target = np.zeros(shape, bool)
    target[3:5, 5:8, 5:8] = True
    oar = np.zeros(shape, bool)
    oar[3:5, 2:4, 2:4] = True
    structures = {"target": target, "OAR_1": oar}
    dose_b = np.zeros(shape)  # MCsquare reference
    dose_b[target] = 60.0
    dose_b[oar] = 10.0
    dose_a = dose_b * 0.95  # ADoTA, 5% low
    return structures, dose_a, dose_b


def test_compute_structure_dvhs() -> None:
    structures, dose_a, dose_b = _scene()
    dvhs = compute_structure_dvhs(structures, dose_a, dose_b, (1.0, 1.0, 1.0))
    assert set(dvhs) == {"target", "OAR_1"}
    dvh_a_target, dvh_b_target = dvhs["target"]
    assert dvh_b_target.Dmax == 60.0
    assert dvh_a_target.Dmax == 57.0  # 0.95 * 60


def test_figure_writes_only_images(tmp_path: Path) -> None:
    structures, dose_a, dose_b = _scene()
    paths = dvh_comparison_figure(
        structures, dose_a, dose_b, (1.0, 1.0, 1.0), str(tmp_path / "dvh"), dpi=80
    )
    assert {p.suffix for p in paths} == {".svg", ".pdf", ".png"}
    # The figure no longer writes any metrics file.
    assert not (tmp_path / "dvh_metrics.csv").exists()


def test_metrics_are_structure_type_dependent() -> None:
    structures, dose_a, dose_b = _scene()
    report = dvh_metrics(structures, dose_a, dose_b, (1.0, 1.0, 1.0))
    assert report["units"] == "Gy"
    assert report["doses"] == ["ADoTA", "MCsquare"]
    target = report["structures"]["target"]
    oar = report["structures"]["OAR_1"]
    assert target["type"] == "target" and oar["type"] == "OAR"
    # Target carries D95/D98; OARs do not.
    assert set(target["ADoTA"]) == {"Dmin", "Dmean", "Dmax", "D95", "D98"}
    assert set(oar["ADoTA"]) == {"Dmin", "Dmean", "Dmax"}
    # Difference = ADoTA - MCsquare.
    assert target["difference"]["Dmax"] == round(57.0 - 60.0, 4)


def test_write_metrics_json(tmp_path: Path) -> None:
    structures, dose_a, dose_b = _scene()
    out = tmp_path / "dvh_metrics.json"
    write_dvh_metrics_json(out, structures, dose_a, dose_b, (1.0, 1.0, 1.0))
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert "D95" in loaded["metric_definitions"]
    assert loaded["structures"]["target"]["n_voxels"] == int(structures["target"].sum())
