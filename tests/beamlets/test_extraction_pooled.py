"""``run_extraction`` (serial) vs ``run_extraction_pooled`` (threads) equivalence.

The pooled path is a speed option only: it must produce **byte-identical** per-spot
files (``*_ct.npy``, ``*_flux.npy``, ``*_sim_res.json``) so the serial run stays a
valid reference for comparison. Several spots across two fields and >1 worker
exercise the concurrency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.beamlets.extraction import (
    ExtractionConfig,
    _union_seconds,
    run_extraction,
    run_extraction_pooled,
)


def test_union_seconds() -> None:
    assert _union_seconds([]) == 0.0
    # Disjoint intervals -> sum of lengths (the serial case).
    assert _union_seconds([(0.0, 1.0), (2.0, 3.5)]) == pytest.approx(2.5)
    # Overlapping (concurrent threads) -> merged wall time, not the 4.0 sum.
    assert _union_seconds([(0.0, 2.0), (1.0, 3.0)]) == pytest.approx(3.0)
    # Nested interval contributes nothing extra.
    assert _union_seconds([(0.0, 5.0), (1.0, 2.0)]) == pytest.approx(5.0)
    # Order independence.
    assert _union_seconds([(2.0, 3.0), (0.0, 2.5)]) == pytest.approx(3.0)
from src.loaders.plan_directory import PlanDirectory
from src.loaders.plan_parser import ControlPoint, Field, Fraction, Plan, Spot

ROI = (4, 4, 8)

_BDL = """\
Nozzle exit to Isocenter distance
400.0
SMX to Isocenter distance
2000.0
SMY to Isocenter distance
2500.0
NominalEnergy MeanEnergy EnergySpread ProtonsMU Weight1 SpotSize1x Divergence1x Correlation1x SpotSize1y Divergence1y Correlation1y
100.0 100.0 1.0 1000.0 1.0 4.0 0.003 0.5 3.0 0.004 0.6
150.0 150.0 0.8 1500.0 1.0 3.5 0.003 0.4 2.8 0.004 0.5
"""


def _cp(n_spots: int, energy: float) -> ControlPoint:
    return ControlPoint(
        index=0,
        spot_tuned_id=0,
        cumulative_msw=0.0,
        energy_mev=energy,
        range_shifter_setting="",
        iso_to_rs_distance=0.0,
        rs_wet=0.0,
        spots=[Spot(x=float(i), y=float(-i), weight=float(i + 1)) for i in range(n_spots)],
    )


def _plan() -> Plan:
    iso = (10.0, 10.0, 5.0)
    f1 = Field(1, 0.0, 90.0, 0.0, iso, "", "", [_cp(6, 100.0)])
    f2 = Field(2, 0.0, -90.0, 0.0, iso, "", "", [_cp(6, 150.0)])
    return Plan(name="synth", total_msw=10.0, fractions=[Fraction(1, [1, 2], [f1, f2])])


@pytest.fixture()
def plan_directory(tmp_path: Path) -> PlanDirectory:
    arr = np.random.default_rng(0).uniform(-200, 200, size=(10, 20, 20)).astype(np.float32)
    ct = sitk.GetImageFromArray(arr)
    ct.SetOrigin((-5.0, -5.0, 2.0))
    ct.SetSpacing((1.0, 1.0, 1.0))
    bdl_path = tmp_path / "bdl.txt"
    bdl_path.write_text(_BDL)
    return PlanDirectory(
        plan_dir=tmp_path,
        plan=_plan(),
        ct=ct,
        contours={},
        config={},
        bdl_path=bdl_path,
        bdl_text=_BDL,
        mc_dose_path=None,
    )


def test_pooled_is_byte_identical_to_serial(plan_directory, tmp_path: Path) -> None:
    cfg_kwargs = dict(roi_size=ROI, save_overlays=False)
    serial_dir = tmp_path / "serial"
    pooled_dir = tmp_path / "pooled"

    m_serial = run_extraction(plan_directory, serial_dir, ExtractionConfig(**cfg_kwargs))
    m_pooled = run_extraction_pooled(
        plan_directory, pooled_dir, ExtractionConfig(**cfg_kwargs), workers=4
    )

    assert m_serial["n_spots"] == m_pooled["n_spots"] == 12
    assert m_serial["spot_ids"] == m_pooled["spot_ids"]  # same ids, same order

    for spot_id in m_serial["spot_ids"]:
        ct_s = np.load(serial_dir / f"{spot_id}_ct.npy")
        ct_p = np.load(pooled_dir / f"{spot_id}_ct.npy")
        np.testing.assert_array_equal(ct_s, ct_p)

        flux_s = np.load(serial_dir / f"{spot_id}_flux.npy")
        flux_p = np.load(pooled_dir / f"{spot_id}_flux.npy")
        np.testing.assert_array_equal(flux_s, flux_p)

        meta_s = json.loads((serial_dir / f"{spot_id}_sim_res.json").read_text())
        meta_p = json.loads((pooled_dir / f"{spot_id}_sim_res.json").read_text())
        assert meta_s == meta_p


def test_pooled_auto_workers(plan_directory, tmp_path: Path) -> None:
    """workers=0 (auto) still runs and writes a complete manifest."""
    out = tmp_path / "auto"
    manifest = run_extraction_pooled(
        plan_directory, out, ExtractionConfig(roi_size=ROI, save_overlays=False), workers=0
    )
    assert manifest["n_spots"] == 12
    assert (out / "manifest.json").is_file()
    for spot_id in manifest["spot_ids"]:
        assert (out / f"{spot_id}_ct.npy").is_file()
