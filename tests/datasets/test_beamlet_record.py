"""Tests for :mod:`src.datasets.beamlet_record` (spec ``docs/dev/h5_v3_spec.md`` 6.1).

Synthetic data only, ``device="cpu"`` only; no host BDL, no `/RadiotherapyData`,
no `/scratch`. The float64-spacing/dtype tests are the regression that pins the
v2-reproduction recipe (spec section 2.7): they paste a verbatim copy of the old
``flux_projection`` (float64 spacing explicit) as the ground truth and prove both
that the module reproduces it and that the float32 shortcuts it must not take
would visibly disagree.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import src.datasets.beamlet_record as beamlet_record_module
from src.adota.config import DEFAULT_SCALE, denormalize_energy
from src.beamlets.bdl import BeamDataLibrary
from src.beamlets.flux import flux_projection_gpu_batched
from src.datasets.beamlet_record import (
    FLUX_SPACING_MM,
    FluxInputs,
    downsample_grid,
    finish_record,
    flux_batch,
    flux_inputs,
    list_record_ids,
    load_raw_record,
    preprocess_flux,
    skip_reason,
    spots_reason,
)
from src.utils.scallers import inverse_minmax
from tests.utils.bdl import build_bdl_text
from tests.utils.beamlet_records import make_metadata, write_synthetic_record


def _bdl(tmp_path_factory) -> BeamDataLibrary:
    bdl_dir = tmp_path_factory.mktemp("bdl")
    bdl_path = bdl_dir / "bdl.txt"
    bdl_path.write_text(build_bdl_text())
    return BeamDataLibrary.from_file(bdl_path)


@pytest.fixture
def bdl(tmp_path_factory) -> BeamDataLibrary:
    return _bdl(tmp_path_factory)


# --- Round trip ---------------------------------------------------------------


def test_round_trip_shapes_and_values(tmp_path, bdl) -> None:
    write_synthetic_record(tmp_path, "sample-1", shape=(80, 80, 400), seed=1)
    raw = load_raw_record(tmp_path, "sample-1")
    finfo = flux_inputs(raw, bdl)
    flux_f64 = flux_batch([finfo], device="cpu")[0]
    pre = finish_record(raw, flux_f64, DEFAULT_SCALE, finfo.sigmas_xy)

    assert pre.ct.dtype == np.float32 and pre.ct.shape == (40, 40, 200)
    assert pre.dose.dtype == np.float32 and pre.dose.shape == (40, 40, 200)
    assert pre.flux.dtype == np.float32 and pre.flux.shape == (40, 40, 200)

    ct_pooled_raw = downsample_grid(raw.ct.astype(np.float64), "average")
    dose_pooled_raw = downsample_grid(raw.ds.astype(np.float64), "average")

    ct_recovered = inverse_minmax(pre.ct.astype(np.float64), DEFAULT_SCALE["min_ct"], DEFAULT_SCALE["max_ct"])
    dose_recovered = inverse_minmax(pre.dose.astype(np.float64), DEFAULT_SCALE["min_ds"], DEFAULT_SCALE["max_ds"])

    # float32-tolerance round trip: normalise-then-pool commutes with pool-then-denormalise
    # for an affine map only up to the float32 cast baked into downsample_grid.
    np.testing.assert_allclose(ct_recovered, ct_pooled_raw, rtol=0.0, atol=1.0)
    np.testing.assert_allclose(dose_recovered, dose_pooled_raw, rtol=1e-5, atol=1.0)

    energy_recovered = denormalize_energy(pre.initial_energy_norm, DEFAULT_SCALE)
    assert energy_recovered == pytest.approx(pre.energy_mev, rel=1e-9)


# --- The float64 regression -----------------------------------------------------


def _reference_flux_projection(
    beamlet_entrence,
    beamlet_direction,
    sigmas_xy,
    shape,
    initial_energy=None,
    spacing=np.asarray([1, 1, 1], dtype=np.float32),
):
    """Verbatim copy of ``src.beamlets.flux.flux_projection`` (lines 68-126).

    Kept byte-for-byte so this test pins the old NumPy numerics independently of
    the production module; do not "clean up" this copy.
    """
    R_x = lambda theta: np.array(  # noqa: E731
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    R_y = lambda theta: np.array(  # noqa: E731
        [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    )

    x_0, y_0, z_0 = beamlet_entrence
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    z = np.arange(0, shape[2], 1)
    [xx, yy, zz] = np.meshgrid(x, y, z)

    theta_x_deg, theta_y_deg = beamlet_direction
    theta_x = theta_x_deg / 180 * np.pi
    theta_y = theta_y_deg / 180 * np.pi

    [x_t, y_t, z_t] = R_y(theta_x) @ R_x(theta_y) @ np.array(
        [xx.flatten() - x_0, yy.flatten() - y_0, zz.flatten() - z_0]
    )
    x_t = x_t.reshape(xx.shape)
    y_t = y_t.reshape(yy.shape)
    z_t = z_t.reshape(zz.shape)

    sigma_x = sigmas_xy[0] / spacing[0]
    sigma_y = sigmas_xy[1] / spacing[1]
    coef = 1 / (2 * np.pi * sigma_x * sigma_y)
    flux = coef * np.exp(-(x_t**2) / 2 / (sigma_x**2) - y_t**2 / 2 / (sigma_y**2))

    if initial_energy is not None:
        flux = flux * initial_energy

    return flux


_F64_SPACING = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

# (entrance, direction, sigmas) -- the second case uses real BDL sigma values with
# many significant digits so the float32 demotion is visible (negative control 2).
_CASES_400 = [
    ((30.0, 30.0, 0.0), (0.0, 0.0), (4.0, 3.0)),
    ((28.5, 31.2, 1.0), (5.0, -3.0), (3.8107859986, 3.4413068387)),
    ((10.0, 50.0, 2.0), (12.0, 6.5), (2.5, 4.5)),
]
_CASE_372 = ((20.0, 45.0, 3.0), (-4.0, 8.0), (3.2, 2.8))


@pytest.mark.parametrize("entrance,direction,sigmas", _CASES_400)
def test_flux_batch_matches_float64_reference_depth_400(entrance, direction, sigmas) -> None:
    shape = (80, 80, 400)
    ref = _reference_flux_projection(entrance, direction, sigmas, shape, spacing=_F64_SPACING)
    ref_pooled = downsample_grid(ref, "average")

    inputs = [
        FluxInputs(
            sample_id="s", entrance_proj=entrance, beamlet_angles=direction, sigmas_xy=sigmas,
            shape=shape, energy_mev=100.0,
        )
    ]
    flux_f64 = flux_batch(inputs, device="cpu")[0]
    got_pooled = preprocess_flux(flux_f64)

    assert np.array_equal(got_pooled, ref_pooled)


def test_flux_batch_matches_float64_reference_depth_372() -> None:
    shape = (80, 80, 372)
    entrance, direction, sigmas = _CASE_372
    ref = _reference_flux_projection(entrance, direction, sigmas, shape, spacing=_F64_SPACING)
    ref_pooled = downsample_grid(ref, "average")

    inputs = [
        FluxInputs(
            sample_id="s", entrance_proj=entrance, beamlet_angles=direction, sigmas_xy=sigmas,
            shape=shape, energy_mev=100.0,
        )
    ]
    flux_f64 = flux_batch(inputs, device="cpu")[0]
    got_pooled = preprocess_flux(flux_f64)

    assert np.array_equal(got_pooled, ref_pooled)


def test_negative_control_float32_dtype_batched_disagrees() -> None:
    """The float32 dtype shortcut in the batched path is visibly wrong (not used)."""
    shape = (80, 80, 400)
    entrance, direction, sigmas = _CASES_400[1]
    ref = _reference_flux_projection(entrance, direction, sigmas, shape, spacing=_F64_SPACING)
    ref_pooled = downsample_grid(ref, "average")

    f32 = flux_projection_gpu_batched(
        [entrance], [direction], [sigmas], shape, initial_energies=None,
        spacing=FLUX_SPACING_MM, device="cpu", dtype=torch.float32, return_numpy=True,
    )[0]
    f32_pooled = downsample_grid(f32, "average")

    assert not np.array_equal(f32_pooled, ref_pooled)


def test_negative_control_numpy_default_float32_spacing_disagrees() -> None:
    """The NumPy path's default float32 ``spacing`` silently demotes the sigmas."""
    shape = (80, 80, 400)
    entrance, direction, sigmas = _CASES_400[1]
    ref64 = _reference_flux_projection(entrance, direction, sigmas, shape, spacing=_F64_SPACING)
    ref64_pooled = downsample_grid(ref64, "average")

    ref32_default = _reference_flux_projection(entrance, direction, sigmas, shape)
    ref32_default_pooled = downsample_grid(ref32_default, "average")

    assert not np.array_equal(ref32_default_pooled, ref64_pooled)


# --- Shape grouping --------------------------------------------------------------


def test_flux_batch_groups_by_shape_and_preserves_order(monkeypatch) -> None:
    calls = []
    real = beamlet_record_module.flux_projection_gpu_batched

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(beamlet_record_module, "flux_projection_gpu_batched", counting)

    shapes = [(80, 80, 400), (80, 80, 372), (80, 80, 400), (80, 80, 372)]
    inputs = [
        FluxInputs(
            sample_id=f"s{i}", entrance_proj=(30.0, 30.0, 0.0), beamlet_angles=(0.0, 0.0),
            sigmas_xy=(4.0, 3.0), shape=shape, energy_mev=100.0,
        )
        for i, shape in enumerate(shapes)
    ]

    results = beamlet_record_module.flux_batch(inputs, device="cpu")

    assert len(calls) == 2
    assert [r.shape for r in results] == list(shapes)


# --- downsample_grid --------------------------------------------------------------


def test_downsample_grid_average_bad_lateral_shape() -> None:
    grid = np.zeros((78, 80, 400), dtype=np.float64)
    pooled = downsample_grid(grid, "average")
    assert pooled.shape == (39, 40, 200)
    assert pooled.dtype == np.float32
    dose_pooled = np.ones((39, 40, 200), dtype=np.float32)
    assert skip_reason(pooled, dose_pooled) == "bad_shape"


def test_downsample_grid_linear_and_trilinear() -> None:
    grid = np.random.default_rng(0).random((80, 80, 400))
    for method in ("linear", "trilinear"):
        pooled = downsample_grid(grid, method)
        assert pooled.shape == (40, 40, 200)
        assert pooled.dtype == np.float32


def test_downsample_grid_unknown_method_raises() -> None:
    grid = np.zeros((80, 80, 400), dtype=np.float64)
    with pytest.raises(ValueError):
        downsample_grid(grid, "cubic")


# --- skip_reason / spots_reason ----------------------------------------------------


def test_skip_reason_order_and_values() -> None:
    good_ct = np.zeros((40, 40, 200), dtype=np.float32)
    good_dose = np.ones((40, 40, 200), dtype=np.float32)
    assert skip_reason(good_ct, good_dose) is None

    zero_dose = np.zeros((40, 40, 200), dtype=np.float32)
    assert skip_reason(good_ct, zero_dose) == "zero_dose"

    short_ct = np.zeros((40, 40, 150), dtype=np.float32)
    short_dose = np.ones((40, 40, 150), dtype=np.float32)
    assert skip_reason(short_ct, short_dose) == "short_depth"


def test_spots_reason() -> None:
    assert spots_reason(make_metadata("s", n_spots=1)) is None
    assert spots_reason(make_metadata("s", n_spots=2)) == "n_spots"


# --- list_record_ids / load_raw_record ----------------------------------------------


def test_list_record_ids_sorted_and_ignores_stray_file(tmp_path) -> None:
    write_synthetic_record(tmp_path, "b-id", shape=(80, 80, 160))
    write_synthetic_record(tmp_path, "a-id", shape=(80, 80, 160))
    (tmp_path / "faulty_ids.txt").write_text("something\n")
    assert list_record_ids(tmp_path) == ["a-id", "b-id"]


def test_load_raw_record_missing_ds_raises(tmp_path) -> None:
    write_synthetic_record(tmp_path, "sample", shape=(80, 80, 160))
    (tmp_path / "sample_ds.npy").unlink()
    with pytest.raises(FileNotFoundError):
        load_raw_record(tmp_path, "sample")


def test_load_raw_record_missing_energy_key_raises_keyerror(tmp_path) -> None:
    write_synthetic_record(tmp_path, "sample", shape=(80, 80, 160))
    meta_path = tmp_path / "sample_metadata.json"
    metadata = json.loads(meta_path.read_text())
    del metadata["simulation_log"]["energy"]
    meta_path.write_text(json.dumps(metadata))

    with pytest.raises(KeyError) as excinfo:
        load_raw_record(tmp_path, "sample")
    assert "simulation_log.energy" in str(excinfo.value)
