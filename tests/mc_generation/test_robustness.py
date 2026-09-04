"""Fast (no-MC) tests for the robustness orchestration logic."""
import numpy as np
import pytest

from src.mc_generation.sweep import (
    RobustnessConfig,
    build_angle_grid,
    resolve_gantry,
)


def test_angle_grid_shape_indices_and_values():
    grid = build_angle_grid((-2.0, 2.0), (-2.0, 2.0), 18)
    assert len(grid) == 18 * 18
    ixs = {g[0] for g in grid}
    iys = {g[1] for g in grid}
    assert ixs == set(range(18)) and iys == set(range(18))
    # endpoints of the sweep are exactly the requested range
    txs = sorted({g[2] for g in grid})
    assert txs[0] == pytest.approx(-2.0) and txs[-1] == pytest.approx(2.0)
    # row-major over theta_x: first 18 entries share ix=0 and span all theta_y
    assert all(g[0] == 0 for g in grid[:18])
    assert sorted(g[3] for g in grid[:18]) == pytest.approx(list(np.linspace(-2, 2, 18)))


def test_gantry_fixed():
    cfg = RobustnessConfig(gantry_mode="fixed", gantry_value=90.0)
    assert resolve_gantry(cfg, "any/uid") == 90.0


def test_gantry_bimodal_is_seeded_and_in_range():
    cfg = RobustnessConfig(gantry_mode="bimodal_random",
                           gantry_ranges=((30.0, 120.0), (240.0, 330.0)), gantry_seed=1234)
    # reproducible per patient uid
    a = resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")
    b = resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")
    assert a == b
    # different patients generally differ
    c = resolve_gantry(cfg, "NSCLC/LUNG1-006/9.9.9")
    assert a != c
    # every draw falls in one of the two bimodal lobes
    for uid in [f"p/{i}" for i in range(50)]:
        g = resolve_gantry(cfg, uid)
        assert (30.0 <= g <= 120.0) or (240.0 <= g <= 330.0)


def test_gantry_uniform_random_is_seeded_and_in_range():
    cfg = RobustnessConfig(gantry_mode="uniform_random", gantry_min=0.0, gantry_max=360.0,
                           gantry_seed=1234)
    a = resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")
    assert a == resolve_gantry(cfg, "NSCLC/LUNG1-001/1.2.3")   # reproducible
    assert a != resolve_gantry(cfg, "NSCLC/LUNG1-006/9.9.9")   # patient-specific
    draws = [resolve_gantry(cfg, f"p/{i}") for i in range(200)]
    assert all(0.0 <= g < 360.0 for g in draws)
    assert max(draws) - min(draws) > 180.0                     # actually spread out


def test_gantry_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_gantry(RobustnessConfig(gantry_mode="spiral"), "x")


def test_robustness_config_from_dict_defaults_and_overrides():
    from src.mc_generation.sweep import robustness_config_from_dict
    r = {"energies": [80.0, 120.0, 140.0], "gantry_mode": "uniform_random",
         "gantry_max": 360.0, "grid_n": 18, "num_primaries": 1e6,
         "experiment_prefix": "patient_set"}
    cfg = robustness_config_from_dict(r, grid_n=3, num_primaries=5e5)
    assert cfg.energies == [80.0, 120.0, 140.0]
    assert cfg.gantry_mode == "uniform_random"
    assert cfg.grid_n == 3                       # CLI override wins over yaml
    assert cfg.num_primaries == 5e5              # CLI override wins
    assert cfg.experiment_prefix == "patient_set"
    assert cfg.rotate_to_canonical is True       # default


def test_ct_rotation_preserves_isocenter_at_grid_center():
    """Rotate+expand about the grid-centre isocenter must keep it the new centre."""
    import SimpleITK as sitk

    from src.beamlets.rotation import rotate_ct_around_isocenter
    from src.mc_generation.geometry import extraction_isocenter_physical

    arr = np.zeros((40, 60, 80), dtype=np.float32)  # (z, y, x); non-cubic on purpose
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    iso0 = extraction_isocenter_physical(img)
    rot = rotate_ct_around_isocenter(img, -(210.0 - 90.0), iso0, expand=True)
    iso1 = extraction_isocenter_physical(rot)
    # grid grows (no clipping) but the isocenter stays put to sub-voxel accuracy
    assert rot.GetSize() != img.GetSize()
    assert np.allclose(np.asarray(iso0), np.asarray(iso1), atol=1.0)


def test_explicit_angle_list_maps_to_lattice_cells():
    """The smoke-test sweep (4 corners + centre) indexes onto the 3x3 lattice."""
    from src.mc_generation.sweep import build_angle_grid as bag
    grid = bag((-2.0, 2.0), (-2.0, 2.0), 3,
               [(2.0, -2.0), (-2.0, -2.0), (-2.0, 2.0), (2.0, 2.0), (0.0, 0.0)])
    assert [(g[0], g[1]) for g in grid] == [(2, 0), (0, 0), (0, 2), (2, 2), (1, 1)]
    assert [(g[2], g[3]) for g in grid][-1] == (0.0, 0.0)   # values kept verbatim


def test_explicit_angle_off_lattice_raises():
    with pytest.raises(ValueError, match="lattice"):
        build_angle_grid((-2.0, 2.0), (-2.0, 2.0), 3, [(1.0, 0.0)])


def test_energy_and_angle_tags_are_filename_safe():
    from src.mc_generation.sweep import angle_tag, energy_tag
    assert energy_tag(140.0) == "140"        # integer energies keep the old naming
    assert energy_tag(102.6) == "102p6"
    assert angle_tag(247.34) == "247p3"
    assert angle_tag(90.0) == "90"


def test_resolve_gantries_is_seeded_distinct_and_extends_single_draw():
    from src.mc_generation.sweep import resolve_gantries
    cfg = RobustnessConfig(gantry_mode="uniform_random", n_gantry=3, gantry_seed=4242)
    g = resolve_gantries(cfg, "LungDx/G0035/1.2.3")
    assert len(g) == 3 and len(set(g)) == 3
    assert g == resolve_gantries(cfg, "LungDx/G0035/1.2.3")          # reproducible
    assert g != resolve_gantries(cfg, "Colorectal/207/9.9.9")        # patient-specific
    # first draw is exactly the single-gantry result -> n_gantry:1 reruns are stable
    assert g[0] == resolve_gantry(cfg, "LungDx/G0035/1.2.3")


def test_resolve_gantries_fixed_mode_yields_one_angle():
    from src.mc_generation.sweep import resolve_gantries
    cfg = RobustnessConfig(gantry_mode="fixed", gantry_value=90.0, n_gantry=3)
    assert resolve_gantries(cfg, "any/uid") == [90.0]


def test_experiment_dir_carries_gantry_only_for_multi_gantry_runs():
    from types import SimpleNamespace

    from src.mc_generation.robustness import _experiment_dir
    rec = SimpleNamespace(anatomy="thoracic", patient_id="Lung_Dx-G0035")
    single = RobustnessConfig(experiment_prefix="p", experiment_version=2, n_gantry=1)
    assert _experiment_dir(single, rec, 140.0, 90.0).name == "p_thoracic_Lung_Dx-G0035_e140_v2"
    multi = RobustnessConfig(experiment_prefix="p", experiment_version=1, n_gantry=3)
    assert (_experiment_dir(multi, rec, 102.6, 247.34).name
            == "p_thoracic_Lung_Dx-G0035_e102p6_g247p3_v1")


def test_sweep_z_extent_requirement_matches_the_steered_axis():
    """theta_x steers along the slice axis: +-2 deg needs ~240 mm of z coverage."""
    from src.mc_generation.sweep import max_theta_x_deg, sweep_z_half_extent_mm
    d_smy = 2584.1
    cfg = RobustnessConfig(grid_n=3, roi_size=(60, 60, 320),
                           angles=[(2.0, -2.0), (-2.0, 2.0), (0.0, 0.0)])
    # d_smy * tan(2 deg) = 90.2 mm off-centre, plus half the 60-voxel window
    assert sweep_z_half_extent_mm(cfg, d_smy) == pytest.approx(90.25 + 30, abs=0.1)
    # a 218 mm thoracic CT cannot hold it; the 342 mm one used for the paper can
    assert max_theta_x_deg(218.0, cfg, d_smy) < 2.0
    assert max_theta_x_deg(342.0, cfg, d_smy) > 2.0
    # theta_y does not consume z: a y-only sweep needs just the ROI half-window
    cfg_y = RobustnessConfig(grid_n=3, roi_size=(60, 60, 320), angles=[(0.0, 2.0)])
    assert sweep_z_half_extent_mm(cfg_y, d_smy) == pytest.approx(30.0)


def test_sweep_fits_ct_reports_extent_and_limit():
    import SimpleITK as sitk

    from src.mc_generation.sweep import sweep_fits_ct
    cfg = RobustnessConfig(grid_n=3, roi_size=(60, 60, 320), angles=[(2.0, 0.0)])
    short = sitk.GetImageFromArray(np.zeros((218, 40, 40), dtype=np.int16))
    short.SetSpacing((1.0, 1.0, 1.0))
    fits, z_extent, max_tx = sweep_fits_ct(short, cfg, 2584.1)
    assert not fits and z_extent == pytest.approx(218.0) and max_tx == pytest.approx(1.75, abs=0.01)
    tall = sitk.GetImageFromArray(np.zeros((342, 40, 40), dtype=np.int16))
    tall.SetSpacing((1.0, 1.0, 1.0))
    assert sweep_fits_ct(tall, cfg, 2584.1)[0]


def test_beam_entrance_index_finds_the_patient_surface():
    """The entrance index is the first slab holding tissue, inside the beam window."""
    from src.mc_generation.geometry import beam_entrance_index
    arr = np.full((40, 40, 200), -1024.0)
    arr[18:22, 18:22, 120:160] = 50.0          # a patient blob starting at x=120
    arr[0:2, 0:2, 10:20] = 200.0               # couch rail far off-axis, starts at x=10
    assert beam_entrance_index(arr) == 10                       # unrestricted: sees the rail
    window = (slice(10, 30), slice(10, 30))                     # only where the beamlets pass
    assert beam_entrance_index(arr, window) == 120
    assert beam_entrance_index(np.full((4, 4, 10), -1024.0)) == 0   # no tissue -> 0


def test_trim_beam_axis_windows_the_grid_and_clamps():
    """Trimming keeps the requested beam-axis extent and never leaves the grid."""
    import SimpleITK as sitk

    from src.mc_generation.geometry import trim_beam_axis
    ct = sitk.GetImageFromArray(np.arange(4 * 5 * 100, dtype=np.float32).reshape(4, 5, 100))
    ct.SetSpacing((1.0, 1.0, 1.0))
    trimmed = trim_beam_axis(ct, 60, 20)
    assert trimmed.GetSize()[0] == 60
    assert trimmed.GetOrigin()[0] == ct.GetOrigin()[0] + 20      # origin follows the window
    np.testing.assert_allclose(sitk.GetArrayFromImage(trimmed),
                               sitk.GetArrayFromImage(ct)[:, :, 20:80])
    assert trim_beam_axis(ct, 60, 90).GetSize()[0] == 60         # clamped to the far edge
    assert trim_beam_axis(ct, 60, -5).GetOrigin()[0] == ct.GetOrigin()[0]
    assert trim_beam_axis(ct, 10_000, 0) is ct                   # wider than the grid: no-op


def test_sweep_lateral_half_extents_cover_steering_plus_roi():
    """The lateral search window spans the outermost beamlet plus half the ROI."""
    from src.mc_generation.sweep import sweep_lateral_half_extents
    cfg = RobustnessConfig(grid_n=3, roi_size=(60, 60, 320),
                           angles=[(2.0, -2.0), (-2.0, 2.0), (0.0, 0.0)])
    half_z, half_y = sweep_lateral_half_extents(cfg, 2014.9, 2584.1)
    assert half_z == pytest.approx(2584.1 * np.tan(np.deg2rad(2)) + 30, abs=0.1)
    assert half_y == pytest.approx(2014.9 * np.tan(np.deg2rad(2)) + 30, abs=0.1)


def test_field_geometry_puts_the_entrance_face_before_the_patient():
    """A rotated field keeps the gantry-90 entrance convention, not the expanded grid."""
    import SimpleITK as sitk

    from src.mc_generation.robustness import _field_geometry

    class _BDL:
        d_smx, d_smy = 2014.9, 2584.1

    arr = np.full((80, 300, 300), -1024.0, dtype=np.float32)
    zz, yy, xx = np.mgrid[0:80, 0:300, 0:300]
    body = ((yy - 150) ** 2 / 90 ** 2 + (xx - 150) ** 2 / 60 ** 2) < 1.0   # off-round torso
    arr[body] = 30.0
    ct = sitk.GetImageFromArray(arr)
    ct.SetSpacing((1.0, 1.0, 1.0))
    cfg = RobustnessConfig(grid_n=3, roi_size=(60, 60, 200), angles=[(0.0, 0.0)],
                           rotate_to_canonical=True, beam_entrance_standoff_mm=20.0)

    geom = _field_geometry(ct, 135.0, cfg, _BDL())
    assert geom.mc_gantry == 90.0 and geom.ct_rotation_deg == pytest.approx(-45.0)
    assert geom.ct.GetSize()[0] == ct.GetSize()[0]        # beam axis back to the original
    assert geom.ct.GetSize()[1] > ct.GetSize()[1]         # lateral expansion kept
    prof = geom.ct_array[40, 130:170, :].max(axis=0)
    gap = int(np.argmax(prof > -300))
    assert 15 <= gap <= 25                                # the requested standoff, +-a voxel

    # gantry 90 is untouched: no rotation, no trim
    same = _field_geometry(ct, 90.0, cfg, _BDL())
    assert same.ct.GetSize() == ct.GetSize() and same.ct_rotation_deg == 0.0
