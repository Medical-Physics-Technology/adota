"""HU -> RLSP from MCsquare-format files, and the pencil-beam model tables."""
from __future__ import annotations

import numpy as np
import pytest

from src.pencil_beam.beam_model import LUT_LENGTH, PencilBeamModel, hu_to_rlsp, read_rlsp_curve
from src.pencil_beam.rlsp import hu_to_rlsp_table, write_rlsp_csv


def _fake_install(root):
    """Two-material MCsquare install: water below HU 0, 'Bone' (SPR 1.5) from HU 0 up."""
    scanner = root / "Scanners" / "default"
    scanner.mkdir(parents=True)
    (scanner / "HU_Density_Conversion.txt").write_text("# HU density\n-1000 0.0\n0 1.0\n1000 2.0\n")
    (scanner / "HU_Material_Conversion.txt").write_text("# HU label\n-1050 17 # Water\n0 63 # Bone\n")
    mats = root / "Materials"
    mats.mkdir()
    (mats / "list.dat").write_text("17 Water\n63 Bone\n")
    energies = np.array([50.0, 100.0, 150.0])
    for name, scale in (("Water", 1.0), ("Bone", 1.5)):
        (mats / name).mkdir(parents=True)
        np.savetxt(mats / name / "G4_Stop_Pow.dat", np.column_stack([energies, scale * 7.0 * np.ones(3)]))
    return root


def test_rlsp_follows_mcsquare_density_and_material_rules(tmp_path):
    hu, rlsp = hu_to_rlsp_table(_fake_install(tmp_path), hu_range=(-1000, 1000))
    at = dict(zip(hu.astype(int), rlsp))
    assert at[-1000] == pytest.approx(0.0)
    assert at[-500] == pytest.approx(0.5)            # linear density, water
    assert at[0] == pytest.approx(1.0)               # HU 0 is NOT > 0 threshold: still water
    assert at[1] == pytest.approx(1.001 * 1.5)       # strictly above the threshold: bone
    assert at[1000] == pytest.approx(2.0 * 1.5)


def test_rlsp_csv_roundtrip(tmp_path):
    hu = np.arange(-5, 6, dtype=float)
    write_rlsp_csv(tmp_path / "HU_RLSP.csv", hu, hu * 0.1 + 1.0)
    curve = read_rlsp_curve(tmp_path)
    assert np.allclose(hu_to_rlsp(np.array([[-5.0, 2.5]]), curve), [[0.5, 1.25]])


def _bragg_like(r80, depths):
    return np.exp(-((depths - 0.97 * r80) / (0.02 * r80)) ** 2) + 0.3 * (depths < r80)


def _model(energies=(100.0, 150.0), r80s=(77.0, 158.0)):
    depths = np.array([np.linspace(0, 1.3 * r, LUT_LENGTH) for r in r80s])
    return PencilBeamModel(
        vsadx=2014.9, vsady=2584.1, energies=np.array(energies), r80=np.array(r80s),
        sigma_air_coef=np.array([[0.0, 1e-3, 1.0], [0.0, 2e-3, 0.5]]), depths=depths,
        sigmas=depths * 0.01, idds=np.array([_bragg_like(r, d) for r, d in zip(r80s, depths)]),
        reference_sigma=np.array([4.0, 3.5]), meta={"note": "test"})


def test_model_write_read_roundtrip(tmp_path):
    m = _model()
    m.write(tmp_path)
    back = PencilBeamModel.read(tmp_path)
    assert back.vsadx == m.vsadx and back.vsady == m.vsady
    assert np.allclose(back.energies, m.energies)
    assert np.allclose(back.idds, m.idds, rtol=1e-5)
    assert np.allclose(back.sigma_air_coef, m.sigma_air_coef, rtol=1e-5)
    assert back.meta == {"note": "test"}


def test_at_energy_node_and_interpolation():
    m = _model()
    node = m.at_energy(100.0)
    assert node.energies.tolist() == [100.0] and np.array_equal(node.idds[0], m.idds[0])
    mid = m.at_energy(125.0)
    assert m.r80[0] < mid.r80[0] < m.r80[1]
    peak = mid.depths[0][np.argmax(mid.idds[0])]
    assert peak == pytest.approx(0.97 * mid.r80[0], rel=0.02)   # range-scaled blend keeps one peak
    assert mid.meta["interpolated_from"] == [100.0, 150.0]
    with pytest.raises(ValueError):
        m.at_energy(160.0)


def test_kernel_model_layout():
    k = _model().at_energy(150.0).kernel_model()
    assert k.divergence_params.shape == (1, 5) and k.divergence_params.dtype == np.float32
    assert k.divergence_params[0, :2].tolist() == pytest.approx([150.0, 158.0])
    assert k.lut_depths.shape == k.lut_idds.shape == k.lut_sigmas.shape == (1, LUT_LENGTH)
