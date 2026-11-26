"""Synthetic, CPU-only tests for :mod:`src.figures.dataset_overview`."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import QuadMesh

from src.figures import dataset_overview as dov

SITES = ["thoracic", "abdominal", "pelvic"]


def _hist(body: bool = False) -> pd.DataFrame:
    lows = np.arange(-1024, 3071, 10).astype(float)
    rows = []
    for k, site in enumerate(SITES):
        counts = np.zeros(lows.size, dtype=int)
        counts[(lows > -200) & (lows < 200)] = 1000 * (k + 1)
        counts[-1] = 7
        if body:
            counts[lows < -500] = 0
        rows.append(pd.DataFrame({"site": site, "bin_low_hu": lows, "count": counts}))
    return pd.concat(rows, ignore_index=True)


def _energy_frames():
    rng = np.random.default_rng(0)
    metadata = pd.DataFrame({"site": np.repeat(SITES, 200), "energy_mev": rng.uniform(70, 230, 600)})
    # Deliberately disagrees with metadata: a pass proves the texts come from stats, not a recomputation.
    stats = pd.DataFrame({"site": ["pelvic", "thoracic", "abdominal"], "energy_min_mev": [91.04, 70.0, 75.54],
                          "energy_median_mev": [170.0, 119.84, 150.24], "energy_max_mev": [230.0, 218.26, 229.9]})
    return metadata, stats


def _lattice_cells() -> pd.DataFrame:
    """3 x 3 lattice with unequal pitches (6 mm in x, 8 mm in y), one empty cell and a distinctive corner."""
    rows = []
    for x in (-6.0, 0.0, 6.0):
        for y in (-8.0, 0.0, 8.0):
            if (x, y) == (0.0, 0.0):
                continue
            rows.append({"site": "thoracic", "spot_x_mm": x, "spot_y_mm": y, "beamlets": 99 if (x, y) == (6.0, -8.0)
                         else 5})
    return pd.DataFrame(rows)


def _sectors() -> pd.DataFrame:
    lows = np.arange(0, 360, 15)
    return pd.DataFrame([{"site": s, "gantry_bin_low_deg": lo, "beamlets": (k + 1) * 100 + (900 if lo == 90 else 0)}
                         for k, s in enumerate(SITES) for lo in lows])


def _assert_two_files(paths, directory) -> None:
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
    assert sorted(p.name for p in directory.iterdir()) == sorted(p.name for p in paths)
    assert all(p.stat().st_size > 0 for p in paths)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("body", [False, True])
def test_hu_histograms_write_png_and_pdf(tmp_path, body):
    paths = dov.site_hu_histograms_figure(_hist(body), title="HU", figure_path=tmp_path / "hu")
    _assert_two_files(paths, tmp_path)


def test_hu_histograms_all_zero_and_missing_rows_do_not_raise(tmp_path):
    hist = _hist(body=True)
    hist.loc[hist["site"] == "pelvic", "count"] = 0
    hist = hist[~((hist["site"] == "abdominal") & (hist["bin_low_hu"] < -500))]
    paths = dov.site_hu_histograms_figure(hist, title="HU", figure_path=tmp_path / "hu", x_range=(-600.0, 2000.0))
    _assert_two_files(paths, tmp_path)


def test_hu_bin_grid_is_gapless_and_caps_the_last_bin():
    edges = dov.hu_bin_grid([-1024.0, -1014.0, 3066.0], 3071.0)
    assert edges[0] == -1024.0 and edges[-1] == 3071.0
    assert np.allclose(np.diff(edges[:-1]), 10.0)
    assert edges[-2] == 3066.0


def test_energy_texts_come_from_stats(tmp_path):
    metadata, stats = _energy_frames()
    fig = dov._build_energy_by_site(metadata, stats, title="Energy")
    texts = {t.get_text() for ax in fig.axes for t in ax.texts}
    plt.close(fig)
    assert "min 91.0 / median 170.0 / max 230.0 MeV" in texts
    assert "min 70.0 / median 119.8 / max 218.3 MeV" in texts
    assert "min 75.5 / median 150.2 / max 229.9 MeV" in texts
    for _, row in stats.iterrows():
        assert dov.energy_summary_text(row) in texts
    paths = dov.energy_by_site_figure(metadata, stats, title="Energy", figure_path=tmp_path / "energy")
    _assert_two_files(paths, tmp_path)


def test_lattice_edges_are_midpoints_extended_by_half_a_pitch():
    assert np.allclose(dov.lattice_edges([6.0, -6.0, 0.0, 0.0]), [-9.0, -3.0, 3.0, 9.0])
    assert np.allclose(dov.lattice_edges([-8.0, 0.0, 8.0]), [-12.0, -4.0, 4.0, 12.0])


def test_spot_panel_mesh_has_lattice_edges_and_orientation(tmp_path):
    cells = _lattice_cells()
    fig = dov._build_spot_positions(cells, title="Spots", value_label="beamlets")
    ax = fig.axes[0]
    mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
    coords = mesh.get_coordinates()
    values = np.asarray(mesh.get_array()).reshape(3, 3)
    aspect = ax.get_aspect()
    plt.close(fig)
    assert np.allclose(coords[0, :, 0], [-9.0, -3.0, 3.0, 9.0])
    assert np.allclose(coords[:, 0, 1], [-12.0, -4.0, 4.0, 12.0])
    assert values[1, 1] == 0          # unoccupied centre drawn as 0
    assert values[0, 2] == 99         # row = y (lowest y first), column = x
    assert mesh.norm.vmin == 0.0
    assert aspect == 1.0
    paths = dov.spot_positions_figure(cells, title="Spots", figure_path=tmp_path / "spots")
    _assert_two_files(paths, tmp_path)


def test_gantry_polar_orientation_and_shared_radius(tmp_path):
    sectors = _sectors()
    fig = dov._build_gantry_polar(sectors, title="Gantry", bin_deg=15.0)
    polar = [ax for ax in fig.axes if ax.name == "polar"]
    offsets = [ax.get_theta_offset() for ax in polar]
    directions = [ax.get_theta_direction() for ax in polar]
    rmax = [ax.get_rmax() for ax in polar]
    titles = [ax.get_title(loc="left") for ax in polar]
    plt.close(fig)
    assert len(polar) == 3
    assert np.allclose(offsets, np.pi / 2) and directions == [-1, -1, -1]
    assert len(set(rmax)) == 1 and rmax[0] >= sectors["beamlets"].max()
    assert [t.split(":")[0] for t in titles] == SITES
    paths = dov.gantry_polar_figure(sectors, title="Gantry", figure_path=tmp_path / "gantry")
    _assert_two_files(paths, tmp_path)
