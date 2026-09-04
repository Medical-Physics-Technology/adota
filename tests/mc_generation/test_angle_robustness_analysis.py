"""Fast tests for the robustness grid aggregation + shared-scale logic (no inference)."""
import numpy as np
import pytest

from src.mc_generation.angle_robustness_analysis import (
    GammaCriterion,
    Panel,
    aggregate_panels,
    shared_scale_per_criterion,
)

C = GammaCriterion(2, 2, 10)


def _panel(anatomy, energy, patient, grid):
    return Panel(anatomy=anatomy, energy=energy, patient=patient, n_patients=1,
                 grids={C.key: np.asarray(grid, float)})


def test_criterion_labels():
    assert C.key == "g2_2_10"
    assert GammaCriterion(1, 3, 0.1).cbar_label == "Γ(1%, 3mm, 0.1%) [%]"


def test_aggregate_is_per_cell_nanmean():
    a = _panel("pelvic", 140, "P1", [[90.0, 100.0], [np.nan, 80.0]])
    b = _panel("pelvic", 140, "P2", [[80.0, 90.0], [100.0, 60.0]])
    agg = aggregate_panels([a, b])
    assert agg.patient is None and agg.n_patients == 2
    np.testing.assert_allclose(agg.grids[C.key], [[85.0, 95.0], [100.0, 70.0]])


def test_shared_scale_spans_all_panels_ignoring_nan():
    a = _panel("pelvic", 140, "P1", [[90.0, 100.0], [np.nan, 80.0]])
    b = _panel("thoracic", 140, "T1", [[83.0, 97.0], [61.0, 99.0]])
    vmin, vmax = shared_scale_per_criterion([a, b], [C])[C.key]
    assert vmin == pytest.approx(61.0)   # global min across both panels
    assert vmax == pytest.approx(100.0)  # global max across both panels


def test_extreme_indices_picks_lowest_and_highest():
    import numpy as np

    from src.mc_generation.angle_robustness_analysis import extreme_indices
    g = np.array([[90.0, 70.0, np.nan],
                  [60.0, 99.0, 85.0],
                  [50.0, 95.0, 80.0]])
    worst, best = extreme_indices(g, n=3)
    assert [round(w[2], 0) for w in worst] == [50.0, 60.0, 70.0]      # 3 lowest
    assert [round(b[2], 0) for b in best] == [99.0, 95.0, 90.0]       # 3 highest, desc
    # NaN cell (0,2) is excluded
    assert (0, 2) not in [(w[0], w[1]) for w in worst + best]
    # worst[0] maps to the min-GPR cell (2,0)=50
    assert (worst[0][0], worst[0][1]) == (2, 0)


def test_panel_name_includes_gantry_only_when_set():
    from src.mc_generation.angle_robustness_analysis import panel_name
    p = _panel("thoracic", 102.6, "Lung_Dx-G0035", [[1.0]])
    assert panel_name(p) == "thoracic_Lung_Dx-G0035_e102p6"     # historical naming
    p.gantry = 247.34
    assert panel_name(p) == "thoracic_Lung_Dx-G0035_e102p6_g247p3"
    p.patient, p.n_patients, p.gantry = None, 3, None
    assert panel_name(p) == "thoracic_aggregate3_e102p6"


def test_aggregate_keeps_a_shared_gantry_and_drops_mixed_ones():
    a = _panel("thoracic", 135.0, "P1", [[90.0]])
    b = _panel("thoracic", 135.0, "P2", [[80.0]])
    a.gantry = b.gantry = 41.2
    assert aggregate_panels([a, b]).gantry == pytest.approx(41.2)
    b.gantry = 300.7
    assert aggregate_panels([a, b]).gantry is None


def test_saved_grids_round_trip_for_cheap_rerender(tmp_path):
    """A saved panel reloads exactly, so re-rendering needs no inference or gamma."""
    from src.mc_generation.angle_robustness_analysis import (
        load_panel_grids,
        panel_name,
        save_panel_grids,
    )
    p = _panel("thoracic", 102.6, "Lung_Dx-G0037", [[90.0, np.nan], [np.nan, 80.0]])
    p.gantry = 137.9
    back = load_panel_grids(save_panel_grids(p, tmp_path))
    assert (back.anatomy, back.patient, back.n_patients) == ("thoracic", "Lung_Dx-G0037", 1)
    assert back.energy == pytest.approx(102.6) and back.gantry == pytest.approx(137.9)
    np.testing.assert_array_equal(back.grids[C.key], p.grids[C.key])   # NaNs included
    assert panel_name(back) == panel_name(p)

    agg = aggregate_panels([p, _panel("thoracic", 102.6, "Lung_Dx-G0049", [[70.0, 1.0], [2.0, 3.0]])])
    back_agg = load_panel_grids(save_panel_grids(agg, tmp_path))
    assert back_agg.patient is None and back_agg.n_patients == 2   # stays an aggregate
    assert back_agg.gantry is None                                 # mixed gantries dropped
