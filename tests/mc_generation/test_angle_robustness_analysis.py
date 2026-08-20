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
    from src.mc_generation.angle_robustness_analysis import extreme_indices
    import numpy as np
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
