"""The analysis and figures of difficulty_engine_grids.py, on synthetic beamlets."""
import numpy as np
import pandas as pd
import pytest

from src.active_learning.ensemble.engine_grids import (
    add_within_field_decile,
    choose_examples,
    decile_table,
    downsample2,
    mape_above_cutoff,
    metric_columns,
    reference_percentile,
    spearman_summary,
    within_field_spearman,
)
from src.figures.difficulty_engine_grids import StoryCase, difficulty_story_figure, difficulty_trend_figure
from src.pencil_beam.saved_beamlets import Engine
from src.pencil_beam.saved_stages import load_adota_engine

ENGINES = [Engine("A", "adota", "_ds_pred.npy"), Engine("B", "pencil_beam", "_ds_pba.npy")]
METRICS = {"gpr_2pct": "g2", "mape_1mm": "mape_1mm"}


def _frame(n_fields=4, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fields):
        score = rng.normal(size=n)
        rows.append(pd.DataFrame({
            "block": f"f{f}", "stem": [f"a{i:02d}_00" for i in range(n)], "patient_id": f"p{f // 2}",
            "anatomy": "thoracic", "energy": 100.0, "d_score": score,
            "a_g2": 99 - score, "a_mape_1mm": 5 + score, "b_g2": rng.normal(95, 1, n),
            "b_mape_1mm": rng.normal(10, 1, n)}))
    return pd.concat(rows, ignore_index=True)


def test_mape_uses_only_voxels_above_the_cutoff_of_the_reference():
    reference = np.array([100.0, 50.0, 5.0])
    test = np.array([110.0, 50.0, 500.0])       # the 5.0 voxel is below 10% of the maximum
    assert mape_above_cutoff(reference, test, 0.1) == pytest.approx(5.0)
    assert np.isnan(mape_above_cutoff(np.zeros(3), test))


def test_downsample2_averages_blocks_and_keeps_the_mean():
    volume = np.arange(4 * 4 * 8, dtype=float).reshape(4, 4, 8)
    out = downsample2(volume)
    assert out.shape == (2, 2, 4)
    assert out[0, 0, 0] == pytest.approx(volume[:2, :2, :2].mean())
    assert out.mean() == pytest.approx(volume.mean())


def test_examples_come_from_the_scores_and_skip_missing_ones():
    frame = pd.DataFrame({"block": ["b"] * 5, "stem": list("abcde"), "d_score": [0.0, 1.0, np.nan, 2.0, 3.0]})
    picks = choose_examples(frame, {"Low": 0, "High": 100}).set_index("case")
    assert picks.loc["Low", "stem"] == "a" and picks.loc["High", "stem"] == "e"
    assert set(choose_examples(frame, {"Mid": 50})["stem"]) <= {"a", "b", "d", "e"}


def test_reference_percentile():
    assert reference_percentile(np.array([0.5, 2.0]), np.array([0.0, 1.0, 2.0, 3.0])).tolist() == [25.0, 75.0]


def test_within_field_spearman_reads_the_built_in_trend_and_its_sign():
    frame = _frame()
    columns = metric_columns(ENGINES, METRICS)
    per_field = within_field_spearman(frame, columns)
    assert len(per_field) == 4 * 2 * 2
    summary = spearman_summary(frame, per_field, columns, n_bootstrap=50).set_index(["engine", "metric"])
    assert summary.loc[("A", "gpr_2pct"), "median_within_field"] == pytest.approx(-1.0)
    assert summary.loc[("A", "gpr_2pct"), "share_fields_expected_sign"] == 1.0
    assert summary.loc[("A", "mape_1mm"), "median_within_field"] == pytest.approx(1.0)
    assert summary.loc[("A", "mape_1mm"), "n_patients"] == 2
    assert abs(summary.loc[("B", "mape_1mm"), "median_within_field"]) < 0.5


def test_within_field_deciles_are_per_field_and_cover_one_to_ten():
    out = add_within_field_decile(_frame())
    assert out.groupby("block")["field_decile"].agg(["min", "max"]).eq([1, 10]).all().all()
    table = decile_table(out, metric_columns(ENGINES, METRICS))
    rising = table[(table["engine"] == "A") & (table["metric"] == "mape_1mm")].sort_values("field_decile")
    assert rising["median"].is_monotonic_increasing


def test_an_engine_without_a_model_is_refused():
    with pytest.raises(ValueError, match="needs model_dir"):
        load_adota_engine({}, None)


def test_figures_are_written_in_all_formats(tmp_path):
    shape = (60, 60, 320)
    reference = np.zeros(shape, dtype=np.float32)
    reference[28:32, 28:32, :100] = 1.0
    reference[28:32, 28:32, 100:110] = 3.0
    case = StoryCase("Low difficulty", "d = 0.1", np.full(shape, -500.0), reference / 3.0, reference,
                     {"A": reference * 0.9, "B": reference}, {"A": "GPR 99.0%  MAPE 4.0%", "B": "GPR 95.0%  MAPE 9.0%"})
    frame = add_within_field_decile(_frame())
    table = decile_table(frame, metric_columns(ENGINES, METRICS))
    paths = difficulty_story_figure([case, case], ["A", "B"], figure_path=str(tmp_path / "story"), title="Story",
                                    caption="A caption.", formats=("png",))
    assert [p.suffix for p in paths] == [".png"] and paths[0].exists()
    paths = difficulty_trend_figure(table, "mape_1mm", ["A", "B"], figure_path=str(tmp_path / "trend"),
                                    ylabel="MAPE [%]", title="Trend", formats=("png",))
    assert [p.suffix for p in paths] == [".png"] and paths[0].exists()
    assert not (tmp_path / "trend.pdf").exists()
