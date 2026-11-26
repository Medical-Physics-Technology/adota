from pathlib import Path

import pandas as pd

from src.acquisition.explain.tables import example_table, latex_escape, write_csv_and_latex

CAPTION = "Shares & rates: 50% of 100_x"


def _synthetic():
    placement = pd.DataFrame({
        "beamlet_id": ["b1", "b1", "b1", "b2", "b2", "b2"],
        "feature": ["f_a", "f_b", "f_c", "f_a", "f_b", "f_c"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "percentile_ref": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "robust_z": [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    })
    contributions = pd.DataFrame({
        "beamlet_id": ["b1", "b1", "b1", "b2", "b2"],
        "feature": ["f_a", "f_b", "f_c", "f_a", "f_b"],
        "contribution": [0.5, 0.25, 0.125, 0.75, 0.0],
        "scorer": "s",
    })
    groups = {"f_a": "Geometry", "f_b": "Dose", "f_c": "Geometry"}
    top_features = {"b1": ["f_b"]}
    return placement, contributions, groups, top_features


def test_latex_escape_specials():
    assert latex_escape("a_b & 50%") == r"a\_b \& 50\%"


def test_example_table_columns_order_and_flags():
    placement, contributions, groups, top_features = _synthetic()
    table = example_table(placement, contributions, groups, top_features)

    expected_columns = ["feature", "group"] + [
        f"{b} {s}" for b in ["b1", "b2"] for s in ["value", "percentile", "z", "contribution", "top"]
    ]
    assert list(table.columns) == expected_columns
    assert list(table["feature"]) == ["f_a", "f_b", "f_c"]
    assert list(table["group"]) == ["Geometry", "Dose", "Geometry"]
    assert list(table["b1 top"]) == [False, True, False]
    assert not table["b2 top"].any()
    assert table.loc[table["feature"] == "f_a", "b2 value"].item() == 4.0
    assert pd.isna(table.loc[table["feature"] == "f_c", "b2 contribution"].item())


def test_write_csv_and_latex(tmp_path: Path):
    placement, contributions, groups, top_features = _synthetic()
    frame = example_table(placement, contributions, groups, top_features)

    csv_path, tex_path = write_csv_and_latex(
        frame, tmp_path / "out" / "tab_example", caption=CAPTION, label="tab:example",
    )

    assert csv_path.exists() and tex_path.exists()
    assert csv_path.name == "tab_example.csv" and tex_path.name == "tab_example.tex"

    round_trip = pd.read_csv(csv_path)
    assert round_trip.shape == frame.shape

    tex = tex_path.read_text(encoding="utf-8")
    assert r"\toprule" in tex
    assert r"\bottomrule" in tex
    assert latex_escape(CAPTION) in tex
    assert r"\label{tab:example}" in tex
    assert "--" in tex
    assert r"\textbf{" in tex
    assert "b1 top" not in tex

    rows = {line.split(" & ")[0]: line for line in tex.splitlines() if " & " in line}
    assert r"\textbf{" in rows[latex_escape("f_b")]
    assert r"\textbf{" not in rows[latex_escape("f_a")]
