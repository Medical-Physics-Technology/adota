"""Unit tests for src/evaluation/outputs.py (shared CSV writer)."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.outputs import CsvColumn, save_results_csv


@dataclass
class _R:
    sample_id: str
    energy: float
    metric: float


def _columns():
    return [
        CsvColumn("sample_id", lambda r: r.sample_id),
        CsvColumn("energy_mev", lambda r: f"{r.energy:.2f}"),
        CsvColumn("metric", lambda r: f"{r.metric:.4f}", lambda r: r.metric, ".4f"),
    ]


def test_header_rows_and_summary(tmp_path):
    results = [_R("a", 100.0, 1.0), _R("b", 90.0, 3.0)]
    out = tmp_path / "r.csv"
    save_results_csv(
        results, out, _columns(),
        sort_key=lambda r: r.energy, label_column="sample_id",
    )
    lines = out.read_text().splitlines()

    # Header exact.
    assert lines[0] == "sample_id,energy_mev,metric"
    # Rows sorted by energy ascending.
    assert lines[1] == "b,90.00,3.0000"
    assert lines[2] == "a,100.00,1.0000"
    # Blank separator row (all empty fields).
    assert lines[3] == ",,"
    # Summary block: mean/std/min/max over metric; label in sample_id; energy empty.
    assert lines[4] == "mean,,2.0000"
    assert lines[5] == "std,,1.0000"
    assert lines[6] == "min,,1.0000"
    assert lines[7] == "max,,3.0000"
    assert len(lines) == 8


def test_no_summary_when_no_column_declares_it(tmp_path):
    results = [_R("a", 100.0, 1.0)]
    out = tmp_path / "r.csv"
    cols = [
        CsvColumn("sample_id", lambda r: r.sample_id),
        CsvColumn("energy_mev", lambda r: f"{r.energy:.2f}"),
    ]
    save_results_csv(results, out, cols)
    lines = out.read_text().splitlines()
    assert lines == ["sample_id,energy_mev", "a,100.00"]


def test_summary_uses_all_results_not_sorted_subset(tmp_path):
    # Stats are order-independent; ensure all rows contribute.
    results = [_R("a", 1.0, 10.0), _R("b", 2.0, 20.0), _R("c", 3.0, 30.0)]
    out = tmp_path / "r.csv"
    save_results_csv(results, out, _columns(), sort_key=lambda r: -r.energy)
    lines = out.read_text().splitlines()
    # Rows now in descending energy order.
    assert lines[1].startswith("c,")
    # mean of metric = 20.0
    assert lines[5] == "mean,,20.0000"
