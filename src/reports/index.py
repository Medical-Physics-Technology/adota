"""Flattening of the record tree into queryable tables.

The Markdown records are the source of truth; this module derives the
"database" view of them: one row per record in ``index.csv``, one row per
reported number in ``metrics.csv``, and the same two tables in a SQLite file for
ad-hoc queries. Nothing here is authoritative -- the outputs are regenerated
from the records at any time, which is why they are cheap to throw away.

Where a record names an ``artifacts.run_dir`` that is reachable on this machine,
its ``manifest.json`` (written by every training and evaluation run: git commit,
GPU, dataset fingerprint) is folded into the index row, so a result can be tied
back to the exact code and data that produced it.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.reports.records import Record

INDEX_COLUMNS = ["id", "kind", "title", "date", "status", "branch", "pr", "tags",
                 "publication_target", "run_dir", "git_commit", "dataset_fingerprint",
                 "n_metrics", "path"]
METRIC_COLUMNS = ["record_id", "date", "name", "value", "unit", "split", "n"]


def _manifest_fields(run_dir: Optional[str]) -> Dict[str, str]:
    """Git commit and dataset fingerprint from a run's manifest, if readable."""
    if not run_dir:
        return {}
    manifest = Path(run_dir) / "manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    git = data.get("git") or {}
    dataset = data.get("dataset") or {}
    return {"git_commit": str(git.get("commit", data.get("git_commit", ""))),
            "dataset_fingerprint": str(dataset.get("fingerprint",
                                                   data.get("dataset_fingerprint", "")))}


def index_row(record: Record) -> Dict[str, object]:
    """One flat row describing a record."""
    front = record.front
    artifacts = front.get("artifacts") or {}
    publication = front.get("publication") or {}
    run_dir = artifacts.get("run_dir")
    row = {
        "id": record.id,
        "kind": record.kind,
        "title": front.get("title", ""),
        "date": str(front.get("date", "")),
        "status": front.get("status", ""),
        "branch": front.get("branch", ""),
        "pr": front.get("pr", ""),
        "tags": ";".join(str(t) for t in (front.get("tags") or [])),
        "publication_target": publication.get("target", ""),
        "run_dir": run_dir or "",
        "git_commit": "",
        "dataset_fingerprint": "",
        "n_metrics": len(front.get("metrics") or []),
        "path": str(record.path),
    }
    row.update(_manifest_fields(run_dir))
    return row


def metric_rows(record: Record) -> List[Dict[str, object]]:
    """The record's metric rows in long format, one number per row."""
    date = str(record.front.get("date", ""))
    rows = []
    for metric in record.front.get("metrics") or []:
        rows.append({"record_id": record.id, "date": date,
                     "name": metric.get("name", ""), "value": metric.get("value", ""),
                     "unit": metric.get("unit", ""), "split": metric.get("split", ""),
                     "n": metric.get("n", "")})
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _write_sqlite(path: Path, index: Sequence[Dict], metrics: Sequence[Dict]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as connection:
        for table, columns, rows in (("records", INDEX_COLUMNS, index),
                                     ("metrics", METRIC_COLUMNS, metrics)):
            connection.execute(
                f"CREATE TABLE {table} ({', '.join(f'{c} TEXT' for c in columns)})")
            connection.executemany(
                f"INSERT INTO {table} VALUES ({', '.join('?' * len(columns))})",
                [tuple(str(row.get(c, '')) for c in columns) for row in rows])


def build_index(records: Sequence[Record], out_dir: Path,
                sqlite_path: Optional[Path] = None) -> Dict[str, Path]:
    """Write ``index.csv`` and ``metrics.csv`` (and optionally the SQLite file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = [index_row(record) for record in records]
    metrics = [row for record in records for row in metric_rows(record)]

    written = {"index": out_dir / "index.csv", "metrics": out_dir / "metrics.csv"}
    _write_csv(written["index"], INDEX_COLUMNS, index)
    _write_csv(written["metrics"], METRIC_COLUMNS, metrics)
    if sqlite_path is not None:
        _write_sqlite(sqlite_path, index, metrics)
        written["sqlite"] = sqlite_path
    return written
