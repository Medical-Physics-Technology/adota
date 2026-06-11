"""Shared output writers for the evaluation scripts.

The Tier-1 scripts each emitted a near-identical ``save_results_csv``: header,
per-sample rows (with per-field float formatting), a blank separator row, then
``mean`` / ``std`` / ``min`` / ``max`` summary rows. The only differences are the
column set, the per-field format, the sort key, and which columns get summary
statistics.

:func:`save_results_csv` captures that pattern. Each script passes an explicit
list of :class:`CsvColumn` specs, so column order, header names, and float
precision are preserved exactly (the golden tests verify byte-stable output).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

# (label, numpy reducer) applied in order to build the summary block.
_SUMMARY_STATS = (
    ("mean", np.mean),
    ("std", np.std),
    ("min", np.min),
    ("max", np.max),
)


@dataclass
class CsvColumn:
    """One CSV column.

    Attributes:
        name: Header / field name.
        row: Maps a result to the per-row cell (returns a formatted string, or a
            raw value the csv writer will stringify).
        summary_extract: If set, the column gets summary statistics; this maps a
            result to the numeric value reduced by mean/std/min/max.
        summary_fmt: Python format spec (e.g. ``".4f"``) applied to each summary
            statistic. Required when ``summary_extract`` is set.
    """

    name: str
    row: Callable[[Any], Any]
    summary_extract: Optional[Callable[[Any], float]] = None
    summary_fmt: Optional[str] = None


def save_results_csv(
    results: Sequence[Any],
    output_path: Path,
    columns: Sequence[CsvColumn],
    *,
    sort_key: Optional[Callable[[Any], Any]] = None,
    label_column: Optional[str] = None,
    logger: Optional[Any] = None,
) -> None:
    """Write per-sample results to CSV, with an optional summary block.

    The summary block (blank separator row + mean/std/min/max rows) is written
    iff at least one column declares ``summary_extract``. Summary statistics are
    computed over ``results`` in their original order (the reducers are
    order-independent); only the row order is affected by ``sort_key``.

    Args:
        results: Result objects (one per sample).
        output_path: Destination CSV path.
        columns: Column specs, in output order.
        sort_key: Optional key for ordering the per-sample rows.
        label_column: Column that holds the stat label ("mean", ...) in the
            summary rows. Defaults to the first column.
        logger: Optional logger; an info line is emitted on success.
    """
    fieldnames = [c.name for c in columns]
    if label_column is None:
        label_column = fieldnames[0]

    ordered = sorted(results, key=sort_key) if sort_key is not None else list(results)
    has_summary = any(c.summary_extract is not None for c in columns)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ordered:
            writer.writerow({c.name: c.row(r) for c in columns})

        if has_summary:
            writer.writerow({})  # blank separator
            for label, reducer in _SUMMARY_STATS:
                out: dict[str, Any] = {}
                for c in columns:
                    if c.summary_extract is not None:
                        value = reducer([c.summary_extract(r) for r in results])
                        out[c.name] = format(float(value), c.summary_fmt)
                    elif c.name == label_column:
                        out[c.name] = label
                    else:
                        out[c.name] = ""
                writer.writerow(out)

    if logger is not None:
        logger.info(f"Results CSV saved to {output_path}")
