"""The research report record system.

Records (one Markdown file per change or experiment) live in a separate private
repository mounted at ``reports/``; this package is the public tooling that
parses, validates and flattens them. See ``docs/reporting.md``.
"""
from src.reports.index import build_index, index_row, metric_rows
from src.reports.records import (
    Record,
    RecordError,
    load_record,
    load_records,
    load_schema,
    next_id,
    slugify,
)
from src.reports.validation import validate_collection, validate_record, validate_tree

__all__ = [
    "Record", "RecordError", "load_record", "load_records", "load_schema", "next_id",
    "slugify", "validate_record", "validate_collection", "validate_tree",
    "build_index", "index_row", "metric_rows",
]
