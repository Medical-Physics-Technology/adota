"""Shared inference-evaluation infrastructure for ADoTA analysis scripts.

The granular one-scenario-per-script layout is preserved; the logic those
scripts duplicate (CLI/config merge, device resolution, data sources, the
batched/per-sample evaluation engine, and output writers) lives here so each
script stays a thin typer CLI over shared code.

See ``docs/scripts_refactor_phase1_plan.md`` for the migration plan.
"""
