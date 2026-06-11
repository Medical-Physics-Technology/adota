"""Shared helpers for the characterization (golden) tests.

These tests pin the current numeric behavior of the Tier-1 inference scripts
before the ``src/evaluation/`` refactor touches them. Reference CSVs live under
``$ADOTA_GOLDEN_DIR`` (default ``/scratch/mstryja/tmp_adota``), never in the repo
or home directory, because home storage is limited.

Comparison policy (timing-aware, see docs/scripts_refactor_phase1_plan.md):

* header line must match exactly (locks column order, names, and float
  precision);
* non-numeric columns (``sample_id``, ``label``, ...) must match exactly;
* metric columns are compared numerically within tolerance;
* timing columns (anything ending ``_time_s``) and their summary rows are
  ignored, since they vary run to run.

Modes:

* if the golden file is missing, or ``ADOTA_GOLDEN_UPDATE`` is set, the current
  output is written as the new baseline (``check_against_golden`` returns
  ``"captured"``);
* otherwise the current output is compared against the stored golden.
"""

from __future__ import annotations

import csv
import io
import math
import os
from pathlib import Path

DEFAULT_GOLDEN_DIR = Path(os.environ.get("ADOTA_GOLDEN_DIR", "/scratch/mstryja/tmp_adota"))
GOLDEN_SUBDIR = "golden"


def golden_dir() -> Path:
    """Return (creating if needed) the directory holding reference CSVs."""
    d = DEFAULT_GOLDEN_DIR / GOLDEN_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def should_update() -> bool:
    """True when ``ADOTA_GOLDEN_UPDATE`` requests re-baselining."""
    return os.environ.get("ADOTA_GOLDEN_UPDATE", "") not in ("", "0", "false", "False")


def set_determinism(seed: int = 1234) -> None:
    """Pin RNG + cuDNN so repeated runs minimize numeric drift."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_timing(col: str) -> bool:
    return col.endswith("_time_s") or col in {"calc_time_s", "extract_time_s"}


def compare_csv(
    actual_text: str,
    golden_text: str,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> list[str]:
    """Return a list of human-readable differences; empty means a match."""
    diffs: list[str] = []
    a_lines = actual_text.splitlines()
    g_lines = golden_text.splitlines()
    if not a_lines or not g_lines:
        return ["one of the CSVs is empty"]

    if a_lines[0] != g_lines[0]:
        return [
            "header differs:\n"
            f"  actual: {a_lines[0]}\n"
            f"  golden: {g_lines[0]}"
        ]

    a_rows = list(csv.DictReader(io.StringIO(actual_text)))
    g_rows = list(csv.DictReader(io.StringIO(golden_text)))
    if len(a_rows) != len(g_rows):
        return [f"row count differs: actual={len(a_rows)} golden={len(g_rows)}"]

    header = a_lines[0].split(",")
    for i, (ar, gr) in enumerate(zip(a_rows, g_rows)):
        for col in header:
            if _is_timing(col):
                continue
            av = ar.get(col, "")
            gv = gr.get(col, "")
            if av == gv:
                continue
            try:
                af = float(av)
                gf = float(gv)
            except (ValueError, TypeError):
                diffs.append(f"row {i} col {col!r}: actual={av!r} golden={gv!r}")
                continue
            if not math.isclose(af, gf, rel_tol=rel_tol, abs_tol=abs_tol):
                diffs.append(
                    f"row {i} col {col!r}: actual={af} golden={gf} "
                    f"(rel_tol={rel_tol}, abs_tol={abs_tol})"
                )
    return diffs


def check_against_golden(
    name: str,
    actual_text: str,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> tuple[str, list[str]]:
    """Capture a baseline or compare against it.

    Returns ``(status, messages)`` where status is one of ``"captured"``,
    ``"match"``, or ``"mismatch"``.
    """
    path = golden_dir() / f"{name}.csv"
    if should_update() or not path.exists():
        path.write_text(actual_text)
        return "captured", [f"baseline written to {path}"]

    golden_text = path.read_text()
    diffs = compare_csv(actual_text, golden_text, rel_tol=rel_tol, abs_tol=abs_tol)
    return ("match" if not diffs else "mismatch"), diffs
