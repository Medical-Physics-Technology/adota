"""Check the gamma-acceleration report's headline numbers against its data.

Two checks, both mechanical:

1. Every macro in ``tables/numbers.tex`` is recomputed from the CSV files under
   ``data/`` and must match exactly. The macros are what the abstract, results
   and conclusion quote, so this ties the prose to the data without trusting
   the script that wrote both.
2. Every numeric literal (``\\num{...}``, ``\\SI{...}{...}``) in the abstract and
   the conclusion of ``gamma_acceleration.tex`` is listed, and any that does not
   equal a macro value or a value found in the CSVs is reported. Those are the
   numbers a reader would have to take on trust.

Exit status is non-zero if a macro disagrees with the data; unexplained
literals are reported but do not fail the audit, because some are legitimate
(grid dimensions, years, counts stated in the methods).

Example::

    uv run python scripts/analysis/audit_gamma_report.py \\
        --report-dir reports/technical-reports/gamma-pass-rate
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import typer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.gamma_evidence_latex import headline_values  # noqa: E402

app = typer.Typer(help="Audit the report's headline numbers against its CSV data.", add_completion=False)

MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")
LITERAL = re.compile(r"\\(?:num|SI)\{([^}]*)\}")


def read_macros(path: Path) -> Dict[str, str]:
    return {name: value for name, value in MACRO.findall(path.read_text())}


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value in ("", "None"):
                row[key] = None
                continue
            try:
                row[key] = int(value) if re.fullmatch(r"-?\d+", value) else float(value)
            except ValueError:
                pass
    return rows


def section(tex: str, start: str, end: str) -> str:
    begin = tex.find(start)
    stop = tex.find(end, begin + 1) if end else len(tex)
    return tex[begin:stop] if begin >= 0 else ""


def csv_value_set(data_dir: Path) -> Set[str]:
    values: Set[str] = set()
    for path in data_dir.glob("*.csv"):
        for row in read_csv(path):
            for value in row.values():
                if isinstance(value, (int, float)):
                    values.add(f"{value}")
                    for digits in (0, 1, 2, 3, 4, 6):
                        values.add(f"{value:.{digits}f}")
    return values


@app.command()
def main(
    report_dir: Path = typer.Option(..., help="Directory holding gamma_acceleration.tex, tables/ and data/."),
) -> None:
    """Recompute the macros from the CSVs and list unexplained literals."""
    data = report_dir / "data"
    macros = read_macros(report_dir / "tables" / "numbers.tex")
    recomputed = headline_values(
        read_csv(data / "matched.csv"),
        read_csv(data / "beamlet_maps.csv"),
        read_csv(data / "plan_maps.csv"),
        read_csv(data / "pools.csv"),
        read_csv(data / "scaling.csv"),
        read_csv(data / "profile.csv") or None,
    )
    for row in read_csv(data / "plan_timing.csv"):
        suffix = {"1%/1mm/10%": "One", "2%/2mm/10%": "Two", "3%/3mm/10%": "Three"}.get(row["criterion"])
        if suffix:
            recomputed[f"planPairedSingle{suffix}"] = f"{row['rung4_paired_median']:.1f}"
            recomputed[f"planPairedDouble{suffix}"] = f"{row['rung3_paired_median']:.1f}"

    mismatches = {
        name: (value, recomputed.get(name)) for name, value in macros.items() if recomputed.get(name) != value
    }
    missing = sorted(set(macros) - set(recomputed))
    typer.echo(f"{len(macros)} macros in numbers.tex, {len(recomputed)} recomputed from data/")
    for name, (written, actual) in sorted(mismatches.items()):
        typer.echo(f"  MISMATCH {name}: numbers.tex={written!r} data={actual!r}")

    tex = (report_dir / "gamma_acceleration.tex").read_text()
    used = {name for name in macros if f"\\{name}" in tex}
    typer.echo(f"{len(used)} of {len(macros)} macros are used in the report")
    known = set(macros.values()) | csv_value_set(data) | set(json.loads((data / "numbers.json").read_text()).values())
    unexplained = {}
    for label, start, end in (("abstract", r"\begin{abstract}", r"\end{abstract}"),
                              ("conclusion", r"\section{Conclusion}", r"\begin{thebibliography}")):
        text = section(tex, start, end)
        literals = [lit for lit in LITERAL.findall(text) if not any(ch.isalpha() for ch in lit)]
        unexplained[label] = sorted({lit for lit in literals if lit.replace(",", "") not in known})
        typer.echo(
            f"{label}: {len(literals)} numeric literals, {len(unexplained[label])} not backed by data: "
            f"{unexplained[label]}"
        )

    if mismatches or missing:
        typer.echo(f"AUDIT FAILED: {len(mismatches)} mismatches, {len(missing)} macros without data: {missing}")
        raise typer.Exit(code=1)
    typer.echo("AUDIT PASSED: every macro matches the data")


if __name__ == "__main__":
    app()
