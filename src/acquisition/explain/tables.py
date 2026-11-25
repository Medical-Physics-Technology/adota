"""CSV and LaTeX export of the explanation tables.

The LaTeX is written by hand in booktabs style, so this module needs neither jinja2 nor
``DataFrame.to_latex`` / ``Styler``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Sequence

import pandas as pd

LATEX_SPECIALS: Mapping[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in one pass, so no escape is escaped twice."""
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def example_table(placement: pd.DataFrame, contributions: pd.DataFrame, groups: Mapping[str, str],
                  top_features: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Features by example, wide.

    placement: long, columns beamlet_id, feature, value, percentile_ref, robust_z (other columns ignored).
    contributions: long, columns beamlet_id, feature, contribution (one scorer only; other columns ignored).
    groups: {feature: group name}.
    top_features: {beamlet_id: features to flag}.

    Returns one row per feature (order of first appearance in ``placement``), columns:
    feature, group, then for each beamlet_id in order of first appearance in ``placement``:
    f"{b} value", f"{b} percentile", f"{b} z", f"{b} contribution", f"{b} top" (bool).
    """
    features = list(dict.fromkeys(placement["feature"]))
    beamlets = list(dict.fromkeys(placement["beamlet_id"]))
    out = pd.DataFrame({
        "feature": features,
        "group": [groups.get(f, "") for f in features],
    })
    for b in beamlets:
        here = placement[placement["beamlet_id"] == b].set_index("feature")
        contrib = contributions[contributions["beamlet_id"] == b].set_index("feature")["contribution"]
        flagged = set(top_features.get(b, ()))
        out[f"{b} value"] = here["value"].reindex(features).to_numpy()
        out[f"{b} percentile"] = here["percentile_ref"].reindex(features).to_numpy()
        out[f"{b} z"] = here["robust_z"].reindex(features).to_numpy()
        out[f"{b} contribution"] = contrib.reindex(features).to_numpy()
        out[f"{b} top"] = [f in flagged for f in features]
    return out


def _is_flag(column: pd.Series, name: str, suffix: str) -> bool:
    return bool(suffix) and str(name).endswith(suffix) and pd.api.types.is_bool_dtype(column)


def _is_numeric(column: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def _cell(value, dtype, float_format: str) -> str:
    if pd.isna(value):
        return "--"
    if pd.api.types.is_bool_dtype(dtype):
        return str(bool(value))
    if pd.api.types.is_integer_dtype(dtype):
        return str(int(value))
    if pd.api.types.is_float_dtype(dtype):
        return float_format.format(float(value))
    return latex_escape(str(value))


def write_csv_and_latex(frame: pd.DataFrame, path_stem: Path, *, caption: str, label: str,
                        float_format: str = "{:.2f}", bold_column_suffix: str = " top",
                        drop_columns: Sequence[str] = ()) -> List[Path]:
    """Write ``{path_stem}.csv`` (all columns, full precision, index=False) and ``{path_stem}.tex``.

    LaTeX: a ``table`` environment with ``\\centering``, ``\\caption{caption}``, ``\\label{label}``,
    a ``tabular`` with ``l`` for text columns and ``r`` for numeric ones, ``\\toprule``/``\\midrule``/``\\bottomrule``,
    escaped header and text cells, floats via ``float_format``, NaN as ``--``. Boolean columns whose name ends
    with ``bold_column_suffix`` are not printed; instead, in that row, every printed cell whose column name
    starts with the same prefix (the part before the suffix) is wrapped in ``\\textbf{}``.
    ``drop_columns`` are omitted from the .tex only. Creates parent directories. Returns [csv_path, tex_path].
    """
    path_stem = Path(path_stem)
    csv_path = path_stem.parent / f"{path_stem.name}.csv"
    tex_path = path_stem.parent / f"{path_stem.name}.tex"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(csv_path, index=False)

    cols = list(frame.columns)
    flags = {
        c: str(c)[: -len(bold_column_suffix)]
        for c in cols
        if _is_flag(frame[c], c, bold_column_suffix)
    }
    shown = [c for c in cols if c not in drop_columns and c not in flags]
    align = "".join("r" if _is_numeric(frame[c]) else "l" for c in shown)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(latex_escape(str(c)) for c in shown) + " \\\\",
        "\\midrule",
    ]
    for rec in frame.itertuples(index=False, name=None):
        values = dict(zip(cols, rec))
        bold = set()
        for flag_col, prefix in flags.items():
            if not pd.isna(values[flag_col]) and bool(values[flag_col]):
                bold.update(c for c in shown if str(c) == prefix or str(c).startswith(prefix + " "))
        cells = []
        for c in shown:
            text = _cell(values[c], frame[c].dtype, float_format)
            cells.append(f"\\textbf{{{text}}}" if c in bold else text)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, tex_path]
