"""One row per test beamlet: metadata, difficulty score and both engines' errors (stage ``score_table``).

The difficulty score is score v2 of EXP-0022, read from its population decomposition
(``analysis.score_contributions``): ``difficulty_score = total + baseline``, where ``total`` is the
score minus the reference set's mean and equals the sum of the group contributions (the score is
linear). The analyse stage builds the score the same way. Beamlets whose analytic peak lies outside
the crop have no features and therefore no score; they keep their row with empty score columns.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

__all__ = ["ERROR_COLUMNS", "beamlet_score_table", "contribution_name", "run_score_table"]

ERROR_COLUMNS = ("gpr_2pct_2mm_10pct", "gpr_3pct_3mm_10pct", "mape_pct", "rde_pct", "dr80_mm", "peak_ratio",
                 "integral_ratio")
META_COLUMNS = ("sample_id", "patient_id", "site", "field_id", "gantry_deg", "energy_mev", "spot_x_mm",
                "spot_y_mm", "in_analysis")
_NON_GROUP = {"sample_id", "total", "baseline", "decile", "patient_id", "anatomy"}


def contribution_name(group: str) -> str:
    """``"depth and energy [max_grad_depth_mm]"`` -> ``"contrib_max_grad_depth_mm"``: the group's lead
    feature, since two groups can share a plain-language label."""
    match = re.search(r"\[([^\]]+)\]\s*$", group)
    label = match.group(1) if match else group
    return "contrib_" + re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def beamlet_score_table(metadata: pd.DataFrame, contributions: pd.DataFrame,
                        errors: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Join metadata (all beamlets), the score with its group contributions, and each engine's errors
    (columns prefixed by the mapping's key, e.g. ``adota_mape_pct``).

    Raises ValueError if a scored beamlet is missing from the metadata, its site disagrees, a scored
    beamlet is not in the analysis population, two groups map to one column name, or an engine's
    error table does not cover exactly the metadata's beamlets.
    """
    meta = metadata[list(META_COLUMNS)].copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    contrib = contributions.copy()
    contrib["sample_id"] = contrib["sample_id"].astype(str)
    unknown = sorted(set(contrib["sample_id"]) - set(meta["sample_id"]))
    if unknown:
        raise ValueError(f"{len(unknown)} scored beamlets are not in the metadata, e.g. {unknown[:3]}")
    groups = [c for c in contrib.columns if c not in _NON_GROUP]
    names = {g: contribution_name(g) for g in groups}
    if len(set(names.values())) != len(names):
        raise ValueError(f"group names collide after renaming: {names}")
    score = pd.DataFrame({"sample_id": contrib["sample_id"],
                          "difficulty_score": contrib["total"] + contrib["baseline"],
                          "score_decile": contrib["decile"].astype("Int64"),
                          "score_baseline": contrib["baseline"],
                          **{names[g]: contrib[g] for g in groups}})
    table = meta.merge(score, on="sample_id", how="left", validate="1:1")
    if "anatomy" in contrib.columns:
        site = meta.set_index("sample_id")["site"].reindex(contrib["sample_id"]).to_numpy()
        if (site != contrib["anatomy"].to_numpy()).any():
            raise ValueError("score file anatomy disagrees with the metadata site")
    scored = table["difficulty_score"].notna()
    if (scored & ~table["in_analysis"].astype(bool)).any():
        raise ValueError("a beamlet outside the analysis population has a score")
    for prefix, err in errors.items():
        err = err.copy()
        err["sample_id"] = err["sample_id"].astype(str)
        if err["sample_id"].duplicated().any() or set(err["sample_id"]) != set(meta["sample_id"]):
            raise ValueError(f"{prefix} errors do not cover exactly the metadata's beamlets")
        cols = [c for c in ERROR_COLUMNS if c in err.columns]
        table = table.merge(err[["sample_id", *cols]].rename(columns={c: f"{prefix}_{c}" for c in cols}),
                            on="sample_id", how="left", validate="1:1")
    return table.sort_values("sample_id").reset_index(drop=True)


def run_score_table(cfg: Mapping, run: Path) -> List[str]:
    """The stage: writes ``analysis/score_table/beamlet_score_and_errors.csv`` and returns the summary lines."""
    out = run / "analysis" / "score_table"
    out.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(run / "analysis" / "dataset_overview" / "beamlet_metadata.csv", dtype={"sample_id": str})
    contributions = pd.read_csv(cfg["analysis"]["score_contributions"], dtype={"sample_id": str})
    errors: Dict[str, pd.DataFrame] = {e: pd.read_csv(run / "tables" / f"errors_{e}.csv", dtype={"sample_id": str})
                                       for e in cfg["engines"]}
    table = beamlet_score_table(metadata, contributions, errors)
    table.to_csv(out / "beamlet_score_and_errors.csv", index=False)
    groups = [c for c in contributions.columns if c not in _NON_GROUP]
    pd.DataFrame({"column": [contribution_name(g) for g in groups], "group": groups}).to_csv(
        out / "contribution_groups.csv", index=False)
    scored = table["difficulty_score"].notna()
    lines = ["# Difficulty score and errors per beamlet", "",
             f"`beamlet_score_and_errors.csv`: {len(table)} beamlets, {int(scored.sum())} with a difficulty score "
             "(the analysis population; the others have the analytic peak outside the crop, so no features).", "",
             "- Score: score v2 (EXP-0022), `difficulty_score = total + baseline` from "
             f"`{cfg['analysis']['score_contributions']}`; "
             "`score_decile` 1 (easiest) to 10; `contrib_<lead feature>` are the group contributions (group labels in "
             "`contribution_groups.csv`), which add up to "
             "`difficulty_score - score_baseline`.",
             "- Errors: `<engine>_` prefix (adota = EXP-0019 model, pba = pencil beam), from "
             "`tables/errors_<engine>.csv`; "
             "GPR in %, MAPE and RDE in %, dR80 in mm (engine minus Monte Carlo), peak and integral dose ratios "
             "engine / Monte Carlo.", "",
             "| site | beamlets | scored | score min | score median | score max |", "|---|---|---|---|---|---|"]
    for site, part in table.groupby("site"):
        s = part["difficulty_score"].dropna()
        lines.append(f"| {site} | {len(part)} | {s.size} | {s.min():.3f} | {s.median():.3f} | {s.max():.3f} |")
    residual = (table.loc[scored, [c for c in table.columns if c.startswith("contrib_")]].sum(axis=1)
                - (table.loc[scored, "difficulty_score"] - table.loc[scored, "score_baseline"])).abs().max()
    lines += ["", f"Largest |sum of contributions - (score - baseline)|: {float(np.nan_to_num(residual)):.2e}.", ""]
    return lines
