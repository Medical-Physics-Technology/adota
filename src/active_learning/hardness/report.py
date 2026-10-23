"""The hardness report: ``what_makes_a_beamlet_hard.md`` from the result CSVs.

The report opens with the three sentences the analysis exists for, filled in
from the results (which clusters matter, in which direction, with their
intervals), followed by one table per analysis step and the limitations. It is
regenerated on every run, so a rerun on retrained members rewrites it; any
interpretation beyond what the numbers say belongs in the experiment record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.active_learning.hardness.analysis import CONSENSUS, residual_name
from src.active_learning.hardness.cells import MetricClusters

METRIC_TEXT: Dict[str, str] = {
    "energy_mev": "beam energy",
    "ct_max_hu": "the densest voxel in the beam's path (bone, metal)",
    "bp_range_min_mm": "depth of the proximal edge of the Bragg-peak zone",
    "bp_range_max_mm": "depth of the distal edge of the Bragg-peak zone",
    "max_grad_depth_mm": "depth of the steepest depth-dose gradient",
    "n_density_regions": "number of tissue classes crossed in the Bragg-peak zone",
    "total_hu_change": "summed HU change between tissues in the Bragg-peak zone",
    "max_hu_jump": "largest HU jump between tissues in the Bragg-peak zone",
    "sigma_hu_bp": "spread of the beam-weighted HU along the Bragg-peak zone",
    "max_hu_gradient": "steepest slice-to-slice HU change in the Bragg-peak zone",
    "lateral_hu_var_bp": "lateral HU variance across the beam at the Bragg peak",
    "hetero_fraction": "share of the Bragg-peak zone outside its dominant tissue",
    "interface_bp_distance": "distance from the Bragg peak to the nearest tissue interface",
    "mean_sobel_axial": "mean CT edge strength along the beam axis",
    "p95_sobel_bp": "strong CT edges around the Bragg peak",
    "sum_sobel_bp": "total CT edge strength around the Bragg peak",
    "sobel_dw_mean": "dose-weighted CT edge strength around the peak",
    "sobel_dw_anisotropy": "directionality of the dose-weighted CT edges",
    "sobel_dw_edge_energy": "dose-weighted CT edge energy around the peak",
    "sobel_th_mean": "CT edge strength in the high-dose region",
    "sobel_th_anisotropy": "directionality of the CT edges in the high-dose region",
    "sobel_th_edge_energy": "CT edge energy in the high-dose region",
    "lateral_edge_energy": "interfaces running along the beam near the peak (lateral edges)",
    "pflugfelder_hi": "lateral heterogeneity of the water-equivalent path (Pflugfelder index)",
    "wepl_mean": "water-equivalent depth of the Bragg peak",
    "wepl_std": "spread of the water-equivalent path across the beam",
    "isi_sum": "summed severity of the density interfaces crossed",
    "isi_max": "the most severe density interface crossed",
    "isi_mean": "mean severity of the density interfaces crossed",
    "isi_axial_sum": "summed severity of the interfaces along the beam axis",
}


def _fmt(value: float, lo: Optional[float] = None, hi: Optional[float] = None, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        text += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return text


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def clusters_from_csv(path: Path) -> MetricClusters:
    """The clusters as ``analyse`` wrote them to ``clusters.csv``."""
    table = pd.read_csv(path)
    assignment = {m: int(r.cluster) for r in table.itertuples() for m in r.members.split()}
    signs = {m: (1.0 if s == "+" else -1.0) for r in table.itertuples()
             for m, s in zip(r.members.split(), r.signs.split())}
    return MetricClusters(assignment, dict(zip(table.cluster, table.name)), dict(zip(table.cluster, table.medoid)),
                          signs, pd.DataFrame(), float("nan"))


def figure_tables(specs: Mapping[str, Any], metrics: Sequence[str], res: Path, clusters: MetricClusters
                  ) -> Iterator[Tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]]:
    """Per table and error metric, the metrics-by-targets matrices of the
    univariate step (value, interval) with rows in cluster order."""
    order = [m for c in clusters.ids for m in clusters.members(c)]
    for name in specs:
        for metric in metrics:
            uni = pd.read_csv(res / f"univariate_{name}_{metric}.csv").query("kind == 'metric'")
            cols = list(dict.fromkeys(uni["target"]))
            grids = [uni.pivot(index="name", columns="target", values=v).loc[order, cols]
                     for v in ("median_cell_spearman", "ci_lo", "ci_hi")]
            yield (name, metric, *grids, [f"c{clusters.assignment[m]}" for m in order])


def _importance(res: Path, table: str, metric: str, target: str) -> pd.DataFrame:
    path = res / f"importance_{table}_{metric}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).query("target == @target").sort_values("importance", ascending=False)


def _evaluation(res: Path, table: str, metric: str, target: str) -> Optional[pd.Series]:
    path = res / f"fit_evaluation_{table}_{metric}.csv"
    if not path.exists():
        return None
    part = pd.read_csv(path).query("score == @target and target == @target")
    return part.iloc[0] if len(part) else None


def _direction(res: Path, table: str, metric: str, target: str, cluster_name: str) -> Tuple[str, pd.Series]:
    uni = pd.read_csv(res / f"univariate_{table}_{metric}.csv")
    row = uni.query("kind == 'cluster' and target == @target and name == @cluster_name").iloc[0]
    return ("larger" if row["median_cell_spearman"] > 0 else "smaller"), row


def sentence(res: Path, table: str, metric: str, target: str, clusters: MetricClusters, opening: str,
             max_clusters: int = 2) -> str:
    """One of the three sentences: the target's leading clusters, each with its
    direction and univariate Spearman, then the out-of-fold Spearman of the
    whole score.

    A cluster leads only when both its importance interval and its own
    within-cell Spearman interval exclude zero, in importance order. Importance
    alone is not enough: a cluster that matters in opposite directions for
    different engines (peak depth for the U-Net against the attention models)
    can help the multivariate fit while its own association with the target is
    nil, and naming it would misstate what is hard.
    """
    imp = _importance(res, table, metric, target)
    ev = _evaluation(res, table, metric, target)
    if imp.empty or ev is None:
        return f"{opening} ... (no result for {target} on table {table})."
    rows = []
    for row in imp[imp["ci_lo"] > 0].itertuples():
        direction, uni = _direction(res, table, metric, target, row.cluster_name)
        if uni["ci_lo"] > 0 or uni["ci_hi"] < 0:
            rows.append((row, direction, uni))
    weak = ""
    if not rows:
        row = imp.iloc[0]
        rows = [(row, *_direction(res, table, metric, target, row["cluster_name"]))]
        weak = " (no cluster is both important and individually associated; the leading one is shown)"
    parts = []
    for row, direction, uni in rows[:max_clusters]:
        row = pd.Series(row._asdict()) if hasattr(row, "_asdict") else row
        medoid = clusters.medoid[int(row["cluster"])]
        size = len(clusters.members(int(row["cluster"])))
        family = f" and {size - 1} correlated metrics" if size > 1 else ""
        parts.append(f"the {METRIC_TEXT.get(medoid, medoid)} is {direction} ({medoid}{family}; within-cell "
                     f"Spearman {_fmt(uni['median_cell_spearman'], uni['ci_lo'], uni['ci_hi'])}, importance "
                     f"{_fmt(row['importance'], row['ci_lo'], row['ci_hi'], 3)})")
    return (f"{opening} " + "; and when ".join(parts) + f"{weak}. The thirty-metric score fitted to it reaches an "
            f"out-of-fold within-cell Spearman of {_fmt(ev['median_cell_spearman'], ev['ci_lo'], ev['ci_hi'])} "
            f"(table {table}, {int(ev['n_records'])} beamlets, {int(ev['n_scans'])} scans).")


def _table_b_coverage(tables_dir: Path, metrics: Sequence[str]) -> str:
    path = tables_dir / "table_B.csv"
    if not path.exists():
        return "Table B was not built."
    frame = pd.read_csv(path, usecols=lambda c: c in {*metrics, "cell", "peak_inside_crop"})
    rows = []
    for m in metrics:
        missing = int(frame[m].isna().sum())
        constant = int((frame.groupby("cell")[m].nunique() <= 1).sum())
        rows.append({"metric": m, "undefined": missing, "cells_without_variation": constant})
    table = pd.DataFrame(rows)
    flagged = table[(table["undefined"] > 0) | (table["cells_without_variation"] > 0)]
    head = (f"Every one of the thirty metrics was computed for all {len(frame):,} beamlets of table B."
            if (table["undefined"] == 0).all() else "Some metrics are undefined for some beamlets of table B.")
    if flagged.empty:
        return head
    return (head + " Metrics that carry no within-cell information in table B (constant inside a cell, so their "
            "within-cell Spearman is undefined there):\n\n" + markdown_table(flagged))


def write_report(out: Path, cfg: Mapping[str, Any], engines: Sequence[Any]) -> Path:
    """Write ``what_makes_a_beamlet_hard.md`` into ``out``."""
    res = out / "results"
    metric = str(cfg["primary_metric"])
    clusters = clusters_from_csv(res / "clusters.csv")
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    labels = {e.label for e in engines}
    u0 = next((r for r in cfg["residuals"] if r in labels and r.startswith("U")), None)
    pba = next((e.label for e in engines if e.family == "pencil_beam"), None)
    lines = [f"# What makes a beamlet hard, per dose engine ({cfg['experiment']})", ""]
    lines += ["Error: gamma failure (1 - pass rate, 2%/2 mm, 10% cutoff, global) unless stated. Within-cell: inside "
              f"one scan x 20 MeV cell (n >= {int(cfg['min_cell_size'])}); intervals: 95% bootstrap over scans. "
              "Associations, not causes.", ""]
    lines.append("1. " + sentence(res, "A", metric, CONSENSUS, clusters, "A beamlet is hard for every engine when"))
    if u0:
        lines.append("2. " + sentence(res, "A", metric, residual_name(u0), clusters,
                                      f"It is additionally hard for the U-Net ({u0}, no sequence mixer) when"))
    if pba:
        lines.append("3. " + sentence(res, "B", metric, residual_name(pba), clusters,
                                      "It is hard for the pencil beam (beyond the consensus) when"))
    lines.append("")
    check = summary.get("agreement_check")
    if check:
        lines += [f"Sanity check: within-cell gamma Spearman between {check['a']} and {check['b']} on table A is "
                  f"{_fmt(check['median_cell_spearman'], check['ci_lo'], check['ci_hi'])} (agreement analysis: "
                  f"{check['expected']}).", ""]
    lines += ["## Data", ""]
    pops = [{"table": t, "error": m, **v} for t, by in summary.get("tables", {}).items() for m, v in by.items()]
    if pops:
        lines += [markdown_table(pd.DataFrame(pops)), ""]
    tables_dir = Path(cfg.get("tables_dir") or out / "tables")
    lines += [_table_b_coverage(tables_dir, list(METRIC_TEXT)), ""]
    leak = tables_dir / "table_b_geometry_leakage.csv"
    if leak.exists():
        frame = pd.read_csv(leak)
        lines += [f"Leakage check of table B: {int((frame['n_matches'] > 0).sum())} of {len(frame)} CTs share their "
                  "grid (x size, z size, z origin) with a scan of the training HDF5.", ""]
    lines += ["## 2. Metric clusters", "", markdown_table(pd.read_csv(res / "clusters.csv")
                                                      [["cluster", "name", "n_metrics", "members", "min_abs_rho"]]
                                                      .round(2)), ""]
    for step, title in (("univariate", "1. Univariate, cluster composites"), ("fit", "3-4. Out-of-fold scores"),
                        ("importance", "3-4. Cluster importance"), ("transfer", "5. Cross transfer"),
                        ("strata", "6. Strata")):
        lines += [f"## {title}", ""]
        for table in ("A", "B"):
            block = _step_table(res, step, table, metric, clusters)
            if block:
                lines += [f"Table {table}:", "", block, ""]
    examples = out / "examples" / "examples.csv"
    if examples.exists():
        ex = pd.read_csv(examples)
        lines += ["## 7. Examples", "", markdown_table(ex[["cluster_name", "group", "rank", "cell", "sample_id",
                                                           "composite"]].round(3)), ""]
    lines += ["## Limitations", ""] + [f"- {t}" for t in LIMITATIONS] + [""]
    path = out / "what_makes_a_beamlet_hard.md"
    path.write_text("\n".join(lines))
    return path


def _step_table(res: Path, step: str, table: str, metric: str, clusters: MetricClusters) -> str:
    tag = f"{table}_{metric}"
    if step == "univariate":
        path = res / f"univariate_{tag}.csv"
        if not path.exists():
            return ""
        uni = pd.read_csv(path).query("kind == 'cluster'")
        uni["value"] = [_fmt(v, lo, hi) for v, lo, hi in zip(uni["median_cell_spearman"], uni["ci_lo"], uni["ci_hi"])]
        return markdown_table(uni.pivot(index="name", columns="target", values="value").reset_index())
    if step == "fit":
        path = res / f"fit_evaluation_{tag}.csv"
        if not path.exists():
            return ""
        ev = pd.read_csv(path)
        ev["out_of_fold_spearman"] = [_fmt(v, lo, hi) for v, lo, hi in
                                      zip(ev["median_cell_spearman"], ev["ci_lo"], ev["ci_hi"])]
        return markdown_table(ev[["target", "out_of_fold_spearman", "n_cells", "n_scans", "n_records"]])
    if step == "importance":
        path = res / f"importance_{tag}.csv"
        if not path.exists():
            return ""
        imp = pd.read_csv(path)
        imp["value"] = [_fmt(v, lo, hi, 3) for v, lo, hi in zip(imp["importance"], imp["ci_lo"], imp["ci_hi"])]
        top = imp.groupby("target")["importance"].max()
        keep = imp.groupby("cluster")["importance"].max().sort_values(ascending=False).head(8).index
        grid = imp[imp["cluster"].isin(keep)].pivot(index="cluster_name", columns="target", values="value")
        return markdown_table(grid[[t for t in top.index]].reset_index())
    if step == "transfer":
        path, summary = res / f"transfer_{tag}.csv", res / f"transfer_summary_{tag}.csv"
        if not path.exists():
            return ""
        m = pd.read_csv(path)
        grid = m.pivot(index="score", columns="target", values="median_cell_spearman").round(2)
        s = pd.read_csv(summary)
        text = ", ".join(f"{r.quantity} {_fmt(r.value, r.ci_lo, r.ci_hi)}" for r in s.itertuples())
        return markdown_table(grid.reset_index()) + f"\n\nOver the engines: {text}."
    if step == "strata":
        path = res / f"strata_evaluation_{tag}.csv"
        if not path.exists():
            return ""
        ev = pd.read_csv(path)
        imp = pd.read_csv(res / f"strata_importance_{tag}.csv")
        best = imp.sort_values("importance", ascending=False).groupby(["stratum", "target"]).head(1) \
            .set_index(["stratum", "target"])
        rows = []
        for r in ev.itertuples():
            b = best.loc[(r.stratum, r.target)] if (r.stratum, r.target) in best.index else None
            rows.append({"stratum": r.stratum, "target": r.target, "n": r.n_records, "scans": r.n_scans,
                         "out_of_fold_spearman": _fmt(r.median_cell_spearman, r.ci_lo, r.ci_hi),
                         "top_cluster": b["cluster_name"] if b is not None else "",
                         "importance": _fmt(b["importance"], b["ci_lo"], b["ci_hi"], 3) if b is not None else ""})
        return markdown_table(pd.DataFrame(rows))
    return ""


LIMITATIONS = (
    "Correlation inside cells is not causation: a cluster is associated with an engine's error, and a metric "
    "correlated with the cluster's members could carry the same association.",
    "Table A has 8 validation scans and table B 6 CTs at 2 energies; the scan bootstrap over so few scans gives "
    "wide, possibly optimistic intervals, and table B's 12 cells make every median a median of few values.",
    "The HPO finalists are single, partly unstable trainings (C1 stopped at 9 epochs and is left out of the "
    "consensus); the residual of one engine mixes its architecture with its training noise. The same command reruns "
    "on retrained members.",
    "The metrics are computed with the analytic dose surrogate (input-only arm), so the peak location and the "
    "dose-weighted edge metrics are those of the surrogate, not of the Monte Carlo dose. Beamlets whose analytic "
    "peak lies outside the 320 mm crop, and energies above 250 MeV (beyond MCsquare's nuclear data), are excluded "
    "from the primary population; the sensitivity rows refit on every beamlet.",
    "Table B's published ADoTA errors use the saved prediction, upsampled to 1 mm when it was written and resampled "
    "back to 2 mm here; the comparison file's own 1 mm errors are kept beside them as a cross-check.",
)
