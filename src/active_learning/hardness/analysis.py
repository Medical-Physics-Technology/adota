"""The analysis steps on one table, for one error metric.

Every step uses the cells of :mod:`.cells` (scan times 20 MeV bin, at least
``min_cell_size`` records) and the scan bootstrap for its intervals.

1. univariate: median within-cell Spearman of every metric (and every cluster
   composite) with every target;
2. the metric clusters are computed by the caller once and passed in;
3. per target, the score refit out of fold by scan, its standardised
   coefficients and the permutation importance of every cluster;
4. the targets include the **consensus** (mean within-cell error rank over the
   consensus engines) and per engine the **residual**, its own within-cell rank
   minus the consensus;
5. cross transfer: every engine-fitted score against every engine's error;
6. strata: steps 1 and 3-4 inside energy terciles and sites, with cells cut at
   the stratum boundary.

A *target* is an engine's error or a derived rank; all of them are oriented so
that larger means harder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.active_learning.hardness.cells import (
    MetricClusters,
    cell_ranks,
    cell_spearman,
    cluster_composites,
    eligible_cells,
    summarise_correlation,
)
from src.active_learning.hardness.fit import (
    coefficients,
    evaluate_scores,
    out_of_fold,
    permutation_importance,
    transfer_summary,
)
from src.active_learning.hardness.tables import error_column

logger = logging.getLogger(__name__)

CONSENSUS = "consensus"


def residual_name(engine: str) -> str:
    return f"{engine} residual"


@dataclass
class Settings:
    """How every step is run; the defaults are the agreement analysis's."""

    metrics_used: List[str]                    # the physics metrics (all thirty by default)
    min_cell_size: int = 10
    n_bootstrap: int = 2000
    n_bootstrap_coef: int = 500
    n_perm_repeats: int = 5
    seed: int = 20261023
    ci: float = 0.95
    population: str = "inside_crop"            # inside_crop | all
    max_energy_mev: Optional[float] = 250.0
    energy_terciles: Tuple[Tuple[float, float], ...] = ((70.0, 110.0), (110.0, 152.0), (152.0, 270.0))


@dataclass
class TableSpec:
    """One table and the roles of its engines."""

    name: str
    frame: pd.DataFrame
    engines: List[str]
    consensus: List[str]
    residuals: List[str] = field(default_factory=list)


@dataclass
class Population:
    """The rows one analysis runs on and its targets, aligned by position."""

    frame: pd.DataFrame
    targets: pd.DataFrame
    engines: List[str]
    excluded: Dict[str, int]


def population(table: TableSpec, metric: str, settings: Settings, cell_column: str = "cell",
               rows: Optional[pd.Series] = None) -> Population:
    """The analysis rows of ``table`` for ``metric``: valid metrics, inside the
    crop (unless ``population`` is ``all``), at most ``max_energy_mev``, a defined
    error for every engine, in a cell of at least ``min_cell_size`` records; and
    the targets (engine errors, consensus, residuals) on those rows."""
    frame = table.frame if rows is None else table.frame[rows.to_numpy()]
    keep = pd.Series(True, index=frame.index)
    excluded: Dict[str, int] = {"n_table": len(frame)}

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal keep
        excluded[reason] = int((keep & ~mask).sum())
        keep &= mask

    drop(frame[settings.metrics_used].notna().all(axis=1), "metrics_undefined")
    if settings.population == "inside_crop":
        drop(frame["peak_inside_crop"].astype(bool), "peak_outside_crop")
    elif settings.population != "all":
        raise ValueError(f"population must be inside_crop or all, got {settings.population!r}")
    if settings.max_energy_mev is not None:
        drop(frame["energy_mev"] <= settings.max_energy_mev, "above_max_energy")
    errors = pd.DataFrame({e: frame[error_column(metric, e)] for e in table.engines})
    drop(errors.notna().all(axis=1), "error_undefined")
    sub = frame[keep]
    drop_cells = eligible_cells(sub[cell_column], settings.min_cell_size)
    excluded["small_cell"] = int((~drop_cells).sum())
    sub = sub[drop_cells.to_numpy()].reset_index(drop=True)
    excluded["n_used"] = len(sub)
    sub = sub.assign(cell=sub[cell_column])

    targets = pd.DataFrame({e: sub[error_column(metric, e)].to_numpy() for e in table.engines})
    ranks = cell_ranks(targets, sub["cell"])
    if table.consensus:
        targets[CONSENSUS] = ranks[table.consensus].mean(axis=1).to_numpy()
        for e in table.residuals:
            if e in ranks:
                targets[residual_name(e)] = (ranks[e] - targets[CONSENSUS]).to_numpy()
    return Population(sub, targets, list(table.engines), excluded)


def univariate(pop: Population, clusters: MetricClusters, settings: Settings) -> pd.DataFrame:
    """Step 1: per target, every metric's and every cluster composite's median
    within-cell Spearman with the target."""
    feats = pop.frame[settings.metrics_used]
    composites = cluster_composites(clusters, feats, pop.frame["cell"])
    rows = []
    for t in pop.targets.columns:
        for kind, x in (("metric", feats), ("cluster", composites)):
            corr = cell_spearman(x, pop.targets[t], pop.frame["cell"], pop.frame["scan"], settings.min_cell_size)
            table = summarise_correlation(corr, settings.n_bootstrap, settings.seed, settings.ci, label="name")
            table.insert(0, "kind", kind)
            table.insert(0, "target", t)
            if kind == "metric":
                table["cluster"] = [clusters.assignment[m] for m in table["name"]]
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


@dataclass
class FitResults:
    evaluation: pd.DataFrame
    coefficients: pd.DataFrame
    importance: pd.DataFrame
    oof_scores: pd.DataFrame
    fits: Dict[str, object]


def fit_targets(pop: Population, clusters: MetricClusters, settings: Settings,
                targets: Optional[Sequence[str]] = None, with_coefficients: bool = True) -> FitResults:
    """Steps 3-4: per target the out-of-fold score, its within-cell Spearman with
    that target, the standardised coefficients and the cluster importance."""
    metrics, cells, scans = settings.metrics_used, pop.frame["cell"], pop.frame["scan"]
    names = list(targets or pop.targets.columns)
    scores, evals, coefs, imps, fits = {}, [], [], [], {}
    for t in names:
        oof = out_of_fold(pop.frame, metrics, pop.targets[t], cells, scans)
        scores[t] = oof.score
        fits[t] = oof
        evals.append(evaluate_scores(pd.DataFrame({t: oof.score}), pop.targets[[t]], cells, scans,
                                     settings.min_cell_size, settings.n_bootstrap, settings.seed, settings.ci))
        if with_coefficients:
            coefs.append(coefficients(pop.frame, metrics, pop.targets[t], cells, scans, clusters,
                                      settings.n_bootstrap_coef, settings.seed, settings.ci).assign(target=t))
        imps.append(permutation_importance(oof, pop.targets[t], cells, scans, clusters, settings.min_cell_size,
                                           settings.n_perm_repeats, settings.n_bootstrap, settings.seed,
                                           settings.ci).assign(target=t))
        logger.info("fit %-22s out-of-fold median within-cell Spearman %.3f", t,
                    float(evals[-1]["median_cell_spearman"].iloc[0]))
    return FitResults(pd.concat(evals, ignore_index=True),
                      pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame(),
                      pd.concat(imps, ignore_index=True), pd.DataFrame(scores), fits)


def cross_transfer(pop: Population, fits: FitResults, settings: Settings,
                   extra_scores: Sequence[str] = (CONSENSUS,)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Step 5: the out-of-fold score fitted on each engine (and on the consensus)
    against every engine's error; the long matrix and the diagonal summary over
    the engines."""
    engines = [e for e in pop.engines if e in fits.oof_scores]
    rows = [e for e in [*engines, *extra_scores] if e in fits.oof_scores]
    matrix = evaluate_scores(fits.oof_scores[rows], pop.targets[engines], pop.frame["cell"], pop.frame["scan"],
                             settings.min_cell_size, settings.n_bootstrap, settings.seed, settings.ci)
    per_cell = {s: cell_spearman(pop.targets[engines], pd.Series(fits.oof_scores[s].to_numpy()), pop.frame["cell"],
                                 pop.frame["scan"], settings.min_cell_size) for s in engines}
    scans_of_cells = next(iter(per_cell.values())).cells["scan"]
    summary = transfer_summary(matrix, {s: c.values for s, c in per_cell.items()}, scans_of_cells, engines,
                               settings.n_bootstrap, settings.seed, settings.ci)
    return matrix, summary


def strata(table: TableSpec) -> Dict[str, pd.Series]:
    """Row masks of the strata of step 6; the energy terciles come from the
    caller's settings, the sites from the table."""
    out: Dict[str, pd.Series] = {}
    for site in sorted(table.frame["site"].dropna().unique()):
        out[f"site={site}"] = table.frame["site"] == site
    return out


def stratified(table: TableSpec, metric: str, clusters: MetricClusters, settings: Settings,
               targets: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Step 6: univariate and the consensus / residual fits inside each stratum.

    Cells are cut at the stratum boundary (a cell straddling 152 MeV becomes two),
    so a stratum's within-cell ranks are its own. A stratum with fewer than two
    scans cannot be fitted out of fold and is skipped for the fits.
    """
    masks = {f"energy={lo:g}-{hi:g}": (table.frame["energy_mev"] >= lo) & (table.frame["energy_mev"] < hi)
             for lo, hi in settings.energy_terciles}
    masks.update(strata(table))
    uni, evals, imps, pops = [], [], [], []
    for name, mask in masks.items():
        frame = table.frame.assign(stratum_cell=table.frame["cell"] + "|" + name)
        sub_table = TableSpec(table.name, frame, table.engines, table.consensus, table.residuals)
        try:
            pop = population(sub_table, metric, settings, cell_column="stratum_cell", rows=mask)
        except ValueError as exc:
            logger.warning("stratum %s of table %s: %s", name, table.name, exc)
            continue
        pops.append({"stratum": name, "n_scans": pop.frame["scan"].nunique(), **pop.excluded})
        if pop.frame.empty or pop.frame["cell"].nunique() == 0:
            continue
        uni.append(univariate(pop, clusters, settings).assign(stratum=name))
        present = [t for t in targets if t in pop.targets.columns]
        if pop.frame["scan"].nunique() >= 2 and present:
            res = fit_targets(pop, clusters, settings, present, with_coefficients=False)
            evals.append(res.evaluation.assign(stratum=name))
            imps.append(res.importance.assign(stratum=name))
    cat = lambda frames: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  # noqa: E731
    return {"univariate": cat(uni), "evaluation": cat(evals), "importance": cat(imps),
            "populations": pd.DataFrame(pops)}


def composite_examples(pop: Population, clusters: MetricClusters, cluster_ids: Sequence[int],
                       settings: Settings, n: int = 3, cell: Optional[str] = None) -> pd.DataFrame:
    """Step 7: per cluster, the ``n`` records with the highest and the ``n`` with
    the lowest composite inside one cell (the largest by default), so scan and
    energy are held fixed between the two groups."""
    feats = pop.frame[settings.metrics_used]
    comp = cluster_composites(clusters, feats, pop.frame["cell"], cluster_ids)
    cell = cell or str(pop.frame["cell"].value_counts().idxmax())
    in_cell = (pop.frame["cell"] == cell).to_numpy()
    rows = []
    for c in cluster_ids:
        name = clusters.names[c]
        values = comp[name][in_cell].sort_values()
        for group, picked in (("low", values.index[:n]), ("high", values.index[::-1][:n])):
            for rank, i in enumerate(picked, 1):
                rows.append({"cluster": c, "cluster_name": name, "group": group, "rank": rank, "cell": cell,
                             "composite": float(comp.loc[i, name]), "sample_id": pop.frame.loc[i, "sample_id"],
                             **{f"target:{t}": float(pop.targets.loc[i, t]) for t in pop.targets.columns},
                             **{m: float(pop.frame.loc[i, m]) for m in clusters.members(c)}})
    return pd.DataFrame(rows)


def agreement_check(table: TableSpec, metric: str, pair: Tuple[str, str], settings: Settings) -> Dict[str, float]:
    """Within-cell Spearman of two engines' errors on every record whose error is
    defined for every engine, in an eligible cell: the population of
    ``scripts/active_learning/ensemble_agreement.py`` (complete cases over all
    members), not this analysis's. Median over cells with its scan interval."""
    ok = table.frame[[error_column(metric, e) for e in table.engines]].notna().all(axis=1).to_numpy()
    frame = table.frame[ok].reset_index(drop=True)
    corr = cell_spearman(pd.DataFrame({"b": frame[error_column(metric, pair[1])]}),
                         frame[error_column(metric, pair[0])], frame["cell"], frame["scan"], settings.min_cell_size)
    row = summarise_correlation(corr, settings.n_bootstrap, settings.seed, settings.ci).iloc[0]
    return {"a": pair[0], "b": pair[1], "metric": metric, **{k: row[k] for k in
            ("median_cell_spearman", "ci_lo", "ci_hi", "n_cells", "n_scans", "n_records")}}


def importance_ranking(importance: pd.DataFrame, target: str, top: int) -> List[int]:
    """The ``top`` clusters by importance for ``target``."""
    part = importance[importance["target"] == target].sort_values("importance", ascending=False)
    return [int(c) for c in part["cluster"].head(top)]


def describe_population(pop: Population) -> Dict[str, float]:
    return {**pop.excluded, "n_cells": int(pop.frame["cell"].nunique()), "n_scans": int(pop.frame["scan"].nunique())}
