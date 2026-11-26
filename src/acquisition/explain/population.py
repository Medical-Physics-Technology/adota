"""The difficulty explanation across a whole population, and example beamlets drawn from it.

Two questions the three-beamlet explanation cannot answer:

1. Do the reasons hold for other patients? :func:`select_examples` draws an easy,
   a medium and a hard beamlet per anatomy (nearest given percentiles of the score
   within that anatomy), each from a different patient.
2. Do the reasons hold across the population? :func:`population_groups` decomposes
   every reference beamlet at once (vectorised, with the same sum check as
   :func:`src.acquisition.explain.contributions.decompose`), and
   :func:`decile_profiles` averages the group contributions within score deciles.
   The score is linear, so a decile's mean contributions add up exactly to its mean
   score minus the baseline.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.acquisition.explain.contributions import SUM_ATOL, percentile_frame
from src.acquisition.scorer import DifficultyScorer
from src.active_learning.ensemble.engine_grids import reference_percentile

ALL = "all"
"""Panel label of the whole population in :func:`decile_profiles`."""


def pool_metadata(index_csv: str) -> pd.DataFrame:
    """Patient and anatomy of every record of a beamlet-pool export, indexed by sample id."""
    frame = pd.read_csv(index_csv, usecols=["sample_id", "pool_patient_id", "pool_anatomy"])
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame.rename(columns={"pool_patient_id": "patient_id", "pool_anatomy": "anatomy"}).set_index("sample_id")


def select_examples(scores: pd.Series, meta: pd.DataFrame, percentiles: Sequence[float],
                    levels: Sequence[str], by: str = "anatomy") -> pd.DataFrame:
    """Per stratum of ``meta[by]``, the beamlet nearest each percentile of the score within that stratum.

    No two picks share a patient: within a stratum the levels are filled in order, each
    taking the nearest beamlet of a patient not yet used. Ties go to the smaller sample id.

    Returns:
        One row per pick: ``label`` (``<stratum>_<level>``), ``sample_id``, ``stratum``, ``level``,
        ``patient_id``, ``target_percentile``, ``stratum_percentile``, ``reference_percentile``,
        ``score``, ``n_stratum``.
    """
    if len(percentiles) != len(levels):
        raise ValueError(f"{len(percentiles)} percentiles for {len(levels)} levels")
    frame = meta.join(scores.rename("score"), how="inner").dropna(subset=["score"]).sort_index()
    frame["reference_percentile"] = reference_percentile(frame["score"].to_numpy(), frame["score"].to_numpy())
    rows, used = [], set()
    for stratum, sub in frame.groupby(by, sort=True):
        pct = reference_percentile(sub["score"].to_numpy(), sub["score"].to_numpy())
        sub = sub.assign(stratum_percentile=pct)
        for target, level in zip(percentiles, levels):
            free = sub[~sub["patient_id"].isin(used)]
            if free.empty:
                raise ValueError(f"{by}={stratum}: no unused patient left for level {level!r}")
            dist = (free["stratum_percentile"] - float(target)).abs()
            pick = dist[dist == dist.min()].index.min()
            r = free.loc[pick]
            used.add(r["patient_id"])
            rows.append({"label": f"{stratum}_{level}", "sample_id": pick, "stratum": stratum, "level": level,
                         "patient_id": r["patient_id"], "target_percentile": float(target),
                         "stratum_percentile": float(r["stratum_percentile"]),
                         "reference_percentile": float(r["reference_percentile"]), "score": float(r["score"]),
                         "n_stratum": len(sub)})
    return pd.DataFrame(rows)


def within_stratum_pairs(selection: pd.DataFrame, levels: Sequence[str]) -> List[Tuple[str, str]]:
    """``(harder, easier)`` labels of consecutive levels within every stratum, e.g. medium vs easy."""
    pairs = []
    for _, sub in selection.groupby("stratum", sort=False):
        lab = dict(zip(sub["level"], sub["label"]))
        pairs += [(lab[h], lab[e]) for e, h in zip(levels[:-1], levels[1:])]
    return pairs


def population_groups(scorer: DifficultyScorer, beamlets: pd.DataFrame, reference: pd.DataFrame,
                      groups: Mapping[str, str], atol: float = SUM_ATOL) -> Tuple[pd.DataFrame, pd.Series, float]:
    """Every beamlet's contributions summed within groups, vectorised.

    Args:
        groups: ``{feature: group name}``; a feature without a group is its own.

    Returns:
        ``(contributions, total, baseline)``: beamlets x groups, score minus baseline per beamlet, and the
        baseline (mean reference score).

    Raises:
        AssertionError: if a beamlet's contributions do not add up to its score minus the baseline.
    """
    metrics = list(scorer.metrics)
    p_ref = percentile_frame(scorer, reference, metrics).mean()
    contrib = (percentile_frame(scorer, beamlets, metrics) - p_ref) * pd.Series(scorer.weights)[metrics]
    base = float(np.mean(scorer.score(reference)))
    total = pd.Series(np.asarray(scorer.score(beamlets), dtype=float), index=beamlets.index) - base
    residual = float((total - contrib.sum(axis=1)).abs().max())
    assert residual <= atol, f"population contributions do not add up (max residual {residual:.2e})"
    grouped = contrib.T.groupby(lambda f: groups.get(f, f), sort=False).sum().T
    return grouped, total, base


def score_deciles(scores: pd.Series, n_bins: int = 10) -> pd.Series:
    """Decile (1 = easiest) of each score within the scores themselves; equal-count bins by rank."""
    ranks = scores.rank(method="first").to_numpy()
    return pd.Series(np.ceil(ranks * n_bins / len(scores)).astype(int), index=scores.index, name="decile")


def decile_profiles(grouped: pd.DataFrame, total: pd.Series, deciles: pd.Series, meta: pd.DataFrame,
                    by: str = "anatomy") -> pd.DataFrame:
    """Mean, median and quartiles of every group's contribution per decile, for all beamlets and per stratum.

    Returns:
        Long: ``panel, decile, group, mean_contribution, median_contribution, q25, q75, n_beamlets,
        n_patients, mean_total``. Within a panel and decile the group means add up to ``mean_total``.
    """
    strata = meta.reindex(grouped.index)[by]
    patients = meta.reindex(grouped.index)["patient_id"]
    panels: Dict[str, pd.Index] = {ALL: grouped.index}
    panels.update({str(s): grouped.index[strata == s] for s in sorted(strata.dropna().unique())})
    rows = []
    for panel, ids in panels.items():
        for d, sub in grouped.loc[ids].groupby(deciles.reindex(ids)):
            n, n_pat, mean_total = len(sub), patients.loc[sub.index].nunique(), float(total.loc[sub.index].mean())
            for g in grouped.columns:
                v = sub[g]
                rows.append({"panel": panel, "decile": int(d), "group": g, "mean_contribution": float(v.mean()),
                             "median_contribution": float(v.median()), "q25": float(v.quantile(0.25)),
                             "q75": float(v.quantile(0.75)), "n_beamlets": n, "n_patients": n_pat,
                             "mean_total": mean_total})
    return pd.DataFrame(rows)


def variance_share(grouped: pd.DataFrame, total: pd.Series, meta: pd.DataFrame, by: str = "anatomy") -> pd.DataFrame:
    """Share of the score's variance carried by each group: ``cov(group, total) / var(total)``.

    Exact and additive, because the total is the sum of the groups: the shares of a panel add up to 1.
    A negative share is a group that, on average, moves against the score. Columns: ``panel`` (``all`` and
    every stratum), ``group``, ``variance_share``, ``n_beamlets``.
    """
    strata = meta.reindex(grouped.index)[by]
    panels: Dict[str, pd.Index] = {ALL: grouped.index}
    panels.update({str(s): grouped.index[strata == s] for s in sorted(strata.dropna().unique())})
    rows = []
    for panel, ids in panels.items():
        t = total.loc[ids].to_numpy()
        t = t - t.mean()
        for g in grouped.columns:
            v = grouped.loc[ids, g].to_numpy()
            rows.append({"panel": panel, "group": g, "variance_share": float(np.dot(v - v.mean(), t) / np.dot(t, t)),
                         "n_beamlets": len(ids)})
    return pd.DataFrame(rows)


def leading_group_share(grouped: pd.DataFrame, deciles: pd.Series) -> pd.DataFrame:
    """Per decile, the share of beamlets whose largest positive contribution comes from each group.

    A beamlet with no positive contribution counts as ``"none positive"``.
    """
    lead = grouped.idxmax(axis=1).where(grouped.max(axis=1) > 0, "none positive")
    share = pd.crosstab(deciles.reindex(grouped.index), lead, normalize="index")
    return share.rename_axis(index="decile", columns="leading_group").reset_index()


def profile_agreement(profiles: Mapping[str, pd.DataFrame], reference: str, panel: str = ALL) -> pd.DataFrame:
    """How closely each scorer's decile profile follows the reference scorer's, in one panel.

    Per scorer: Pearson correlation over all (decile, group) mean contributions, the largest absolute
    difference, and whether the group with the largest mean contribution in the top decile is the same.
    """
    def matrix(p: pd.DataFrame) -> pd.DataFrame:
        sub = p[p["panel"] == panel]
        return sub.pivot(index="decile", columns="group", values="mean_contribution")

    ref = matrix(profiles[reference])
    top = int(ref.index.max())
    rows = []
    for name, prof in profiles.items():
        if name == reference:
            continue
        mine = matrix(prof).reindex(index=ref.index, columns=ref.columns, fill_value=0.0)
        a, b = mine.to_numpy().ravel(), ref.to_numpy().ravel()
        rows.append({"scorer": name, "panel": panel, "pearson_vs_reference": float(np.corrcoef(a, b)[0, 1]),
                     "max_abs_difference": float(np.abs(a - b).max()),
                     "top_decile_leading_group": str(mine.loc[top].idxmax()),
                     "same_top_decile_leader": mine.loc[top].idxmax() == ref.loc[top].idxmax()})
    return pd.DataFrame(rows)


__all__ = ["ALL", "decile_profiles", "leading_group_share", "pool_metadata", "population_groups",
           "profile_agreement", "score_deciles", "select_examples", "variance_share", "within_stratum_pairs"]
