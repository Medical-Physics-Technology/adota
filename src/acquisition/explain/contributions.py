"""Why a beamlet gets its difficulty score: exact additive contributions.

The deployed score is linear in percentile-transformed features
(:class:`~src.acquisition.scorer.DifficultyScorer`),

    score(x) = b + sum_k w_k p_k(x),

so with a reference population R as background the score splits exactly into

    score(x) - mean_R(score) = sum_k w_k (p_k(x) - mean_R p_k).

``mean_R(score)`` is the baseline. For a linear model with an independent
(interventional) background this is also the SHAP value of each feature, so no
sampling and no approximation is involved. Every decomposition asserts that its
contributions add up to ``score - baseline``.

Several scorers that share percentile grids and whose weights average to an
ensemble's weights (the per-member ridges behind scorer v2, see
:mod:`src.acquisition.explain.members`) have contributions that average to the
ensemble's, because the decomposition is linear in the weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.acquisition.scorer import DifficultyScorer

SUM_ATOL = 1e-9


@dataclass(frozen=True)
class Decomposition:
    """One scorer's contributions for a set of beamlets.

    ``table`` is long form, one row per (beamlet, feature) with columns
    ``beamlet_id, scorer, feature, percentile, mean_percentile_ref, weight,
    contribution``. ``percentile`` is in [0, 1], on the scorer's own grids.
    """

    scorer: str
    baseline: float
    scores: pd.Series            # beamlet_id -> score
    table: pd.DataFrame

    def wide(self) -> pd.DataFrame:
        """Contributions as features x beamlets."""
        return self.table.pivot(index="feature", columns="beamlet_id", values="contribution")


def percentile_frame(scorer: DifficultyScorer, features: pd.DataFrame,
                     metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Every metric of ``features`` through the scorer's percentile transform, in [0, 1]."""
    metrics = list(metrics) if metrics is not None else list(scorer.metrics)
    return pd.DataFrame({m: scorer.percentile(m, features[m].to_numpy(dtype=float)) for m in metrics},
                        index=features.index)


def baseline(scorer: DifficultyScorer, reference: pd.DataFrame) -> float:
    """Mean score over the reference population."""
    return float(np.mean(scorer.score(reference)))


def decompose(scorer: DifficultyScorer, beamlets: pd.DataFrame, reference: pd.DataFrame, name: str,
              atol: float = SUM_ATOL) -> Decomposition:
    """Contributions ``w_k (p_k(x) - mean_R p_k)`` of every weighted metric.

    Args:
        scorer: A frozen linear scorer.
        beamlets: Feature rows, indexed by beamlet id.
        reference: Feature rows of the background population.
        name: Label of the scorer in the output.
        atol: Tolerance of the sum check.

    Raises:
        AssertionError: if any beamlet's contributions do not add up to
            ``score - baseline`` within ``atol``.
    """
    metrics = list(scorer.metrics)
    p_x = percentile_frame(scorer, beamlets, metrics)
    p_ref = percentile_frame(scorer, reference, metrics).mean()
    weights = pd.Series(scorer.weights)[metrics]
    contrib = (p_x - p_ref) * weights
    base = baseline(scorer, reference)
    scores = pd.Series(np.asarray(scorer.score(beamlets), dtype=float), index=beamlets.index)
    residual = (scores - base) - contrib.sum(axis=1)
    bad = residual[np.abs(residual) > atol]
    assert bad.empty, f"{name}: contributions do not add up for {list(bad.index)} (max {np.abs(residual).max():.2e})"

    rows = []
    for bid in beamlets.index:
        for m in metrics:
            rows.append({"beamlet_id": bid, "scorer": name, "feature": m, "percentile": float(p_x.at[bid, m]),
                         "mean_percentile_ref": float(p_ref[m]), "weight": float(weights[m]),
                         "contribution": float(contrib.at[bid, m])})
    return Decomposition(scorer=name, baseline=base, scores=scores, table=pd.DataFrame(rows))


def sum_check(decompositions: Sequence[Decomposition]) -> pd.DataFrame:
    """Per scorer and beamlet: score, baseline, sum of contributions and the residual."""
    rows = []
    for d in decompositions:
        sums = d.table.groupby("beamlet_id")["contribution"].sum()
        for bid, score in d.scores.items():
            rows.append({"scorer": d.scorer, "beamlet_id": bid, "score": float(score), "baseline": d.baseline,
                         "sum_contributions": float(sums[bid]),
                         "residual": float(score - d.baseline - sums[bid])})
    return pd.DataFrame(rows)


def mean_decomposition(members: Sequence[Decomposition], name: str = "member_mean") -> Decomposition:
    """The mean of several decompositions over the same beamlets and features."""
    tables = [m.table.set_index(["beamlet_id", "feature"]) for m in members]
    mean = sum(t[["contribution", "weight", "percentile", "mean_percentile_ref"]] for t in tables) / len(tables)
    table = mean.reset_index().assign(scorer=name)
    scores = sum(m.scores for m in members) / len(members)
    return Decomposition(scorer=name, baseline=float(np.mean([m.baseline for m in members])), scores=scores,
                         table=table[["beamlet_id", "scorer", "feature", "percentile", "mean_percentile_ref",
                                      "weight", "contribution"]])


def ensemble_consistency(ensemble: Decomposition, members: Sequence[Decomposition]) -> float:
    """Largest absolute difference between the ensemble's contributions and the
    mean of the members' (zero up to rounding when member weights average to the
    ensemble's)."""
    mean = mean_decomposition(members).table.set_index(["beamlet_id", "feature"])["contribution"]
    ens = ensemble.table.set_index(["beamlet_id", "feature"])["contribution"]
    joined = pd.concat([ens, mean], axis=1, keys=["ens", "mean"]).fillna(0.0)
    return float((joined["ens"] - joined["mean"]).abs().max())


def group_contributions(table: pd.DataFrame, groups: Mapping[str, str]) -> pd.DataFrame:
    """Contributions summed within feature groups.

    Args:
        table: A decomposition table (or several concatenated).
        groups: ``{feature: group name}``; a feature without a group is its own.

    Returns:
        ``beamlet_id, scorer, group, contribution, n_features, features``.
    """
    frame = table.assign(group=table["feature"].map(lambda f: groups.get(f, f)))
    out = frame.groupby(["beamlet_id", "scorer", "group"], sort=False).agg(
        contribution=("contribution", "sum"), n_features=("feature", "size"),
        features=("feature", lambda s: " ".join(s))).reset_index()
    return out


def pair_contrast(frame: pd.DataFrame, harder: str, easier: str, level: str = "group",
                  top_k: int = 3) -> pd.DataFrame:
    """Rank features or groups by how much more they add to ``harder`` than to ``easier``.

    Args:
        frame: Group contributions (:func:`group_contributions`) or a decomposition
            table (``level="feature"``), for one or more scorers.
        harder, easier: Beamlet ids.
        level: ``"group"`` or ``"feature"``.
        top_k: How many items per scorer are flagged ``top``.

    Returns:
        Per scorer and item: both contributions, ``difference = harder - easier``,
        its share of the total score difference, the rank by difference (largest
        positive first) and the ``top`` flag.
    """
    key = "group" if level == "group" else "feature"
    rows = []
    for scorer, sub in frame.groupby("scorer", sort=False):
        piv = sub.pivot(index=key, columns="beamlet_id", values="contribution")
        diff = piv[harder] - piv[easier]
        total = float(diff.sum())
        order = diff.sort_values(ascending=False)
        for rank, (item, d) in enumerate(order.items(), start=1):
            rows.append({"scorer": scorer, "level": level, "item": item, "harder": harder, "easier": easier,
                         "contribution_harder": float(piv.at[item, harder]),
                         "contribution_easier": float(piv.at[item, easier]), "difference": float(d),
                         "share_of_gap": float(d / total) if total != 0 else np.nan, "score_gap": total,
                         "rank": rank, "top": rank <= top_k})
    return pd.DataFrame(rows)


__all__ = ["Decomposition", "SUM_ATOL", "baseline", "decompose", "ensemble_consistency", "group_contributions",
           "mean_decomposition", "pair_contrast", "percentile_frame", "sum_check"]
