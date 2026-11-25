"""What drives a dose engine's error, measured on held-out patients.

For each engine and error metric:

1. **One property at a time** (:func:`univariate`): Spearman correlation of every
   feature with the error, a 95% interval from resampling patients.
2. **All properties together** (:func:`grouped_importance`): a gradient-boosted tree
   model of the error from all features, fitted with folds grouped by patient. Its
   out-of-fold accuracy says how predictable the error is at all. A feature group's
   importance is the increase of the held-out squared error when the group's columns
   are permuted together (within the held-out fold); per beamlet, so a patient
   bootstrap gives its interval. Groups of correlated features are permuted jointly,
   so correlated features cannot hide each other's importance.
3. **Shape** (:func:`partial_dependence_table`): partial dependence of the fitted model
   on the lead feature of the top groups.

Errors are oriented so that larger is worse (gamma as the failing share, ``100 - GPR``)
and modelled on the scale the difficulty score uses (``log(1 + x)``), except the signed
range error, which is modelled as it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Target:
    """An error metric: its column in the error table, how it is oriented and modelled."""

    name: str
    column: str
    label: str
    transform: str          # "fail_log" (log(1 + 100 - x)), "log" (log(1 + x)) or "none"

    def values(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.column].to_numpy(dtype=float)
        if self.transform == "fail_log":
            return np.log1p(np.clip(100.0 - x, 0.0, None))
        if self.transform == "log":
            return np.log1p(np.clip(x, 0.0, None))
        return x


TARGETS: Sequence[Target] = (
    Target("fail_2_2", "gpr_2pct_2mm_10pct", "gamma fail 2%/2mm (100 - GPR)", "fail_log"),
    Target("fail_3_3", "gpr_3pct_3mm_10pct", "gamma fail 3%/3mm (100 - GPR)", "fail_log"),
    Target("mape", "mape_pct", "MAPE", "log"),
    Target("rde", "rde_pct", "RDE", "log"),
    Target("dr80", "dr80_mm", "dR80 signed (prediction - MC)", "none"),
)


def _patient_resamples(groups: np.ndarray, n_boot: int, seed: int) -> List[np.ndarray]:
    """Row indices of ``n_boot`` resamples drawing patients with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_of = [np.nonzero(inverse == i)[0] for i in range(len(uniq))]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([rows_of[i] for i in pick]))
    return out


def univariate(frame: pd.DataFrame, features: Sequence[str], target: Target, groups: np.ndarray,
               n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Spearman correlation of each feature with the (worse-is-larger) error, patient-bootstrap 95% CI.

    The interval uses the Pearson correlation of the full-sample ranks within each resample,
    the usual fast approximation of a cluster-bootstrapped Spearman.
    """
    y = rankdata(target.values(frame))
    X = np.column_stack([rankdata(frame[f].to_numpy(dtype=float)) for f in features])

    def corr(rows: np.ndarray) -> np.ndarray:
        xs, ys = X[rows], y[rows]
        xs = xs - xs.mean(axis=0)
        ys = ys - ys.mean()
        denom = np.sqrt((xs ** 2).sum(axis=0) * (ys ** 2).sum())
        return np.where(denom > 0, (xs * ys[:, None]).sum(axis=0) / np.where(denom > 0, denom, 1.0), np.nan)

    point = corr(np.arange(len(y)))
    boot = np.array([corr(r) for r in _patient_resamples(groups, n_boot, seed)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"feature": list(features), "spearman": point, "ci_low": lo, "ci_high": hi})
    out["ci_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    out["target"] = target.name
    return out.sort_values("spearman", key=np.abs, ascending=False).reset_index(drop=True)


def _model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                         early_stopping=False, random_state=seed)


@dataclass(frozen=True)
class ImportanceResult:
    """Out-of-fold fit quality and grouped permutation importance of one engine and target."""

    fit: Dict[str, float]
    importance: pd.DataFrame
    model: HistGradientBoostingRegressor       # refitted on all rows, for partial dependence


def grouped_importance(frame: pd.DataFrame, features: Sequence[str], groups_of: Mapping[str, str],
                       target: Target, patients: np.ndarray, n_folds: int = 5, n_repeats: int = 5,
                       n_boot: int = 500, seed: int = 0) -> ImportanceResult:
    """Grouped permutation importance on held-out patients (folds grouped by patient).

    ``importance`` has one row per group: the mean increase of the held-out squared error when
    the group is permuted (``delta_mse``), its share of the summed positive increases, a patient
    bootstrap 95% interval, and the group's members.
    """
    X = frame[list(features)].to_numpy(dtype=float)
    y = target.values(frame)
    names = sorted(set(groups_of.get(f, f) for f in features))
    cols = {g: [i for i, f in enumerate(features) if groups_of.get(f, f) == g] for g in names}
    oof = np.full(len(y), np.nan)
    deltas = {g: np.zeros(len(y)) for g in names}
    rng = np.random.default_rng(seed)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, patients)):
        model = _model(seed + k).fit(X[tr], y[tr])
        base = model.predict(X[te])
        oof[te] = base
        loss0 = (base - y[te]) ** 2
        for g in names:
            acc = np.zeros(len(te))
            for _ in range(n_repeats):
                Xp = X[te].copy()
                perm = rng.permutation(len(te))
                Xp[:, cols[g]] = Xp[perm][:, cols[g]]
                acc += (model.predict(Xp) - y[te]) ** 2 - loss0
            deltas[g][te] = acc / n_repeats
    resid = y - oof
    fit = {"n": int(len(y)), "n_patients": int(len(np.unique(patients))),
           "oof_r2": float(1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)),
           "oof_spearman": float(pd.Series(oof).corr(pd.Series(y), method="spearman")),
           "mse_baseline": float(np.var(y))}
    resamples = _patient_resamples(patients, n_boot, seed)
    rows = []
    for g in names:
        d = deltas[g]
        boot = np.array([d[r].mean() for r in resamples])
        rows.append({"group": g, "delta_mse": float(d.mean()), "ci_low": float(np.percentile(boot, 2.5)),
                     "ci_high": float(np.percentile(boot, 97.5)),
                     "members": " ".join(f for f in features if groups_of.get(f, f) == g)})
    imp = pd.DataFrame(rows)
    pos = imp["delta_mse"].clip(lower=0).sum()
    imp["share"] = imp["delta_mse"].clip(lower=0) / pos if pos > 0 else np.nan
    imp["relative_to_baseline_mse"] = imp["delta_mse"] / fit["mse_baseline"]
    imp["target"] = target.name
    imp = imp.sort_values("delta_mse", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return ImportanceResult(fit=fit, importance=imp, model=_model(seed).fit(X, y))


def partial_dependence_table(result: ImportanceResult, frame: pd.DataFrame, features: Sequence[str],
                             lead_features: Sequence[str], target: Target, grid_points: int = 20) -> pd.DataFrame:
    """Partial dependence of the all-rows model on each lead feature, on its 5th to 95th percentile."""
    X = frame[list(features)].to_numpy(dtype=float)
    rows = []
    for f in lead_features:
        i = list(features).index(f)
        pd_ = partial_dependence(result.model, X, [i], grid_resolution=grid_points, percentiles=(0.05, 0.95),
                                 kind="average")
        grid = pd_["grid_values"][0] if "grid_values" in pd_ else pd_["values"][0]
        for x, v in zip(grid, pd_["average"][0]):
            rows.append({"target": target.name, "feature": f, "value": float(x), "partial_dependence": float(v)})
    return pd.DataFrame(rows)


def score_relation(frame: pd.DataFrame, score: str, target: Target, patients: np.ndarray, n_boot: int = 500,
                   seed: int = 0, n_bins: int = 10, deciles: Optional[pd.Series] = None) -> tuple:
    """``(correlation row, decile table)``: Spearman of the difficulty score with the error, with a
    patient-bootstrap CI, and the error's median, quartiles and mean per score decile."""
    corr = univariate(frame, [score], target, patients, n_boot, seed).iloc[0].to_dict()
    dec = deciles if deciles is not None else pd.Series(
        np.ceil(rankdata(frame[score], method="ordinal") * n_bins / len(frame)).astype(int), index=frame.index)
    vals = frame[target.column]
    table = vals.groupby(dec).agg(n="size", median="median", q25=lambda s: s.quantile(0.25),
                                  q75=lambda s: s.quantile(0.75), mean="mean").reset_index()
    table = table.rename(columns={table.columns[0]: "decile"})
    table["target"] = target.name
    return corr, table


__all__ = ["ImportanceResult", "TARGETS", "Target", "grouped_importance", "partial_dependence_table",
           "score_relation", "univariate"]
