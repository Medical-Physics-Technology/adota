"""The per-member fits implied by scorer v2, reconstructed and checked.

Scorer v2 (``analytic_scorer_v2_exp0014.json``) is one ridge fitted to the mean
of five members' standardised ``log(1 + RDE)``. A ridge with fixed design and
penalty is linear in its target, so the five ridges fitted to each member's own
standardised error, on the same rows, average exactly to the deployed one. They
were never saved; this module refits them on the frozen fit rows for analysis
only and refuses to return them unless their mean reproduces the deployed
weights, intercept and percentile grids.

The fit rows are assembled with the same helpers and the same config keys as
``scripts/active_learning/difficulty_independence.py`` (the run that produced
scorer v2), so the population cannot drift from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from src.acquisition.features import FEATURE_NAMES
from src.acquisition.scorer import FULL, DifficultyScorer
from src.active_learning.ensemble.independence import (
    Standardiser,
    common_population,
    fit_scorer,
    inside_crop_features,
    log_errors,
    read_ids,
)

METRICS = list(FEATURE_NAMES)
IDENTITY_ATOL = 1e-8


@dataclass(frozen=True)
class MemberFits:
    """The per-member ridges and how closely their mean matches the deployed score."""

    members: Dict[str, DifficultyScorer]
    deployed: DifficultyScorer
    max_weight_diff: float
    intercept_diff: float
    max_grid_diff: float
    n_fit: int
    n_scans: int

    def check_table(self) -> pd.DataFrame:
        return pd.DataFrame([{"check": "mean member weight vs deployed", "max_abs_diff": self.max_weight_diff},
                             {"check": "mean member intercept vs deployed", "max_abs_diff": self.intercept_diff},
                             {"check": "fit-row percentile grids vs deployed", "max_abs_diff": self.max_grid_diff}])


def fit_population(cfg: Mapping) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Fit-row features and standardised member errors, as the independence run built them.

    Args:
        cfg: The independence config (``config_difficulty_independence.yaml``):
            ``ladder``, ``v3_features``, ``v3_index``, ``v3_fit_ids``, ``error_metric``.

    Returns:
        ``(features[METRICS], z[members], n_scans)`` on the fit rows.
    """
    metric = cfg.get("error_metric", "rde_pct")
    v3_feat = inside_crop_features(cfg["v3_features"])
    v3_err = log_errors(cfg["ladder"], metric)
    fit_ids = common_population(v3_feat, v3_err, read_ids(cfg["v3_fit_ids"]))
    z = Standardiser.fit(v3_err.loc[fit_ids])(v3_err.loc[fit_ids])
    scans = pd.read_csv(cfg["v3_index"], usecols=["sample_id", "patient_key"]).astype(str) \
        .set_index("sample_id")["patient_key"]
    return v3_feat.loc[fit_ids, METRICS], z[list(cfg["ladder"])], int(scans.loc[fit_ids].nunique())


def reconstruct_member_fits(cfg: Mapping, deployed: DifficultyScorer, variant: str = FULL,
                            atol: float = IDENTITY_ATOL) -> MemberFits:
    """One ridge per member on the frozen fit rows, accepted only if they average to ``deployed``.

    Raises:
        ValueError: if the mean of the member weights or intercepts, or the
            percentile grids, differ from the deployed scorer by more than ``atol``.
    """
    if variant != FULL:
        raise ValueError("only the ridge is linear in its target; the Lasso has no averaging identity")
    features, z, n_scans = fit_population(cfg)
    members = {k: fit_scorer(features, METRICS, z[k].to_numpy(), variant, f"member:{k}") for k in z.columns}
    names: List[str] = list(deployed.weights)
    mean_w = np.mean([[s.weights.get(m, 0.0) for m in names] for s in members.values()], axis=0)
    w_diff = float(np.max(np.abs(mean_w - np.array([deployed.weights[m] for m in names]))))
    b_diff = abs(float(np.mean([s.intercept for s in members.values()])) - deployed.intercept)
    any_member = next(iter(members.values()))
    g_diff = float(max(np.max(np.abs(any_member.grids[m] - deployed.grids[m])) for m in names))
    fits = MemberFits(members=members, deployed=deployed, max_weight_diff=w_diff, intercept_diff=b_diff,
                      max_grid_diff=g_diff, n_fit=len(features), n_scans=n_scans)
    if max(w_diff, b_diff, g_diff) > atol:
        raise ValueError(f"member fits do not reproduce the deployed score: weights {w_diff:.2e}, "
                         f"intercept {b_diff:.2e}, grids {g_diff:.2e} (tolerance {atol:.0e})")
    return fits


def member_weight_table(fits: MemberFits) -> pd.DataFrame:
    """Weights of every member and of the deployed score, features x scorers."""
    cols = {k: pd.Series(s.weights) for k, s in fits.members.items()}
    cols["ensemble"] = pd.Series(fits.deployed.weights)
    return pd.DataFrame(cols).loc[list(fits.deployed.weights)]


__all__ = ["IDENTITY_ATOL", "MemberFits", "fit_population", "member_weight_table", "reconstruct_member_fits"]
