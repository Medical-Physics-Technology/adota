"""The frozen difficulty score: percentile-normalised metrics, weighted and summed.

``score(x) = b + sum_k w_k * p_k(x)``, where ``p_k`` is the percentile rank of
metric ``k`` within the reference pool (``research/acquisition_function_final_summary.md``,
Section 2.2). The triple of weights, intercept and percentile grids is frozen in
a JSON file and never refit at selection time; this module only applies it.

Two file layouts are read: the study's ``frozen_final_scorer.json`` (one set of
grids, ``weights[variant]``) and the arms-keyed ``analytic_scorer.json`` written
by ``scripts/analysis/acquisition_input_only_refit.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

SPARSE, FULL = "sparse (Lasso)", "full (ridge)"
N_GRID = 101


@dataclass(frozen=True)
class DifficultyScorer:
    """A frozen linear difficulty score."""

    variant: str
    intercept: float
    weights: Dict[str, float]                 # only the metrics with a non-zero weight
    grids: Dict[str, np.ndarray]              # 101 quantiles per metric, over every metric
    source: str = ""

    @classmethod
    def load(cls, path: Union[str, Path], variant: str = SPARSE, arm: Optional[str] = None) -> "DifficultyScorer":
        """Load a frozen scorer.

        Args:
            path: ``frozen_final_scorer.json`` or ``analytic_scorer.json``.
            variant: ``"sparse (Lasso)"`` or ``"full (ridge)"``.
            arm: For the arms-keyed layout, which ``arm/population`` entry to use
                (for example ``"analytic/both_inside_crop"``). Ignored otherwise.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "arms" in data:
            if arm is None or arm not in data["arms"]:
                raise KeyError(f"choose an arm from {sorted(data['arms'])}, got {arm!r}")
            block = data["arms"][arm]
        else:
            block = data
        if variant not in block["weights"]:
            raise KeyError(f"variant {variant!r} not in {sorted(block['weights'])}")
        w = block["weights"][variant]
        grids = {k: np.asarray(v, dtype=float) for k, v in block["percentile_grids_dev"].items()}
        return cls(variant=variant, intercept=float(w["intercept"]),
                   weights={k: float(v) for k, v in w["coefficients"].items()}, grids=grids,
                   source=f"{path}::{arm or ''}::{variant}")

    @property
    def metrics(self) -> Sequence[str]:
        """The metrics the score actually uses."""
        return tuple(self.weights)

    def percentile(self, metric: str, values: np.ndarray) -> np.ndarray:
        """The study's transform: right-side rank into the 101-point grid, over 101, clipped."""
        return np.clip(np.searchsorted(self.grids[metric], np.asarray(values, dtype=float), side="right") / N_GRID,
                       0.0, 1.0)

    def score(self, features: Union[Mapping[str, float], pd.DataFrame]) -> Union[float, np.ndarray]:
        """Score one feature mapping or a frame of them. Higher is harder."""
        if isinstance(features, pd.DataFrame):
            total = np.full(len(features), self.intercept, dtype=float)
            for metric, weight in self.weights.items():
                total += weight * self.percentile(metric, features[metric].to_numpy())
            return total
        total = self.intercept
        for metric, weight in self.weights.items():
            total += weight * float(self.percentile(metric, np.array([features[metric]]))[0])
        return float(total)

    def contributions(self, features: Mapping[str, float]) -> Dict[str, float]:
        """Per-metric ``w_k * p_k`` for one beamlet, for reading a score."""
        return {m: w * float(self.percentile(m, np.array([features[m]]))[0]) for m, w in self.weights.items()}

    def missing(self, columns: Iterable[str]) -> Sequence[str]:
        """The metrics this scorer needs that ``columns`` does not provide."""
        have = set(columns)
        return tuple(m for m in self.weights if m not in have)
