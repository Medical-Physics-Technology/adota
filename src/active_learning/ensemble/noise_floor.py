"""The Monte Carlo noise floor under the EXP-0014 errors.

Every engine's error on a pool beamlet is measured against one Monte Carlo
realisation at 1e6 primaries (realisation A). A second realisation of some of
the same beamlets with another seed (B, :mod:`src.active_learning.beamlet_pool.noise`)
gives three things:

- **The floor.** B scored against A through the very metric code of the
  engines (:func:`src.active_learning.retrospective.validation.record_metrics`),
  per beamlet: the error a perfect engine would show, doubled in variance
  because both sides are noisy.
- **Whether the score predicts noise.** If the score ranked the floor as well as
  it ranks the engines' errors, it would be measuring the reference's noise
  instead of the engines' difficulty.
- **Shared-noise agreement.** Two engines scored against the same realisation
  agree partly because they share its noise. Scoring one against A and the
  other against B removes that share; the drop in their correlation is the
  agreement the noise made.
"""
from __future__ import annotations

import itertools
from typing import Dict, Sequence

import h5py
import numpy as np
import pandas as pd

from src.acquisition.reference import read_stored_record
from src.active_learning.retrospective.validation import MetricSettings, record_metrics


def mc_pair_errors(h5_reference: str, h5_other: str, ids: Sequence[str], settings: MetricSettings,
                   backend_options: Dict = None) -> pd.DataFrame:
    """Per beamlet, realisation ``h5_other`` scored as a prediction of ``h5_reference``."""
    rows = []
    with h5py.File(h5_reference, "r") as ref, h5py.File(h5_other, "r") as other:
        for sample_id in ids:
            a = read_stored_record(ref[sample_id], sample_id, settings.scale)
            b = read_stored_record(other[sample_id], sample_id, settings.scale)
            rows.append({"sample_id": sample_id, "energy_mev": a.energy_mev,
                         **record_metrics(a.dose, b.dose, settings, backend_options, sample_id=sample_id)})
    return pd.DataFrame(rows)


def cross_reference_agreement(errors_a: pd.DataFrame, errors_b: pd.DataFrame) -> pd.DataFrame:
    """Per pair of engines, Pearson of their ``log(1 + error)`` when both are
    scored against A, and when the first is scored against A and the second
    against B (both orders averaged). Columns are engines, rows beamlets."""
    rows = []
    la, lb = np.log1p(errors_a), np.log1p(errors_b)
    for j, k in itertools.combinations(errors_a.columns, 2):
        same = float(np.corrcoef(la[j], la[k])[0, 1])
        cross = float(np.mean([np.corrcoef(la[j], lb[k])[0, 1], np.corrcoef(lb[j], la[k])[0, 1]]))
        rows.append({"engine_a": j, "engine_b": k, "pearson_same_reference": same,
                     "pearson_cross_reference": cross, "shared_noise_share": same - cross})
    return pd.DataFrame(rows)


def floor_ratio(floor: pd.Series, errors: pd.DataFrame) -> pd.DataFrame:
    """Per engine, the floor relative to the engine's own error: the median over
    beamlets of ``floor / error`` and the ratio of the means. The floor is A
    against B, so one realisation's own noise is about ``floor / sqrt(2)``."""
    rows = []
    for engine in errors.columns:
        e = errors[engine].to_numpy()
        f = floor.loc[errors.index].to_numpy()
        rows.append({"engine": engine, "median_ratio": float(np.median(f / e)),
                     "ratio_of_means": float(f.mean() / e.mean()),
                     "ratio_of_means_one_realisation": float(f.mean() / np.sqrt(2.0) / e.mean())})
    return pd.DataFrame(rows)
