"""The Monte Carlo noise floor under the EXP-0014 errors.

Every engine's error on a pool beamlet is measured against one Monte Carlo
realisation at 1e6 primaries. The noise set
(:mod:`src.active_learning.beamlet_pool.noise`) simulates new beamlets at 1e6 and
at 1e7, so each engine has an error against either reference, and:

- **The floor.** The 1e6 realisation scored against the 1e7 one through the very
  metric code of the engines
  (:func:`src.active_learning.retrospective.validation.record_metrics`), per
  beamlet: the error a perfect engine would show against the 1e6 reference.
- **Whether the score tracks noise.** :func:`partial_pearson_with_interval`, the
  score against the floor with the 1e7 target held fixed: what the score sees of
  the noise beyond the difficulty it was fitted to.
- **Shared-noise agreement.** Two engines scored against the same realisation
  agree partly because they share its noise. Scoring one against 1e6 and the
  other against 1e7 removes that share; :func:`agreement_drop` is the drop in
  their correlation.
- **The hottest voxel.** RDE and gamma are normalised by the reference's maximum,
  which a single deposit in a low-density voxel can set at 1e6.
  :func:`peak_statistics` measures how often, and
  :func:`renormalised_errors` repeats an RDE with a robust maximum, without
  inference, because the normaliser is one factor of the RDE.

:func:`noise_floor_checks` applies the decision rule, fixed before the per-beamlet
results existed.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from src.acquisition.reference import read_stored_record
from src.active_learning.ensemble.independence import pearson_from_sums, scan_sums, verdict
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


def floor_ratio(floor: pd.Series, errors: pd.DataFrame, primaries_ratio: float = 1.0) -> pd.DataFrame:
    """Per engine, the floor relative to the engine's own error: the median over
    beamlets of ``floor / error`` and the ratio of the means. The floor holds both
    realisations' noise; with ``primaries_ratio`` times the primaries in the better
    one, the noisier one's own share is about ``floor / sqrt(1 + 1 / primaries_ratio)``."""
    own_share = np.sqrt(1.0 + 1.0 / primaries_ratio)
    rows = []
    for engine in errors.columns:
        e = errors[engine].to_numpy()
        f = floor.loc[errors.index].to_numpy()
        rows.append({"engine": engine, "median_ratio": float(np.median(f / e)),
                     "ratio_of_means": float(f.mean() / e.mean()),
                     "ratio_of_means_one_realisation": float(f.mean() / own_share / e.mean())})
    return pd.DataFrame(rows)


def peak_statistics(h5_reference: str, h5_other: str, ids: Sequence[str], scale: Dict[str, float],
                    robust_percentile: float = 99.9) -> pd.DataFrame:
    """Per beamlet, the maximum and a robust maximum of both realisations, the depth
    of each maximum (voxels of the model grid), the HU at the other's maximum, and
    the summed L1 difference per unit of reference dose (noise without the factor of
    dose volume over maximum that every RDE carries)."""
    rows = []
    with h5py.File(h5_reference, "r") as ref, h5py.File(h5_other, "r") as other:
        for sample_id in ids:
            a = read_stored_record(ref[sample_id], sample_id, scale)
            b = read_stored_record(other[sample_id], sample_id, scale)
            at_a = np.unravel_index(int(np.argmax(a.dose)), a.dose.shape)
            at_b = np.unravel_index(int(np.argmax(b.dose)), b.dose.shape)
            rows.append({"sample_id": sample_id, "max_reference": float(a.dose.max()),
                         "max_other": float(b.dose.max()),
                         "robust_reference": float(np.percentile(a.dose, robust_percentile)),
                         "robust_other": float(np.percentile(b.dose, robust_percentile)),
                         "max_depth_reference": int(at_a[0]), "max_depth_other": int(at_b[0]),
                         "volume_over_max": float(a.dose.sum() / a.dose.max()),
                         "l1_per_unit_dose": float(np.abs(b.dose.astype(np.float64) - a.dose).sum() / a.dose.sum()),
                         "hu_at_max_other": float(b.ct_hu[at_b])})
    return pd.DataFrame(rows).set_index("sample_id")


def renormalised_errors(rde: pd.DataFrame, maximum: pd.Series, robust: pd.Series) -> pd.DataFrame:
    """RDE with a robust maximum in place of the reference's maximum, which is its
    only factor that depends on the normaliser. Columns are engines, rows beamlets."""
    return rde.mul((maximum / robust).loc[rde.index], axis=0)


def _resampled_pearson(x: np.ndarray, y: np.ndarray, scans: np.ndarray, order: Sequence[str],
                       counts: np.ndarray) -> Tuple[float, np.ndarray]:
    sums = scan_sums(np.asarray(x, dtype=float), np.asarray(y, dtype=float), scans, order)
    return float(pearson_from_sums(sums.sum(axis=0))), pearson_from_sums(counts @ sums)


def _partial(r_xy, r_xz, r_yz):
    return (r_xy - r_xz * r_yz) / np.sqrt((1.0 - r_xz ** 2) * (1.0 - r_yz ** 2))


def partial_pearson_with_interval(x: np.ndarray, y: np.ndarray, z: np.ndarray, scans: np.ndarray,
                                  counts: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """Pearson of ``x`` and ``y`` with ``z`` held fixed, with an interval from the
    scan resamples ``counts``."""
    scans = np.asarray(scans).astype(str)
    order = sorted(pd.unique(scans))
    xy, xy_b = _resampled_pearson(x, y, scans, order, counts)
    xz, xz_b = _resampled_pearson(x, z, scans, order, counts)
    yz, yz_b = _resampled_pearson(y, z, scans, order, counts)
    lo, hi = np.nanquantile(_partial(xy_b, xz_b, yz_b), [(1 - ci) / 2, (1 + ci) / 2])
    return float(_partial(xy, xz, yz)), float(lo), float(hi)


def agreement_drop(errors_a: pd.DataFrame, errors_b: pd.DataFrame, scans: np.ndarray, counts: np.ndarray,
                   ci: float = 0.95) -> Dict[str, float]:
    """Mean over engine pairs of the Pearson of ``log(1 + error)`` against the same
    reference (``a`` and ``b``) and across references, and the drop from ``a`` to
    across, the agreement ``a``'s shared noise made, with its scan interval."""
    scans = np.asarray(scans).astype(str)
    order = sorted(pd.unique(scans))
    la, lb = np.log1p(errors_a), np.log1p(errors_b)
    points: Dict[str, list] = {"same_a": [], "same_b": [], "cross": []}
    drops = []
    for j, k in itertools.combinations(errors_a.columns, 2):
        same_a, boot_a = _resampled_pearson(la[j], la[k], scans, order, counts)
        same_b, _ = _resampled_pearson(lb[j], lb[k], scans, order, counts)
        cross_1, boot_1 = _resampled_pearson(la[j], lb[k], scans, order, counts)
        cross_2, boot_2 = _resampled_pearson(lb[j], la[k], scans, order, counts)
        points["same_a"].append(same_a)
        points["same_b"].append(same_b)
        points["cross"].append((cross_1 + cross_2) / 2.0)
        drops.append(boot_a - (boot_1 + boot_2) / 2.0)
    lo, hi = np.nanquantile(np.mean(np.stack(drops), axis=0), [(1 - ci) / 2, (1 + ci) / 2])
    means = {name: float(np.mean(values)) for name, values in points.items()}
    return {"pearson_same_a": means["same_a"], "pearson_same_b": means["same_b"], "pearson_cross": means["cross"],
            "drop": means["same_a"] - means["cross"], "drop_ci_lo": float(lo), "drop_ci_hi": float(hi)}


def top_k_overlap(a: np.ndarray, b: np.ndarray, fractions: Sequence[float] = (0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Share of the top ``fraction`` by score ``a`` that is also in the top by ``b``."""
    rows = []
    for fraction in fractions:
        k = max(1, int(round(fraction * len(a))))
        top_a, top_b = set(np.argsort(-np.asarray(a))[:k]), set(np.argsort(-np.asarray(b))[:k])
        rows.append({"fraction": fraction, "k": k, "overlap": len(top_a & top_b) / k})
    return pd.DataFrame(rows)


def score_decile_table(score: np.ndarray, floor_rde: np.ndarray, engine_rde: np.ndarray,
                       l1_per_unit_dose: np.ndarray) -> pd.DataFrame:
    """Exploratory, per decile of the score: engine error, floor, the floor's share of
    the engine error per beamlet, and the noise per unit dose."""
    frame = pd.DataFrame({"score": score, "engine_rde": engine_rde, "floor_rde": floor_rde,
                          "noise_share": np.asarray(floor_rde) / np.asarray(engine_rde),
                          "l1_per_unit_dose": l1_per_unit_dose})
    frame["decile"] = pd.qcut(frame["score"], 10, labels=False) + 1
    return frame.groupby("decile").median().reset_index()


@dataclass(frozen=True)
class NoiseFloorThresholds:
    """The pass rule of the noise-floor checks, fixed before the per-beamlet results.
    The gap's own thresholds are those of :func:`src.active_learning.ensemble.independence.verdict`."""

    target_min_pearson: float = 0.95
    agreement_max_drop: float = 0.02
    score_max_partial: float = 0.20


def noise_floor_checks(median_gap: float, gap_ci_hi: float, target_pearson: float, drop: float, partial: float,
                       thresholds: Optional[NoiseFloorThresholds] = None) -> Tuple[pd.DataFrame, str]:
    """The four checks and the outcome: ``model-independent`` when all pass, else
    ``review`` (no automatic label, and no thresholds moved afterwards)."""
    t = thresholds or NoiseFloorThresholds()
    label = verdict(median_gap, gap_ci_hi)
    checks = pd.DataFrame([
        {"check": "gap_against_1e7", "value": median_gap, "rule": f"verdict() is model-independent, here {label}",
         "passes": label == "model-independent"},
        {"check": "target_1e6_vs_1e7", "value": target_pearson, "rule": f">= {t.target_min_pearson}",
         "passes": target_pearson >= t.target_min_pearson},
        {"check": "agreement_drop", "value": drop, "rule": f"<= {t.agreement_max_drop}",
         "passes": drop <= t.agreement_max_drop},
        {"check": "score_vs_noise_partial", "value": partial, "rule": f"abs <= {t.score_max_partial}",
         "passes": abs(partial) <= t.score_max_partial},
    ])
    return checks, "model-independent" if checks["passes"].all() else "review"
