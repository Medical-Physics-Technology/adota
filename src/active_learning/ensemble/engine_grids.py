"""Does an input-only difficulty score predict the error of dose engines on saved beamlet grids?

A saved block holds every beamlet of one field: one patient, energy and gantry, over
a grid of beamlet angles. The primary reading is therefore **within a field**, because
active-learning selection rescales the score within an energy layer, so comparing
beamlets of one field is what it relies on. Pooled numbers mix patients and energies
and are reported beside it.

Example beamlets are chosen from the scores alone (:func:`choose_examples`), before
any error is read, so they cannot be picked for telling a clean story.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pencil_beam.saved_beamlets import Engine, load_doses

GROUP_FIELD = "block"
GROUP_PATIENT = "patient_id"


def downsample2(volume: np.ndarray) -> np.ndarray:
    """Average every 2 x 2 x 2 block: the 1 mm crop on the 2 mm model grid."""
    a, b, c = (s // 2 for s in volume.shape)
    trimmed = volume[:2 * a, :2 * b, :2 * c].reshape(a, 2, b, 2, c, 2)
    return trimmed.mean(axis=(1, 3, 5))


def mape_above_cutoff(reference: np.ndarray, test: np.ndarray, cutoff_fraction: float = 0.1) -> float:
    """Mean ``|test - reference| / reference`` in %, over voxels above ``cutoff_fraction``
    of the reference maximum. NaN for an empty reference."""
    peak = float(np.max(reference))
    if peak <= 0:
        return float("nan")
    mask = reference > cutoff_fraction * peak
    return float(np.mean(np.abs(test[mask] - reference[mask]) / reference[mask]) * 100.0)


def beamlet_features(task: Tuple[str, str, str, float]) -> dict:
    """Worker: the score's analytic features of one saved beamlet, from its input only.

    ``task`` is ``(block_dir, stem, block, energy_mev)``. The crops go through
    :func:`~src.acquisition.candidates.model_grid_inputs`, the path of the model and
    of a candidate: normalise, then trilinear to the 2 mm grid.
    """
    from src.acquisition.candidates import analytic_features, model_grid_inputs

    block_dir, stem, block, energy = task
    ct = np.load(Path(block_dir) / f"{stem}_ct.npy")
    flux = np.load(Path(block_dir) / f"{stem}_flux.npy")
    features, inside = analytic_features(*model_grid_inputs(ct, flux, energy), energy)
    return {"block": block, "stem": stem, "peak_inside_crop": inside, **features}


def beamlet_mape(task: Tuple[str, str, str, Tuple[Engine, ...], float]) -> dict:
    """Worker: MAPE of each engine against the MC, on the 1 mm crop and on the 2 mm grid.

    ``task`` is ``(block_dir, stem, block, engines, cutoff_fraction)``. Columns are
    ``{tag}_mape_1mm`` and ``{tag}_mape_2mm``.
    """
    block_dir, stem, block, engines, cutoff = task
    doses = load_doses(Path(block_dir), stem, engines)
    mc = doses["MC"]
    mc_2mm = downsample2(mc)
    row = {"block": block, "stem": stem}
    for e in engines:
        row[f"{e.tag}_mape_1mm"] = mape_above_cutoff(mc, doses[e.name], cutoff)
        row[f"{e.tag}_mape_2mm"] = mape_above_cutoff(mc_2mm, downsample2(doses[e.name]), cutoff)
    return row


def reference_percentile(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The percentile of each score within a reference population's scores."""
    ordered = np.sort(np.asarray(reference, dtype=float))
    return 100.0 * np.searchsorted(ordered, np.asarray(scores, dtype=float), side="right") / len(ordered)


def choose_examples(scores: pd.DataFrame, percentiles: Mapping[str, float]) -> pd.DataFrame:
    """The beamlet nearest each named percentile of ``d_score``, from the scores alone.

    Ties go to the first in ``(block, stem)`` order, so the choice is deterministic.
    ``scores`` needs ``block``, ``stem`` and ``d_score``; a NaN score is never chosen.
    """
    usable = scores.dropna(subset=["d_score"]).sort_values(["block", "stem"]).reset_index(drop=True)
    picks = []
    for name, percentile in percentiles.items():
        target = float(np.percentile(usable["d_score"], percentile))
        row = usable.iloc[int((usable["d_score"] - target).abs().to_numpy().argmin())]
        picks.append({"case": name, "percentile_of_set": float(percentile), "target_score": target,
                      "block": row["block"], "stem": row["stem"], "d_score": float(row["d_score"])})
    return pd.DataFrame(picks)


def metric_columns(engines: Sequence[Engine], metrics: Mapping[str, str]) -> Dict[Tuple[str, str], str]:
    """``{(engine name, metric key): column}`` for ``metrics`` = ``{key: column suffix}``."""
    return {(e.name, key): f"{e.tag}_{suffix}" for e in engines for key, suffix in metrics.items()}


def within_field_spearman(frame: pd.DataFrame, columns: Mapping[Tuple[str, str], str]) -> pd.DataFrame:
    """Spearman of ``d_score`` with each metric, per field. One row per field, engine and metric."""
    rows = []
    for field_name, group in frame.groupby(GROUP_FIELD, sort=True):
        for (engine, metric), column in columns.items():
            pair = group[["d_score", column]].dropna()
            spread = len(pair) > 2 and pair["d_score"].nunique() > 1 and pair[column].nunique() > 1
            rho = float(pair["d_score"].corr(pair[column], method="spearman")) if spread else float("nan")
            rows.append({GROUP_FIELD: field_name, GROUP_PATIENT: group[GROUP_PATIENT].iloc[0],
                         "anatomy": group["anatomy"].iloc[0], "energy": group["energy"].iloc[0],
                         "engine": engine, "metric": metric, "spearman": rho, "n": len(pair)})
    return pd.DataFrame(rows)


def spearman_summary(frame: pd.DataFrame, per_field: pd.DataFrame, columns: Mapping[Tuple[str, str], str],
                     n_bootstrap: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per engine and metric: the median within-field Spearman with a patient-bootstrap
    interval, the share of fields with the expected sign, and the pooled Spearman.

    A pass rate should fall with difficulty (negative), a MAPE rise (positive).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (engine, metric), column in columns.items():
        part = per_field[(per_field["engine"] == engine) & (per_field["metric"] == metric)].dropna(subset=["spearman"])
        by_patient = [g["spearman"].to_numpy() for _, g in part.groupby(GROUP_PATIENT)]
        draws = [np.median(np.concatenate([by_patient[i] for i in rng.integers(0, len(by_patient), len(by_patient))]))
                 for _ in range(n_bootstrap)] if by_patient else [float("nan")]
        expected = -1.0 if metric.startswith("gpr") else 1.0
        pair = frame[["d_score", column]].dropna()
        rows.append({"engine": engine, "metric": metric, "median_within_field": float(part["spearman"].median()),
                     "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)),
                     "share_fields_expected_sign": float(np.mean(np.sign(part["spearman"]) == expected)),
                     "n_fields": len(part), "n_patients": len(by_patient),
                     "pooled_spearman": float(pair["d_score"].corr(pair[column], method="spearman")),
                     "n_beamlets": len(pair)})
    return pd.DataFrame(rows)


def add_within_field_decile(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy with ``field_decile``: 1 to 10, the score's decile within its own field."""
    out = frame.copy()
    ranks = out.groupby(GROUP_FIELD)["d_score"].rank(method="first", pct=True)
    out["field_decile"] = np.ceil(ranks * 10).clip(1, 10)
    return out


def decile_table(frame: pd.DataFrame, columns: Mapping[Tuple[str, str], str]) -> pd.DataFrame:
    """Median, quartiles and count of each metric per within-field score decile."""
    rows = []
    for decile, group in frame.dropna(subset=["field_decile"]).groupby("field_decile"):
        for (engine, metric), column in columns.items():
            values = group[column].dropna().to_numpy(dtype=float)
            rows.append({"field_decile": int(decile), "engine": engine, "metric": metric, "n": len(values),
                         "median": float(np.median(values)), "q25": float(np.percentile(values, 25)),
                         "q75": float(np.percentile(values, 75))})
    return pd.DataFrame(rows)


def per_engine_summary(frame: pd.DataFrame, columns: Mapping[Tuple[str, str], str]) -> pd.DataFrame:
    """Each engine's overall median, mean and 5th and 95th percentile of each metric."""
    rows = []
    for (engine, metric), column in columns.items():
        v = frame[column].dropna().to_numpy(dtype=float)
        rows.append({"engine": engine, "metric": metric, "n": len(v), "median": float(np.median(v)),
                     "mean": float(np.mean(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))})
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Mapping) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")


__all__: List[str] = [
    "add_within_field_decile", "beamlet_features", "beamlet_mape", "choose_examples", "decile_table",
    "downsample2", "mape_above_cutoff", "metric_columns", "per_engine_summary", "reference_percentile",
    "spearman_summary", "within_field_spearman", "write_json",
]
