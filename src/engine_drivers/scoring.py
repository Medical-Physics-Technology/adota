"""Per-beamlet errors of saved engine doses against the Monte Carlo, on the model grid.

Every engine is scored against the same reference, the export's 2 mm Monte Carlo
dose de-normalised with the export's own scale, through the one definition of the
per-record errors, :func:`src.active_learning.retrospective.validation.record_metrics`
(MAPE, RDE, signed dR80 and the gamma pass rate). Gamma runs once per criterion; all
criteria are global, normalised to the Monte Carlo maximum, with their own low-dose
cutoff.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

from src.active_learning.retrospective.validation import MetricSettings, record_metrics
from src.utils.scallers import inverse_minmax

logger = logging.getLogger(__name__)

CRITERIA: Tuple[Tuple[float, float, float], ...] = ((2.0, 2.0, 10.0), (3.0, 3.0, 10.0))
"""``(dose %, distance mm, cutoff %)``, global gamma."""


def criterion_key(c: Sequence[float]) -> str:
    """``gpr_2pct_2mm_10pct``."""
    return f"gpr_{c[0]:g}pct_{c[1]:g}mm_{c[2]:g}pct"


def export_scale(dataset_path: Path) -> Dict[str, float]:
    """The dose scale the export was normalised with."""
    with h5py.File(dataset_path, "r") as handle:
        raw = handle.attrs["scale_json"]
    scale = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    return {k: float(v) for k, v in scale.items()}


def mc_model_grid(handle: h5py.File, sample_id: str, scale: Mapping[str, float]) -> np.ndarray:
    """The export's Monte Carlo dose of one record, de-normalised, model layout ``(D, H, W)``."""
    dose = np.asarray(handle[sample_id]["dose"][...], dtype=np.float32)
    return np.ascontiguousarray(inverse_minmax(dose, scale["min_ds"], scale["max_ds"]).transpose(2, 0, 1))


def _settings(c: Sequence[float], scale: Mapping[str, float], backend: str) -> MetricSettings:
    params = {"dose_percent_threshold": c[0], "distance_mm_threshold": c[1], "interp_fraction": 10,
              "max_gamma": 2, "lower_percent_dose_cutoff": c[2], "random_subset": None, "local_gamma": False,
              "quiet": True}
    return MetricSettings(scale=dict(scale), gamma_params=params, gamma_cutoff_percent=c[2], gamma_backend=backend)


def score_doses(dataset_path: Path, dose_dir: Path, ids: Sequence[str], engine: str,
                criteria: Sequence[Sequence[float]] = CRITERIA, backend: str = "torch",
                backend_options: Optional[dict] = None) -> pd.DataFrame:
    """One row per id: every gamma criterion (%), MAPE, RDE, signed dR80 and dose ratios.

    A missing dose file is an error, never a silent gap.
    """
    scale = export_scale(dataset_path)
    settings = [_settings(c, scale, backend) for c in criteria]
    rows: List[dict] = []
    started = perf_counter()
    heartbeat = max(1, len(ids) // 10)
    with h5py.File(dataset_path, "r") as handle:
        for n, sid in enumerate(ids, 1):
            mc = mc_model_grid(handle, sid, scale)
            test = np.load(Path(dose_dir) / f"{sid}.npy")
            if test.shape != mc.shape:
                raise ValueError(f"{engine} {sid}: dose shape {test.shape}, Monte Carlo {mc.shape}")
            row = {"sample_id": sid, "engine": engine}
            for c, s in zip(criteria, settings):
                m = record_metrics(mc, test, s, backend_options, sample_id=sid)
                row[criterion_key(c)] = 100.0 * m["gpr"]
            row.update({k: m[k] for k in ("mape_pct", "rde_pct", "dr80_mm", "r80_gt_mm", "r80_pred_mm",
                                          "peak_dose_gt")})
            row["peak_ratio"] = float(test.max() / mc.max()) if mc.max() > 0 else np.nan
            row["integral_ratio"] = float(test.sum() / mc.sum()) if mc.sum() > 0 else np.nan
            rows.append(row)
            if n % heartbeat == 0:
                logger.info("score %s %d/%d  %.0f s", engine, n, len(ids), perf_counter() - started)
    return pd.DataFrame(rows)


__all__ = ["CRITERIA", "criterion_key", "export_scale", "mc_model_grid", "score_doses"]
