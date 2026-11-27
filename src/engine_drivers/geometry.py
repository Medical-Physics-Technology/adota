"""Two path features the 30 difficulty features lack: what the beam crosses before its peak.

EXP-0022 found that the score's depth term measures geometric depth, so air in front
of the patient and lung on the way to the peak both raise it. These two features
separate them, on the same flux-weighted beam-axis HU the difficulty features use
(:func:`src.acquisition.explain.profiles.flux_weighted_hu`):

- ``air_gap_mm``: depth of the skin, the first slice of a run of at least ``min_run``
  slices above ``skin_hu``: the air (and any couch or immobilisation shell, too thin to
  form such a run) the beam crosses before the patient;
- ``lowdensity_path_mm``: slices below ``skin_hu`` between the skin and the analytic
  peak, i.e. lung (or bowel gas) on the way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from src.acquisition.explain.profiles import flux_weighted_hu
from src.acquisition.features import FeatureConfig
from src.active_learning.retrospective.scoring import read_stored_inputs

SKIN_HU = -500.0
"""HU above which a beam-axis slice counts as dense."""
MIN_RUN = 5
"""Consecutive dense slices (10 mm on the model grid) that mark the skin; a couch shell is thinner."""


def path_lengths(hu_axis: np.ndarray, peak_slice: int, dz_mm: float = 2.0, skin_hu: float = SKIN_HU,
                 min_run: int = MIN_RUN) -> tuple:
    """``(air_gap_mm, lowdensity_path_mm)`` of one flux-weighted beam-axis HU profile."""
    dense = np.asarray(hu_axis) > skin_hu
    run = np.convolve(dense.astype(int), np.ones(min_run, dtype=int), mode="valid") == min_run
    starts = np.nonzero(run)[0]
    skin = int(starts[0]) if starts.size else len(hu_axis)
    stop = max(skin, min(int(peak_slice), len(hu_axis)))
    low = int(np.sum(hu_axis[skin:stop] <= skin_hu))
    return skin * dz_mm, low * dz_mm


def path_features(dataset_path: Path, peak_slices: Mapping[str, int], scale: Mapping[str, float],
                  dz_mm: float = 2.0, skin_hu: float = SKIN_HU,
                  flux_frac: float = FeatureConfig().flux_threshold_frac) -> pd.DataFrame:
    """``air_gap_mm`` and ``lowdensity_path_mm`` of every id of ``peak_slices`` (id -> analytic peak slice)."""
    rows = []
    with h5py.File(dataset_path, "r") as handle:
        for sid, peak in peak_slices.items():
            rec = read_stored_inputs(handle[sid], sid, dict(scale))
            hu = flux_weighted_hu(rec.ct_hu, rec.flux, flux_frac)
            air, low = path_lengths(hu, int(peak), dz_mm, skin_hu)
            rows.append({"sample_id": sid, "air_gap_mm": air, "lowdensity_path_mm": low})
    return pd.DataFrame(rows).set_index("sample_id")


PATH_FEATURES: Sequence[str] = ("air_gap_mm", "lowdensity_path_mm")

__all__ = ["MIN_RUN", "PATH_FEATURES", "SKIN_HU", "path_features", "path_lengths"]
