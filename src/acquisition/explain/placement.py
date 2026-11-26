"""Place beamlets against a reference population, feature by feature.

For each (beamlet, feature) pair this gives the mid-rank percentile of the value within the reference and a
robust z score. The robust z is ``(value - median) / scale`` where the scale is expressed on the scale of a
normal SD: ``IQR / 1.349``. The median and IQR are used instead of the mean and SD because many difficulty
features are strongly skewed or have many zeros, where the SD is dominated by the tail.

Scale fallback order: IQR / 1.349 (method ``"iqr"``); if the IQR is 0, MAD * 1.4826 (method ``"mad"``); if that
is 0 too, the scale is NaN (method ``"degenerate"``) and the robust z is NaN. Note that an IQR of 0 means at
least half of the reference sits on the median, which already forces the MAD to 0, so in practice a feature
with a zero IQR ends up ``"degenerate"``; the MAD branch is a safety net for the exact-boundary cases.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

IQR_TO_SD = 1.349  # IQR of a standard normal
MAD_TO_SD = 1.4826  # MAD of a standard normal

TABLE_COLUMNS = (
    "beamlet_id",
    "feature",
    "value",
    "percentile_ref",
    "robust_z",
    "ref_median",
    "ref_q25",
    "ref_q75",
    "ref_scale",
    "scale_method",
    "n_reference",
)


def _finite_sorted(reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float).ravel()
    return np.sort(ref[~np.isnan(ref)])


def percentile_of(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value within ``reference``, in [0, 100]:
    100 * (n_below + 0.5 * n_equal) / n. NaNs in reference are dropped; a NaN value gives NaN."""
    ref = _finite_sorted(reference)
    vals = np.asarray(values, dtype=float)
    n = ref.size
    if n == 0:
        return np.full(vals.shape, np.nan)
    below = np.searchsorted(ref, vals, side="left")
    at_or_below = np.searchsorted(ref, vals, side="right")
    out = 100.0 * (below + 0.5 * (at_or_below - below)) / n
    return np.where(np.isnan(vals), np.nan, out)


def robust_scale(reference: np.ndarray) -> Tuple[float, float, str]:
    """(median, scale, method). scale = IQR / IQR_TO_SD (method "iqr"); if the IQR is 0,
    scale = MAD * MAD_TO_SD (method "mad"); if that is 0 too, scale = NaN (method "degenerate")."""
    ref = _finite_sorted(reference)
    if ref.size == 0:
        return float("nan"), float("nan"), "degenerate"
    q25, med, q75 = np.quantile(ref, [0.25, 0.5, 0.75])
    iqr = q75 - q25
    if iqr > 0:
        return float(med), float(iqr / IQR_TO_SD), "iqr"
    mad = float(np.median(np.abs(ref - med)))
    if mad > 0:
        return float(med), mad * MAD_TO_SD, "mad"
    return float(med), float("nan"), "degenerate"


def placement_table(reference: pd.DataFrame, beamlets: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Long form, one row per (beamlet, metric), beamlet order then metric order kept.
    Columns: beamlet_id (the index of ``beamlets``), feature, value, percentile_ref,
    robust_z = (value - median) / scale (NaN when degenerate), ref_median, ref_q25, ref_q75,
    ref_scale, scale_method, n_reference."""
    ids = list(beamlets.index)
    frames = []
    for metric in metrics:
        ref = _finite_sorted(reference[metric].to_numpy(dtype=float))
        med, scale, method = robust_scale(ref)
        q25, q75 = (np.quantile(ref, [0.25, 0.75]) if ref.size else (np.nan, np.nan))
        vals = beamlets[metric].to_numpy(dtype=float)
        z = (vals - med) / scale if np.isfinite(scale) else np.full(vals.shape, np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "beamlet_id": ids,
                    "feature": metric,
                    "value": vals,
                    "percentile_ref": percentile_of(ref, vals),
                    "robust_z": z,
                    "ref_median": med,
                    "ref_q25": q25,
                    "ref_q75": q75,
                    "ref_scale": scale,
                    "scale_method": method,
                    "n_reference": ref.size,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(TABLE_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    # beamlet order first, then metric order: stable sort on the beamlet position
    pos = np.tile(np.arange(len(ids)), len(frames))
    mpos = np.repeat(np.arange(len(frames)), len(ids))
    order = np.lexsort((mpos, pos))
    return out.iloc[order].reset_index(drop=True)[list(TABLE_COLUMNS)]
