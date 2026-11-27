"""HU histograms of the CT crops the model sees, per anatomical site.

Each export stores ``ct`` as a ``(30, 30, 160)`` float32 crop on the 2 mm model grid,
every value the average of a 2x2x2 block of 1 mm int16 HU, min-max normalised with the
file attribute ``scale_json``: ``v = (HU - min_ct) / (max_ct - min_ct)``. The inverse is
rounded to the nearest 1/8 HU (an average of eight integers is a multiple of 1/8), so the
float32 round trip never moves a value across a bin edge. Binning and the class counts
work on integer eighths of a HU, which makes every comparison exact.

Padding and range: for the pool test export (``v1_apollo_1e6_test.h5``) it was verified
before this module was written that no crop is padded (every lateral window lies inside
the CT grid and every depth grid holds at least 334 voxels, more than the 320 the crop
needs) and that no stored HU lies outside ``[-1024, 3071]``. All voxels are therefore
histogrammed without a mask. Values outside the range after the inverse are still
clipped into the end bins and counted per site, so a different export that breaks the
assumption shows it in :func:`clipping_frame` instead of silently.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Union

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HU_MIN = -1024
HU_MAX = 3071
HU_EDGES: np.ndarray = np.r_[np.arange(HU_MIN, 3067, 10), HU_MAX].astype(float)
"""410 bins: low edges -1024..3066 in 10 HU steps; the last bin [3066, 3071] is partial and closed on the right."""
N_BINS = len(HU_EDGES) - 1
BODY_THRESHOLD_HU = -500.0
CLASS_THRESHOLD_HIGH_HU = 300.0
CLASS_LABELS = ("below_-500", "-500_to_300", "above_300")
"""HU < -500 ; -500 <= HU <= 300 ; HU > 300."""
SITE_ORDER = ("thoracic", "abdominal", "pelvic")

_EIGHTHS = 8
_BIN_WIDTH_EIGHTHS = 10 * _EIGHTHS


@dataclass
class SiteHU:
    """Accumulated HU counts of one site."""

    counts: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS, dtype=np.int64))
    body_counts: np.ndarray = field(default_factory=lambda: np.zeros(N_BINS, dtype=np.int64))
    class_counts: np.ndarray = field(default_factory=lambda: np.zeros(len(CLASS_LABELS), dtype=np.int64))
    n_voxels: int = 0
    n_crops: int = 0
    n_clipped_low: int = 0
    n_clipped_high: int = 0


def export_hu(ct_normalised: np.ndarray, scale: Mapping) -> np.ndarray:
    """Invert the export's min-max normalisation and round to the nearest 1/8 HU (float64)."""
    lo, hi = float(scale["min_ct"]), float(scale["max_ct"])
    hu = np.asarray(ct_normalised, dtype=np.float64) * (hi - lo) + lo
    return np.round(hu * _EIGHTHS) / _EIGHTHS


def _add_crop(acc: SiteHU, hu: np.ndarray) -> None:
    """Add one crop of HU values (already rounded to 1/8 HU) to ``acc``."""
    e = np.rint(np.ravel(hu) * _EIGHTHS).astype(np.int64)
    lo, hi = HU_MIN * _EIGHTHS, HU_MAX * _EIGHTHS
    acc.n_clipped_low += int(np.count_nonzero(e < lo))
    acc.n_clipped_high += int(np.count_nonzero(e > hi))
    e = np.clip(e, lo, hi)
    # Classes and the body mask come from the HU values themselves; -500 and 300 are not bin edges.
    below = e < int(BODY_THRESHOLD_HU) * _EIGHTHS
    above = e > int(CLASS_THRESHOLD_HIGH_HU) * _EIGHTHS
    n_below, n_above = int(np.count_nonzero(below)), int(np.count_nonzero(above))
    acc.class_counts += np.array([n_below, e.size - n_below - n_above, n_above], dtype=np.int64)
    # Bin i holds [edge_i, edge_i+1); the last bin also takes 3071 (closed on the right).
    idx = np.minimum((e - lo) // _BIN_WIDTH_EIGHTHS, N_BINS - 1)
    acc.counts += np.bincount(idx, minlength=N_BINS)
    # Body is HU > -500 strictly, so a voxel at exactly -500 is in the middle class but not in the body.
    acc.body_counts += np.bincount(idx[e > int(BODY_THRESHOLD_HU) * _EIGHTHS], minlength=N_BINS)
    acc.n_voxels += e.size
    acc.n_crops += 1


def accumulate_hu(export_h5: Union[str, Path], sample_sites: Mapping[str, str],
                  log_every: int = 500) -> Dict[str, SiteHU]:
    """HU histograms per site over the given sample ids of one export.

    Args:
        export_h5: v3-layout HDF5 export, one group per sample id holding ``ct``.
        sample_sites: sample id to site name.
        log_every: log progress every this many crops (0 or less disables it).

    Raises:
        KeyError: if a sample id is not a group of the file.
    """
    per_site: Dict[str, SiteHU] = {}
    t0 = perf_counter()
    with h5py.File(export_h5, "r") as f:
        scale = json.loads(f.attrs["scale_json"])
        n = len(sample_sites)
        for i, (sid, site) in enumerate(sample_sites.items(), start=1):
            if sid not in f:
                raise KeyError(f"sample id {sid!r} not in {export_h5}")
            hu = export_hu(f[sid]["ct"][()], scale)
            _add_crop(per_site.setdefault(site, SiteHU()), hu)
            if log_every > 0 and (i % log_every == 0 or i == n):
                logger.info("HU histogram: %d/%d crops (%.1f s)", i, n, perf_counter() - t0)
    return per_site


def _ordered_sites(per_site: Mapping[str, SiteHU]) -> List[str]:
    known = [s for s in SITE_ORDER if s in per_site]
    return known + sorted(s for s in per_site if s not in SITE_ORDER)


def histogram_frame(per_site: Mapping[str, SiteHU], body: bool = False) -> pd.DataFrame:
    """Long table ``site, bin_low_hu, count``; ``body=True`` keeps only HU > -500."""
    frames = []
    for site in _ordered_sites(per_site):
        acc = per_site[site]
        frames.append(pd.DataFrame({"site": site, "bin_low_hu": HU_EDGES[:-1],
                                    "count": acc.body_counts if body else acc.counts}))
    if not frames:
        return pd.DataFrame(columns=["site", "bin_low_hu", "count"])
    return pd.concat(frames, ignore_index=True)


def class_frame(per_site: Mapping[str, SiteHU]) -> pd.DataFrame:
    """Table ``site, hu_class, n_voxels, share``; share is of the site's voxels (0-1)."""
    rows = []
    for site in _ordered_sites(per_site):
        acc = per_site[site]
        for label, n in zip(CLASS_LABELS, acc.class_counts):
            rows.append({"site": site, "hu_class": label, "n_voxels": int(n),
                         "share": float(n) / acc.n_voxels if acc.n_voxels else float("nan")})
    return pd.DataFrame(rows, columns=["site", "hu_class", "n_voxels", "share"])


def clipping_frame(per_site: Mapping[str, SiteHU]) -> pd.DataFrame:
    """Table ``site, n_crops, n_voxels, n_clipped_low, n_clipped_high``."""
    rows = [{"site": s, "n_crops": per_site[s].n_crops, "n_voxels": per_site[s].n_voxels,
             "n_clipped_low": per_site[s].n_clipped_low, "n_clipped_high": per_site[s].n_clipped_high}
            for s in _ordered_sites(per_site)]
    return pd.DataFrame(rows, columns=["site", "n_crops", "n_voxels", "n_clipped_low", "n_clipped_high"])
