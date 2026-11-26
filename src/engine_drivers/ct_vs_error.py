"""CT features against per-beamlet dose error: the tables behind ``analysis/ct_vs_error``.

Input-only features only: the 30 difficulty features of the pool test split, computed in
``analytic`` mode (the peak and the edge weights come from the analytic surrogate of CT,
flux and energy, :func:`src.acquisition.reference.record_features`), plus the two path
features of :mod:`src.engine_drivers.geometry`. The two diagnostic columns of the feature
CSV that compare the analytic peak with the Monte Carlo peak are dropped.

Statistics, per engine, error and feature, on the beamlets with the analytic peak inside
the crop:

- Spearman correlation, pooled and per site, oriented so that positive means harder (the
  gamma pass rate enters as its failing share), with 95% intervals from resampling patients
  (:func:`src.engine_drivers.importance.univariate`);
- partial Spearman controlling for beam energy: the correlation of the rank residuals
  after a linear fit on the energy rank, with the same patient resampling;
- binned medians in both directions: the feature per error bin (bins of the error within
  each site; tied error values stay in one bin, so a saturated pass rate gives fewer bins)
  and the error per feature decile (same tie rule).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.acquisition.features import FEATURE_NAMES
from src.engine_drivers.geometry import PATH_FEATURES
from src.engine_drivers.importance import Target, _patient_resamples, univariate

MC_DIAGNOSTICS = ("peak_depth_err_mm", "peak_lateral_err_vox")
"""Columns of the feature CSV computed against the Monte Carlo peak: not input-only, never exported."""

ERRORS: Mapping[str, Target] = {
    "gpr_2pct_2mm_10pct": Target("fail_2_2", "gpr_2pct_2mm_10pct", "gamma fail 2%/2mm (100 - GPR)", "fail_log"),
    "mape_pct": Target("mape", "mape_pct", "MAPE", "log"),
}
"""The two error measures of the figures, oriented so that larger is worse for the statistics."""

CT_FEATURES = ("total_hu_change", "isi_sum", "sobel_th_mean", "sobel_dw_anisotropy", "pflugfelder_hi",
               "interface_bp_distance", "lowdensity_path_mm", "air_gap_mm", "ct_max_hu")
"""The medoid of every correlated group except depth and energy (EXP-0023 ``groups.csv``)."""
DEPTH_FEATURES = ("max_grad_depth_mm", "energy_mev")

FEATURE_LABELS: Mapping[str, str] = {
    "total_hu_change": "Tissue change in the peak zone\ntotal_hu_change [HU]",
    "isi_sum": "Interface severity, sum of RSP jumps squared\nisi_sum [dimensionless]",
    "sobel_th_mean": "Edge strength around the peak\nsobel_th_mean [a.u.]",
    "sobel_dw_anisotropy": "Edge anisotropy around the peak\nsobel_dw_anisotropy [0-1]",
    "pflugfelder_hi": "WEPL spread across the beamlet (std / mean)\npflugfelder_hi [dimensionless]",
    "interface_bp_distance": "Peak to nearest interface\ninterface_bp_distance [2 mm slices]",
    "lowdensity_path_mm": "Low-density path to the peak\nlowdensity_path_mm [mm]",
    "air_gap_mm": "Air (and couch) before the skin\nair_gap_mm [mm]",
    "ct_max_hu": "Densest voxel in the crop\nct_max_hu [HU]",
    "max_grad_depth_mm": "Depth of the steepest dose rise\nmax_grad_depth_mm [mm]",
    "energy_mev": "Beam energy\nenergy_mev [MeV]",
}
ERROR_LABELS: Mapping[str, str] = {
    "gpr_2pct_2mm_10pct": "GPR 2%/2mm, 10% cutoff [%]",
    "mape_pct": "MAPE [%]",
}
SITES = ("thoracic", "abdominal", "pelvic")


def beamlet_feature_table(features_csv: Path, index_csv: Path, path_features: pd.DataFrame) -> pd.DataFrame:
    """One row per beamlet: ids, site, patient, energy, gantry, the 32 input-only features, the flag.

    Raises:
        ValueError: if the feature CSV, the export index and the path features do not cover
            the same beamlets.
    """
    feats = pd.read_csv(features_csv, dtype={"sample_id": str}).set_index("sample_id")
    index = pd.read_csv(index_csv, dtype={"sample_id": str}).set_index("sample_id")
    for name, other in (("export index", index), ("path features", path_features)):
        if set(other.index) != set(feats.index):
            raise ValueError(f"the feature CSV ({len(feats)}) and the {name} ({len(other)}) cover different beamlets")
    if (index["energy_mev"] - feats["energy_mev"]).abs().max() > 1e-6:
        raise ValueError("energy_mev of the feature CSV and the export index differ")
    meta = pd.DataFrame({"patient_id": index["pool_patient_id"], "site": index["pool_anatomy"],
                         "gantry_deg": index["gantry_angle"]})
    table = meta.join(feats[list(FEATURE_NAMES)]).join(path_features[list(PATH_FEATURES)])
    table["peak_inside_crop"] = feats["peak_inside_crop"].astype(bool)
    table["analytic_bp_slice"] = feats["bp_slice"].astype(int)
    cols = ["patient_id", "site", "energy_mev", "gantry_deg"] + [c for c in table.columns
                                                                    if c not in ("patient_id", "site", "energy_mev",
                                                                                 "gantry_deg")]
    assert not set(MC_DIAGNOSTICS) & set(cols)
    return table[cols].rename_axis("sample_id").reset_index()


def join_errors(table: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    """The feature table with one engine's errors, asserting that no beamlet is lost on either side."""
    joined = table.merge(errors.drop(columns=["engine"], errors="ignore"), on="sample_id", how="inner",
                         validate="one_to_one")
    if not (len(joined) == len(table) == len(errors)):
        raise ValueError(f"join lost rows: features {len(table)}, errors {len(errors)}, joined {len(joined)}")
    return joined


def partial_spearman(frame: pd.DataFrame, features: Sequence[str], target: Target, control: str,
                     groups: np.ndarray, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Spearman of each feature with the error after removing ``control`` (rank residuals), with a
    patient-bootstrap 95% interval. A feature equal to the control gets NaN."""
    y = rankdata(target.values(frame))
    z = rankdata(frame[control].to_numpy(dtype=float))
    X = np.column_stack([rankdata(frame[f].to_numpy(dtype=float)) for f in features])

    def corr(rows: np.ndarray) -> np.ndarray:
        zc = z[rows] - z[rows].mean()
        zz = float(zc @ zc)

        def resid(v: np.ndarray) -> np.ndarray:
            vc = v - v.mean(axis=0)
            return vc - np.outer(zc, zc @ vc / zz) if vc.ndim == 2 else vc - zc * (zc @ vc / zz)
        rx, ry = resid(X[rows]), resid(y[rows])
        den = np.sqrt((rx ** 2).sum(axis=0) * (ry ** 2).sum())
        return np.where(den > 1e-9 * len(rows), (rx * ry[:, None]).sum(axis=0) / np.where(den > 0, den, 1.0), np.nan)

    point = corr(np.arange(len(y)))
    boot = np.array([corr(r) for r in _patient_resamples(groups, n_boot, seed)])
    with np.errstate(all="ignore"):
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"feature": list(features), "spearman": point, "ci_low": lo, "ci_high": hi})
    out.loc[out["feature"] == control, ["spearman", "ci_low", "ci_high"]] = np.nan
    return out


def correlations(points: pd.DataFrame, features: Sequence[str], error: str, n_boot: int, seed: int,
                 sites: Sequence[str] = SITES) -> pd.DataFrame:
    """Pooled, per-site and energy-controlled Spearman of every feature with one error (positive = worse)."""
    target = ERRORS[error]
    rows = []
    uni = univariate(points, features, target, points["patient_id"].to_numpy(), n_boot, seed)
    rows.append(uni.assign(scope="pooled"))
    for s in sites:
        sub = points[points["site"] == s]
        rows.append(univariate(sub, features, target, sub["patient_id"].to_numpy(), n_boot, seed).assign(scope=s))
    rows.append(partial_spearman(points, features, target, "energy_mev", points["patient_id"].to_numpy(), n_boot,
                                 seed).assign(scope="energy_partial"))
    out = pd.concat(rows, ignore_index=True)[["feature", "scope", "spearman", "ci_low", "ci_high"]]
    out["ci_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    out.insert(0, "error", error)
    return out


def _bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Equal-count bins with ties kept together (``qcut`` on the values, duplicate edges dropped)."""
    return pd.qcut(values, n_bins, labels=False, duplicates="drop").astype(int) + 1


def binned(points: pd.DataFrame, by: str, of: str, n_bins: int = 10, sites: Sequence[str] = SITES) -> pd.DataFrame:
    """Per site: bins of ``by``; in each, the median of ``by`` and the median and quartiles of ``of``."""
    rows = []
    for s in sites:
        sub = points[points["site"] == s]
        b = _bins(sub[by], n_bins)
        for k, g in sub.groupby(b):
            rows.append({"site": s, "bin": int(k), "n": len(g), f"{by}_median": float(g[by].median()),
                         f"{of}_median": float(g[of].median()), f"{of}_q25": float(g[of].quantile(0.25)),
                         f"{of}_q75": float(g[of].quantile(0.75))})
    return pd.DataFrame(rows)


def error_bins(points: pd.DataFrame, features: Sequence[str], error: str, n_bins: int = 10) -> pd.DataFrame:
    """The figure schema ``error, feature, site, bin, error_median, n, feature_median, feature_q25, feature_q75``."""
    out = []
    for f in features:
        b = binned(points, error, f, n_bins).rename(columns={f"{error}_median": "error_median",
                                                             f"{f}_median": "feature_median",
                                                             f"{f}_q25": "feature_q25", f"{f}_q75": "feature_q75"})
        out.append(b.assign(error=error, feature=f))
    return pd.concat(out, ignore_index=True)[["error", "feature", "site", "bin", "error_median", "n",
                                              "feature_median", "feature_q25", "feature_q75"]]


def feature_bins(points: pd.DataFrame, features: Sequence[str], error: str, n_bins: int = 10) -> pd.DataFrame:
    """The figure schema ``error, feature, site, bin, feature_median, n, error_median, error_q25, error_q75``."""
    out = []
    for f in features:
        b = binned(points, f, error, n_bins).rename(columns={f"{f}_median": "feature_median",
                                                             f"{error}_median": "error_median",
                                                             f"{error}_q25": "error_q25", f"{error}_q75": "error_q75"})
        out.append(b.assign(error=error, feature=f))
    return pd.concat(out, ignore_index=True)[["error", "feature", "site", "bin", "feature_median", "n",
                                              "error_median", "error_q25", "error_q75"]]


def verdict(stats: pd.DataFrame, error: str, feature: str, sites: Sequence[str] = SITES) -> Tuple[str, Dict[str, bool]]:
    """One plain sentence on how a feature relates to an error, by fixed rules, and the flags behind it.

    A site counts when its interval excludes zero, and its sign is always taken into account. Rules:
    pooled relation (interval excludes zero) -> its direction, the sites where it holds in the same
    direction, the sites where it runs the other way, "between sites" when it holds in the same direction
    within no site, and "disappears" when the energy-controlled interval includes zero. No pooled relation
    -> the sites where one exists, and "cancel" when they disagree in sign.
    """
    s = stats[(stats["error"] == error) & (stats["feature"] == feature)].set_index("scope")
    pooled = s.loc["pooled"]
    sig = bool(pooled["ci_excludes_zero"])
    partial = s.loc["energy_partial"] if "energy_partial" in s.index else None
    partial_sig = bool(partial["ci_excludes_zero"]) if partial is not None and np.isfinite(partial["spearman"]) \
        else None
    site_sig = {x: bool(s.loc[x, "ci_excludes_zero"]) for x in sites if x in s.index}
    site_rho = {x: float(s.loc[x, "spearman"]) for x in site_sig}
    flags = {"pooled": sig, "energy_partial": partial_sig, **{f"site_{k}": v for k, v in site_sig.items()}}
    what = "a worse error" if error != "gpr_2pct_2mm_10pct" else "more gamma failures"

    def listing(names):
        return ", ".join(f"{x} ({site_rho[x]:+.2f})" for x in names)

    pos = [x for x, v in site_sig.items() if v and site_rho[x] > 0]
    neg = [x for x, v in site_sig.items() if v and site_rho[x] < 0]
    if not sig:
        text = f"No pooled relation with {what} (rho {pooled['spearman']:+.2f})"
        if pos and neg:
            text += (f"; the sites disagree, positive within {listing(pos)} and negative within {listing(neg)}, "
                     "and cancel when pooled")
        elif pos or neg:
            text += f"; within {listing(pos or neg)} there is one"
        return text + ".", flags
    same, other = (pos, neg) if pooled["spearman"] > 0 else (neg, pos)
    direction = "higher" if pooled["spearman"] > 0 else "lower"
    text = f"A {direction} value goes with {what} (pooled rho {pooled['spearman']:+.2f})"
    if not same:
        text += (", but within no single site in that direction: the pooled relation comes from differences "
                 "between sites")
    else:
        text += f"; it holds within {listing(same)}"
    if other:
        text += f"; within {listing(other)} it runs the other way"
    if partial_sig is False:
        text += ". It disappears after controlling for beam energy"
    elif partial_sig:
        text += f". It survives controlling for beam energy (rho {partial['spearman']:+.2f})"
    return text + ".", flags


__all__ = ["CT_FEATURES", "DEPTH_FEATURES", "ERRORS", "ERROR_LABELS", "FEATURE_LABELS", "MC_DIAGNOSTICS", "SITES",
           "beamlet_feature_table", "binned", "correlations", "error_bins", "feature_bins", "join_errors",
           "partial_spearman", "verdict"]
