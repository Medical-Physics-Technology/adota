"""The ``analyse`` stage of ``engine_error_drivers.py``: every engine, error metric and site.

Population: the beamlets of the run with the analytic peak inside the crop (where the
30 difficulty features are defined), plus the two path features of
:mod:`src.engine_drivers.geometry`. Features are grouped by correlation exactly as in
EXP-0022 (:func:`src.acquisition.explain.groups.feature_groups`, ``|rho|`` cut
``group_threshold``). Panels: all beamlets and each anatomy. Results go to
``<run>/analysis/`` as CSV, with a markdown summary of the numbers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd
from threadpoolctl import threadpool_limits

from src.acquisition.explain.groups import feature_groups
from src.acquisition.features import FEATURE_NAMES
from src.active_learning.ensemble.independence import inside_crop_features
from src.engine_drivers.geometry import PATH_FEATURES, path_features
from src.engine_drivers.importance import (
    TARGETS,
    grouped_importance,
    partial_dependence_table,
    score_relation,
    univariate,
)
from src.engine_drivers.scoring import export_scale

logger = logging.getLogger(__name__)
ALL = "all"


def load_table(cfg: Mapping, run: Path) -> pd.DataFrame:
    """One row per (beamlet, engine): features, path features, errors, patient, anatomy, score."""
    feats = inside_crop_features(cfg["analysis"]["features_csv"])
    cache = run / "analysis" / "path_features.csv"
    if cache.exists():
        paths = pd.read_csv(cache, dtype={"sample_id": str}).set_index("sample_id")
    else:
        paths = path_features(Path(cfg["dataset_path"]), feats["bp_slice"].astype(int).to_dict(),
                              export_scale(Path(cfg["dataset_path"])))
        paths.to_csv(cache)
    index_csv = Path(cfg["dataset_path"]).with_name(Path(cfg["dataset_path"]).stem + "_index.csv")
    meta = pd.read_csv(index_csv, dtype={"sample_id": str}).set_index("sample_id")[["pool_patient_id", "pool_anatomy"]]
    meta.columns = ["patient_id", "anatomy"]
    score = pd.read_csv(cfg["analysis"]["score_contributions"], dtype={"sample_id": str}).set_index("sample_id")
    score = (score["total"] + score["baseline"]).rename("difficulty_score").to_frame().join(
        score["decile"].rename("score_decile"))
    base = feats[list(FEATURE_NAMES)].join(paths, how="inner").join(meta, how="inner").join(score, how="inner")
    frames = []
    for engine in cfg["engines"]:
        err = pd.read_csv(run / "tables" / f"errors_{engine}.csv", dtype={"sample_id": str}).set_index("sample_id")
        frames.append(base.join(err.drop(columns=["engine"]), how="inner").assign(engine=engine))
    return pd.concat(frames).rename_axis("sample_id").reset_index()


def _panels(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {ALL: frame}
    out.update({a: sub for a, sub in frame.groupby("anatomy", sort=True)})
    return out


def run_analysis(cfg: Mapping, run: Path) -> List[str]:
    """Write every table to ``<run>/analysis`` and return the summary lines."""
    acfg = cfg["analysis"]
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    table = load_table(cfg, run)
    features = list(FEATURE_NAMES) + list(PATH_FEATURES)
    one = table[table["engine"] == list(cfg["engines"])[0]].set_index("sample_id")
    groups = feature_groups(one[features], float(acfg.get("group_threshold", 0.7)))
    groups.table().to_csv(out / "groups.csv", index=False)
    n_boot, seed = int(acfg.get("n_boot", 500)), int(acfg.get("seed", 20261125))
    uni, imp, fits, pdp, corr, dec = [], [], [], [], [], []
    with threadpool_limits(limits=int(acfg.get("threads", 16))):
        for engine, eng_frame in table.groupby("engine", sort=False):
            for panel, sub in _panels(eng_frame).items():
                patients = sub["patient_id"].to_numpy()
                for t in TARGETS:
                    tag = {"engine": engine, "panel": panel}
                    uni.append(univariate(sub, features, t, patients, n_boot, seed).assign(**tag))
                    res = grouped_importance(sub, features, groups.names, t, patients,
                                             n_folds=int(acfg.get("n_folds", 5)),
                                             n_repeats=int(acfg.get("n_repeats", 5)), n_boot=n_boot, seed=seed)
                    imp.append(res.importance.assign(**tag))
                    fits.append({**tag, "target": t.name, **res.fit})
                    c, d = score_relation(sub, "difficulty_score", t, patients, n_boot, seed,
                                          deciles=sub["score_decile"])
                    corr.append({**tag, **c})
                    dec.append(d.assign(**tag))
                    if panel == ALL:
                        u = uni[-1].set_index("feature")["spearman"].abs()
                        leads = [max(res.importance.loc[i, "members"].split(), key=lambda f: u.get(f, 0.0))
                                 for i in range(min(3, len(res.importance)))]
                        pdp.append(partial_dependence_table(res, sub, features, leads, t).assign(**tag))
                    logger.info("analyse %s %s %s: OOF R2 %.2f, top %s", engine, panel, t.name, res.fit["oof_r2"],
                                res.importance["group"].iloc[0])
    results = {"univariate": pd.concat(uni), "importance": pd.concat(imp), "fit": pd.DataFrame(fits),
               "partial_dependence": pd.concat(pdp), "score_correlation": pd.DataFrame(corr),
               "score_deciles": pd.concat(dec)}
    for name, frame in results.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return summary_lines(results, table)


def summary_lines(results: Mapping[str, pd.DataFrame], table: pd.DataFrame) -> List[str]:
    """Numbers only: per engine and target, fit quality, the top three groups and the score's correlation."""
    imp, fit, corr = results["importance"], results["fit"].set_index(["engine", "panel", "target"]), \
        results["score_correlation"].set_index(["engine", "panel", "target"])
    n = table.groupby("engine")["sample_id"].nunique().to_dict()
    lines = ["# What drives each engine's error (generated; numbers only)", "",
             f"Beamlets with the analytic peak inside the crop: {n}. Folds grouped by patient; importance = increase "
             "of the held-out squared error when a feature group is permuted, share of the positive increases; "
             "95% intervals from resampling patients.", ""]
    for (engine, panel), sub in imp.groupby(["engine", "panel"], sort=False):
        lines += [f"## {engine}, {panel}", "", "| target | OOF R2 | OOF Spearman | top 3 groups (share) | "
                  "score Spearman [95% CI] |", "|---|---|---|---|---|"]
        for t in TARGETS:
            s = sub[sub["target"] == t.name].sort_values("rank")
            f = fit.loc[(engine, panel, t.name)]
            c = corr.loc[(engine, panel, t.name)]
            top = "; ".join(f"{r.group} {100 * r.share:.0f}%" for r in s.head(3).itertuples())
            lines.append(f"| {t.label} | {f['oof_r2']:.2f} | {f['oof_spearman']:.2f} | {top} | "
                         f"{c['spearman']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] |")
        lines.append("")
    return lines


__all__ = ["load_table", "run_analysis", "summary_lines"]
