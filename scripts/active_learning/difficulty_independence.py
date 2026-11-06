"""EXP-0014 closing analysis: is the difficulty score model-independent?

    uv run python scripts/active_learning/difficulty_independence.py \
        --config scripts/active_learning/config_difficulty_independence.yaml

Reads the per-record errors that ``ensemble_agreement.py`` wrote for the HPO
ladder (the target members) and the external engines, and the analytic
features of the same beamlets. The score is fitted on the v3 fit scans only;
the beamlet pool's test split is the primary test set, F the secondary. See
:mod:`src.active_learning.ensemble.independence` for the method.

Writes into a timestamped directory under ``output_dir``:

- ``lomo_members.csv`` and ``lomo_summary.csv``: the leave-one-member-out gap
  per member and its median, per test set, variant and member set, plus the
  pool's per-anatomy rows;
- ``cv_gap_pool.csv``: the gap with both scores fitted inside the pool, for
  every engine including the external ones;
- ``transfer.csv``: Pearson of scorer v2, the deployed v1, and energy alone
  with the ensemble target and every engine's error;
- ``spread.csv``: the members' disagreement, descriptively;
- ``pool_scores.csv``: scorer v2 and the ensemble target per pool beamlet;
- ``scorer_v2_fit.json`` (fit scans only, the evaluated one) and
  ``scorer_v2_unseen.json`` (fit scans plus F, for deployment);
- ``summary.json`` with the verdict, the resolved ``config.yaml`` and ``manifest.json``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import numpy as np
import pandas as pd
import typer

from src.acquisition.features import FEATURE_NAMES
from src.acquisition.scorer import FULL, SPARSE, DifficultyScorer
from src.active_learning.ensemble.independence import (
    EvalSet,
    Standardiser,
    common_population,
    energy_baseline,
    fit_scorer,
    inside_crop_features,
    leave_one_member_out,
    log_errors,
    read_ids,
    scorer_json,
    transfer_table,
    verdict,
    within_test_cv_gap,
)
from src.evaluation.cli import load_yaml_config, merge_config, setup_logging, setup_run_directory
from src.training.run_dir import save_resolved_config, write_manifest

logger = logging.getLogger(__name__)
METRICS = list(FEATURE_NAMES)
DEFAULTS = {"output_dir": None, "error_metric": "rde_pct", "n_bootstrap": 2000, "seed": 0, "cv_splits": 5,
            "gate_members": []}


def _test_set(name: str, features: pd.DataFrame, z: pd.DataFrame, scans: pd.Series, ids: List[str]) -> EvalSet:
    return EvalSet(name, features.loc[ids, METRICS], z.loc[ids], scans.loc[ids].astype(str).to_numpy())


def main(
    config: Annotated[Path, typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_difficulty_independence.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Parent of the run directory.")] = None,
    n_bootstrap: Annotated[Optional[int], typer.Option(help="Scan-bootstrap replicates.")] = None,
) -> None:
    cfg = merge_config({"output_dir": str(output_dir) if output_dir else None, "n_bootstrap": n_bootstrap},
                       load_yaml_config(config), DEFAULTS)
    out = setup_run_directory(Path(cfg["output_dir"]), prefix="difficulty_independence_", subdirs=())
    setup_logging(out, log_filename="difficulty_independence.log")
    save_resolved_config(cfg, out / "config.yaml")
    metric, n_boot, seed = cfg["error_metric"], int(cfg["n_bootstrap"]), int(cfg["seed"])
    ladder = list(cfg["ladder"])
    gate = [m for m in cfg["gate_members"] if m in ladder]

    # v3: fit scans and F.
    v3_feat = inside_crop_features(cfg["v3_features"])
    v3_scans = pd.read_csv(cfg["v3_index"], usecols=["sample_id", "patient_key"]).astype(str) \
        .set_index("sample_id")["patient_key"]
    v3_err = log_errors(cfg["ladder"], metric)
    fit_ids = common_population(v3_feat, v3_err, read_ids(cfg["v3_fit_ids"]))
    f_ids = common_population(v3_feat, v3_err, read_ids(cfg["v3_f_ids"]))
    std = Standardiser.fit(v3_err.loc[fit_ids])
    v3_z = std(v3_err)
    ext_err = log_errors(cfg["external_f"], metric)
    f_ids = [i for i in f_ids if i in set(ext_err.dropna().index)]
    v3_z = v3_z.join(Standardiser.fit(ext_err.loc[f_ids])(ext_err), how="left")
    logger.info("v3: %d fit records on %d scans, %d F records on %d scans", len(fit_ids),
                v3_scans.loc[fit_ids].nunique(), len(f_ids), v3_scans.loc[f_ids].nunique())

    # Pool test split.
    pool_dir = Path(cfg["pool_members_dir"])
    pool_paths = {p.name[:-len("_errors.csv")]: str(p) for p in sorted(pool_dir.glob("*_errors.csv"))}
    missing = [m for m in ladder + list(cfg["pool_external"]) if m not in pool_paths]
    if missing:
        raise typer.BadParameter(f"no pool errors for {missing} in {pool_dir}")
    pool_err = log_errors(pool_paths, metric)
    pool_feat = inside_crop_features(cfg["pool_features"])
    pool_index = pd.read_csv(cfg["pool_index"], usecols=["sample_id", "pool_patient_id", "pool_anatomy"]) \
        .astype(str).set_index("sample_id")
    pool_ids = common_population(pool_feat, pool_err, None)
    ext_pool = list(cfg["pool_external"])
    pool_z = std(pool_err[ladder]).join(Standardiser.fit(pool_err.loc[pool_ids, ext_pool])(pool_err[ext_pool]))
    pool_scans = pool_index["pool_patient_id"]
    logger.info("pool: %d of %d records with the peak inside the crop, %d patients", len(pool_ids),
                len(pool_err), pool_scans.loc[pool_ids].nunique())

    fit_features = v3_feat.loc[fit_ids, METRICS]
    fit_z = v3_z.loc[fit_ids, ladder]
    pool = _test_set("pool_test", pool_feat, pool_z, pool_scans, pool_ids)
    f_set = _test_set("F", v3_feat, v3_z, v3_scans, f_ids)
    by_anatomy = [_test_set(f"pool_{a}", pool_feat, pool_z, pool_scans,
                            [i for i in pool_ids if pool_index.loc[i, "pool_anatomy"] == a])
                  for a in sorted(pool_index.loc[pool_ids, "pool_anatomy"].unique())]

    # 1. Leave one member out.
    member_rows, summary_rows = [], []
    runs = [("ladder", ladder, FULL, [pool, f_set] + by_anatomy), ("ladder", ladder, SPARSE, [pool, f_set])]
    if len(gate) >= 3:
        runs.append(("gate", gate, FULL, [pool, f_set]))
    for member_set, members, variant, tests in runs:
        rows, summ = leave_one_member_out(fit_features, fit_z, METRICS, tests, members, variant, n_boot, seed)
        member_rows.append(rows.assign(member_set=member_set))
        summary_rows.append(summ.assign(member_set=member_set))
        for r in summ.itertuples():
            logger.info("LOMO %-6s %-14s %-16s median gap %+.3f [%+.3f, %+.3f]", member_set, variant,
                        r.test_set, r.median_gap, r.ci_lo, r.ci_hi)
    lomo = pd.concat(summary_rows, ignore_index=True)
    lomo["verdict"] = [verdict(g, hi) for g, hi in zip(lomo["median_gap"], lomo["ci_hi"])]
    pd.concat(member_rows, ignore_index=True).to_csv(out / "lomo_members.csv", index=False)
    lomo.to_csv(out / "lomo_summary.csv", index=False)

    # 2. Gap with both scores fitted inside the pool (external engines included).
    cv = within_test_cv_gap(pool, METRICS, ladder + ext_pool, ladder, int(cfg["cv_splits"]))
    cv.to_csv(out / "cv_gap_pool.csv", index=False)
    logger.info("pool CV gaps:\n%s", cv.to_string(index=False, float_format="%.3f"))

    # 3. Scorer v2 and what it transfers to.
    target_fit = fit_z.mean(axis=1).to_numpy()
    v2 = {v: fit_scorer(fit_features, METRICS, target_fit, v, "v2_fit") for v in (FULL, SPARSE)}
    unseen_ids = fit_ids + f_ids
    target_unseen = v3_z.loc[unseen_ids, ladder].mean(axis=1).to_numpy()
    v2_unseen = {v: fit_scorer(v3_feat.loc[unseen_ids, METRICS], METRICS, target_unseen, v, "v2_unseen")
                 for v in (FULL, SPARSE)}
    pd.DataFrame({"sample_id": pool_ids, "patient_id": pool.scans,
                  "anatomy": pool_index.loc[pool_ids, "pool_anatomy"].to_numpy(),
                  "energy_mev": pool.features["energy_mev"].to_numpy(), "score": v2[FULL].score(pool.features),
                  "ensemble_target": pool.z[ladder].mean(axis=1).to_numpy()}).to_csv(out / "pool_scores.csv",
                                                                                     index=False)
    v1 = {v: DifficultyScorer.load(variant=v) for v in (FULL, SPARSE)}
    fit_energy = fit_features["energy_mev"].to_numpy()
    transfer = []
    for test, engines in ((pool, ladder + ext_pool), (f_set, ladder + list(cfg["external_f"]))):
        scores = {"v2 ridge": v2[FULL].score(test.features), "v2 sparse": v2[SPARSE].score(test.features),
                  "v1 ridge (deployed)": v1[FULL].score(test.features),
                  "v1 sparse": v1[SPARSE].score(test.features),
                  "energy only": energy_baseline(fit_energy, target_fit, test.features["energy_mev"].to_numpy())}
        targets: Dict[str, np.ndarray] = {"ensemble": test.z[ladder].mean(axis=1).to_numpy()}
        targets.update({e: test.z[e].to_numpy() for e in engines})
        transfer.append(transfer_table(scores, test, targets, n_boot, seed))
    transfer = pd.concat(transfer, ignore_index=True)
    transfer.to_csv(out / "transfer.csv", index=False)
    logger.info("transfer to the ensemble target:\n%s",
                transfer[transfer["target"] == "ensemble"].to_string(index=False, float_format="%.3f"))

    # 4. Spread across the ladder, descriptively.
    spread_rows = []
    for test in (pool, f_set):
        sd = test.z[ladder].std(axis=1, ddof=0).to_numpy()
        ens = test.z[ladder].mean(axis=1).to_numpy()
        spread_rows.append({"test_set": test.name, "mean_spread": float(sd.mean()),
                            "var_shared_fraction": float(np.var(ens) / test.z[ladder].var(ddof=0).mean()),
                            "pearson_spread_ensemble": float(np.corrcoef(sd, ens)[0, 1]),
                            "pearson_spread_v2": float(np.corrcoef(sd, v2[FULL].score(test.features))[0, 1])})
    pd.DataFrame(spread_rows).to_csv(out / "spread.csv", index=False)

    common = {"target": f"mean over {ladder} of log(1 + {metric}), each standardised on the fit scans",
              "members": ladder, "standardiser": {"mean": std.mean.to_dict(), "sd": std.sd.to_dict()},
              "population": "peak_inside_crop (analytic)"}
    (out / "scorer_v2_fit.json").write_text(json.dumps(scorer_json(
        v2, "EXP-0014 scorer v2, fitted on the 19 v3 fit scans only (F and the pool held out).",
        {**common, "n_fit": len(fit_ids)}), indent=1))
    (out / "scorer_v2_unseen.json").write_text(json.dumps(scorer_json(
        v2_unseen, "EXP-0014 scorer v2 for deployment, fitted on all 26 v3 scans the HPO never trained on.",
        {**common, "n_fit": len(unseen_ids)}), indent=1))

    primary = lomo[(lomo["member_set"] == "ladder") & (lomo["variant"] == FULL)].set_index("test_set")
    summary = {"verdict": primary.loc["pool_test", "verdict"],
               "primary": primary.loc["pool_test", ["median_gap", "ci_lo", "ci_hi"]].to_dict(),
               "secondary_F": primary.loc["F", ["median_gap", "ci_lo", "ci_hi", "verdict"]].to_dict(),
               "n_fit": len(fit_ids), "n_f": len(f_ids), "n_pool": len(pool_ids), "outputs": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_manifest(out, config=cfg, dataset_path=Path(cfg["pool_index"]), extra={
        "kind": "difficulty_independence", "inputs": {k: str(cfg[k]) for k in (
            "pool_members_dir", "pool_features", "pool_index", "v3_features", "v3_index", "v3_fit_ids", "v3_f_ids")}})
    logger.info("verdict: %s (%s)", summary["verdict"], summary["primary"])


if __name__ == "__main__":
    typer.run(main)
