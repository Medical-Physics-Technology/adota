"""EXP-0014 noise floor: does the model-independence verdict survive a cleaner reference?

    uv run python scripts/active_learning/difficulty_noise_floor.py \
        --config scripts/active_learning/config_difficulty_noise_floor.yaml --device-index 0

Reads the errors of every engine against the 1e6 and the 1e7 realisation of the
noise set, and the analytic features of its beamlets. No model is loaded; the GPU
only runs the gamma of the 1e6 dose against the 1e7 one. The method and the
decision rule are in :mod:`src.active_learning.ensemble.noise_floor`; the guide is
``scripts/docs/difficulty_noise_floor.md``.

Writes into a timestamped directory under ``output_dir``:

- ``checks.csv`` and ``summary.json``: the four checks and the outcome;
- ``per_beamlet.csv``: floor, maxima, target against either reference and scores;
- ``engines.csv``: per engine, the median RDE and pass rate against either
  reference and the floor's share of its error;
- ``lomo_members.csv`` and ``lomo_summary.csv``: the leave-one-member-out gap against
  1e7, against 1e6 and with a robust maximum, per variant, member set and anatomy;
- ``agreement_pairs.csv`` and ``agreement_summary.csv``: agreement within and across
  references;
- ``score_noise.csv``: every score against the floor, raw and with the target fixed;
- ``transfer.csv``: every score against the targets and engines on this set;
- ``selection_overlap.csv``: top-k overlap of the deployed score and scorer v2;
- ``exploratory_*``: not part of the rule. Added after the score-vs-noise check failed
  in the first run: that check repeated without the volume factor of the RDE, and
  the noise share per score decile;
- ``noise_floor_rde.{svg,pdf,png}``, and the resolved ``config.yaml``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Dict, Optional

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
    inside_crop_features,
    leave_one_member_out,
    log_errors,
    pearson_with_interval,
    read_ids,
    scan_resample_counts,
    transfer_table,
    verdict,
)
from src.active_learning.ensemble.noise_floor import (
    NoiseFloorThresholds,
    agreement_drop,
    cross_reference_agreement,
    floor_ratio,
    mc_pair_errors,
    noise_floor_checks,
    partial_pearson_with_interval,
    peak_statistics,
    renormalised_errors,
    score_decile_table,
    top_k_overlap,
)
from src.active_learning.retrospective.validation import MetricSettings
from src.active_learning.validation import MODEL_DZ_MM
from src.adota.config import DEFAULT_SCALE
from src.evaluation.cli import load_yaml_config, merge_config, resolve_device, setup_logging, setup_run_directory
from src.figures.ensemble_agreement import noise_floor_figure
from src.training.run_dir import save_resolved_config, write_manifest

logger = logging.getLogger(__name__)
METRICS = list(FEATURE_NAMES)
PRIMARIES_RATIO = 10.0
DEFAULTS = {"output_dir": None, "device_index": None, "n_bootstrap": 2000, "seed": 0, "robust_percentile": 99.9,
            "spike_depth_mm": 10.0, "thresholds": {}, "gamma_cutoff_percent": 10.0, "gamma_backend": "torch",
            "gamma_resolution_mm": [2.0, 2.0, 2.0]}


def _member_paths(directory: str, engines) -> Dict[str, str]:
    paths = {p.name[:-len("_errors.csv")]: str(p) for p in sorted(Path(directory).glob("*_errors.csv"))}
    missing = [e for e in engines if e not in paths]
    if missing:
        raise typer.BadParameter(f"no errors for {missing} in {directory}")
    return {e: paths[e] for e in engines}


def main(
    config: Annotated[Path, typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_difficulty_noise_floor.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Parent of the run directory.")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA index for the floor's gamma.")] = None,
    n_bootstrap: Annotated[Optional[int], typer.Option(help="Patient-bootstrap replicates.")] = None,
) -> None:
    cfg = merge_config({"output_dir": str(output_dir) if output_dir else None, "device_index": device_index,
                        "n_bootstrap": n_bootstrap}, load_yaml_config(config), DEFAULTS)
    out = setup_run_directory(Path(cfg["output_dir"]), prefix="difficulty_noise_floor_", subdirs=())
    setup_logging(out, log_filename="difficulty_noise_floor.log")
    save_resolved_config(cfg, out / "config.yaml")
    n_boot, seed = int(cfg["n_bootstrap"]), int(cfg["seed"])

    # The fit side is the closing analysis's, unchanged.
    base = load_yaml_config(Path(cfg["independence_config"]))
    ladder, external = list(base["ladder"]), list(base["pool_external"])
    gate = [m for m in base.get("gate_members", []) if m in ladder]
    metric = base.get("error_metric", "rde_pct")
    v3_feat = inside_crop_features(base["v3_features"])
    v3_err = log_errors(base["ladder"], metric)
    fit_ids = common_population(v3_feat, v3_err, read_ids(base["v3_fit_ids"]))
    std = Standardiser.fit(v3_err.loc[fit_ids])
    fit_features, fit_z = v3_feat.loc[fit_ids, METRICS], std(v3_err).loc[fit_ids, ladder]

    # The noise set: one population, whose engines have an error against both references.
    engines = ladder + external
    log_a = log_errors(_member_paths(cfg["members_1e6"], engines), metric)
    log_b = log_errors(_member_paths(cfg["members_1e7"], engines), metric)
    features = inside_crop_features(cfg["features"])
    index = pd.read_csv(cfg["index"], usecols=["sample_id", "pool_patient_id", "pool_anatomy"]).astype(str) \
        .set_index("sample_id")
    ids = common_population(features, log_a.join(log_b, rsuffix="_1e7"), None)
    log_a, log_b = log_a.loc[ids], log_b.loc[ids]
    rde_a, rde_b = np.expm1(log_a), np.expm1(log_b)
    scans = index.loc[ids, "pool_patient_id"].to_numpy()
    anatomy = index.loc[ids, "pool_anatomy"].to_numpy()
    logger.info("%d beamlets with the peak inside the crop, %d patients", len(ids), len(set(scans)))

    # The floor: the 1e6 dose as a prediction of the 1e7 one, through the engines' metric code.
    settings = MetricSettings(scale=dict(DEFAULT_SCALE), gamma_params=dict(cfg["gamma_params"]),
                              resolution_mm=tuple(cfg["gamma_resolution_mm"]),
                              gamma_cutoff_percent=float(cfg["gamma_cutoff_percent"]),
                              gamma_backend=cfg["gamma_backend"])
    device = resolve_device(cfg["device_index"])
    floor = mc_pair_errors(cfg["h5_1e7"], cfg["h5_1e6"], ids, settings, {"device": str(device)}).set_index("sample_id")
    peaks = peak_statistics(cfg["h5_1e7"], cfg["h5_1e6"], ids, settings.scale, float(cfg["robust_percentile"]))

    # 1. The gap against either reference, and with a robust maximum in the 1e6 reference.
    rde_robust = renormalised_errors(rde_a, peaks["max_other"], peaks["robust_other"])
    rde_robust_b = renormalised_errors(rde_b, peaks["max_reference"], peaks["robust_reference"])
    z_a, z_b = std(log_a[ladder]), std(log_b[ladder])
    robust_std = Standardiser.fit(np.log1p(rde_robust[ladder]))
    z_robust = robust_std(np.log1p(rde_robust[ladder]))
    tests = {"noise_1e7": z_b, "noise_1e6": z_a, "noise_1e6_robust_max": z_robust}

    def test_set(name, z, keep=None):
        keep = np.ones(len(ids), dtype=bool) if keep is None else keep
        kept = [i for i, k in zip(ids, keep) if k]
        return EvalSet(name, features.loc[kept, METRICS], z.loc[kept], scans[keep].astype(str))

    sets = [test_set(name, z) for name, z in tests.items()]
    by_anatomy = [test_set(f"noise_1e7_{a}", z_b, anatomy == a) for a in sorted(set(anatomy))]
    runs = [("ladder", ladder, FULL, sets + by_anatomy), ("ladder", ladder, SPARSE, sets)]
    if len(gate) >= 3:
        runs.append(("gate", gate, FULL, sets))
    member_rows, summary_rows = [], []
    for member_set, members, variant, tests_run in runs:
        rows, summ = leave_one_member_out(fit_features, fit_z[members], METRICS, tests_run, members, variant, n_boot,
                                          seed)
        member_rows.append(rows.assign(member_set=member_set))
        summary_rows.append(summ.assign(member_set=member_set))
    lomo = pd.concat(summary_rows, ignore_index=True)
    lomo["verdict"] = [verdict(g, hi) for g, hi in zip(lomo["median_gap"], lomo["ci_hi"])]
    pd.concat(member_rows, ignore_index=True).to_csv(out / "lomo_members.csv", index=False)
    lomo.to_csv(out / "lomo_summary.csv", index=False)
    logger.info("gap:\n%s", lomo.to_string(index=False, float_format="%.3f"))
    primary = lomo[(lomo["member_set"] == "ladder") & (lomo["variant"] == FULL)].set_index("test_set")

    # 2. The target against either reference.
    counts = scan_resample_counts(len(set(scans)), n_boot, seed)
    target_a, target_b = z_a.mean(axis=1).to_numpy(), z_b.mean(axis=1).to_numpy()
    target_robust = z_robust.mean(axis=1).to_numpy()
    paired = test_set("noise", z_b)
    target_r = pearson_with_interval(target_a, target_b, paired, counts)
    target_robust_b = robust_std(np.log1p(rde_robust_b[ladder])).mean(axis=1).to_numpy()
    target_r_robust = pearson_with_interval(target_robust, target_robust_b, paired, counts)

    # 3. Agreement within and across references.
    agreement = [{"members": name, **agreement_drop(rde_a[cols], rde_b[cols], scans, counts)}
                 for name, cols in (("ladder", ladder), ("all engines", engines))]
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(out / "agreement_summary.csv", index=False)
    cross_reference_agreement(rde_a[engines], rde_b[engines]).to_csv(out / "agreement_pairs.csv", index=False)

    # 4. The scores against the floor, with the 1e7 target fixed.
    v2 = {v: DifficultyScorer.load(cfg["scorer_v2"], variant=v) for v in (FULL, SPARSE)}
    v1 = {v: DifficultyScorer.load(variant=v) for v in (FULL, SPARSE)}
    x = features.loc[ids, METRICS]
    scores = {"v2 ridge": v2[FULL].score(x), "v2 sparse": v2[SPARSE].score(x),
              "v1 ridge (deployed)": v1[FULL].score(x), "v1 sparse": v1[SPARSE].score(x),
              "energy only": energy_baseline(fit_features["energy_mev"].to_numpy(), fit_z.mean(axis=1).to_numpy(),
                                             x["energy_mev"].to_numpy())}
    log_floor = np.log1p(floor["rde_pct"].loc[ids].to_numpy())
    noise_rows = []
    for name, s in scores.items():
        raw = pearson_with_interval(s, log_floor, paired, counts)
        partial = partial_pearson_with_interval(s, log_floor, target_b, scans, counts)
        noise_rows.append({"score": name, "pearson": raw[0], "pearson_ci_lo": raw[1], "pearson_ci_hi": raw[2],
                           "partial_given_1e7_target": partial[0], "partial_ci_lo": partial[1],
                           "partial_ci_hi": partial[2]})
    score_noise = pd.DataFrame(noise_rows)
    score_noise.to_csv(out / "score_noise.csv", index=False)
    z_external = Standardiser.fit(log_b[external])(log_b[external])
    targets = {"ensemble_1e6": target_a, "ensemble_1e7": target_b, "floor": log_floor,
               **{e: z_b[e].to_numpy() for e in ladder}, **{e: z_external[e].to_numpy() for e in external}}
    transfer_table(scores, paired, targets, n_boot, seed).to_csv(out / "transfer.csv", index=False)

    # Exploratory, after the check failed: the check without the volume factor of the RDE.
    log_noise = np.log(peaks["l1_per_unit_dose"].loc[ids].to_numpy())
    volume = np.log(peaks["volume_over_max"].loc[ids].to_numpy())
    explore_rows = []
    v2_score = scores["v2 ridge"]
    for name, y_val, z_val in (("score, noise per unit dose | 1e7 target", log_noise, target_b),
                               ("score, floor RDE | volume over maximum", log_floor, volume)):
        r, lo, hi = partial_pearson_with_interval(v2_score, y_val, z_val, scans, counts)
        explore_rows.append({"quantity": name, "partial": r, "ci_lo": lo, "ci_hi": hi})
    for name, x_val, y_val in (("floor RDE, volume over maximum", log_floor, volume),
                               ("score, volume over maximum", v2_score, volume),
                               ("1e7 target, volume over maximum", target_b, volume),
                               ("score, noise per unit dose", v2_score, log_noise)):
        r, lo, hi, _ = pearson_with_interval(x_val, y_val, paired, counts)
        explore_rows.append({"quantity": name, "pearson": r, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(explore_rows).to_csv(out / "exploratory_correlations.csv", index=False)
    deciles = score_decile_table(v2_score, floor["rde_pct"].loc[ids].to_numpy(), rde_b[ladder].mean(axis=1).to_numpy(),
                                 peaks["l1_per_unit_dose"].loc[ids].to_numpy())
    deciles.to_csv(out / "exploratory_deciles.csv", index=False)
    top_k_overlap(target_a, target_b).assign(pair="target 1e6, target 1e7").to_csv(
        out / "exploratory_target_overlap.csv", index=False)

    # Would switching change the selection? On this set and on the pool's test split.
    pool_features = inside_crop_features(base["pool_features"])[METRICS]
    overlap = pd.concat([
        top_k_overlap(v1[FULL].score(x), v2[FULL].score(x)).assign(population="noise set"),
        top_k_overlap(v1[FULL].score(pool_features), v2[FULL].score(pool_features)).assign(population="pool test")])
    overlap.to_csv(out / "selection_overlap.csv", index=False)

    # Per engine, per beamlet, and the hottest voxel.
    table = floor_ratio(floor["rde_pct"], rde_a[engines], PRIMARIES_RATIO)
    def mean_gpr(directory: str) -> list:
        paths = _member_paths(directory, engines)
        return [100 * pd.read_csv(paths[e], usecols=["sample_id", "gpr"]).set_index("sample_id")["gpr"].loc[ids].mean()
                for e in engines]

    label = settings.gamma_label
    table = table.assign(median_rde_1e6=rde_a[engines].median().to_numpy(),
                         median_rde_1e7=rde_b[engines].median().to_numpy(),
                         **{f"gpr_{label}_1e6_pct": mean_gpr(cfg["members_1e6"]),
                            f"gpr_{label}_1e7_pct": mean_gpr(cfg["members_1e7"])})
    table.to_csv(out / "engines.csv", index=False)
    beamlets = pd.DataFrame({"anatomy": anatomy, "patient_id": scans, "target_1e6": target_a, "target_1e7": target_b,
                             "target_1e6_robust_max": target_robust, "score": scores["v2 ridge"],
                             "score_v1": scores["v1 ridge (deployed)"], "floor_rde": floor["rde_pct"].loc[ids].values,
                             f"floor_gpr_{label}": floor["gpr"].loc[ids].values,
                             "floor_dr80_mm": floor["dr80_mm"].loc[ids].values}, index=pd.Index(ids, name="sample_id"))
    beamlets = beamlets.join(peaks)
    beamlets.to_csv(out / "per_beamlet.csv")
    displaced = (beamlets["max_depth_other"] - beamlets["max_depth_reference"]).abs() * MODEL_DZ_MM \
        > float(cfg["spike_depth_mm"])
    ratio = beamlets["max_other"] / beamlets["max_reference"]
    spike = {"max_ratio_1e6_over_1e7_median": float(ratio.median()), "max_ratio_p95": float(ratio.quantile(0.95)),
             "share_max_displaced": float(displaced.mean()),
             "share_1e6_max_below_minus500_hu": float((beamlets["hu_at_max_other"] < -500).mean()),
             "target_pearson_both_robust_max": target_r_robust[0]}

    # The decision.
    thresholds = NoiseFloorThresholds(**dict(cfg["thresholds"] or {}))
    gap, gap_hi = float(primary.loc["noise_1e7", "median_gap"]), float(primary.loc["noise_1e7", "ci_hi"])
    drop = float(agreement.set_index("members").loc["ladder", "drop"])
    partial = float(score_noise.set_index("score").loc["v2 ridge", "partial_given_1e7_target"])
    checks, outcome = noise_floor_checks(gap, gap_hi, target_r[0], drop, partial, thresholds)
    checks.to_csv(out / "checks.csv", index=False)
    noise_floor_figure(beamlets.reset_index(), table, figure_path=str(out / "noise_floor_rde"),
                       title=f"Monte Carlo noise floor: {len(ids)} beamlets, {len(set(scans))} patients, 1e6 vs 1e7",
                       caption=f"RDE as in the engines' metric code on the 2 mm model grid; target: mean over the "
                               f"{len(ladder)} ladder members of log(1 + RDE), standardised on the fit scans. "
                               f"Floor: the 1e6 dose scored as a prediction of the 1e7 dose. Score: scorer v2, ridge.")
    summary = {"outcome": outcome, "gap_verdict": verdict(gap, gap_hi),
               "gap": primary.loc["noise_1e7", ["median_gap", "ci_lo", "ci_hi"]].to_dict(),
               "gap_against_1e6": primary.loc["noise_1e6", ["median_gap", "ci_lo", "ci_hi"]].to_dict(),
               "target_pearson": {"point": target_r[0], "ci_lo": target_r[1], "ci_hi": target_r[2]},
               "agreement": agreement.set_index("members").loc["ladder"].to_dict(),
               "score_partial": partial, "floor_median_rde": float(floor["rde_pct"].loc[ids].median()),
               f"floor_mean_gpr_{label}": float(100 * floor["gpr"].loc[ids].mean()), "hottest_voxel": spike,
               "n_beamlets": len(ids), "n_patients": len(set(scans)), "outputs": str(out)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_manifest(out, config=cfg, dataset_path=Path(cfg["h5_1e7"]), extra={
        "kind": "difficulty_noise_floor", "inputs": {k: str(cfg[k]) for k in (
            "members_1e6", "members_1e7", "h5_1e6", "h5_1e7", "features", "index", "scorer_v2")}})
    logger.info("checks:\n%s", checks.to_string(index=False))
    logger.info("outcome: %s; outputs in %s", outcome, out)


if __name__ == "__main__":
    typer.run(main)
