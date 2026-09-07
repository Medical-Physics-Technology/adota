"""Refit the difficulty score on input-only features, under the study's protocol.

The protocol of ``acquisition_frozen_test.py`` unchanged: target ``log(1 + RDE)``,
percentile features with grids fitted on the training rows only, Lasso
(alpha 0.0008) for the sparse score, ridge (alpha 1) for the full one, a
gradient-boosted ceiling, five-fold patient-grouped cross-validation on the
development patients and one evaluation on the same frozen seven test patients.
Run on two feature arms side by side: ``gt`` (the current code located by the
Monte Carlo dose, the study's own features under today's physics) and
``analytic`` (located by the surrogate), so any gap is attributable to the
surrogate alone.

Two populations: every record, for comparability with the study's 0.852; and
the beamlets whose peak both arms place inside the crop, which is the
population the deployed score will actually see.

Leave-one-anatomy-out for both linear variants on the analytic arm is the
criterion for choosing the deployed variant (head-and-neck and brain are unseen
anatomies arriving later).

Usage:
    uv run --with scikit-learn python scripts/analysis/acquisition_input_only_refit.py --features-dir <dir>
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Dict, List

import numpy as np
import pandas as pd
import typer
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold

from src.acquisition.features import FEATURE_NAMES

RUN_DIR = Path("/scratch/mstryja/adota_runs/20260707_124010")
REFERENCE_RESULTS = RUN_DIR / "results.csv"
PROVENANCE = RUN_DIR / "figures/acquisition/uuid_provenance_map.csv"
FROZEN_TEST_IDS = RUN_DIR / "figures/acquisition/frozen_test_ids.csv"
SPARSE_ALPHA = 0.0008
VARIANTS = ("sparse (Lasso)", "full (ridge)", "gbm (non-linear)")
FEATS = list(FEATURE_NAMES)


def make_model(name: str):
    if name == "sparse (Lasso)":
        return Lasso(alpha=SPARSE_ALPHA, max_iter=5000)
    if name == "full (ridge)":
        return Ridge(alpha=1.0)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)


def grids(df: pd.DataFrame, rows: np.ndarray) -> Dict[str, np.ndarray]:
    return {c: np.quantile(df[c].values[rows], np.linspace(0, 1, 101)) for c in FEATS}


def transform(df: pd.DataFrame, rows: np.ndarray, G: Dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.clip(np.searchsorted(G[c], df[c].values[rows], side="right") / 101, 0, 1)
                            for c in FEATS])


def correlations(pred: np.ndarray, truth: np.ndarray) -> tuple:
    return float(pearsonr(pred, truth)[0]), float(spearmanr(pred, truth).correlation)


def fit_protocol(df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, dev: np.ndarray, test: np.ndarray,
                 arm: str, population: str) -> tuple:
    """Dev-CV plus frozen-test correlations per variant; returns rows and the fitted linear weights."""
    rows, weights = [], {}
    G_dev = grids(df, dev)
    X_dev, X_test = transform(df, dev, G_dev), transform(df, test, G_dev)
    for name in VARIANTS:
        pe, sp = [], []
        for tr, te in GroupKFold(5).split(dev, groups=groups[dev]):
            rtr, rte = dev[tr], dev[te]
            G = grids(df, rtr)
            model = make_model(name).fit(transform(df, rtr, G), y[rtr])
            p, s = correlations(model.predict(transform(df, rte, G)), y[rte])
            pe.append(p)
            sp.append(s)
        model = make_model(name).fit(X_dev, y[dev])
        tp, ts = correlations(model.predict(X_test), y[test])
        n_terms = int(np.sum(np.abs(model.coef_) > 1e-6)) if hasattr(model, "coef_") else len(FEATS)
        rows.append({"arm": arm, "population": population, "variant": name, "n_terms": n_terms,
                     "n_dev": len(dev), "n_test": len(test),
                     "dev_cv_pearson": float(np.mean(pe)), "dev_cv_spearman": float(np.mean(sp)),
                     "test_pearson": tp, "test_spearman": ts})
        if hasattr(model, "coef_"):
            weights[name] = {"intercept": float(model.intercept_),
                             "coefficients": {f: float(c) for f, c in zip(FEATS, model.coef_) if abs(c) > 1e-6}}
    return rows, weights, {c: G_dev[c].tolist() for c in FEATS}


def anatomy_transfer(df: pd.DataFrame, y: np.ndarray, anatomy: np.ndarray, arm: str) -> List[Dict]:
    """Train on one anatomy, evaluate on the other, both directions, both linear variants."""
    rows = []
    for held_out in sorted(pd.unique(anatomy)):
        train, test = np.where(anatomy != held_out)[0], np.where(anatomy == held_out)[0]
        G = grids(df, train)
        for name in VARIANTS[:2]:
            model = make_model(name).fit(transform(df, train, G), y[train])
            tp, ts = correlations(model.predict(transform(df, test, G)), y[test])
            rows.append({"arm": arm, "variant": name, "held_out_anatomy": held_out,
                         "n_train": len(train), "n_test": len(test), "pearson": tp, "spearman": ts})
    return rows


def load_arm(features_dir: Path, arm: str, targets: pd.DataFrame) -> pd.DataFrame:
    feats = pd.read_csv(features_dir / f"features_{arm}.csv")
    return feats.merge(targets, on="sample_id", how="inner").reset_index(drop=True)


def main(
    features_dir: Annotated[Path, typer.Option(help="Directory with features_gt.csv and features_analytic.csv.")],
    results: Annotated[Path, typer.Option()] = REFERENCE_RESULTS,
    provenance: Annotated[Path, typer.Option()] = PROVENANCE,
    frozen_ids: Annotated[Path, typer.Option()] = FROZEN_TEST_IDS,
    arms: Annotated[List[str], typer.Option(help="Feature arms to fit.")] = ["gt", "analytic"],
) -> None:
    """Write refit_results.csv, anatomy_transfer.csv and analytic_scorer.json into features_dir."""
    targets = (pd.read_csv(results, usecols=["sample_id", "rde"])
               .merge(pd.read_csv(provenance), on="sample_id", how="left"))
    test_ids = set(pd.read_csv(frozen_ids)["sample_id"])
    inside_by_arm = {arm: pd.read_csv(features_dir / f"features_{arm}.csv", usecols=["sample_id", "peak_inside_crop"])
                     for arm in arms}
    both_inside = set.intersection(*(set(t.loc[t.peak_inside_crop, "sample_id"]) for t in inside_by_arm.values()))

    results_rows, transfer_rows, scorer = [], [], {}
    for arm in arms:
        df_all = load_arm(features_dir, arm, targets)
        for population, df in (("all", df_all), ("both_inside_crop", df_all[df_all.sample_id.isin(both_inside)]
                                                   .reset_index(drop=True))):
            y = np.log1p(df["rde"].values)
            groups, anatomy = df["patient_key"].values, df["anatomy"].values
            test = np.where(df["sample_id"].isin(test_ids))[0]
            dev = np.where(~df["sample_id"].isin(test_ids))[0]
            rows, weights, grid = fit_protocol(df, y, groups, dev, test, arm, population)
            results_rows += rows
            transfer_rows += [dict(r, population=population) for r in anatomy_transfer(df, y, anatomy, arm)]
            scorer[f"{arm}/{population}"] = {"weights": weights, "percentile_grids_dev": grid,
                                             "n_dev": int(len(dev)), "n_test": int(len(test))}
            typer.echo(f"\n[{arm} / {population}] dev={len(dev)} test={len(test)}")
            for r in rows:
                typer.echo(f"  {r['variant']:18s} terms={r['n_terms']:2d}  dev-CV P/S {r['dev_cv_pearson']:.3f}/"
                           f"{r['dev_cv_spearman']:.3f}   TEST P/S {r['test_pearson']:.3f}/{r['test_spearman']:.3f}")

    res = pd.DataFrame(results_rows)
    res.to_csv(features_dir / "refit_results.csv", index=False)
    tr = pd.DataFrame(transfer_rows)
    tr.to_csv(features_dir / "anatomy_transfer.csv", index=False)
    typer.echo("\nleave-one-anatomy-out Pearson (mean over the two directions):")
    typer.echo(tr.groupby(["arm", "population", "variant"]).pearson.mean().round(3).to_string())
    (features_dir / "analytic_scorer.json").write_text(json.dumps(
        {"features": FEATS, "frozen_test_ids": str(frozen_ids), "arms": scorer}, indent=1))
    typer.secho(f"\nwrote refit_results.csv, anatomy_transfer.csv, analytic_scorer.json in {features_dir}",
                fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.run(main)
