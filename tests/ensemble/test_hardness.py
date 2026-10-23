"""The per-engine hardness analysis (EXP-0014) on synthetic tables: within-cell
Spearman against scipy, the shared scan resampling, clusters of near-duplicate
metrics, the within-cell score fit recovering a planted metric, consensus and
residual targets, and the resampling of a saved dose onto the model grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch.nn.functional as F
from scipy.stats import spearmanr

from src.acquisition.features import FEATURE_NAMES
from src.acquisition.scorer import DifficultyScorer, percentile_grids, percentile_transform
from src.active_learning.ensemble.agreement import scan_bootstrap, scan_resamples
from src.active_learning.hardness import analysis as an
from src.active_learning.hardness.cells import cell_spearman, cluster_composites, cluster_metrics
from src.active_learning.hardness.engines import parse_engines, saved_layout, to_model_grid
from src.active_learning.hardness.fit import out_of_fold, permutation_importance
from src.active_learning.hardness.tables import error_column
from src.active_learning.retrospective.validation import MetricSettings, record_metrics
from src.adota.config import DEFAULT_GAMMA_PARAMS


def synthetic_table(n_scans: int = 5, per_cell: int = 40, seed: int = 0) -> pd.DataFrame:
    """Five scans x two energy cells; ``a`` drives every engine's error, ``b`` only
    engine U's, ``a2`` duplicates ``a`` and ``c`` is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scans):
        for e, energy in enumerate((95.0, 135.0)):
            a, b, c = rng.normal(size=(3, per_cell))
            shift = rng.normal() * 3.0      # cell-level offsets the analysis must ignore
            for i in range(per_cell):
                rows.append({"sample_id": f"s{s}_e{e}_{i}", "scan": f"scan{s}", "cell": f"scan{s}|{e}",
                             "site": "lung" if s % 2 else "colorectal", "energy_mev": energy + i * 0.1,
                             "peak_inside_crop": True, "a": a[i], "a2": a[i] + 0.05 * rng.normal(), "na": -a[i],
                             "b": b[i], "c": c[i],
                             error_column("gamma_fail", "C"): shift + a[i] + 0.3 * rng.normal(),
                             error_column("gamma_fail", "X"): shift + a[i] + 0.3 * rng.normal(),
                             error_column("gamma_fail", "U"): shift + a[i] + 1.5 * b[i] + 0.3 * rng.normal()})
    return pd.DataFrame(rows)


METRICS = ["a", "a2", "na", "b", "c"]


def test_cell_spearman_matches_scipy_per_cell():
    t = synthetic_table()
    corr = cell_spearman(t[["a", "c"]], t[error_column("gamma_fail", "C")], t["cell"], t["scan"], 10)
    for k, cell in enumerate(corr.cells["cell"]):
        part = t[t["cell"] == cell]
        expected = spearmanr(part["a"], part[error_column("gamma_fail", "C")])[0]
        assert corr.values[k, 0] == pytest.approx(expected)
    assert corr.median()["a"] > 0.8 > abs(corr.median()["c"])


def test_scan_bootstrap_draws_are_the_shared_resampling():
    rng = np.random.default_rng(1)
    corr = rng.uniform(size=(12, 3))
    scans = [f"s{i % 4}" for i in range(12)]
    ci, _ = scan_bootstrap(corr, scans, ["unknown"] * 3, n_boot=50, seed=9)
    draws = np.array([np.nanmedian(corr[rows], axis=0) for rows in scan_resamples(scans, 50, 9)])
    assert ci["median_cell_spearman_lo"].to_numpy() == pytest.approx(np.percentile(draws, 2.5, axis=0))


def test_clusters_join_near_duplicates_and_orient_them():
    t = synthetic_table()
    clusters = cluster_metrics(t[METRICS], t["cell"], threshold=0.3)
    assert clusters.assignment["a"] == clusters.assignment["a2"] == clusters.assignment["na"]
    assert len({clusters.assignment[m] for m in ("a", "b", "c")}) == 3
    assert clusters.signs["na"] == -clusters.signs["a"]
    composite = cluster_composites(clusters, t[METRICS], t["cell"])
    name = clusters.names[clusters.assignment["a"]]
    assert spearmanr(composite[name], t["a"])[0] > 0.95


def test_percentile_helpers_reproduce_the_scorer():
    t = synthetic_table()
    grids = percentile_grids(t, METRICS)
    scorer = DifficultyScorer("full (ridge)", 0.0, {"a": 1.0}, grids)
    assert percentile_transform(grids["a"], t["a"].to_numpy()) == pytest.approx(scorer.percentile("a", t["a"]))


def table_spec(t: pd.DataFrame) -> an.TableSpec:
    return an.TableSpec("T", t, ["C", "X", "U"], ["C", "X", "U"], ["U"])


def test_population_builds_consensus_and_residuals():
    t = synthetic_table()
    settings = an.Settings(metrics_used=METRICS)
    pop = an.population(table_spec(t), "gamma_fail", settings)
    ranks = pop.targets[["C", "X", "U"]].groupby(pop.frame["cell"].to_numpy()).rank(pct=True)
    assert pop.targets["consensus"].to_numpy() == pytest.approx(ranks.mean(axis=1).to_numpy())
    assert pop.targets["U residual"].to_numpy() == pytest.approx((ranks["U"] - ranks.mean(axis=1)).to_numpy())
    t.loc[0, "peak_inside_crop"] = False
    assert an.population(table_spec(t), "gamma_fail", settings).excluded["peak_outside_crop"] == 1


def test_within_cell_fit_finds_the_planted_metrics():
    t = synthetic_table()
    settings = an.Settings(metrics_used=METRICS, n_bootstrap=50, n_perm_repeats=2)
    pop = an.population(table_spec(t), "gamma_fail", settings)
    clusters = cluster_metrics(pop.frame[METRICS], pop.frame["cell"], 0.3)
    for target, metric in (("consensus", "a"), ("U residual", "b")):
        oof = out_of_fold(pop.frame, METRICS, pop.targets[target], pop.frame["cell"], pop.frame["scan"])
        corr = cell_spearman(pd.DataFrame({"s": oof.score}), pop.targets[target], pop.frame["cell"],
                             pop.frame["scan"], 10)
        assert corr.median()["s"] > 0.6
        imp = permutation_importance(oof, pop.targets[target], pop.frame["cell"], pop.frame["scan"], clusters,
                                     10, 2, 50, 0, 0.95)
        assert int(imp.sort_values("importance").iloc[-1]["cluster"]) == clusters.assignment[metric]


def test_record_metrics_of_a_one_percent_overdose():
    import torch

    z, yy, xx = np.meshgrid(np.arange(160), np.arange(30), np.arange(30), indexing="ij")
    y = (np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / 20.0) * (1 + z / 100.0) * (z < 120)).astype(np.float32)
    settings = MetricSettings(scale={}, gamma_params=dict(DEFAULT_GAMMA_PARAMS), gamma_backend="torch")
    out = record_metrics(y, 1.01 * y, settings, {"device": torch.device("cpu")})
    # Every voxel is within 2 % (an exact copy would give gamma 0 everywhere,
    # which the pass-rate definition counts as undefined).
    assert out["gpr"] == pytest.approx(1.0)
    assert out["mape_pct"] == pytest.approx(1.0 / 1.01, rel=1e-3)
    assert out["dr80_mm"] == pytest.approx(0.0, abs=1e-6)


def test_saved_dose_resampling_is_the_two_by_two_average():
    rng = np.random.default_rng(0)
    hwd = rng.uniform(size=(60, 60, 320)).astype(np.float32)
    assert saved_layout(hwd) == "hwd" and saved_layout(hwd.transpose(2, 0, 1)[None, None]) == "dhw"
    import torch

    pooled = F.avg_pool3d(torch.from_numpy(hwd).permute(2, 0, 1)[None, None], 2)[0, 0]
    assert to_model_grid(hwd, "hwd").numpy() == pytest.approx(pooled.numpy(), abs=1e-6)
    assert to_model_grid(hwd.transpose(2, 0, 1)[None, None], "dhw").numpy() == pytest.approx(pooled.numpy(),
                                                                                              abs=1e-6)


def test_engine_specs_resolve_trials_and_reject_unknown_keys(tmp_path):
    engines = parse_engines([{"label": "C0", "trial": "t1", "consensus": True},
                             {"label": "PBA", "saved_dose": "_ds_pba", "family": "pencil_beam"}],
                            tmp_path, "2pct_2mm_cutoff10pct")
    assert engines[0].checkpoint.endswith("trials/t1/checkpoints/best.pth")
    assert engines[0].errors_csv.endswith("finalist_2pct_2mm_cutoff10pct.csv")
    assert engines[1].tables == ("B",) and not engines[1].is_model
    with pytest.raises(ValueError):
        parse_engines([{"label": "x", "trial": "t", "colour": "red"}], tmp_path, "g")


def test_feature_names_are_the_thirty():
    assert len(FEATURE_NAMES) == 30
