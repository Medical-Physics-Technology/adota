"""What makes a beamlet hard, per dose engine (EXP-0014, the C1 model-independence test).

    uv run python scripts/active_learning/beamlet_hardness.py \
        --config scripts/active_learning/config_beamlet_hardness.yaml
    uv run python scripts/active_learning/beamlet_hardness.py --config <yaml> --smoke   # minutes, subsets
    uv run python scripts/active_learning/beamlet_hardness.py --config <yaml> \
        --tables-dir <previous run>/tables --stages analyse,examples,report

Stages (comma separated, all by default):

- ``tables``: Table A (the HPO validation records: the thirty input-only
  metrics of the difficulty score, every engine's errors from its finalist CSV)
  and Table B (the engine-comparison beamlets: the same metrics from the saved
  CT and flux, every engine scored on the 2 mm grid against the saved Monte
  Carlo dose; the neural engines run on GPU, the pencil beam and the published
  ADoTA are read from their saved doses). Needs a GPU for Table B.
- ``analyse``: the agreement sanity check (C0 against X0), the metric clusters,
  and per table and error metric the univariate correlations, the out-of-fold
  score per engine, consensus and residual, coefficients, cluster importance,
  cross transfer, the strata and the population sensitivity; heatmaps and the
  slide figure.
- ``examples``: for the two most important clusters of the consensus fit, three
  high and three low beamlets of one cell through the consensus engines.
- ``report``: ``what_makes_a_beamlet_hard.md`` with every result table.

The engines are config entries (:mod:`src.active_learning.hardness.engines`),
so the same command reruns on retrained members. Everything is seeded; the
manifest records the config, the commit and every input's hash and commit.
Library code: :mod:`src.active_learning.hardness`; figures:
:mod:`src.figures.beamlet_hardness` and :func:`src.figures.hpo.finalist_beamlet_figure`.
"""
from __future__ import annotations

import os

# One BLAS thread per process: the metrics are fanned out over a process pool
# and a multithreaded numpy in every worker oversubscribes the machine. Must
# precede the first numpy import, hence the E402 exemption in pyproject.toml.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import logging
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import pandas as pd
import typer

from src.active_learning.ensemble.members import (
    EvaluationSettings,
    held_out_table,
    resolve_member,
    training_config_from_yaml,
)
from src.active_learning.hardness import analysis as an
from src.active_learning.hardness.cells import apply_clusters, cluster_metrics, co_assignment_agreement
from src.active_learning.hardness.engines import parse_engines, predict_h5_records, score_blocks_sharded
from src.active_learning.hardness.fit import fit_score, scorer_json
from src.active_learning.hardness.report import clusters_from_csv, figure_tables, write_report
from src.active_learning.hardness.tables import (
    block_features,
    error_columns,
    feature_names,
    file_error_columns,
    geometry_leakage,
    h5_features,
    input_provenance,
    table_a_metadata,
    table_b_entries,
)
from src.active_learning.retrospective.validation import MetricSettings
from src.adota.config import DEFAULT_GAMMA_PARAMS
from src.evaluation.cli import (
    load_yaml_config,
    merge_config,
    on_root_filesystem,
    resolve_device,
    setup_logging,
    setup_run_directory,
)
from src.figures.beamlet_hardness import TARGET_COLORS, annotated_heatmap, cluster_importance_figure
from src.figures.hpo import finalist_beamlet_figure, model_styles
from src.training.run_dir import save_resolved_config, write_manifest

app = typer.Typer(help="What makes a beamlet hard, per dose engine.", add_completion=False)
logger = logging.getLogger("beamlet_hardness")

STAGES = ("tables", "analyse", "examples", "report")
DEFAULTS: Dict[str, Any] = {
    "experiment": "EXP-0014", "stage": "beamlet_hardness", "hpo_dir": None, "output_dir": None,
    "dataset_path": None, "index_csv": None, "records_csv": None, "record_ids": None, "training_ids": None,
    "comparison_csv": None, "comparison_file_prefixes": {}, "tables_dir": None, "stages": list(STAGES),
    "engines": [], "residuals": [], "device_index": None, "device_indices": None, "feature_workers": 8,
    "block_batch_size": 16,
    "block_num_workers": 4, "gamma_params": dict(DEFAULT_GAMMA_PARAMS), "gamma_resolution_mm": [2.0, 2.0, 2.0],
    "gamma_cutoff_percent": 10.0, "gamma_backend": "torch",
    "metrics": ["gamma_fail", "rde_pct", "mape_pct", "abs_dr80_mm"], "primary_metric": "gamma_fail",
    "min_cell_size": 10, "n_bootstrap": 2000, "n_bootstrap_coef": 500, "n_perm_repeats": 5, "seed": 20261023,
    "ci": 0.95, "population": "inside_crop", "sensitivity_populations": ["all"], "max_energy_mev": 250.0,
    "energy_terciles": [[70, 110], [110, 152], [152, 270]], "cluster_threshold": 0.5, "cluster_method": "average",
    "agreement_check": {"pair": ["C0", "X0"], "metric": "gamma_fail", "expected": 0.82, "tolerance": 0.01},
    "slide": {"top": 8, "bars": []}, "examples": {"table": "A", "n": 3, "n_clusters": 2, "engines": [], "cell": None},
    "smoke": False, "smoke_n_a": 400, "smoke_n_b": 900,
}


def _ids(path: Path) -> List[str]:
    frame = pd.read_csv(path, dtype=str)
    return frame[("sample_id" if "sample_id" in frame.columns else frame.columns[0])].tolist()


def build_table_a(cfg: Dict[str, Any], engines, out: Path, device) -> pd.DataFrame:
    ids = _ids(Path(cfg["record_ids"]))
    if cfg["smoke"]:
        ids = sorted(ids)[: int(cfg["smoke_n_a"])]
    meta = table_a_metadata(Path(cfg["index_csv"]), cfg["records_csv"] and Path(cfg["records_csv"]), ids)
    settings = EvaluationSettings(dataset_path=str(cfg["dataset_path"]), gamma_params=dict(cfg["gamma_params"]),
                                  gamma_resolution_mm=tuple(cfg["gamma_resolution_mm"]),
                                  gamma_cutoff_percent=float(cfg["gamma_cutoff_percent"]),
                                  gamma_backend=str(cfg["gamma_backend"]))
    members = []
    for e in [e for e in engines if "A" in e.tables]:
        spec = {**e.member_spec(), **({"training_ids": cfg["training_ids"]} if cfg["training_ids"] else {})}
        members.append(resolve_member(spec, ids, settings, source="auto", device=device, cache_dir=out))
    held = held_out_table(members, ids, _scan_of_record(cfg))
    held.to_csv(out / "table_a_held_out.csv", index=False)
    if (held["n_shared_groups"].fillna(0) > 0).any():
        raise RuntimeError(f"an engine trained on a validation scan:\n{held.to_string()}")
    features = h5_features(Path(cfg["dataset_path"]), ids, workers=int(cfg["feature_workers"]))
    errors = pd.concat([error_columns(m.errors, m.label, cfg["metrics"]) for m in members], axis=1)
    table = _merge_features(meta, features).merge(errors, left_on="sample_id", right_index=True)
    pd.DataFrame([{"label": m.label, "source": m.source, **m.provenance} for m in members]) \
        .to_csv(out / "table_a_engines.csv", index=False)
    return table


def _merge_features(meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """``meta`` with the metrics; the metric ``energy_mev`` is the metadata's, checked equal."""
    both = meta[["sample_id", "energy_mev"]].merge(features[["sample_id", "energy_mev"]], on="sample_id")
    gap = float((both["energy_mev_x"] - both["energy_mev_y"]).abs().max())
    if gap > 1e-3:
        raise ValueError(f"record energies disagree with the metric energies by up to {gap:.4f} MeV")
    return meta.merge(features.drop(columns=["energy_mev"]), on="sample_id", how="left")


def _scan_of_record(cfg: Dict[str, Any]) -> pd.Series:
    """``sample_id -> patient_key`` over the whole HDF5, for the held-out check."""
    index = pd.read_csv(cfg["index_csv"], usecols=["sample_id", "patient_key"], dtype=str)
    return index.drop_duplicates("sample_id").set_index("sample_id")["patient_key"]


def build_table_b(cfg: Dict[str, Any], engines, out: Path, device) -> pd.DataFrame:
    entries = table_b_entries(Path(cfg["comparison_csv"]))
    if cfg["smoke"]:
        entries = entries.sample(int(cfg["smoke_n_b"]), random_state=int(cfg["seed"])).reset_index(drop=True)
    geometry_leakage(entries, Path(cfg["index_csv"])).to_csv(out / "table_b_geometry_leakage.csv", index=False)
    on_b = [e for e in engines if "B" in e.tables]
    scales = [dict(training_config_from_yaml(Path(e.model_config), str(cfg["dataset_path"])).scale)
              for e in on_b if e.is_model]
    if not scales or any(s != scales[0] for s in scales):
        raise ValueError("table B needs neural engines that share one MinMax scale")
    settings = MetricSettings(scale=scales[0], gamma_params=dict(cfg["gamma_params"]),
                              resolution_mm=tuple(cfg["gamma_resolution_mm"]),
                              gamma_cutoff_percent=float(cfg["gamma_cutoff_percent"]),
                              gamma_backend=str(cfg["gamma_backend"]))
    pairs = list(zip(entries["block_dir"], entries["stem"]))
    sample_of = {(d, s): i for d, s, i in zip(entries["block_dir"], entries["stem"], entries["sample_id"])}
    blocks = []
    devices = cfg["device_indices"] or [device.index or 0]
    scored = score_blocks_sharded(on_b, str(cfg["dataset_path"]), pairs, settings, devices,
                                  int(cfg["block_batch_size"]), int(cfg["block_num_workers"]))
    for label, frame in scored.items():
        frame.insert(0, "sample_id", [sample_of[(d, s)] for d, s in zip(frame["directory"], frame["stem"])])
        frame.to_csv(out / f"table_b_errors_{label}.csv", index=False)
        blocks.append(error_columns(frame, label, cfg["metrics"]))
    features = block_features(entries, workers=int(cfg["feature_workers"]))
    table = _merge_features(entries, features).merge(pd.concat(blocks, axis=1), left_on="sample_id",
                                                     right_index=True)
    return pd.concat([table, file_error_columns(Path(cfg["comparison_csv"]), entries,
                                                cfg["comparison_file_prefixes"])], axis=1)


def analyse(cfg: Dict[str, Any], engines, tables: Dict[str, pd.DataFrame], out: Path) -> Dict[str, Any]:
    metrics = feature_names()
    settings = an.Settings(metrics_used=metrics, min_cell_size=int(cfg["min_cell_size"]),
                           n_bootstrap=int(cfg["n_bootstrap"]), n_bootstrap_coef=int(cfg["n_bootstrap_coef"]),
                           n_perm_repeats=int(cfg["n_perm_repeats"]), seed=int(cfg["seed"]), ci=float(cfg["ci"]),
                           population=str(cfg["population"]), max_energy_mev=cfg["max_energy_mev"],
                           energy_terciles=tuple(tuple(map(float, t)) for t in cfg["energy_terciles"]))
    specs = {}
    for name, frame in tables.items():
        on = [e for e in engines if name in e.tables]
        specs[name] = an.TableSpec(name, frame, [e.label for e in on], [e.label for e in on if e.consensus],
                                   [r for r in cfg["residuals"] if r in {e.label for e in on}])
    res, primary = out / "results", str(cfg["primary_metric"])
    summary: Dict[str, Any] = {"tables": {}}

    check = cfg["agreement_check"]
    agree = an.agreement_check(specs["A"], check["metric"], tuple(check["pair"]), settings)
    gap = abs(agree["median_cell_spearman"] - float(check["expected"]))
    agree["matches_expected"] = bool(gap <= float(check["tolerance"]))
    summary["agreement_check"] = {**agree, "expected": check["expected"]}
    logger.info("agreement check %s vs %s (%s): %.3f [%.3f, %.3f], expected %.2f", *check["pair"], check["metric"],
                agree["median_cell_spearman"], agree["ci_lo"], agree["ci_hi"], float(check["expected"]))

    pop_a = an.population(specs["A"], primary, settings)
    clusters = cluster_metrics(pop_a.frame[metrics], pop_a.frame["cell"], float(cfg["cluster_threshold"]),
                               str(cfg["cluster_method"]))
    clusters.table().to_csv(res / "clusters.csv", index=False)
    clusters.correlation.to_csv(res / "metric_correlation_table_A.csv")
    summary["clusters"] = {"n": len(clusters.ids), "threshold": clusters.threshold}
    if "B" in specs:
        pop_b = an.population(specs["B"], primary, settings)
        own_b = cluster_metrics(pop_b.frame[metrics], pop_b.frame["cell"], float(cfg["cluster_threshold"]),
                                str(cfg["cluster_method"]))
        own_b.table().to_csv(res / "clusters_table_B_own.csv", index=False)
        apply_clusters(clusters, pop_b.frame[metrics], pop_b.frame["cell"]).correlation \
            .to_csv(res / "metric_correlation_table_B.csv")
        summary["clusters"]["adjusted_rand_A_vs_B"] = co_assignment_agreement(clusters, own_b)

    for name, spec in specs.items():
        summary["tables"][name] = {}
        for metric in cfg["metrics"]:
            tag = f"{name}_{metric}"
            pop = an.population(spec, metric, settings)
            summary["tables"][name][metric] = an.describe_population(pop)
            logger.info("table %s, %s: %s", name, metric, summary["tables"][name][metric])
            an.univariate(pop, clusters, settings).to_csv(res / f"univariate_{tag}.csv", index=False)
            fits = an.fit_targets(pop, clusters, settings)
            fits.evaluation.to_csv(res / f"fit_evaluation_{tag}.csv", index=False)
            fits.coefficients.to_csv(res / f"coefficients_{tag}.csv", index=False)
            fits.importance.to_csv(res / f"importance_{tag}.csv", index=False)
            fits.oof_scores.assign(sample_id=pop.frame["sample_id"], cell=pop.frame["cell"]) \
                .to_csv(res / f"oof_scores_{tag}.csv", index=False)
            matrix, transfer = an.cross_transfer(pop, fits, settings)
            matrix.to_csv(res / f"transfer_{tag}.csv", index=False)
            transfer.to_csv(res / f"transfer_summary_{tag}.csv", index=False)
            full = {t: fit_score(pop.frame, metrics, pop.targets[t], pop.frame["cell"],
                                 np.ones(len(pop.frame), dtype=bool)) for t in pop.targets.columns}
            (res / f"scorers_{tag}.json").write_text(json.dumps(scorer_json(
                full, "within-cell scores: rank-equivalent inside a scan x 20 MeV cell; intercept arbitrary",
                {"table": name, "metric": metric}), indent=1))
            if metric == primary:
                derived = [t for t in pop.targets.columns if t == an.CONSENSUS or t.endswith(" residual")]
                for key, frame in an.stratified(spec, metric, clusters, settings, derived).items():
                    frame.to_csv(res / f"strata_{key}_{tag}.csv", index=False)
                for alt in cfg["sensitivity_populations"]:
                    alt_settings = an.Settings(**{**settings.__dict__, "population": alt})
                    alt_pop = an.population(spec, metric, alt_settings)
                    an.fit_targets(alt_pop, clusters, alt_settings, derived, with_coefficients=False) \
                        .evaluation.assign(population=alt, **an.describe_population(alt_pop)) \
                        .to_csv(res / f"sensitivity_{alt}_{tag}.csv", index=False)
    figures(cfg, engines, specs, res, out / "figures", clusters)
    return summary


def figures(cfg, engines, specs, res: Path, fig_dir: Path, clusters) -> None:
    for name, metric, values, lo, hi, groups in figure_tables(specs, cfg["metrics"], res, clusters):
        annotated_heatmap(values, figure_path=str(fig_dir / f"univariate_{name}_{metric}"), lo=lo, hi=hi,
                          groups=groups, title=f"Table {name}: within-cell Spearman of each metric with {metric}",
                          value_label="median within-cell Spearman",
                          caption=(f"Cells: scan x 20 MeV, n >= {cfg['min_cell_size']}. Bold: 95% scan-bootstrap "
                                   "interval excludes zero. Rows grouped by metric cluster."))
        values.to_csv(fig_dir / f"univariate_{name}_{metric}.csv")
    for name in specs:
        for metric in cfg["metrics"]:
            m = pd.read_csv(res / f"transfer_{name}_{metric}.csv")
            grid = m.pivot(index="score", columns="target", values="median_cell_spearman")
            grid = grid.loc[[r for r in [*specs[name].engines, an.CONSENSUS] if r in grid.index], specs[name].engines]
            grid.to_csv(fig_dir / f"transfer_{name}_{metric}.csv")
            annotated_heatmap(grid, figure_path=str(fig_dir / f"transfer_{name}_{metric}"), vmax=1.0,
                              title=f"Table {name}, {metric}: score fitted on the row engine, error of the column",
                              value_label="median within-cell Spearman",
                              caption="Out-of-fold (leave one scan out) scores; diagonal: own-engine fit.")
    slide = cfg["slide"]
    if slide["bars"]:
        primary = cfg["primary_metric"]
        imp = {f"{b['table']}:{b['target']}": pd.read_csv(res / f"importance_{b['table']}_{primary}.csv")
               .query("target == @b['target']").set_index("cluster") for b in slide["bars"]}
        first = next(iter(imp.values()))
        order = pd.concat([v["importance"] for v in imp.values()], axis=1).max(axis=1) \
            .sort_values(ascending=False).index[: int(slide["top"])]
        frame = pd.DataFrame({"cluster": order, "cluster_name": first.loc[order, "cluster_name"].to_numpy()})
        family = {e.label: e.family for e in engines}
        colors, series = {}, []
        for key, b in zip(imp, slide["bars"]):
            label = b.get("label") or f"{b['target']} (table {b['table']})"
            series.append(label)
            colors[label] = TARGET_COLORS["consensus" if b["target"] == an.CONSENSUS
                                          else family.get(b["target"].replace(" residual", ""), "consensus")]
            for col, src in ((label, "importance"), (f"{label}:lo", "ci_lo"), (f"{label}:hi", "ci_hi")):
                frame[col] = imp[key].loc[order, src].to_numpy()
        frame.to_csv(fig_dir / f"slide_cluster_importance_{primary}.csv", index=False)
        cluster_importance_figure(frame, series, colors,
                                  figure_path=str(fig_dir / f"slide_cluster_importance_{primary}"),
                                  title=f"What makes a beamlet hard: top {len(order)} metric clusters ({primary})",
                                  caption=("Drop in the out-of-fold (leave one scan out) median within-cell Spearman "
                                           "when the cluster's metrics are permuted inside cells; bars: 95% scan-"
                                           "bootstrap intervals. Consensus: mean within-cell error rank of C0, X0, U0; "
                                           "residual: the engine's rank minus the consensus."))


def examples(cfg, engines, tables, out: Path, device) -> None:
    ex, primary = cfg["examples"], str(cfg["primary_metric"])
    res = out / "results"
    imp = pd.read_csv(res / f"importance_{ex['table']}_{primary}.csv").query("target == @an.CONSENSUS")
    top = imp.sort_values("importance", ascending=False)["cluster"].head(int(ex["n_clusters"])).tolist()
    clusters = clusters_from_csv(res / "clusters.csv")
    settings = an.Settings(metrics_used=feature_names(), min_cell_size=int(cfg["min_cell_size"]),
                           population=str(cfg["population"]), max_energy_mev=cfg["max_energy_mev"])
    on = [e for e in engines if ex["table"] in e.tables]
    spec = an.TableSpec(ex["table"], tables[ex["table"]], [e.label for e in on], [e.label for e in on if e.consensus],
                        [r for r in cfg["residuals"] if r in {e.label for e in on}])
    picks = an.composite_examples(an.population(spec, primary, settings), clusters, top, settings, int(ex["n"]),
                                  ex.get("cell"))
    picks.to_csv(out / "examples" / "examples.csv", index=False)
    chosen = [e for e in engines if e.label in (ex["engines"] or spec.consensus)]
    data = predict_h5_records(chosen, str(cfg["dataset_path"]), picks["sample_id"].unique().tolist(), device)
    table = tables[ex["table"]].set_index("sample_id")
    for row in picks.itertuples():
        sid, d = row.sample_id, data[row.sample_id]
        notes = {e.label: f"1 - pass rate {table.loc[sid, f'err_gamma_fail__{e.label}'] * 100:.1f}%\n"
                          f"MAPE {table.loc[sid, f'err_mape_pct__{e.label}']:.1f}%" for e in chosen}
        notes["Ground truth"] = f"MCsquare\n{table.loc[sid, 'energy_mev']:.1f} MeV"
        stem = out / "examples" / f"cluster{row.cluster}_{row.group}{row.rank}_{sid[:8]}"
        finalist_beamlet_figure(d["ct"], d["gt"], d["pred"], figure_path=str(stem), notes=notes,
                                styles=model_styles({e.label: e.family for e in chosen}),
                                title=f"{row.cluster_name}: {row.group} ({row.rank} of {ex['n']}) in cell {row.cell}",
                                subtitle=f"Beamlet {sid}; cluster composite (within-cell rank) {row.composite:.2f}.",
                                caption="Gamma 2%/2 mm, 10% cutoff, global. Dose in percent of the ground-truth "
                                        "maximum; white contour: ground-truth 50% isodose.")
        logger.info("example %s %s%d: %s", row.cluster_name, row.group, row.rank, sid)


@app.command()
def main(
    config: Annotated[Optional[Path], typer.Option(help="YAML config.")]
    = Path("scripts/active_learning/config_beamlet_hardness.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option(help="Parent of the run directory.")] = None,
    stages: Annotated[Optional[str], typer.Option(help="Comma-separated subset of " + ",".join(STAGES))] = None,
    tables_dir: Annotated[Optional[Path], typer.Option(help="Reuse table_A.csv / table_B.csv from here.")] = None,
    device_index: Annotated[Optional[int], typer.Option(help="CUDA index for Table B and the examples.")] = None,
    smoke: Annotated[bool, typer.Option(help="Small subsets and few bootstrap draws.")] = False,
) -> None:
    raw = load_yaml_config(config) if config and Path(config).exists() else {}
    cfg = merge_config({"output_dir": str(output_dir) if output_dir else None,
                        "stages": stages.split(",") if stages else None,
                        "tables_dir": str(tables_dir) if tables_dir else None,
                        "device_index": device_index, "smoke": smoke or None}, raw, DEFAULTS)
    if cfg["smoke"]:
        cfg.update(n_bootstrap=200, n_bootstrap_coef=50, n_perm_repeats=2)
    bad = sorted(set(cfg["stages"]) - set(STAGES))
    if bad:
        raise typer.BadParameter(f"unknown stages {bad}; choose from {STAGES}")
    if not cfg["output_dir"]:
        cfg["output_dir"] = str(Path(cfg["hpo_dir"]) / "analysis")
    cfg["index_csv"] = cfg["index_csv"] or str(Path(cfg["dataset_path"]).with_name(
        Path(cfg["dataset_path"]).stem + "_index.csv"))
    for what, path in (("output_dir", cfg["output_dir"]), ("the temporary directory (TMPDIR)", tempfile.gettempdir())):
        if on_root_filesystem(Path(path)):
            raise typer.BadParameter(f"{what} {path} is on the root filesystem; use /scratch")
    out = setup_run_directory(Path(cfg["output_dir"]), prefix="beamlet_hardness_" + ("smoke_" if cfg["smoke"] else ""),
                              subdirs=("tables", "results", "figures", "examples"))
    setup_logging(out, log_filename="beamlet_hardness.log")
    save_resolved_config(cfg, out / "config.yaml")
    gamma = EvaluationSettings(dataset_path="", gamma_params=dict(cfg["gamma_params"]),
                               gamma_cutoff_percent=float(cfg["gamma_cutoff_percent"])).gamma_label
    engines = parse_engines(cfg["engines"], Path(cfg["hpo_dir"]) if cfg["hpo_dir"] else None, gamma)
    needs_gpu = "tables" in cfg["stages"] or "examples" in cfg["stages"]
    device = resolve_device(cfg["device_index"]) if needs_gpu else None

    tables: Dict[str, pd.DataFrame] = {}
    if "tables" in cfg["stages"]:
        tables["A"] = build_table_a(cfg, engines, out / "tables", device)
        tables["A"].to_csv(out / "tables" / "table_A.csv", index=False)
        logger.info("table A: %d rows", len(tables["A"]))
        if cfg["comparison_csv"]:
            tables["B"] = build_table_b(cfg, engines, out / "tables", device)
            tables["B"].to_csv(out / "tables" / "table_B.csv", index=False)
            logger.info("table B: %d rows", len(tables["B"]))
    else:
        source = Path(cfg["tables_dir"] or "")
        for name in ("A", "B"):
            if (source / f"table_{name}.csv").exists():
                tables[name] = pd.read_csv(source / f"table_{name}.csv", dtype={"sample_id": str, "scan": str})
        if "A" not in tables:
            raise typer.BadParameter("no table_A.csv: run the tables stage or pass --tables-dir")
    summary: Dict[str, Any] = {}
    if "analyse" in cfg["stages"]:
        summary = analyse(cfg, engines, tables, out)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if "examples" in cfg["stages"]:
        examples(cfg, engines, tables, out, device)
    if "report" in cfg["stages"]:
        write_report(out, cfg, engines)
    inputs = [cfg["hpo_dir"], cfg["dataset_path"], cfg["index_csv"], cfg["records_csv"], cfg["record_ids"],
              cfg["training_ids"], cfg["comparison_csv"], cfg["tables_dir"],
              *[p for e in engines for p in (e.checkpoint, e.errors_csv) if p]]
    if "B" in tables:
        inputs += sorted(tables["B"]["block_dir"].unique())
    write_manifest(out, config=cfg, dataset_path=Path(cfg["dataset_path"]),
                   extra={"experiment": cfg["experiment"], "stage": cfg["stage"], "kind": "beamlet_hardness",
                          "inputs": [input_provenance(Path(p)) for p in inputs if p],
                          "engines": [e.__dict__ for e in engines]})
    logger.info("outputs in %s", out)


if __name__ == "__main__":
    app()
