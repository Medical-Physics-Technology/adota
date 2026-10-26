"""Compare two retrospective active-learning experiments run by run (EXP-0012 against EXP-0013).

    uv run python scripts/al_compare_experiments.py --config scripts/config_al_compare_experiments.yaml

The two experiments must share the splits, the cycle-0 checkpoint, the evaluation
subsample and the training set size at every cycle (the check of
``scripts/al_compare.py`` over all runs of both), and every ``(strategy, seed)``
run of one must have a partner in the other. Writes, into a timestamped directory
under ``output_dir``:

- ``selection_identity.csv``: per pair and cycle, whether both selected the same
  records in the same order; the pairs differ only in the training setting when
  they did, and the script warns when they did not;
- ``cycle_table.csv``: per run and cycle, the full-V boundary metrics, the median
  of the last ``trajectory_last_k`` subsample evaluations, the snapshot noise and
  the divergence flag;
- ``paired_deltas.csv`` and ``paired_summary.csv``: B minus A per pair and cycle,
  and its mean, range and count of pairs where B was better;
- ``figures/``: G1 schedule and validation loss, G2 paired quality, G3 every run
  at the last cycle by strategy, G4 stability, each with the CSV it draws.

Figures come from :mod:`src.figures.al_experiment_comparison`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import pandas as pd
import typer

from src.active_learning.retrospective.compare import check_consistency, read_run, unique_labels
from src.active_learning.retrospective.experiment_compare import (
    cycle_table,
    loss_curves,
    pair_runs,
    paired_deltas,
    selection_identity,
    strategy_runs,
    summarize_deltas,
)
from src.evaluation.cli import load_yaml_config, merge_config, setup_logging, setup_run_directory
from src.figures.al_experiment_comparison import (
    gamma_text,
    paired_quality_figure,
    schedule_and_loss_figure,
    stability_figure,
    strategy_endpoint_figure,
)

app = typer.Typer(help="Compare two retrospective active-learning experiments.", add_completion=False)
logger = logging.getLogger("al_compare_experiments")

DEFAULTS = {"experiments": [], "output_dir": "/scratch/mstryja/adota_runs/al_retro/compare_experiments",
            "title": "Retrospective active learning", "trajectory_last_k": 3, "divergence_ratio": 2.0}

PAIRED_PANELS = (("traj_gpr_mean", "Mean GPR, median of last 3 evals", "Gamma pass rate [%]", 100.0),
                 ("full_gpr_mean", "Mean GPR, cycle end", "Gamma pass rate [%]", 100.0),
                 ("full_gpr_p05", "5th percentile GPR, cycle end", "Gamma pass rate [%]", 100.0))
ENDPOINT_PANELS = PAIRED_PANELS[1::-1]
SUMMARY_METRICS = ("traj_gpr_mean", "traj_gpr_p05", "full_gpr_mean", "full_gpr_p05",
                   "full_gpr_frac_below_95", "full_mape_pct_mean", "full_abs_dr80_p95_mm", "val_loss",
                   "val_last_over_min", "val_peak_over_min")
LOWER_IS_BETTER = ("full_gpr_frac_below_95", "full_mape_pct_mean", "full_abs_dr80_p95_mm", "val_loss",
                   "val_last_over_min", "val_peak_over_min")


def _experiment_runs(spec: Dict) -> List:
    runs_dir = Path(spec["runs_dir"])
    dirs = sorted(runs_dir.glob(spec.get("pattern", "train_*")))
    if not dirs:
        raise typer.BadParameter(f"no run directories match {spec.get('pattern', 'train_*')} in {runs_dir}")
    return unique_labels(strategy_runs([read_run(d) for d in dirs]))


@app.command()
def main(
    config: Annotated[Path, typer.Option(help="YAML config naming the two experiments.")]
    = Path("scripts/config_al_compare_experiments.yaml"),
    output_dir: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    raw = load_yaml_config(config) if Path(config).exists() else {}
    cfg = merge_config({"output_dir": str(output_dir) if output_dir else None}, raw, DEFAULTS)
    specs = cfg["experiments"]
    if len(specs) != 2:
        raise typer.BadParameter(f"exactly two experiments are compared, the config names {len(specs)}")
    names = [str(s["name"]) for s in specs]
    legend = {str(s["name"]): str(s.get("legend", s["name"])) for s in specs}

    out = setup_run_directory(Path(cfg["output_dir"]), prefix="al_compare_experiments_",
                              subdirs=("figures",))
    setup_logging(out, log_filename="al_compare_experiments.log")
    runs = {name: _experiment_runs(spec) for name, spec in zip(names, specs)}
    consistency = check_consistency(runs[names[0]] + runs[names[1]])
    pairs = pair_runs(runs[names[0]], runs[names[1]])
    identity = selection_identity(pairs)
    identity.to_csv(out / "selection_identity.csv", index=False)
    identical = bool(identity["selection_identical"].all() and identity["training_set_identical"].all())
    if not identical:
        logger.warning("selections differ in %d of %d run-cycles: the pairs differ in more than the "
                       "training setting", int((~identity["selection_identical"]).sum()), len(identity))

    last_k = int(cfg["trajectory_last_k"])
    tables = {name: cycle_table(runs[name], name, last_k=last_k,
                                divergence_ratio=float(cfg["divergence_ratio"])) for name in names}
    table = pd.concat(tables.values(), ignore_index=True)
    table = table[table["strategy"] != "cycle0"] if (table["cycle"] > 0).any() else table
    table.to_csv(out / "cycle_table.csv", index=False)
    scored = {n: t[t["cycle"] >= 1] for n, t in tables.items()}
    paired = paired_deltas(scored[names[0]], scored[names[1]], SUMMARY_METRICS)
    paired.to_csv(out / "paired_deltas.csv", index=False)
    summary = summarize_deltas(paired, SUMMARY_METRICS, lower_is_better=LOWER_IS_BETTER)
    summary.to_csv(out / "paired_summary.csv", index=False)
    diverged = {n: int(t.loc[t["cycle"] >= 1, "diverged"].sum()) for n, t in tables.items()}
    (out / "consistency.json").write_text(json.dumps(
        {**{k: v for k, v in consistency.items() if k not in ("labels", "strategies", "seeds")},
         "experiments": names, "n_pairs": len(pairs), "selections_identical": identical,
         "diverged_run_cycles": diverged}, indent=2, default=str))

    gamma = str(consistency["gamma_label"])
    first = runs[names[0]][0]
    n_seeds = len({k[1] for k in pairs})
    base = (f"{len(pairs)} run pairs ({len({k[0] for k in pairs})} strategies x {n_seeds} seeds), "
            f"same selections in both experiments. Gamma {gamma_text(gamma)}.")
    figures = out / "figures"
    title = cfg["title"]

    curves = pd.concat([loss_curves(runs[n], n) for n in names], ignore_index=True)
    curves.to_csv(figures / "G1_schedule_and_val_loss.csv", index=False)
    boundaries = first.boundaries()
    shared = boundaries.loc[boundaries["cycle"] == 0, "cumulative_epoch"]
    schedule_and_loss_figure(curves, boundaries, names, legend,
                             str(figures / "G1_schedule_and_val_loss"),
                             title=f"{title}: learning-rate schedule and validation loss",
                             caption=(f"{len(pairs)} runs per experiment ({len({k[0] for k in pairs})} "
                                      f"strategies x {n_seeds} seeds) on the same selections; cycle 0 is "
                                      f"one shared run. Validation loss every epoch."),
                             shared_until=int(shared.iloc[0]) if not shared.empty else None)

    stem = f"G2_paired_quality_gamma_{gamma}"
    paired_quality_figure(table[table["cycle"] >= 1], paired, PAIRED_PANELS, names, legend,
                          str(figures / stem), title=f"{title}: {names[1]} against {names[0]}, run by run",
                          caption=(f"{base} Top: mean across runs, band = min to max. Bottom: one dot per "
                                   f"pair, bar = mean, count = pairs where {names[1]} is higher.\n"
                                   f"Last 3 evals: median of the last {last_k} evaluations of the cycle "
                                   f"on the {first.manifest['inputs']['eval_subsample_size']}-record "
                                   f"subsample. Cycle end: full validation set at the boundary epoch."))
    table[["experiment", "strategy", "seed", "cycle", "n_train"] + [p[0] for p in PAIRED_PANELS]] \
        .to_csv(figures / f"{stem}.csv", index=False)

    last = int(table["cycle"].max())
    stem = f"G3_strategies_cycle{last}_gamma_{gamma}"
    strategy_endpoint_figure(table, last, ENDPOINT_PANELS, names, legend, str(figures / stem),
                             title=f"{title}: every run at cycle {last}",
                             caption=f"{base} One marker per run, bar = mean of the {n_seeds} seeds.")
    table.loc[table["cycle"] == last, ["experiment", "strategy", "seed", "cycle"]
              + [p[0] for p in ENDPOINT_PANELS]].to_csv(figures / f"{stem}.csv", index=False)

    stability_figure(table, names, legend, str(figures / "G4_stability"),
                     title=f"{title}: training stability",
                     caption=(f"{base}\nDivergence: training loss rising more than "
                              f"{float(cfg['divergence_ratio']):g}-fold between two epochs, the step "
                              f"into the cycle included. Right: validation loss at the boundary epoch "
                              f"over the cycle's minimum, bar = median."))
    table[["experiment", "strategy", "seed", "cycle", "diverged", "max_rise", "val_last_over_min",
           "val_peak_over_min"]].to_csv(figures / "G4_stability.csv", index=False)

    typer.echo(f"comparison written to {out}")
    typer.echo(f"selections identical: {identical}; diverged run-cycles: {diverged}")
    typer.echo(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    app()
