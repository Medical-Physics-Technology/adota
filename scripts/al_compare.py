"""Compare the strategy runs of the retrospective benchmark (EXP-0009 to EXP-0011).

    uv run python scripts/al_compare.py --config scripts/config_al_compare.yaml
    uv run python scripts/al_compare.py --run /scratch/.../train_<ts>_al_EXP-0011_random_seed1234 \
        --run /scratch/.../train_<ts>_al_EXP-0011_score_topk_seed1234 --output-dir /scratch/...

Reads each run's ``metrics.jsonl`` and ``manifest.json``, checks that the runs
are comparable (same splits, same cycle-0 checkpoint, same evaluation subsample,
same training set size at every cycle), and writes:

- ``F1_training_curves``: loss against the cumulative epoch with cycle rules;
- ``F2_quality_vs_n_train`` and ``F3_quality_vs_epochs``: gamma pass rate, its
  tails, MAPE and dR80 against the training set size and the cumulative epoch;
- ``F4_fingerprint_score_decile`` / ``F4_fingerprint_energy``: what each
  strategy selected against the pool;
- ``summary_boundaries.csv``, ``epochs_to_quality.csv``, ``consistency.json``,
  and a CSV of the numbers beside every figure;
- ``divergences.csv``: per run and cycle, the largest epoch-to-epoch rise of the
  training loss and whether it crossed the ``divergence_ratio`` (a flagged
  cycle, and everything after it, measures the recovery, not the data);
- ``summary_trajectory_last<k>.csv``: the median of the last ``trajectory_last_k``
  subsample evaluations of every cycle, the boundary estimate that does not
  depend on which epoch the cycle happened to end on.

When a strategy was run under several seeds (the same run directories, several
``--run`` entries), the runs are labelled ``<strategy>_seed<seed>``, and the
boundary and trajectory summaries plus F2 and F3 are written a second time
``_by_strategy``: mean across seeds with a min-max band.

Figures come from :mod:`src.figures.al_curves`; every output goes to the config's
``output_dir`` on scratch.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import pandas as pd
import typer

from src.active_learning.retrospective.compare import (
    DEFAULT_THRESHOLDS,
    aggregate_over_seeds,
    boundary_table,
    check_consistency,
    divergence_table,
    epochs_to_quality,
    fingerprint_table,
    fingerprints,
    quality_curves,
    read_run,
    trajectory_table,
    unique_labels,
)
from src.evaluation.cli import load_yaml_config, merge_config, setup_logging, setup_run_directory
from src.figures.al_curves import (
    quality_curves_figure,
    selection_fingerprint_figure,
    training_curves_figure,
)

app = typer.Typer(help="Compare retrospective active-learning runs.", add_completion=False)
logger = logging.getLogger("al_compare")

DEFAULTS = {"runs": [], "labels": [], "output_dir": "/scratch/mstryja/adota_runs/al_retro/compare",
            "thresholds": None, "title": "Retrospective active learning (EXP-0011)",
            "divergence_ratio": 2.0, "trajectory_last_k": 3}


def _thresholds(raw: Optional[Dict]) -> Dict[str, Tuple[float, str]]:
    if not raw:
        return dict(DEFAULT_THRESHOLDS)
    return {metric: (float(spec["value"]), str(spec.get("direction", "ge")))
            for metric, spec in raw.items()}


@app.command()
def main(
    config: Annotated[Optional[Path], typer.Option(help="YAML config with runs and output_dir.")]
    = Path("scripts/config_al_compare.yaml"),
    run: Annotated[Optional[List[Path]], typer.Option(help="Run directory; repeatable.")] = None,
    label: Annotated[Optional[List[str]], typer.Option(help="Label per run; repeatable.")] = None,
    output_dir: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    raw = load_yaml_config(config) if config and Path(config).exists() else {}
    cfg = merge_config({"runs": [str(r) for r in run] if run else None,
                        "labels": label if label else None,
                        "output_dir": str(output_dir) if output_dir else None}, raw, DEFAULTS)
    if not cfg["runs"]:
        raise typer.BadParameter("no runs given: pass --run or set `runs` in the config")
    labels = list(cfg["labels"] or [])
    if labels and len(labels) != len(cfg["runs"]):
        raise typer.BadParameter("labels and runs must have the same length")

    out = setup_run_directory(Path(cfg["output_dir"]), prefix="al_compare_", subdirs=("figures",))
    setup_logging(out, log_filename="al_compare.log")
    runs = unique_labels([read_run(Path(r), labels[i] if labels else None)
                          for i, r in enumerate(cfg["runs"])])
    consistency = check_consistency(runs)
    divergences = divergence_table(runs, ratio=float(cfg["divergence_ratio"]))
    divergences.to_csv(out / "divergences.csv", index=False)
    flagged = divergences[divergences["diverged"]] if not divergences.empty else divergences
    consistency["diverged"] = {label: [int(c) for c in group["cycle"]]
                               for label, group in flagged.groupby("label")}
    (out / "consistency.json").write_text(json.dumps(consistency, indent=2))
    logger.info("comparing %d runs: %s", len(runs), consistency["labels"])
    for _, row in flagged.iterrows():
        logger.warning("%s diverged in cycle %d at epoch %d: training loss %.4g -> %.4g "
                       "(x%.1f); that cycle and the ones after it measure the recovery",
                       row["label"], row["cycle"], row["epoch_in_cycle"],
                       row["train_loss_before"], row["train_loss_after"], row["max_rise"])

    gamma_label = str(consistency["gamma_label"])
    boundaries = runs[0].boundaries()
    caption = (f"Gamma {gamma_label.replace('_', ' ')}; the subsample of "
               f"{runs[0].manifest['inputs']['eval_subsample_size']} validation records every "
               f"{runs[0].manifest['config']['eval_every_n_epochs']} epochs, the full validation "
               f"set at the cycle boundaries (markers).")
    figures = out / "figures"

    summary = boundary_table(runs)
    summary.to_csv(out / f"summary_boundaries_gamma_{gamma_label}.csv", index=False)
    last_k = int(cfg["trajectory_last_k"])
    trajectory = trajectory_table(runs, last_k=last_k)
    trajectory.to_csv(out / f"summary_trajectory_last{last_k}_gamma_{gamma_label}.csv", index=False)
    e2q = epochs_to_quality(runs, _thresholds(cfg["thresholds"]))
    e2q.to_csv(out / "epochs_to_quality.csv", index=False)

    losses = {r.label: r.rows[["cumulative_epoch", "cycle", "n_train", "train_loss", "val_loss"]]
              for r in runs}
    for r in runs:
        losses[r.label].assign(label=r.label).to_csv(figures / f"F1_training_curves_{r.label}.csv",
                                                     index=False)
    training_curves_figure(losses, boundaries, str(figures / "F1_training_curves"),
                           title=f"{cfg['title']}: training curves")

    curves = quality_curves(runs)
    for r in runs:
        curves[r.label].to_csv(figures / f"F2_F3_quality_{r.label}_gamma_{gamma_label}.csv",
                               index=False)
    boundary_caption = (f"Gamma {gamma_label.replace('_', ' ')}; the full validation set at the "
                        f"cycle boundaries.")
    quality_curves_figure(curves, x="n_train", x_label="Training records", prefix="full_",
                          gamma_caption=boundary_caption,
                          figure_path=str(figures / f"F2_quality_vs_n_train_gamma_{gamma_label}"),
                          title=cfg["title"])
    quality_curves_figure(curves, x="cumulative_epoch", x_label="Cumulative epoch",
                          gamma_caption=caption, boundaries=boundaries,
                          figure_path=str(figures / f"F3_quality_vs_epochs_gamma_{gamma_label}"),
                          title=cfg["title"])

    seeded = len({r.strategy for r in runs}) < len(runs)
    if seeded:
        by_strategy = aggregate_over_seeds(summary)
        by_strategy.to_csv(out / f"summary_boundaries_by_strategy_gamma_{gamma_label}.csv",
                           index=False)
        aggregate_over_seeds(trajectory).to_csv(
            out / f"summary_trajectory_last{last_k}_by_strategy_gamma_{gamma_label}.csv",
            index=False)
        stacked = pd.concat([frame.assign(strategy=r.strategy) for r in runs
                             for frame in [curves[r.label]]], ignore_index=True)
        agg = aggregate_over_seeds(stacked, keys=("strategy", "cumulative_epoch"),
                                   carry=("cycle", "n_train", "cycle_boundary"))
        agg_curves = {name: group.sort_values("cumulative_epoch").reset_index(drop=True)
                      for name, group in agg.groupby("strategy", sort=False)}
        for name, frame in agg_curves.items():
            frame.to_csv(figures / f"F2_F3_quality_by_strategy_{name}_gamma_{gamma_label}.csv",
                         index=False)
        n_seeds = ", ".join(f"{s}: {n}" for s, n in
                            sorted(pd.Series([r.strategy for r in runs]).value_counts().items()))
        band = f" Mean across seeds ({n_seeds}), band = min to max."
        quality_curves_figure(agg_curves, x="n_train", x_label="Training records", prefix="full_",
                              gamma_caption=boundary_caption + band,
                              figure_path=str(figures / f"F2_quality_vs_n_train_by_strategy_gamma_{gamma_label}"),
                              title=cfg["title"])
        quality_curves_figure(agg_curves, x="cumulative_epoch", x_label="Cumulative epoch",
                              gamma_caption=caption + band, boundaries=boundaries,
                              figure_path=str(figures / f"F3_quality_vs_epochs_by_strategy_gamma_{gamma_label}"),
                              title=cfg["title"])

    prints = fingerprints(runs)
    for key, key_label in (("score_decile", "score decile (0 easiest, 9 hardest)"),
                           ("energy", "energy bin [MeV]")):
        table = fingerprint_table(prints, key)
        if table.empty:
            logger.warning("no %s fingerprint in any run", key)
            continue
        table.to_csv(figures / f"F4_fingerprint_{key}.csv", index=False)
        selection_fingerprint_figure(table, figure_path=str(figures / f"F4_fingerprint_{key}"),
                                     key_label=key_label,
                                     title=f"{cfg['title']}: selection over {key_label}")
    typer.echo(f"comparison written to {out}")
    if not flagged.empty:
        typer.echo("DIVERGED: " + "; ".join(f"{r.label} cycle {r.cycle} epoch {r.epoch_in_cycle} "
                                            f"(x{r.max_rise:.1f})" for r in flagged.itertuples()))
    typer.echo(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    app()
